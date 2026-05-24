"""
SDec-Tiger reward-misalignment experiment (fully decentralized RSSDA).

True reward function = canonical SDec-Tiger reward (split openings = -100).
Proxy reward function = identical except split openings = -30.

We optimize a fully-decentralized policy under the proxy, then evaluate the
resulting policy under BOTH reward functions. For comparison we also solve
under the true reward and evaluate that policy on both.

"Split opening" = one agent opens the tiger door while the other opens the
safe door simultaneously (no one listens). In the canonical reward this is
the worst joint outcome (-100); the proxy makes it look milder (-30).

Evaluation method: exact tree-walk over (belief, agent-1 cluster,
agent-2 cluster) using the (policy, clustering) returned by the RSSDA
solver. No Monte Carlo, no internal Policy object needed.
"""

import math
import os
import sys
import time
from array import array

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
sys.path.insert(0, _here)
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "baselines"))

import sdec_tiger
from RSSDA import SDecPOMDP, SDecPOMDPModel, RSSDAConfig, int_tuple


# ---------------------------------------------------------------------------
# Problem construction
# ---------------------------------------------------------------------------

def build_tiger_arrays(split_penalty, obs_accuracy=0.85, lam=1.0):
    """Build (T, O, R, init_b, nacts_fac, nobs_fac) for SDec-Tiger.

    Equivalent to TigerProblemFactory.generate() but with the split-opening
    coefficient parameterized. All other reward terms (-2, -50, +9, +20,
    -101) are kept at their canonical values.
    """
    nagents = 2
    nstates = 2
    act_per_agent = 3   # OL=0, OR=1, Li=2
    obs_per_agent = 2
    nacts = act_per_agent ** nagents   # 9
    nobs = obs_per_agent ** nagents    # 4
    nsq = nstates ** 2
    nso = nstates * nobs

    transit = [0.0] * (nsq * nacts)
    obs = [0.0] * (nobs * nstates * nacts)
    reward = [0.0] * (nstates * nacts)

    p_corr = obs_accuracy
    p_wrong = 1.0 - obs_accuracy

    for a in range(nacts):
        a1 = a % 3
        a2 = a // 3
        if a == 8:  # both listen
            for s in range(nstates):
                reward[a * nstates + s] = -2
                for t in range(nstates):
                    transit[a * nsq + s * nstates + t] = 1.0 if s == t else 0.0
        else:
            for s in range(nstates):
                # canonical reward formula, with -100 -> split_penalty
                r_val = (
                    -101 * ((a1 == s) * (a2 == 2) + (a2 == s) * (a1 == 2))
                    - 50 * ((a1 == s) * (a2 == s))
                    + split_penalty * ((a1 != 2) * (a1 != s) * (a2 == s)
                                       + (a2 != 2) * (a2 != s) * (a1 == s))
                    + 9 * ((a1 != 2) * (a1 != s) * (a2 == 2)
                           + (a2 != 2) * (a2 != s) * (a1 == 2))
                    + 20 * ((a1 != 2) * (a1 != s) * (a2 != 2) * (a2 != s))
                )
                reward[a * nstates + s] = r_val
                for t in range(nstates):
                    transit[a * nsq + s * nstates + t] = 0.5
            # opening actions: uniform observations
            for t in range(nstates):
                for o in range(nobs):
                    obs[a * nso + t * nobs + o] = 0.25

    # joint-listen (action 8): lam-mix of informative product and uniform
    for t in range(nstates):
        for o in range(nobs):
            b1, b2 = o % 2, o // 2
            p1 = p_corr if b1 == t else p_wrong
            p2 = p_corr if b2 == t else p_wrong
            informative = p1 * p2
            obs[8 * nso + t * nobs + o] = lam * informative + (1 - lam) * 0.25

    # Single-listener observations: uniform (we are in dec mode, AAMAS Mod C
    # gives single-listener informative ONLY for "semi"). Already set above
    # to uniform 0.25 by the opening-action loop, leave as-is.

    init_b = [0.5, 0.5]
    nacts_fac = [act_per_agent] * nagents
    nobs_fac = [obs_per_agent] * nagents
    return transit, obs, reward, init_b, nacts_fac, nobs_fac


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

def solve_tiger(split_penalty, horizon, obs_accuracy=0.85):
    """Solve fully-decentralized RSSDA at horizon `h` for a given split penalty.

    Returns (val, policy, clustering, model_arrays) where model_arrays is
    (T, O, R, nstates, nactions, nobs, act_per_agent, obs_per_agent).
    """
    T, O, R, init_b, nacts_fac, nobs_fac = build_tiger_arrays(
        split_penalty=split_penalty, obs_accuracy=obs_accuracy, lam=1.0,
    )
    model = SDecPOMDPModel(
        nagents=2, nstates=2, nactions=9, nobs=4,
        transitions=T, obs=O, rewards=R, init_beliefs=init_b,
        nacts_factor=nacts_fac, nobs_factor=nobs_fac,
        sync_states=[], sync_actions=[], sync_observations=[],
    )
    cfg = RSSDAConfig(
        maxh=horizon, maxit=200, IEmin2=3, alpha=0.2,
        algorithm="exact", heuristic_type="POMDP", tail_heuristic_type="POMDP",
        TI1=False, TI2=False, TI3=False, TI4=False,
        score_limit=20, cen_threshold=0.6, sm_temperature=0.6,
        iter_limit=1000, rec_limit=2, hybrid_r=1, max_clusters=10,
    )
    solver = SDecPOMDP(model=model, config=cfg)
    init_idx = solver.dist_dict[int_tuple(solver.init_beliefs)]
    val, policy, clustering, cent_vec, _, _ = solver.multi_agent_astar(
        horizon, init_beliefs=init_idx,
    )
    arrays = (T, O, R, 2, 9, 4, 3, 2)
    return val, policy, clustering, arrays


# ---------------------------------------------------------------------------
# Exact tree-walk policy evaluator (independent of solver internals)
# ---------------------------------------------------------------------------

def evaluate_policy_under(policy, clustering, arrays, R_eval, init_belief=(0.5, 0.5)):
    """Compute exact expected return of a fully-decentralized policy under R_eval.

    Walks the joint state distribution belief * agent-1 cluster *
    agent-2 cluster forward through the horizon. R_eval is indexed
    R_eval[a * nstates + s] (joint action a, state s).
    """
    T, O, _, nstates, nactions, nobs, act_per_agent, obs_per_agent = arrays
    nsq = nstates * nstates
    nso = nstates * nobs
    horizon = len(policy)

    def recurse(step, belief, c1, c2):
        if step >= horizon:
            return 0.0

        # Agent actions for current cluster pair
        dec = policy[step][0]
        # Cluster index may exceed list length when zero-prob branch was
        # never expanded; treat as -1 sentinel (skip).
        if c1 < 0 or c2 < 0:
            return 0.0
        if c1 >= len(dec[0]) or c2 >= len(dec[1]):
            return 0.0
        a1 = dec[0][c1]
        a2 = dec[1][c2]
        if a1 < 0 or a2 < 0:
            return 0.0
        a = a1 + act_per_agent * a2

        # Immediate expected reward
        ir = sum(belief[s] * R_eval[a * nstates + s] for s in range(nstates))

        # Post-action state distribution (before observation)
        post = [
            sum(belief[s] * T[a * nsq + s * nstates + sp] for s in range(nstates))
            for sp in range(nstates)
        ]

        future = 0.0
        for o in range(nobs):
            p_o = sum(post[sp] * O[a * nso + sp * nobs + o] for sp in range(nstates))
            if p_o < 1e-15:
                continue
            new_belief = tuple(
                post[sp] * O[a * nso + sp * nobs + o] / p_o for sp in range(nstates)
            )
            o1 = o % obs_per_agent
            o2 = o // obs_per_agent
            if step < len(clustering) and len(clustering[step]) >= 2:
                cl1, cl2 = clustering[step][0], clustering[step][1]
                new_c1 = cl1[c1][o1] if c1 < len(cl1) and o1 < len(cl1[c1]) else -1
                new_c2 = cl2[c2][o2] if c2 < len(cl2) and o2 < len(cl2[c2]) else -1
            else:
                new_c1, new_c2 = 0, 0
            future += p_o * recurse(step + 1, new_belief, new_c1, new_c2)

        return ir + future

    return recurse(0, tuple(init_belief), 0, 0)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def main():
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    obs_accuracy = float(sys.argv[2]) if len(sys.argv) > 2 else 0.85
    true_pen = -100
    proxy_pen = -30

    print(f"SDec-Tiger reward-misalignment experiment (fully decentralized RSSDA)")
    print(f"horizon = {horizon}   obs_accuracy = {obs_accuracy}")
    print(f"true split penalty  = {true_pen}")
    print(f"proxy split penalty = {proxy_pen}")
    print()

    # --- Solve under proxy reward ---
    t0 = time.time()
    val_proxy_solver, pi_proxy, cl_proxy, arr_proxy = solve_tiger(
        proxy_pen, horizon, obs_accuracy,
    )
    t_proxy = time.time() - t0
    print(f"[solve] proxy   val = {val_proxy_solver:.5f}   t = {t_proxy:.2f}s")

    # --- Solve under true reward ---
    t0 = time.time()
    val_true_solver, pi_true, cl_true, arr_true = solve_tiger(
        true_pen, horizon, obs_accuracy,
    )
    t_true = time.time() - t0
    print(f"[solve] true    val = {val_true_solver:.5f}   t = {t_true:.2f}s")
    print()

    # Build true and proxy reward arrays once, for cross-evaluation.
    R_true = arr_true[2]
    R_proxy = arr_proxy[2]
    # Use arr_proxy for proxy policy evaluation, arr_true for true policy.
    # T and O are identical between the two builds, only R differs.

    # --- Cross-evaluate ---
    v_proxy_under_proxy = evaluate_policy_under(pi_proxy, cl_proxy, arr_proxy, R_proxy)
    v_proxy_under_true  = evaluate_policy_under(pi_proxy, cl_proxy, arr_proxy, R_true)
    v_true_under_proxy  = evaluate_policy_under(pi_true,  cl_true,  arr_true,  R_proxy)
    v_true_under_true   = evaluate_policy_under(pi_true,  cl_true,  arr_true,  R_true)

    print("Exact policy values (tree-walk evaluation)")
    print(f"  pi_proxy  evaluated on R_proxy = {v_proxy_under_proxy:>+10.5f}   "
          f"(solver said {val_proxy_solver:>+10.5f})")
    print(f"  pi_proxy  evaluated on R_true  = {v_proxy_under_true:>+10.5f}")
    print(f"  pi_true   evaluated on R_proxy = {v_true_under_proxy:>+10.5f}")
    print(f"  pi_true   evaluated on R_true  = {v_true_under_true:>+10.5f}   "
          f"(solver said {val_true_solver:>+10.5f})")
    print()

    # --- Misalignment summary ---
    # The proxy policy is what an agent that mistakenly trusts the proxy
    # reward function would deploy. The "alignment cost" is the gap, on the
    # true reward, between deploying pi_proxy and deploying the true-optimal
    # pi_true.
    alignment_cost = v_true_under_true - v_proxy_under_true
    print("Misalignment summary")
    print(f"  optimal value under true R               = {v_true_under_true:>+10.5f}")
    print(f"  value of proxy-optimal policy on true R  = {v_proxy_under_true:>+10.5f}")
    print(f"  alignment cost (true - proxy_on_true)    = {alignment_cost:>+10.5f}")
    if alignment_cost > 1e-6:
        print("  -> proxy and true rewards induce DIFFERENT optimal policies "
              "at this horizon.")
    else:
        print("  -> proxy and true rewards induce equivalent optimal policies "
              "(no misalignment penalty at this horizon).")


if __name__ == "__main__":
    main()
