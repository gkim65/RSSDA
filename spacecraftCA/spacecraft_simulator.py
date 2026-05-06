"""
spacecraft_simulator.py

Closed-loop simulation of the spacecraft CA SDec-POMDP policy.

Supports two policy modes:
  - "rssda": Full RS-SDA* policy from multi_agent_astar (default)
  - "greedy": One-step lookahead greedy (baseline/fallback)

Each rollout:
  - Initializes true ECI states from a radial-miss conjunction
  - Steps through N_STAGES decision epochs, applying the policy
  - At each stage: simulates noisy observation, updates belief, takes action
  - At TCA: measures true miss distance with Brahe

Usage:
  python spacecraft_simulator.py                          # RS-SDA* policy, bin 4 init
  python spacecraft_simulator.py --policy greedy          # greedy baseline
  python spacecraft_simulator.py --init-miss-bin 0        # start in collision zone
  python spacecraft_simulator.py --rollouts 100 --dv 0.5
  python spacecraft_simulator.py --compare                # all 3 variants side by side
"""

import os
import sys
import argparse
import math
import numpy as np
from typing import List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from brahe import (
    initialize_eop,
    state_rtn_to_eci, state_eci_to_rtn,
)

from spacecraft_matrices import (
    load_matrices, DV_MAGNITUDE,
    STAGE_EPOCHS, EPOCH_TCA, V_REL_MS,
    N_JOINT_ACTIONS, N_ACT_AGENT, OBS_SIGMA_KM,
    split_joint_action, apply_maneuver, propagate, sc1_eci_at_tca,
    CONTACT_STAGES,
)
from spacecraft_discretizer import (
    N_STAGES, N_STATES, N_STATES_TOTAL, SINK_STATE,
    N_MISS, MISS_EDGES_KM,
    miss_to_bin, bin_center_km,
    state_index, index_to_state,
    sync_trigger_states,
)
from RSSDA import int_tuple

ACT_NAMES = {0: 'WAIT', 1: '+dVT', 2: '-dVT'}
MISS_LABELS = ['COLLISION', 'HIGH', 'MOD', 'LOW', 'NOM', 'SAFE']


# ---------------------------------------------------------------------------
# Greedy policy (baseline)
# ---------------------------------------------------------------------------

def build_greedy_policy(T, R) -> np.ndarray:
    """One-step lookahead greedy: argmax_a [R(a,s) + max_a' R(a', T(s,a))]."""
    policy = np.zeros(N_STATES, dtype=int)
    for s in range(N_STATES):
        best_a, best_q = 0, -1e9
        for a in range(N_JOINT_ACTIONS):
            r = R[a, s]
            next_states = np.where(T[a, s, :] > 0)[0]
            r_next = max((R[:, sp].max() for sp in next_states if sp < N_STATES), default=0.0)
            q = r + r_next
            if q > best_q:
                best_q = q
                best_a = a
        policy[s] = best_a
    return policy


# ---------------------------------------------------------------------------
# Observation simulation
# ---------------------------------------------------------------------------

def simulate_obs(true_miss_km: float, sigma_km: float = None,
                 rng: np.random.Generator = None) -> int:
    if sigma_km is None:
        sigma_km = OBS_SIGMA_KM
    if rng is None:
        rng = np.random.default_rng()
    noisy = max(0.0, rng.normal(true_miss_km, sigma_km))
    return miss_to_bin(noisy)


# ---------------------------------------------------------------------------
# Belief update (for greedy mode)
# ---------------------------------------------------------------------------

def belief_update(belief: np.ndarray, action: int, obs_joint: int,
                  T: np.ndarray, O: np.ndarray) -> np.ndarray:
    b_new = O[action, :, obs_joint] * (T[action].T @ belief)
    total = b_new.sum()
    return b_new / total if total > 1e-15 else belief.copy()


# ---------------------------------------------------------------------------
# RS-SDA* policy action extraction
# ---------------------------------------------------------------------------

def get_rssda_action(policy, cen_dists_map, clustering, clustering_cen,
                     step: int, current_belief_idx: int, current_oh: List[int],
                     act_per_agent: int):
    """
    Extract the joint action from the RS-SDA* policy at a given step.
    Returns (joint_act, a1, a2, is_centralized).
    Mirrors the logic in sdec_mars.py.
    """
    is_centralized = False
    c_ptr = -1

    if step < len(cen_dists_map):
        dists_at_step = cen_dists_map[step]
        if current_belief_idx in dists_at_step:
            is_centralized = True
            c_ptr = dists_at_step.index(current_belief_idx)

    try:
        if not is_centralized:
            act1 = policy[step][0][0][current_oh[0]]
            act2 = policy[step][0][1][current_oh[1]]
            joint_act = act1 + act2 * act_per_agent
        else:
            joint_act = policy[step][1][c_ptr][0]
            act1 = joint_act % act_per_agent
            act2 = joint_act // act_per_agent
    except (IndexError, TypeError):
        # Policy doesn't cover this branch — fall back to WAIT
        joint_act, act1, act2 = 0, 0, 0

    return joint_act, act1, act2, is_centralized


def update_oh_rssda(policy, cen_dists_map, clustering, clustering_cen,
                    step: int, current_belief_idx: int, current_oh: List[int],
                    is_centralized: bool, c_ptr: int, o1: int, o2: int) -> List[int]:
    """Update observation history indices after taking action and receiving obs."""
    try:
        if is_centralized:
            if len(clustering_cen) > step and len(clustering_cen[step]) > 0:
                new_oh_0 = clustering_cen[step][0][c_ptr][o1]
                new_oh_1 = clustering_cen[step][1][c_ptr][o2]
                return [new_oh_0, new_oh_1]
            return [0, 0]
        else:
            next_oh0 = clustering[step][0][current_oh[0]][o1]
            next_oh1 = clustering[step][1][current_oh[1]][o2]
            if next_oh0 == -1 or next_oh1 == -1:
                return [0, 0]
            return [next_oh0, next_oh1]
    except (IndexError, TypeError, KeyError):
        return [0, 0]


# ---------------------------------------------------------------------------
# Single rollout — RS-SDA* policy
# ---------------------------------------------------------------------------

def rollout_rssda(
    T, O, R, sdec,
    full_result,          # (value, policy, clustering, cent_vector, cen_dists_map, clustering_cen)
    contact_stages: List[int],
    true_init_miss_km: float,
    dv_mag: float,
    rng: np.random.Generator,
) -> dict:
    """Closed-loop rollout using the full RS-SDA* policy."""
    value, policy, clustering, cent_vector, cen_dists_map, clustering_cen = full_result

    sc1_tca_eci = sc1_eci_at_tca()
    sc1_true = propagate(EPOCH_TCA, sc1_tca_eci, STAGE_EPOCHS[0])
    sc2_tca = np.array(state_rtn_to_eci(sc1_tca_eci,
                       np.array([true_init_miss_km * 1e3, 0.0, 0.0, 0.0, -V_REL_MS, 0.0])))
    sc2_true = propagate(EPOCH_TCA, sc2_tca, STAGE_EPOCHS[0])

    # Track belief by index in RSSDA's belief dict
    current_belief_idx = sdec.dist_dict[int_tuple(sdec.init_beliefs)]
    current_oh = [0, 0]  # decentralized observation history indices per agent

    total_dv = 0.0
    sync_count = 0
    maneuver_stages = []
    miss_bin_traj = []

    for k in range(N_STAGES):
        if k > 0:
            sc1_true = propagate(STAGE_EPOCHS[k - 1], sc1_true, STAGE_EPOCHS[k])
            sc2_true = propagate(STAGE_EPOCHS[k - 1], sc2_true, STAGE_EPOCHS[k])

        # True miss at TCA from this stage
        sc1_proj = propagate(STAGE_EPOCHS[k], sc1_true, EPOCH_TCA)
        sc2_proj = propagate(STAGE_EPOCHS[k], sc2_true, EPOCH_TCA)
        rtn_tca  = np.array(state_eci_to_rtn(sc1_proj, sc2_proj))
        true_miss_km = np.linalg.norm(rtn_tca[:3]) / 1e3
        miss_bin_traj.append(miss_to_bin(true_miss_km))

        is_sync = k in contact_stages
        obs1 = obs2 = 0
        if is_sync:
            sync_count += 1
            obs1 = simulate_obs(true_miss_km, rng=rng)
            obs2 = simulate_obs(true_miss_km, rng=rng)

        obs_joint = obs1 * N_MISS + obs2

        # Get action from RS-SDA* policy
        joint_act, a1, a2, is_cen = get_rssda_action(
            policy, cen_dists_map, clustering, clustering_cen,
            k, current_belief_idx, current_oh, N_ACT_AGENT)

        c_ptr = -1
        if is_cen and k < len(cen_dists_map):
            dists_at_step = cen_dists_map[k]
            if current_belief_idx in dists_at_step:
                c_ptr = dists_at_step.index(current_belief_idx)

        # Clamp sentinel actions (negative) to WAIT
        if joint_act < 0:
            joint_act, a1, a2 = 0, 0, 0

        # Apply maneuver
        if a1 != 0 or a2 != 0:
            maneuver_stages.append((k, a1, a2))
        if a1 != 0:
            sc1_true = apply_maneuver(sc1_true, a1, dv=dv_mag)
            total_dv += dv_mag
        if a2 != 0:
            sc2_true = apply_maneuver(sc2_true, a2, dv=dv_mag)
            total_dv += dv_mag

        # Advance belief via RSSDA's get_terminal
        try:
            sparse_transitions = sdec.get_terminal(current_belief_idx, joint_act)
            next_belief_idx = -1
            for o, p, d in sparse_transitions:
                if o == obs_joint:
                    next_belief_idx = d
                    break
            if next_belief_idx != -1:
                current_belief_idx = next_belief_idx
        except KeyError:
            pass  # belief idx not in terminal dict, keep current belief

        # Update observation history for next step
        if k < N_STAGES - 1:
            current_oh = update_oh_rssda(
                policy, cen_dists_map, clustering, clustering_cen,
                k, current_belief_idx, current_oh, is_cen, c_ptr, obs1, obs2)

    # Final miss at TCA
    sc1_final = propagate(STAGE_EPOCHS[-1], sc1_true, EPOCH_TCA)
    sc2_final = propagate(STAGE_EPOCHS[-1], sc2_true, EPOCH_TCA)
    rtn_final = np.array(state_eci_to_rtn(sc1_final, sc2_final))
    miss_km_at_tca = np.linalg.norm(rtn_final[:3]) / 1e3

    return {
        'miss_km_at_tca':      miss_km_at_tca,
        'total_dv_ms':         total_dv,
        'sync_count':          sync_count,
        'maneuver_stages':     maneuver_stages,
        'miss_bin_trajectory': miss_bin_traj,
    }


# ---------------------------------------------------------------------------
# Single rollout — greedy policy
# ---------------------------------------------------------------------------

def rollout_greedy(
    T, O, R, init_b,
    greedy_policy: np.ndarray,
    contact_stages: List[int],
    true_init_miss_km: float,
    dv_mag: float,
    rng: np.random.Generator,
) -> dict:
    sc1_tca_eci = sc1_eci_at_tca()
    sc1_true = propagate(EPOCH_TCA, sc1_tca_eci, STAGE_EPOCHS[0])
    sc2_tca = np.array(state_rtn_to_eci(sc1_tca_eci,
                       np.array([true_init_miss_km * 1e3, 0.0, 0.0, 0.0, -V_REL_MS, 0.0])))
    sc2_true = propagate(EPOCH_TCA, sc2_tca, STAGE_EPOCHS[0])

    belief = init_b.copy()
    total_dv = 0.0
    sync_count = 0
    maneuver_stages = []
    miss_bin_traj = []

    for k in range(N_STAGES):
        if k > 0:
            sc1_true = propagate(STAGE_EPOCHS[k - 1], sc1_true, STAGE_EPOCHS[k])
            sc2_true = propagate(STAGE_EPOCHS[k - 1], sc2_true, STAGE_EPOCHS[k])

        sc1_proj = propagate(STAGE_EPOCHS[k], sc1_true, EPOCH_TCA)
        sc2_proj = propagate(STAGE_EPOCHS[k], sc2_true, EPOCH_TCA)
        rtn_tca  = np.array(state_eci_to_rtn(sc1_proj, sc2_proj))
        true_miss_km = np.linalg.norm(rtn_tca[:3]) / 1e3
        miss_bin_traj.append(miss_to_bin(true_miss_km))

        is_sync = k in contact_stages
        obs1 = obs2 = 0
        if is_sync:
            sync_count += 1
            obs1 = simulate_obs(true_miss_km, rng=rng)
            obs2 = simulate_obs(true_miss_km, rng=rng)
            obs_joint = obs1 * N_MISS + obs2
            belief = belief_update(belief, 0, obs_joint, T, O)

        most_likely_s = int(np.argmax(belief[:N_STATES]))
        a = greedy_policy[most_likely_s]
        a1, a2 = split_joint_action(a)

        if a1 != 0 or a2 != 0:
            maneuver_stages.append((k, a1, a2))
        if a1 != 0:
            sc1_true = apply_maneuver(sc1_true, a1, dv=dv_mag)
            total_dv += dv_mag
        if a2 != 0:
            sc2_true = apply_maneuver(sc2_true, a2, dv=dv_mag)
            total_dv += dv_mag

        belief = np.dot(T[a].T, belief)
        s = belief.sum()
        if s > 1e-30:
            belief /= s

    sc1_final = propagate(STAGE_EPOCHS[-1], sc1_true, EPOCH_TCA)
    sc2_final = propagate(STAGE_EPOCHS[-1], sc2_true, EPOCH_TCA)
    rtn_final = np.array(state_eci_to_rtn(sc1_final, sc2_final))
    miss_km_at_tca = np.linalg.norm(rtn_final[:3]) / 1e3

    return {
        'miss_km_at_tca':      miss_km_at_tca,
        'total_dv_ms':         total_dv,
        'sync_count':          sync_count,
        'maneuver_stages':     maneuver_stages,
        'miss_bin_trajectory': miss_bin_traj,
    }


# ---------------------------------------------------------------------------
# Multi-rollout runner
# ---------------------------------------------------------------------------

def run_simulation(
    T, O, R, init_b,
    contact_stages: List[int],
    n_rollouts: int = 100,
    init_miss_bin: int = 4,
    dv_mag: float = None,
    seed: int = 42,
    verbose: bool = True,
    policy_mode: str = "rssda",
    sdec=None,
    full_result=None,
) -> List[dict]:
    if dv_mag is None:
        dv_mag = DV_MAGNITUDE
    rng = np.random.default_rng(seed)

    lo = float(MISS_EDGES_KM[init_miss_bin])
    hi = float(MISS_EDGES_KM[init_miss_bin + 1])
    if hi == float('inf'):
        hi = 1500.0

    if policy_mode == "greedy":
        greedy_pol = build_greedy_policy(T, R)
    else:
        assert sdec is not None and full_result is not None, \
            "sdec and full_result required for rssda policy mode"

    results = []
    for i in range(n_rollouts):
        true_miss_km = rng.uniform(lo, hi)
        if policy_mode == "greedy":
            r = rollout_greedy(T, O, R, init_b, greedy_pol, contact_stages,
                               true_miss_km, dv_mag, rng)
        else:
            r = rollout_rssda(T, O, R, sdec, full_result, contact_stages,
                              true_miss_km, dv_mag, rng)
        results.append(r)
        if verbose and (i + 1) % 10 == 0:
            print(f"  rollout {i+1}/{n_rollouts}  "
                  f"miss={r['miss_km_at_tca']:.2f}km  "
                  f"dv={r['total_dv_ms']:.3f}m/s  "
                  f"syncs={r['sync_count']}")
    return results


def summarize(results: List[dict], label: str, init_miss_bin: int, dv_mag: float):
    misses = [r['miss_km_at_tca'] for r in results]
    dvs    = [r['total_dv_ms']    for r in results]
    syncs  = [r['sync_count']     for r in results]
    n_coll = sum(1 for m in misses if m < 1.0)
    return {
        'label':       label,
        'n':           len(results),
        'init_bin':    init_miss_bin,
        'coll_rate':   100.0 * n_coll / len(results),
        'mean_miss':   float(np.mean(misses)),
        'min_miss':    float(np.min(misses)),
        'mean_dv':     float(np.mean(dvs)),
        'mean_syncs':  float(np.mean(syncs)),
    }


def print_summary(results: List[dict], init_miss_bin: int, dv_mag: float, label: str = ""):
    s = summarize(results, label, init_miss_bin, dv_mag)
    misses = [r['miss_km_at_tca'] for r in results]
    print(f"\n{'='*60}")
    if label:
        print(f"[{label}]  ({s['n']} rollouts, init_bin={init_miss_bin} "
              f"[{MISS_LABELS[init_miss_bin]}], dv={dv_mag} m/s)")
    print(f"{'='*60}")
    print(f"  Collisions:  {s['coll_rate']:.1f}%")
    print(f"  Miss at TCA: min={s['min_miss']:.3f}  mean={s['mean_miss']:.3f}  km")
    print(f"  Total dv:    mean={s['mean_dv']:.4f} m/s")
    print(f"  Sync events: mean={s['mean_syncs']:.1f} of {N_STAGES} stages")
    print(f"\n  Miss-bin distribution at TCA:")
    from collections import Counter
    cnt = Counter(miss_to_bin(m) for m in misses)
    for mb in range(N_MISS):
        bar = '#' * cnt.get(mb, 0)
        print(f"    bin {mb} {MISS_LABELS[mb]:10s}: {cnt.get(mb,0):4d}  {bar}")


def print_comparison_table(summaries: List[dict]):
    print(f"\n{'='*80}")
    print(f"{'Scenario':<22} {'Init bin':<10} {'Coll%':>7} {'Mean miss(km)':>14} "
          f"{'Mean dv(m/s)':>13} {'Mean syncs':>11}")
    print(f"{'-'*80}")
    for s in summaries:
        print(f"  {s['label']:<20} {s['init_bin']:<10} {s['coll_rate']:>7.1f} "
              f"{s['mean_miss']:>14.3f} {s['mean_dv']:>13.4f} {s['mean_syncs']:>11.1f}")
    print(f"{'='*80}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts",      type=int,   default=100)
    parser.add_argument("--init-miss-bin", type=int,   default=4,
                        help="Initial true miss-distance bin (0=collision, 4=nominal)")
    parser.add_argument("--dv",            type=float, default=None)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--policy",        choices=["rssda", "greedy"], default="rssda")
    parser.add_argument("--compare",       action="store_true",
                        help="Run centralized, decentralized, and SDec variants")
    args = parser.parse_args()

    initialize_eop()

    T, O, R, init_b, contact_stages, cached_dv = load_matrices()
    dv_mag = args.dv if args.dv is not None else cached_dv

    from sdec_spacecraft import solve, build_model, build_config
    from RSSDA import SDecPOMDP

    def make_init_b_for_bin(bin_idx):
        """Initial belief uniform over stage-0 states in the given miss bin."""
        b = np.zeros(N_STATES_TOTAL, dtype=np.float64)
        b[state_index(bin_idx, 0)] = 1.0
        return b

    def solve_variant(label, cs, init_belief):
        print(f"\n--- Solving: {label} ---")
        model, _ = build_model(T, O, R, init_belief, cs)
        config = build_config(exact=False)
        sdec_obj = SDecPOMDP(model=model, config=config)
        full_res = sdec_obj.multi_agent_astar(N_STAGES)
        print(f"  Policy value: {full_res[0]:.4f}")
        return sdec_obj, full_res

    if args.compare:
        variants = [
            ("Centralized",     list(range(N_STAGES))),
            ("Decentralized",   []),
            ("SDec (contacts)", contact_stages),
        ]
        all_summaries = []
        for init_bin in [0, 1]:
            init_b_test = make_init_b_for_bin(init_bin)
            for label, cs in variants:
                sdec_obj, full_res = solve_variant(label, cs, init_b_test)
                print(f"  Running {args.rollouts} rollouts (init_bin={init_bin})...")
                results = run_simulation(
                    T, O, R, init_b_test, cs,
                    n_rollouts=args.rollouts,
                    init_miss_bin=init_bin,
                    dv_mag=dv_mag,
                    seed=args.seed,
                    verbose=False,
                    policy_mode="rssda",
                    sdec=sdec_obj,
                    full_result=full_res,
                )
                s = summarize(results, label, init_bin, dv_mag)
                all_summaries.append(s)
                print_summary(results, init_bin, dv_mag, label)

        print_comparison_table(all_summaries)

    else:
        init_b_test = make_init_b_for_bin(args.init_miss_bin)
        if args.policy == "rssda":
            print("Solving with RS-SDA*...")
            sdec_obj, full_res = solve_variant("SDec", contact_stages, init_b_test)
            results = run_simulation(
                T, O, R, init_b_test, contact_stages,
                n_rollouts=args.rollouts,
                init_miss_bin=args.init_miss_bin,
                dv_mag=dv_mag,
                seed=args.seed,
                verbose=True,
                policy_mode="rssda",
                sdec=sdec_obj,
                full_result=full_res,
            )
        else:
            results = run_simulation(
                T, O, R, init_b_test, contact_stages,
                n_rollouts=args.rollouts,
                init_miss_bin=args.init_miss_bin,
                dv_mag=dv_mag,
                seed=args.seed,
                verbose=True,
                policy_mode="greedy",
            )
        print_summary(results, args.init_miss_bin, dv_mag,
                      label=args.policy.upper())
