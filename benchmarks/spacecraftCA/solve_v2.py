"""
solve_v2.py

CHECKPOINT 3 driver: build the v2 reduced-state matrices, solve each variant with
RS-SDA*, and report (a) optimal value, (b) the maneuver TIMING along the most-likely
rollout — to answer the two acceptance questions:
  1. Does the policy DIFFERENTIATE across centralized / sdec / dec? (v1 did not)
  2. Does it still burn only at stage 0 (the v1 null-result symptom), or spread/wait?

Usage:
  python solve_v2.py                       # all 3 variants, default conjunction
  python solve_v2.py --perp 0 --dt0 2      # set conjunction (head-on, 2km)
"""
import os, sys, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCHMARKS = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_BENCHMARKS)
sys.path.insert(0, _ROOT); sys.path.insert(0, _BENCHMARKS); sys.path.insert(0, _HERE)

from RSSDA import SDecPOMDP, SDecPOMDPModel, RSSDAConfig
import spacecraft_discretizer_v2 as D
import spacecraft_transition_v2 as TV

ACT_NAMES = {0: 'WAIT', 1: '+dVT', 2: '-dVT'}


def build_model_v2(T, O, R, init_b, contact_stages):
    sync_states = D.sync_trigger_states(contact_stages)
    model = SDecPOMDPModel(
        nagents=2,
        nstates=D.N_STATES_TOTAL,
        nactions=TV.N_JOINT_ACTIONS,
        nobs=TV.N_JOINT_OBS,
        transitions=T.flatten().tolist(),
        obs=O.flatten().tolist(),
        rewards=R.flatten().tolist(),
        init_beliefs=init_b.tolist(),
        nacts_factor=[TV.N_ACT_AGENT, TV.N_ACT_AGENT],
        nobs_factor=[TV.N_OBS_AGENT, TV.N_OBS_AGENT],
        sync_states=sync_states,
        sync_actions=[],
        sync_observations=[],
    )
    return model, sync_states


def build_config():
    return RSSDAConfig(
        maxh=D.N_STAGES, algorithm="approximate",
        TI1=False, TI2=True, TI3=True, TI4=True,
        iter_limit=2000, rec_limit=2, max_clusters=20, heuristic_type="HYBRID",
    )


def most_likely_burn_timeline(T, R, init_b, greedy_from_R=True):
    """
    Rough policy-free burn-timing probe: from init_b, at each stage pick the joint
    action that maximizes immediate R + one-step lookahead, propagate the belief,
    record which stages burn. (A proxy for inspecting whether burns cluster at s0.)
    """
    b = init_b.copy()
    timeline = []
    for k in range(D.N_STAGES - 1):
        # expected one-step value of each action under current belief
        best_a, best_q = 0, -1e18
        for a in range(TV.N_JOINT_ACTIONS):
            q = float(b @ R[a])
            nb = b @ T[a]
            q += float(sum(nb[s] * R[:, s].max() for s in np.where(nb > 0)[0]))
            if q > best_q:
                best_q, best_a = q, a
        a1, a2 = best_a % TV.N_ACT_AGENT, best_a // TV.N_ACT_AGENT
        timeline.append((k, ACT_NAMES[a1], ACT_NAMES[a2]))
        b = b @ T[best_a]
    return timeline


def run_one(belief_label, init_b, perp):
    """Solve all 3 variants under a given initial belief; return {variant: value}."""
    contact_by_variant = {
        "centralized": list(range(D.N_STAGES)),
        "sdec": list(TV.CONTACT_STAGES),
        "dec": [],
    }
    print(f"\n{'='*70}\nBELIEF: {belief_label}   (perp={perp} km)\n{'='*70}")
    results = {}
    for variant in ("centralized", "sdec", "dec"):
        T, O, R, _ = TV.build_matrices_v2(variant, perp_km=perp, dt0_km=0.0,
                                          init_b=init_b, verbose=False)
        model, sync_states = build_model_v2(T, O, R, init_b, contact_by_variant[variant])
        sdec = SDecPOMDP(model=model, config=build_config())
        value = sdec.multi_agent_astar(D.N_STAGES)[0]
        results[variant] = value
        print(f"  {variant:<12} value = {value:>12.4f}   (sync: {len(sync_states)} states)")
    spread = max(results.values()) - min(results.values())
    print(f"  -> spread = {spread:.4f}  "
          f"{'DIFFERENTIATES' if spread > 1.0 else 'still ~identical'}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perp", type=float, default=0.0)
    ap.add_argument("--miss-bins", type=str, default="50,200",
                    help="candidate |dt| magnitudes (km) for the spread belief")
    args = ap.parse_args()

    from brahe import initialize_eop
    initialize_eop()

    miss_bins = [float(x) for x in args.miss_bins.split(",")]
    print(f"\nv2 CHECKPOINT 3 — v1-matched SPREAD belief over |dt| in {miss_bins} km")
    print("Comparing sign mapping: BOTH (+/-dt ambiguity, faithful to v1 unsigned) vs POS")

    b_both = TV.build_init_b_spread(miss_bins, sign_mode="both")
    b_pos  = TV.build_init_b_spread(miss_bins, sign_mode="pos")

    res_both = run_one(f"BOTH-signs over {miss_bins}", b_both, args.perp)
    res_pos  = run_one(f"POS-only over {miss_bins}",  b_pos,  args.perp)

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"{'variant':<12} {'BOTH-signs':>12} {'POS-only':>12}")
    for v in ("centralized", "sdec", "dec"):
        print(f"{v:<12} {res_both[v]:>12.4f} {res_pos[v]:>12.4f}")
    print(f"\nv1 reference (broken model): Cen ~-22.3, SDec ~-22.5, Dec ~-166")
    print("Looking for: Cen <= SDec << Dec differentiation, driven by belief uncertainty.")


if __name__ == "__main__":
    main()
