"""
sdec_spacecraft.py

Wire spacecraft CA matrices into RSSDA and solve with RS-SDA*.

Usage:
  python sdec_spacecraft.py                   # approximate, dv=0.5 m/s
  python sdec_spacecraft.py --dv 0.1          # rebuild matrices with dv=0.1 m/s
  python sdec_spacecraft.py --exact           # exact algorithm
  python sdec_spacecraft.py --rebuild         # force matrix rebuild
  python sdec_spacecraft.py --sweep-beliefs   # query policy at dangerous states
"""

import os
import sys
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCHMARKS = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_BENCHMARKS)

sys.path.insert(0, _ROOT)
sys.path.insert(0, _BENCHMARKS)
sys.path.insert(0, _HERE)

from RSSDA import SDecPOMDP, SDecPOMDPModel, RSSDAConfig

from spacecraft_matrices import (
    load_matrices, build_matrices, save_matrices,
    CACHE_PATH, CONTACT_STAGES, DV_MAGNITUDE,
    N_JOINT_ACTIONS, N_JOINT_OBS, N_ACT_AGENT, N_OBS_AGENT,
)
from spacecraft_discretizer import (
    N_STAGES, N_STATES, N_STATES_TOTAL,
    N_MISS, DEV_ZERO, DEV_LABELS, index_to_state, state_index,
    sync_trigger_states, miss_bin_label,
)

ACT_NAMES = {0: 'WAIT', 1: '+dVT', 2: '-dVT'}
MISS_LABELS = [miss_bin_label(i) for i in range(N_MISS)]


def build_model(T, O, R, init_b, contact_stages):
    sync_states = sync_trigger_states(contact_stages)
    model = SDecPOMDPModel(
        nagents=2,
        nstates=N_STATES_TOTAL,
        nactions=N_JOINT_ACTIONS,
        nobs=N_JOINT_OBS,
        transitions=T.flatten().tolist(),
        obs=O.flatten().tolist(),
        rewards=R.flatten().tolist(),
        init_beliefs=init_b.tolist(),
        nacts_factor=[N_ACT_AGENT, N_ACT_AGENT],
        nobs_factor=[N_OBS_AGENT, N_OBS_AGENT],
        sync_states=sync_states,
        sync_actions=[],
        sync_observations=[],
    )
    return model, sync_states


def build_config(exact: bool = False, ti1_enabled: bool = True) -> RSSDAConfig:
    if exact:
        return RSSDAConfig(
            maxh=N_STAGES,
            algorithm="exact",
            TI1=False, TI2=False, TI3=False, TI4=False,
        )
    return RSSDAConfig(
        maxh=N_STAGES,
        algorithm="approximate",
        TI1=ti1_enabled, TI2=True, TI3=True, TI4=True,
        iter_limit=2000,
        rec_limit=2,
        max_clusters=20,
        heuristic_type="HYBRID",
    )


def sweep_beliefs(T, R, dv_mag):
    """
    For each (miss_bin, stage), compute the expected value of each joint action
    under a point-mass belief at that state, and show the best action.
    This is a proxy for policy inspection without needing a full belief tree.
    """
    print(f"\n{'='*65}")
    print(f"Belief sweep  (dv = {dv_mag} m/s)")
    print(f"For each state: immediate reward + one-step lookahead value")
    print(f"{'='*65}")

    # One-step lookahead Q(s,a) = R(a,s) + max_a' R(a', s')
    # (rough proxy — not the full policy value, but shows action ordering)
    for k in range(N_STAGES):
        print(f"\n  Stage {k}:")
        for mb in range(N_MISS):
            s = state_index(mb, DEV_ZERO, DEV_ZERO, k)
            q_vals = {}
            for a in range(N_JOINT_ACTIONS):
                a1 = a %  N_ACT_AGENT
                a2 = a // N_ACT_AGENT
                r_now = R[a, s]
                # lookahead: best reward at next state
                next_states = np.where(T[a, s, :] > 0)[0]
                r_next = 0.0
                for sp in next_states:
                    if sp < N_STATES:
                        r_next = max(R[:, sp].max(), r_next)
                q_vals[(a1, a2)] = r_now + r_next
            best = max(q_vals, key=q_vals.get)
            best_q = q_vals[best]
            next_s = np.where(T[0, s, :] > 0)[0]  # WAIT transition
            next_mb = index_to_state(next_s[0])[0] if len(next_s) > 0 and next_s[0] < N_STATES else -1
            print(f"    miss_bin={mb} ({MISS_LABELS[mb]:22s})  "
                  f"best=({ACT_NAMES[best[0]]},{ACT_NAMES[best[1]]}) Q={best_q:.1f}  "
                  f"WAIT->bin{next_mb}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact",         action="store_true")
    parser.add_argument("--rebuild",       action="store_true",
                        help="Force matrix rebuild")
    parser.add_argument("--dv",            type=float, default=None,
                        help="Delta-v per maneuver in m/s (triggers rebuild if changed)")
    parser.add_argument("--sweep-beliefs", action="store_true",
                        help="Print one-step Q-values for all states before solving")
    parser.add_argument("--disable-ti1", action="store_true",
                        help="Disable TI1 interleaving and solve a full-horizon fixed policy")
    args = parser.parse_args()

    from brahe import initialize_eop
    initialize_eop()

    requested_dv = args.dv if args.dv is not None else DV_MAGNITUDE

    # Auto-invalidate cache if dv changed
    need_rebuild = args.rebuild
    if not need_rebuild and os.path.exists(CACHE_PATH):
        _, _, _, _, _, cached_dv = load_matrices()
        if abs(cached_dv - requested_dv) > 1e-9:
            print(f"Cache dv={cached_dv} m/s differs from requested dv={requested_dv} m/s "
                  f"— rebuilding.")
            need_rebuild = True

    if need_rebuild or not os.path.exists(CACHE_PATH):
        print(f"Building matrices (dv={requested_dv} m/s)...")
        T, O, R, init_b = build_matrices(verbose=True, dv_magnitude=requested_dv)
        save_matrices(T, O, R, init_b, dv_magnitude=requested_dv)
        contact_stages = CONTACT_STAGES
        dv_used = requested_dv
    else:
        T, O, R, init_b, contact_stages, dv_used = load_matrices()

    print(f"\nSpacecraft CA SDec-POMDP")
    print(f"  States: {N_STATES_TOTAL}, Actions: {N_JOINT_ACTIONS}, Obs: {N_JOINT_OBS}")
    print(f"  nacts_factor=[{N_ACT_AGENT},{N_ACT_AGENT}], nobs_factor=[{N_OBS_AGENT},{N_OBS_AGENT}]")
    print(f"  Horizon: {N_STAGES}, Algorithm: {'exact' if args.exact else 'approximate'}")
    print(f"  Contact stages: {contact_stages}")
    print(f"  dv_magnitude: {dv_used} m/s")

    if args.sweep_beliefs:
        sweep_beliefs(T, R, dv_used)

    model, sync_states = build_model(T, O, R, init_b, contact_stages)
    config = build_config(args.exact, ti1_enabled=not args.disable_ti1)
    sdec = SDecPOMDP(model=model, config=config)

    print(f"\nRunning RS-SDA* (horizon={N_STAGES})...")
    result = sdec.multi_agent_astar(N_STAGES)
    value, policy = result[0], result[1]

    print(f"\n{'='*55}")
    print(f"Optimal value: {value:.4f}")
    print(f"{'='*55}")

    print("\nInitial belief support:")
    for s, p in enumerate(init_b):
        if p > 0:
            mb, dev1, dev2, k = index_to_state(s)
            print(f"  s={s} (miss_bin={mb} {MISS_LABELS[mb]}, "
                  f"dev=({DEV_LABELS[dev1]},{DEV_LABELS[dev2]}), "
                  f"stage={k})  p={p:.3f}")

    return value, policy, sdec, result


def solve(contact_stages=None, dv_magnitude=None, exact=False, verbose=True,
          ti1_enabled=True):
    """
    Convenience function: load matrices, build model, solve with RS-SDA*.
    Returns (value, sdec, full_result, T, O, R, init_b, contact_stages_used).
    """
    from brahe import initialize_eop
    initialize_eop()

    requested_dv = dv_magnitude if dv_magnitude is not None else DV_MAGNITUDE
    need_rebuild = False
    if os.path.exists(CACHE_PATH):
        _, _, _, _, _, cached_dv = load_matrices()
        if abs(cached_dv - requested_dv) > 1e-9:
            need_rebuild = True
    else:
        need_rebuild = True

    if need_rebuild:
        if verbose:
            print(f"Building matrices (dv={requested_dv} m/s)...")
        T, O, R, init_b = build_matrices(verbose=verbose, dv_magnitude=requested_dv)
        save_matrices(T, O, R, init_b, dv_magnitude=requested_dv)
        cs = CONTACT_STAGES
    else:
        T, O, R, init_b, cs, _ = load_matrices()

    if contact_stages is not None:
        cs = contact_stages

    model, sync_states = build_model(T, O, R, init_b, cs)
    config = build_config(exact, ti1_enabled=ti1_enabled)
    sdec = SDecPOMDP(model=model, config=config)

    if verbose:
        print(f"Running RS-SDA* (horizon={N_STAGES}, dv={requested_dv} m/s, contact_stages={cs})...")
    result = sdec.multi_agent_astar(N_STAGES)
    value = result[0]
    if verbose:
        print(f"Optimal value: {value:.4f}")

    return value, sdec, result, T, O, R, init_b, cs


if __name__ == "__main__":
    main()
