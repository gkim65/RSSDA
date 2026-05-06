"""
scenario3_sweep.py

Scenario 3: Costly Communication Sweep.

For each c_sync in a range, adds a sync-stage penalty to R, re-solves with RS-SDA*,
runs 100 rollouts from bin 0 and bin 1, and records:
  (c_sync, policy_value, collision_rate, mean_miss_km, mean_dv_ms, mean_sync_count)

Then plots collision_rate vs c_sync.

Usage:
  python scenario3_sweep.py
  python scenario3_sweep.py --c-max 200 --c-step 20 --rollouts 100 --init-miss-bin 0
  python scenario3_sweep.py --no-plot        # skip matplotlib plot
"""

import os
import sys
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from brahe import initialize_eop

from spacecraft_matrices import (
    load_matrices, CONTACT_STAGES,
    N_JOINT_ACTIONS, N_ACT_AGENT,
)
from spacecraft_discretizer import (
    N_STAGES, N_STATES, N_STATES_TOTAL, SINK_STATE,
    N_MISS, state_index, index_to_state, sync_trigger_states,
)
from sdec_spacecraft import build_model, build_config
from spacecraft_simulator import (
    run_simulation, summarize,
)
from RSSDA import SDecPOMDP


def add_sync_penalty(R_base: np.ndarray, c_sync: float,
                     contact_stages: list) -> np.ndarray:
    """
    Return a copy of R with -c_sync added to all (action, state) entries
    where the state is in a sync stage.
    Penalty is per-stage, not per-agent.
    """
    R_new = R_base.copy()
    if c_sync == 0.0:
        return R_new
    for k in contact_stages:
        for mb in range(N_MISS):
            s = state_index(mb, k)
            R_new[:, s] -= c_sync
    return R_new


def run_sweep(c_values, contact_stages, T, O, R_base, n_rollouts, init_miss_bin,
              dv_mag, seed, verbose=True):
    results_by_c = []
    init_b = np.zeros(N_STATES_TOTAL, dtype=np.float64)
    init_b[state_index(init_miss_bin, 0)] = 1.0

    for c in c_values:
        R_mod = add_sync_penalty(R_base, c, contact_stages)

        model, sync_states = build_model(T, O, R_mod, init_b, contact_stages)
        config = build_config(exact=False)
        sdec_obj = SDecPOMDP(model=model, config=config)
        full_res = sdec_obj.multi_agent_astar(N_STAGES)
        policy_value = full_res[0]

        rollouts = run_simulation(
            T, O, R_mod, init_b, contact_stages,
            n_rollouts=n_rollouts,
            init_miss_bin=init_miss_bin,
            dv_mag=dv_mag,
            seed=seed,
            verbose=False,
            policy_mode='rssda',
            sdec=sdec_obj,
            full_result=full_res,
        )
        s = summarize(rollouts, f"c={c}", init_miss_bin, dv_mag)

        row = {
            'c_sync':      c,
            'policy_value': policy_value,
            'coll_rate':   s['coll_rate'],
            'mean_miss':   s['mean_miss'],
            'mean_dv':     s['mean_dv'],
            'mean_syncs':  s['mean_syncs'],
        }
        results_by_c.append(row)

        if verbose:
            print(f"  c_sync={c:6.1f}  val={policy_value:8.2f}  "
                  f"coll={s['coll_rate']:5.1f}%  "
                  f"miss={s['mean_miss']:7.3f}km  "
                  f"dv={s['mean_dv']:.4f}m/s  "
                  f"syncs={s['mean_syncs']:.1f}")

    return results_by_c


def print_table(rows):
    print(f"\n{'='*80}")
    print(f"Scenario 3: Costly Communication Sweep")
    print(f"{'='*80}")
    print(f"{'c_sync':>8} {'Policy val':>11} {'Coll%':>7} "
          f"{'Mean miss(km)':>14} {'Mean dv(m/s)':>13} {'Mean syncs':>11}")
    print(f"{'-'*80}")
    for r in rows:
        print(f"{r['c_sync']:8.1f} {r['policy_value']:11.2f} {r['coll_rate']:7.1f} "
              f"{r['mean_miss']:14.3f} {r['mean_dv']:13.4f} {r['mean_syncs']:11.1f}")
    print(f"{'='*80}")


def plot_results(rows, init_miss_bin, out_path=None):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    c_vals    = [r['c_sync']    for r in rows]
    colls     = [r['coll_rate'] for r in rows]
    means     = [r['mean_miss'] for r in rows]
    syncs     = [r['mean_syncs'] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(c_vals, colls, 'o-', color='crimson')
    axes[0].set_xlabel('c_sync')
    axes[0].set_ylabel('Collision rate (%)')
    axes[0].set_title('Collision Rate vs Sync Cost')
    axes[0].set_ylim(-5, 105)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(c_vals, means, 's-', color='steelblue')
    axes[1].set_xlabel('c_sync')
    axes[1].set_ylabel('Mean miss at TCA (km)')
    axes[1].set_title('Miss Distance vs Sync Cost')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(c_vals, syncs, '^-', color='seagreen')
    axes[2].set_xlabel('c_sync')
    axes[2].set_ylabel('Mean sync events')
    axes[2].set_title('Sync Count vs Sync Cost')
    axes[2].set_ylim(-0.5, N_STAGES + 0.5)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f'Scenario 3: Costly Comms (init_bin={init_miss_bin})', fontsize=13)
    plt.tight_layout()

    if out_path is None:
        out_path = os.path.join(_HERE, 'notes', f'scenario3_sweep_bin{init_miss_bin}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--c-max",         type=float, default=200)
    parser.add_argument("--c-step",        type=float, default=20)
    parser.add_argument("--rollouts",      type=int,   default=100)
    parser.add_argument("--init-miss-bin", type=int,   default=0)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--no-plot",       action="store_true")
    args = parser.parse_args()

    initialize_eop()

    T, O, R_base, init_b, contact_stages, dv_mag = load_matrices()

    c_values = list(np.arange(0, args.c_max + 1e-9, args.c_step))
    print(f"Scenario 3 sweep: c_sync = {c_values}")
    print(f"  rollouts={args.rollouts}, init_bin={args.init_miss_bin}, dv={dv_mag}m/s")
    print(f"  contact_stages={contact_stages}\n")

    rows = run_sweep(
        c_values, contact_stages, T, O, R_base,
        n_rollouts=args.rollouts,
        init_miss_bin=args.init_miss_bin,
        dv_mag=dv_mag,
        seed=args.seed,
        verbose=True,
    )

    print_table(rows)

    if not args.no_plot:
        plot_results(rows, args.init_miss_bin)
