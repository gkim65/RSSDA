"""
plot_policy.py

Visualize the RS-SDA* policy as a (miss_bin, stage) heatmap on the neutral-deviation slice.

For each starting (miss_bin, neutral_dev, neutral_dev, stage), re-solves
RS-SDA* with a point-mass initial belief and reads the stage-0 action from the policy.
This correctly captures what the policy would do given certainty about the
current state.

Usage:
  .venv/bin/python spacecraftCA/plot_policy.py
  .venv/bin/python spacecraftCA/plot_policy.py --out notes/policy_grid.png
"""

import os, sys, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCHMARKS = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_BENCHMARKS)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _BENCHMARKS)
sys.path.insert(0, _HERE)

from brahe import initialize_eop
from spacecraft_matrices import load_matrices, CONTACT_STAGES, STAGE_T_BEFORE_TCA_SEC
from spacecraft_discretizer import (
    N_MISS, N_STAGES, N_STATES_TOTAL, SINK_STATE, DEV_ZERO, state_index,
    miss_bin_label,
)
from sdec_spacecraft import build_model, build_config
from RSSDA import SDecPOMDP

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Joint action index → readable label
def joint_action_label(ja: int) -> str:
    if ja < 0:
        return "N/A"
    a1, a2 = ja % 3, ja // 3
    names = {0: "W", 1: "+", 2: "-"}
    n1, n2 = names[a1], names[a2]
    if a1 == 0 and a2 == 0:
        return "WAIT"
    if a1 != 0 and a2 == 0:
        return f"SC1{n1}"
    if a1 == 0 and a2 != 0:
        return f"SC2{n2}"
    if a1 == a2:
        return f"Both{n1}"
    return f"SC1{n1}\nSC2{n2}"

# Color per action category
def action_color(label: str) -> str:
    if label == "WAIT":     return "#cccccc"
    if label.startswith("SC1"): return "#4c9be8"
    if label.startswith("SC2"): return "#e8714c"
    if label.startswith("Both"): return "#9b59b6"
    if "SC1" in label and "SC2" in label: return "#27ae60"
    return "#ffffff"

_LEGACY_BIN_LABELS = [
    "<1 km\nCOLL", "1–5 km\nHIGH", "5–20 km\nMOD",
    "20–100 km\nLOW", "100–500 km\nNOM", ">500 km\nSAFE",
]
BIN_LABELS = [miss_bin_label(i).replace(") ", ")\n") for i in range(N_MISS)]
STAGE_LABELS = [
    f"S{k}\nT-{t_sec / 3600.0:.1f}h"
    for k, t_sec in enumerate(STAGE_T_BEFORE_TCA_SEC)
]


def solve_from_state(T, O, R, contact_stages, miss_bin, stage):
    """Solve with point-mass belief on the neutral-deviation state."""
    init_b = np.zeros(N_STATES_TOTAL)
    init_b[state_index(miss_bin, DEV_ZERO, DEV_ZERO, stage)] = 1.0
    model, _ = build_model(T, O, R, init_b, contact_stages)
    config = build_config(exact=False)
    sdec = SDecPOMDP(model=model, config=config)
    horizon = N_STAGES - stage
    full_result = sdec.multi_agent_astar(horizon)
    return full_result, sdec


def read_stage0_action(full_result, mode="cen"):
    """Read the first action from the policy (step 0, belief index 0)."""
    value, policy, clustering, cent_vector, cen_dists_map, clustering_cen = full_result
    if not policy:
        return -1, -1

    try:
        if mode == "cen":
            # Centralized: step 0, cluster 0, first action
            cen_pol = policy[0][1]
            if cen_pol and cen_pol[0]:
                ja = cen_pol[0][0]
                return ja, value
        else:
            # Decentralized: step 0, history index 0 for each agent
            dec_pol = policy[0][0]
            if dec_pol and len(dec_pol) >= 2:
                a1 = dec_pol[0][0] if dec_pol[0] else 0
                a2 = dec_pol[1][0] if dec_pol[1] else 0
                a1 = max(a1, 0)
                a2 = max(a2, 0)
                ja = a1 + 3 * a2
                return ja, value
    except (IndexError, TypeError):
        pass
    return 0, value  # fallback WAIT


def build_grid(T, O, R, contact_stages, mode="cen"):
    """Build (N_MISS, N_STAGES) grids of action labels and policy values.

    For decentralized mode, solves with contact_stages=[] so RSSDA populates
    the decentralized branch at step 0 (otherwise it's never computed because
    all states are sync states and step 0 is always centralized).
    """
    label_grid = np.full((N_MISS, N_STAGES), "N/A", dtype=object)
    value_grid = np.zeros((N_MISS, N_STAGES))

    # Decentralized solve must use no contact stages so step 0 is dec-only
    solve_stages = [] if mode == "dec" else contact_stages

    for stage in range(N_STAGES):
        for mb in range(N_MISS):
            print(f"  Solving (bin={mb}, stage={stage})...", end="\r", flush=True)
            try:
                full_result, _ = solve_from_state(T, O, R, solve_stages, mb, stage)
                ja, val = read_stage0_action(full_result, mode=mode)
                label_grid[mb, stage] = joint_action_label(ja)
                value_grid[mb, stage] = val
            except Exception as e:
                label_grid[mb, stage] = "ERR"
                value_grid[mb, stage] = 0.0

    print()
    return label_grid, value_grid


def plot_grid(label_grid, value_grid, title, ax):
    color_arr = np.zeros((N_MISS, N_STAGES, 3))
    for mb in range(N_MISS):
        for st in range(N_STAGES):
            hex_c = action_color(label_grid[mb, st])
            color_arr[mb, st] = [int(hex_c[i:i+2], 16)/255 for i in (1,3,5)]

    ax.imshow(color_arr, aspect='auto', origin='upper')

    for mb in range(N_MISS):
        for st in range(N_STAGES):
            lbl = label_grid[mb, st]
            val = value_grid[mb, st]
            ax.text(st, mb, f"{lbl}\n({val:.0f})", ha='center', va='center',
                    fontsize=7, color='black')

    ax.set_xticks(range(N_STAGES))
    ax.set_xticklabels(STAGE_LABELS, fontsize=8)
    ax.set_yticks(range(N_MISS))
    ax.set_yticklabels(BIN_LABELS, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel("Decision stage →  TCA")
    ax.set_ylabel("Miss distance bin (starting state)")

    # Red border around danger bins 0 and 1
    for mb in range(2):
        ax.add_patch(mpatches.FancyBboxPatch(
            (-0.5, mb - 0.5), N_STAGES, 1,
            lw=2, edgecolor='red', facecolor='none',
            boxstyle="square,pad=0"
        ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--mode", choices=["cen", "dec", "both"], default="both")
    args = parser.parse_args()

    initialize_eop()
    T, O, R, init_b, contact_stages, dv_mag = load_matrices()

    if args.mode == "both":
        modes = ["cen", "dec"]
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    else:
        modes = [args.mode]
        fig, axes = plt.subplots(1, 1, figsize=(9, 6))
        axes = [axes]

    for ax, mode in zip(axes, modes):
        print(f"\nBuilding {mode} policy grid ({N_MISS}×{N_STAGES} = {N_MISS*N_STAGES} solves)...")
        label_grid, value_grid = build_grid(T, O, R, contact_stages, mode=mode)
        title = "Centralized Policy" if mode == "cen" else "Decentralized Policy"
        plot_grid(label_grid, value_grid, title, ax)

    # Legend
    legend_items = [
        ("WAIT", "#cccccc"), ("SC1 burn", "#4c9be8"),
        ("SC2 burn", "#e8714c"), ("Both same dir", "#9b59b6"),
        ("Opposite burns", "#27ae60"),
    ]
    patches = [mpatches.Patch(color=c, label=l) for l, c in legend_items]
    fig.legend(handles=patches, loc='lower center', ncol=len(patches),
               fontsize=9, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(f"RS-SDA* Policy Grid  (dv={dv_mag} m/s, each cell = point-mass belief solve)",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0.07, 1, 1])

    out = args.out or os.path.join(_HERE, "notes", "policy_grid.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
