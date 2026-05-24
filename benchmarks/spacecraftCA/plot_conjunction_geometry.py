"""
plot_conjunction_geometry.py

Generates two plots:
  1. Static: RTN relative trajectory of SC2 w.r.t. SC1 over 24h (radial-miss geometry)
  2. GIF: animated version showing SC2 approaching over time

Usage:
  .venv/bin/python spacecraftCA/plot_conjunction_geometry.py
  .venv/bin/python spacecraftCA/plot_conjunction_geometry.py --gif
  .venv/bin/python spacecraftCA/plot_conjunction_geometry.py --miss-km 5.0

Saves to spacecraftCA/notes/conjunction_geometry.png (and .gif if --gif)
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCHMARKS = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_BENCHMARKS)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _BENCHMARKS)
sys.path.insert(0, _HERE)

from brahe import initialize_eop, state_rtn_to_eci, state_eci_to_rtn
from spacecraft_matrices import (
    sc1_eci_at_tca, propagate, STAGE_EPOCHS, EPOCH_TCA, V_REL_MS,
)
from spacecraft_discretizer import bin_center_km, N_MISS, MISS_EDGES_KM


def compute_trajectory(miss_km: float, n_points: int = 200):
    """
    Compute RTN relative trajectory of SC2 w.r.t. SC1 from T-24h to TCA.
    Returns arrays of (dR, dT, dN, t_hours_before_tca).
    """
    initialize_eop()
    sc1_tca_eci = sc1_eci_at_tca()
    sc2_tca = np.array(state_rtn_to_eci(sc1_tca_eci,
                       np.array([miss_km * 1e3, 0.0, 0.0, 0.0, -V_REL_MS, 0.0])))

    t_before_tca = np.linspace(24 * 3600, 0, n_points)  # seconds before TCA

    dR = np.zeros(n_points)
    dT = np.zeros(n_points)
    dN = np.zeros(n_points)

    for i, dt in enumerate(t_before_tca):
        epoch = EPOCH_TCA - dt
        sc1 = propagate(EPOCH_TCA, sc1_tca_eci, epoch)
        sc2 = propagate(EPOCH_TCA, sc2_tca, epoch)
        rtn = np.array(state_eci_to_rtn(sc1, sc2))
        dR[i] = rtn[0] / 1e3   # km
        dT[i] = rtn[1] / 1e3   # km
        dN[i] = rtn[2] / 1e3   # km

    t_hours = t_before_tca / 3600
    return dR, dT, dN, t_hours


def plot_static(miss_km: float, save_path: str):
    dR, dT, dN, t_hours = compute_trajectory(miss_km)

    stage_t_h = [23.39, 9.48, 7.83, 6.17, 2.80, 1.13]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Radial-Miss Conjunction Geometry  (miss = {miss_km:.2f} km)",
                 fontsize=13, fontweight='bold')

    # --- Left: dT vs dR (the approach plane) ---
    ax = axes[0]
    sc = ax.scatter(dT, dR, c=t_hours, cmap='plasma_r', s=8, zorder=2)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label('Hours before TCA', fontsize=9)

    # Mark decision stages
    from spacecraft_matrices import STAGE_EPOCHS
    for k, th in enumerate(stage_t_h):
        idx = np.argmin(np.abs(t_hours - th))
        ax.plot(dT[idx], dR[idx], 'k^', ms=8, zorder=3)
        ax.annotate(f'S{k}', (dT[idx], dR[idx]),
                    textcoords='offset points', xytext=(5, 5), fontsize=8)

    ax.axhline(miss_km, color='red', lw=1, ls='--', label=f'miss = {miss_km:.2f} km')
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.axvline(0, color='gray', lw=0.5, ls=':')
    ax.plot(0, 0, 'ko', ms=10, label='SC1 (TCA)', zorder=4)
    ax.plot(dT[-1], dR[-1], 'r*', ms=12, label='SC2 at TCA', zorder=4)
    ax.set_xlabel('dT (along-track, km)', fontsize=10)
    ax.set_ylabel('dR (radial, km)', fontsize=10)
    ax.set_title('Approach trajectory (dR vs dT)', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(min(dT) * 1.05, max(dT) * 1.05)
    ax.grid(True, alpha=0.3)

    # Draw collision zone
    collision_patch = mpatches.Rectangle((-abs(min(dT))*1.1, 0), abs(min(dT))*1.1*2, 1.0,
                                         color='red', alpha=0.08, label='Collision zone (<1 km)')
    ax.add_patch(collision_patch)

    # --- Right: dR, dT vs time ---
    ax2 = axes[1]
    ax2.plot(t_hours, dR, 'r-', lw=2, label='dR (radial, collision axis)')
    ax2.plot(t_hours, dT, 'b-', lw=2, label='dT (along-track)')
    ax2.axhline(0, color='gray', lw=0.5, ls=':')
    ax2.axhline(miss_km, color='red', lw=1, ls='--', alpha=0.5)
    for k, th in enumerate(stage_t_h):
        ax2.axvline(th, color='green', lw=1, ls=':', alpha=0.6)
        ax2.text(th, ax2.get_ylim()[0] if ax2.get_ylim()[0] < 0 else 0,
                 f'S{k}', fontsize=7, color='green', ha='center')
    ax2.set_xlabel('Hours before TCA', fontsize=10)
    ax2.set_ylabel('Separation (km)', fontsize=10)
    ax2.set_title('RTN components over time', fontsize=10)
    ax2.invert_xaxis()  # time flows left to right toward TCA
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved static plot: {save_path}")
    plt.close()


def plot_gif(miss_km: float, save_path: str):
    dR, dT, dN, t_hours = compute_trajectory(miss_km, n_points=100)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlim(min(dT) * 1.1, max(dT) * 1.1 + 50)
    ax.set_ylim(-abs(miss_km) * 0.5 - 5, max(abs(dR)) * 1.1)
    ax.set_xlabel('dT (along-track, km)')
    ax.set_ylabel('dR (radial, km)')
    ax.set_title(f'Radial-Miss Conjunction  (miss = {miss_km:.2f} km)')

    # SC1 at origin
    ax.plot(0, 0, 'ko', ms=12, label='SC1', zorder=5)
    ax.axhline(miss_km, color='red', lw=1, ls='--', alpha=0.6)
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.axvline(0, color='gray', lw=0.5, ls=':')
    collision_patch = mpatches.Rectangle((-abs(min(dT))*1.2, 0), abs(min(dT))*2.4, 1.0,
                                         color='red', alpha=0.08)
    ax.add_patch(collision_patch)
    ax.text(min(dT)*0.5, 0.5, 'Collision zone', fontsize=8, color='red', alpha=0.7)

    trail_line, = ax.plot([], [], 'b-', lw=1.5, alpha=0.4)
    sc2_dot, = ax.plot([], [], 'bs', ms=10, label='SC2', zorder=4)
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=10,
                        verticalalignment='top')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    def update(frame):
        trail_line.set_data(dT[:frame+1], dR[:frame+1])
        sc2_dot.set_data([dT[frame]], [dR[frame]])
        time_text.set_text(f'T-{t_hours[frame]:.1f}h')
        return trail_line, sc2_dot, time_text

    ani = FuncAnimation(fig, update, frames=len(t_hours), interval=80, blit=True)
    ani.save(save_path, writer=PillowWriter(fps=15))
    print(f"Saved GIF: {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--miss-km", type=float, default=2.0,
                        help="Miss distance in km (default: 2.0)")
    parser.add_argument("--gif", action="store_true",
                        help="Also generate animated GIF")
    args = parser.parse_args()

    notes_dir = os.path.join(_HERE, "notes")
    os.makedirs(notes_dir, exist_ok=True)

    static_path = os.path.join(notes_dir, "conjunction_geometry.png")
    plot_static(args.miss_km, static_path)

    if args.gif:
        gif_path = os.path.join(notes_dir, "conjunction_geometry.gif")
        plot_gif(args.miss_km, gif_path)
