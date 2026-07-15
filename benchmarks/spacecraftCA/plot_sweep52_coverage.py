"""
Sweep-52 coverage figure — the D6 "representative sweep" panel, standalone.

Reproduces ONLY the final row of coverage_5dof_spread.png (fig D6 in
plot_coverage_scope.py) over the 52-conjunction evaluation suite
notes/conj_sweep_spherical_50.json: three ECI projections (XY / XZ / YZ),
each ellipse one SC2 orbit we conjunct with, colored by plane-crossing angle
Δi = |i2 - i1|, with SC1 in black and the Earth as a gray disk. Behind-Earth
arcs are drawn faint-dashed (same occlusion cue as Figure C/D).

No on-figure caption (the paper caption is in the companion text). Follows the
global figure rules: Computer Modern serif, sentence-case labels, transparent
save + light/dark theme flag, PDF + SVG + PNG.

Run from benchmarks/spacecraftCA/:
    PYTHONPATH=. ../../.venv/bin/python -u plot_sweep52_coverage.py
    PYTHONPATH=. ../../.venv/bin/python -u plot_sweep52_coverage.py --dark
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_coverage_scope import (
    koe_ellipse_eci, earth_circle, _draw_occluded, _PROJ, R_E_KM,
)
from spacecraft_matrices import SC1_OE_AT_TCA

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "notes", "figures")
DEFAULT_JSON = os.path.join(HERE, "notes", "conj_sweep_spherical_50.json")


def _apply_style():
    """Computer Modern serif via LaTeX when available, mathtext-cm fallback."""
    use_tex = shutil.which("latex") is not None
    plt.rcParams.update({
        "text.usetex": use_tex,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    })


def plot_sweep(json_path, out_stem, dark=False):
    fg = "white" if dark else "black"
    earth = "0.30" if dark else "0.85"
    sc1_col = "white" if dark else "black"

    with open(json_path) as f:
        specs = json.load(f)

    cmap = plt.get_cmap("turbo")
    norm = plt.Normalize(0, 120)                       # Δi color scale (deg)

    fig = plt.figure(figsize=(8.5, 3.4))
    # 3 projection panels + a slim colorbar column. wspace gives each panel's
    # ECI y-axis label room to clear the neighbor panel's data box (the axes
    # were too tight before, overlapping label text with lines).
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.30)
    axes = [fig.add_subplot(gs[0, j]) for j in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    proj2d = list(zip(axes, ["xy", "xz", "yz"]))

    o1 = koe_ellipse_eci(SC1_OE_AT_TCA)
    ex, ey = earth_circle()
    for ax, key in proj2d:
        h, v, d = _PROJ[key]
        ax.fill(ex, ey, color=earth, zorder=1)
        _draw_occluded(ax, o1, sc1_col, 1.8, 1.0, (0, (1, 1)), h, v, d, zbase=3)
    axes[0].plot([], [], color=sc1_col, lw=1.8, label="SC1")

    drawn = []
    for s in specs:
        sc1 = np.asarray(s["sc1_oe"], float)
        sc2 = np.asarray(s["sc2_oe"], float)
        di = abs(float(sc2[2]) - float(sc1[2]))        # Δi = |i2 - i1|
        o2 = koe_ellipse_eci(sc2)
        drawn.append(o2)
        for ax, key in proj2d:
            h, v, d = _PROJ[key]
            _draw_occluded(ax, o2, cmap(norm(di)), 0.9, 0.55, (0, (4, 3)),
                           h, v, d, zbase=4)

    allpts = np.vstack(drawn + [o1])
    for ax, key in proj2d:
        h, v, _ = _PROJ[key]
        m = 1.08 * np.abs(allpts[:, [h, v]]).max()
        ax.set_xlim(-m, m)
        ax.set_ylim(-m, m)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        # sentence-case, clean axis labels (no raw field names)
        ax.set_xlabel(f"ECI {key[0]} (km)", fontsize=9, color=fg)
        ax.set_ylabel(f"ECI {key[1]} (km)", fontsize=9, color=fg)
        ax.set_title(key.upper(), fontsize=9, color=fg)
        for sp in ax.spines.values():
            sp.set_color(fg)

    axes[0].legend(fontsize=8, loc="upper right", framealpha=0.0,
                   labelcolor=fg)

    # Shrink the colorbar to the panels' actual (square) drawn height instead of
    # the full gridspec cell, so it lines up with the boxes rather than towering
    # over them. Panels are aspect="equal", so their rendered height < cell height.
    fig.canvas.draw()
    p0 = axes[0].get_position()
    cpos = cax.get_position()
    cax.set_position([cpos.x0, p0.y0, cpos.width, p0.height])

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Plane-crossing angle $\\Delta i$ (deg)", fontsize=9, color=fg)
    cb.ax.tick_params(labelsize=8, color=fg, labelcolor=fg)
    cb.outline.set_edgecolor(fg)

    os.makedirs(FIG_DIR, exist_ok=True)
    for ext, dpi in (("png", 200), ("pdf", None), ("svg", None)):
        path = os.path.join(FIG_DIR, f"{out_stem}.{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=True)
        print(f"wrote {path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=DEFAULT_JSON)
    ap.add_argument("--out", default="sweep52_coverage")
    ap.add_argument("--dark", action="store_true",
                    help="light foreground for black-background slides")
    args = ap.parse_args()
    _apply_style()
    plot_sweep(args.json, args.out, dark=args.dark)


if __name__ == "__main__":
    main()
