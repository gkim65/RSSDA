#!/usr/bin/env python
"""plot_summary_violin.py — 9-category box+violin of miss distance from PASTED summary stats.

Draws all approaches side by side so you can eyeball how similar they are:
  POMDP:   Centralized | SDec | Dec        (left group, divider after)
  Operator: the 6 strategy-pair iterations  (right group)

DATA = a JSON dict {category: {n, mean, min, q1, median, q3, max, p05, p95, coll_pct,
band_in_pct, ...}} as printed by summarize_npz.py. Because we only have the summary (not the raw
per-rollout arrays), the "violin" is RECONSTRUCTED from the reported percentiles: a smooth width
profile interpolated through (min, p05, q1, median, q3, p95, max), symmetric about each category's
axis. The box (q1/median/q3, whiskers p05..p95) is drawn ON TOP and is EXACT. So the box is the
truth; the violin is a faithful-to-the-quartiles silhouette, not a KDE of hidden data. Labeled as
such in the caption note.

Feed the JSON via --json <file> or paste it into STATS below. Missing categories are skipped, so
you can render baselines-only now and add centralized/sdec/dec later.

Usage:
  python plot_summary_violin.py --json stats_sweep50.json --tag sweep50
  python plot_summary_violin.py --tag baselines_partial      # uses embedded STATS
  python plot_summary_violin.py --json s.json --tag t --dark --transparent
"""
import argparse
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCA = os.path.dirname(_HERE)

# ---- paste-in default (partial baseline run, 381 cells + POMDP rollouts_sweep50_drag, 269) ----
STATS = {
  "centralized": {"n": 18200, "mean": 6.726, "min": 1.14, "q1": 5.435, "median": 6.534,
    "q3": 7.469, "max": 50.641, "p05": 4.954, "p95": 9.753, "coll_pct": 0.0,
    "band_in_pct": 60.4, "dv_mean": 0.6261},
  "sdec": {"n": 17800, "mean": 6.722, "min": 1.14, "q1": 5.435, "median": 6.509,
    "q3": 7.469, "max": 50.641, "p05": 4.954, "p95": 9.753, "coll_pct": 0.0,
    "band_in_pct": 60.7, "dv_mean": 0.6296},
  "dec": {"n": 17800, "mean": 8.291, "min": 0.773, "q1": 6.002, "median": 7.814,
    "q3": 9.911, "max": 41.35, "p05": 3.72, "p95": 13.968, "coll_pct": 0.337,
    "band_in_pct": 28.3, "dv_mean": 0.8483},
  "threshold-x-threshold": {"n": 13200, "mean": 31.0, "min": 14.975, "q1": 31.873,
    "median": 34.192, "q3": 35.851, "max": 36.682, "p05": 16.994, "p95": 36.5,
    "coll_pct": 0.0, "band_in_pct": 0.0, "dv_mean": 1.0},
  "threshold-x-selfish": {"n": 13000, "mean": 31.191, "min": 14.975, "q1": 32.27,
    "median": 34.27, "q3": 35.854, "max": 36.682, "p05": 16.994, "p95": 36.5,
    "coll_pct": 0.0, "band_in_pct": 0.0, "dv_mean": 1.0},
  "threshold-x-fixedlead": {"n": 12600, "mean": 16.381, "min": 3.313, "q1": 9.667,
    "median": 19.071, "q3": 22.34, "max": 27.888, "p05": 6.114, "p95": 26.871,
    "coll_pct": 0.0, "band_in_pct": 6.2, "dv_mean": 1.3215},
  "selfish-x-selfish": {"n": 12600, "mean": 31.602, "min": 14.975, "q1": 33.082,
    "median": 34.27, "q3": 35.864, "max": 36.682, "p05": 16.994, "p95": 36.5,
    "coll_pct": 0.0, "band_in_pct": 0.0, "dv_mean": 1.0},
  "selfish-x-fixedlead": {"n": 12600, "mean": 15.946, "min": 2.926, "q1": 8.908,
    "median": 17.838, "q3": 22.34, "max": 27.888, "p05": 5.363, "p95": 26.871,
    "coll_pct": 0.0, "band_in_pct": 7.7, "dv_mean": 1.302},
  "fixedlead-x-fixedlead": {"n": 12200, "mean": 13.011, "min": 4.224, "q1": 8.394,
    "median": 15.22, "q3": 17.227, "max": 19.974, "p05": 6.1, "p95": 19.156,
    "coll_pct": 0.0, "band_in_pct": 6.8, "dv_mean": 2.0},
}

# display order + clean labels; POMDP first (left group), then the 6 operator pairs.
_POMDP = ["centralized", "sdec", "dec"]
_PAIRS = ["threshold-x-threshold", "threshold-x-selfish", "threshold-x-fixedlead",
          "selfish-x-selfish", "selfish-x-fixedlead", "fixedlead-x-fixedlead"]
_ORDER = _POMDP + _PAIRS
_LABEL = {
    "centralized": "Centralized", "sdec": "Semi-decentralized", "dec": "Decentralized",
    "threshold-x-threshold": "Threshold $\\times$ threshold",
    "threshold-x-selfish": "Threshold $\\times$ selfish",
    "threshold-x-fixedlead": "Threshold $\\times$ fixed-lead",
    "selfish-x-selfish": "Selfish $\\times$ selfish",
    "selfish-x-fixedlead": "Selfish $\\times$ fixed-lead",
    "fixedlead-x-fixedlead": "Fixed-lead $\\times$ fixed-lead",
}


def _setup_typography():
    import shutil
    import matplotlib.pyplot as plt
    common = {"font.family": "serif", "mathtext.fontset": "cm", "axes.titlesize": 22,
              "axes.labelsize": 22, "font.size": 19, "legend.fontsize": 18,
              "xtick.labelsize": 19, "ytick.labelsize": 19, "savefig.dpi": 300}
    if shutil.which("latex"):
        try:
            plt.rcParams.update({"text.usetex": True,
                                 "font.serif": ["Computer Modern Roman"], **common})
            fig = plt.figure(); fig.text(0.5, 0.5, r"$\delta T$"); fig.canvas.draw(); plt.close(fig)
            print("[typography] using real LaTeX (Computer Modern)")
            return
        except Exception as e:
            plt.rcParams["text.usetex"] = False
            print(f"[typography] LaTeX failed ({e}); mathtext-cm fallback")
    plt.rcParams.update({"text.usetex": False,
                         "font.serif": ["CMU Serif", "cmr10", "STIXGeneral", "DejaVu Serif"],
                         **common})
    print("[typography] using mathtext Computer Modern (no usetex)")


_setup_typography()


def _violin_silhouette(s, npts=200):
    """Reconstruct a symmetric width profile from the reported percentiles. Width peaks where the
    distribution is DENSE (percentiles close together => steep CDF => big density). We put density
    mass proportional to 1/gap between successive percentile knots, smoothed, then map to y."""
    knots_p = np.array([0, 5, 25, 50, 75, 95, 100]) / 100.0
    knots_y = np.array([s["min"], s["p05"], s["q1"], s["median"], s["q3"], s["p95"], s["max"]],
                       dtype=float)
    # guard against non-monotone / degenerate knots (all-equal distributions)
    for i in range(1, len(knots_y)):
        if knots_y[i] <= knots_y[i - 1]:
            knots_y[i] = knots_y[i - 1] + 1e-6
    y = np.linspace(knots_y[0], knots_y[-1], npts)
    # local density ~ dP/dy from the piecewise-linear CDF (percentile vs value)
    dens = np.gradient(np.interp(y, knots_y, knots_p), y)
    dens = np.clip(dens, 0, None)
    # smooth a touch so the silhouette isn't jagged at the knots
    k = np.hanning(max(5, npts // 20))
    dens = np.convolve(dens, k / k.sum(), mode="same")
    if dens.max() > 0:
        dens = dens / dens.max()
    return y, dens


def _theme(dark):
    fg = "#f0f0f0" if dark else "#222222"
    band = "#39d353" if dark else "#2e7d32"       # safe-band green
    unsafe = "#e74c3c" if dark else "#c0392b"     # unsafe-band red
    far = "#f0ad4e" if dark else "#e0a800"        # far/over-mitigated amber
    grid = "#555555" if dark else "#cccccc"
    return fg, band, unsafe, far, grid


def main():
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=None, help="JSON stats file (from summarize_npz.py). "
                    "If omitted, uses the embedded STATS.")
    ap.add_argument("--tag", default="summary")
    ap.add_argument("--band-legend", default="arrows", choices=["corner", "strip", "arrows"],
                    help="how to show the 3-band key: corner legend / top strip / "
                         "'arrows' = two-line words above each band (default).")
    ap.add_argument("--ymax", type=float, default=None, help="miss-axis cap (km); default auto.")
    ap.add_argument("--dark", action="store_true")
    ap.add_argument("--transparent", action="store_true")
    args = ap.parse_args()

    stats = json.load(open(args.json)) if args.json else STATS
    cats = [c for c in _ORDER if c in stats]
    if not cats:
        raise SystemExit("no known categories in the stats")
    fg, band, unsafe, far, grid = _theme(args.dark)
    if args.dark:
        plt.rcParams.update({"axes.facecolor": "none", "figure.facecolor": "none",
                             "text.color": fg, "axes.labelcolor": fg,
                             "xtick.color": fg, "ytick.color": fg})

    # colors: POMDP in a cool family, operator pairs in tab10
    cmap = plt.get_cmap("tab10")
    pomdp_col = {"centralized": "#1b4f72", "sdec": "#2874a6", "dec": "#5dade2"}
    color = {c: pomdp_col.get(c, cmap((_PAIRS.index(c) if c in _PAIRS else 0) % 10)) for c in cats}

    # HORIZONTAL layout: miss on X, methods down Y. Row 0 at TOP -> order reads top-to-bottom.
    rows = list(range(len(cats)))
    ypos = {c: len(cats) - 1 - i for i, c in enumerate(cats)}   # flip so first cat is on top
    # wide landscape ~4:3 (width x height)
    fig_w = 16.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 3.0 / 4.0))

    xmax_data = max(stats[c]["max"] for c in cats)
    # default cap at 40 km so a rare >40 km outlier tail doesn't squish the informative range
    # (whiskers are p05..p95 so the box/violin bulk is always well within this); override via --ymax.
    xhi = args.ymax or min(xmax_data * 1.10, 40.0)
    # bands span the FULL x-range as vertical strips (miss on x now)
    far_start = 10.0
    ax.axvspan(0, 4, color=unsafe, alpha=0.13, lw=0, zorder=0)          # unsafe 0-4 km
    ax.axvspan(4, 7, color=band, alpha=0.16, lw=0, zorder=0)            # safe 4-7 km
    ax.axvspan(far_start, 1e4, color=far, alpha=0.13, lw=0, zorder=0)   # far / over-mitigated
    ax.axvline(1, color="#d32f2f", lw=1.4, ls=":", alpha=0.9, zorder=1) # collision

    half_w = 0.38
    bw = 0.16
    for c in cats:
        s = stats[c]
        col = color[c]
        yc = ypos[c]
        # violin silhouette (reconstructed from percentiles), now horizontal about row yc.
        # Clip to the visible x-range AND drop the near-zero-density tail so a rare far outlier
        # (e.g. a 50 km max) doesn't draw a stray thin line running to the axis edge.
        x, dens = _violin_silhouette(s)
        keep = (x <= xhi) & (dens > 0.02)
        x, dens = x[keep], dens[keep]
        if x.size:
            ax.fill_between(x, yc - dens * half_w, yc + dens * half_w,
                            color=col, alpha=0.28, lw=0, zorder=2)
            ax.plot(x, yc - dens * half_w, color=col, lw=1.0, alpha=0.7, zorder=2)
            ax.plot(x, yc + dens * half_w, color=col, lw=1.0, alpha=0.7, zorder=2)
        # EXACT box: q1..q3, median, whiskers p05..p95 (all horizontal)
        ax.add_patch(plt.Rectangle((s["q1"], yc - bw), s["q3"] - s["q1"], 2 * bw,
                                   facecolor=col, alpha=0.55, edgecolor=fg, lw=1.0, zorder=3))
        ax.plot([s["median"]] * 2, [yc - bw, yc + bw], color=fg, lw=2.0, zorder=4)
        ax.plot([s["p05"], s["q1"]], [yc, yc], color=fg, lw=1.0, zorder=3)
        ax.plot([s["q3"], s["p95"]], [yc, yc], color=fg, lw=1.0, zorder=3)
        ax.plot([s["p05"]] * 2, [yc - bw / 2, yc + bw / 2], color=fg, lw=1.0, zorder=3)
        ax.plot([s["p95"]] * 2, [yc - bw / 2, yc + bw / 2], color=fg, lw=1.0, zorder=3)
        ax.plot(s["mean"], yc, marker="D", ms=5, color="white",
                markeredgecolor=fg, markeredgewidth=0.8, zorder=5)

    # Group the two blocks: SHADE ONLY the heuristic-operator block (planners stay on white).
    # Group labels sit on the RIGHT, rotated 90 deg, spanning each block's rows.
    n_pomdp = sum(1 for c in cats if c in _POMDP)
    top = len(cats) - 1                       # y of the topmost row
    # small, uniform headroom; the "arrows" (words) style places two-line labels just above the
    # top spine, so give it a touch more room than the corner style.
    y_upper = top + (0.9 if args.band_legend in ("strip", "arrows") else 0.55)
    if 0 < n_pomdp < len(cats):
        div_y = top - n_pomdp + 0.5            # boundary between the two blocks
        ax.axhline(div_y, color=fg, lw=1.4, alpha=0.6, zorder=1)   # divider only (no block shade)
        # rotated group labels just outside the right spine, centered on the ROWS of each block
        lbl_x = 1.015
        pomdp_rows = [ypos[c] for c in cats if c in _POMDP]
        op_rows = [ypos[c] for c in cats if c not in _POMDP]
        ax.text(lbl_x, sum(pomdp_rows) / len(pomdp_rows), "Optimized planners",
                transform=ax.get_yaxis_transform(), rotation=270, ha="left", va="center",
                color=fg, fontsize=19, fontweight="bold")
        ax.text(lbl_x, sum(op_rows) / len(op_rows), "Heuristic operators",
                transform=ax.get_yaxis_transform(), rotation=270, ha="left", va="center",
                color=fg, fontsize=19, fontweight="bold")

    ax.set_yticks([ypos[c] for c in cats])
    ax.set_yticklabels([_LABEL.get(c, c) for c in cats])
    ax.set_xlabel("Miss distance at TCA (km)", color=fg)
    ax.set_ylim(-0.7, y_upper)
    ax.set_xlim(0, xhi)

    # ---- 3-band key (3 switchable styles) ----
    usetex = plt.rcParams.get("text.usetex")
    U, S, F = "$<$4\\,km" if usetex else "<4 km", "4--7\\,km" if usetex else "4-7 km", \
              "$>$10\\,km" if usetex else ">10 km"
    bands = [("Unsafe", U, unsafe, 2.0), ("Safe", S, band, 5.5), ("Far", F, far, 25.0)]
    style = args.band_legend
    if style == "corner":
        import matplotlib.patches as mpatches
        handles = [mpatches.Patch(facecolor=c, edgecolor="none", alpha=0.55,
                                  label=f"{n} ({r})") for n, r, c, _ in bands]
        leg = ax.legend(handles=handles, loc="upper right", title="Miss-distance zones",
                        frameon=True, framealpha=0.95, fontsize=17, title_fontsize=18)
        leg.get_frame().set_edgecolor(fg)
    elif style == "strip":
        # slim colored bar spanning the top, above the axes, labeled left-to-right
        for name, rng, c, cx in bands:
            lo, hi = {"Unsafe": (0, 4), "Safe": (4, 7), "Far": (10, xhi)}[name]
            ax.axvspan(lo, hi, ymin=1.005, ymax=1.055, color=c, alpha=0.85, lw=0,
                       clip_on=False, zorder=5)
            ax.text((lo + hi) / 2 if name != "Far" else (10 + xhi) / 2, y_upper + 0.55,
                    f"{name} ({rng})", ha="center", va="bottom", color=fg,
                    fontsize=16, fontweight="bold", clip_on=False)
    else:  # words above the top spine, centered over each band (no arrows). Two-line label
        # (name over range) so Unsafe/Safe fit side by side without colliding.
        word_x = {"Unsafe": 2.0, "Safe": 5.5, "Far": 25.0}
        for name, rng, c, cx in bands:
            lbl = f"{name}\\\\{rng}" if usetex else f"{name}\n{rng}"
            ax.text(word_x[name], 1.015, lbl, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", color=c, fontsize=16, fontweight="bold",
                    multialignment="center", clip_on=False)
    ax.grid(True, axis="x", color=grid, alpha=0.4, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(colors=fg)
    for sp in ax.spines.values():
        sp.set_color(fg)
    fig.text(0.5, -0.02, "Box = exact quartiles (q1/median/q3), whiskers p05--p95, "
             "$\\diamond$ = mean; violin reconstructed from reported percentiles.",
             ha="center", color=fg, fontsize=15)
    fig.tight_layout()

    fig_dir = os.path.join(_SCA, "notes", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    base = os.path.join(fig_dir, f"summary_violin_{args.tag}")
    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(base + ext, dpi=300 if ext == ".png" else None,
                    bbox_inches="tight", transparent=args.transparent)
    print(f"wrote {base}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
