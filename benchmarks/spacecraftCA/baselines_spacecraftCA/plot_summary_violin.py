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
  "threshold-x-threshold": {"n": 31200, "mean": 20.112, "min": 2.186, "q1": 9.548,
    "median": 17.136, "q3": 34.192, "max": 43.909, "p05": 5.001, "p95": 36.5,
    "coll_pct": 0.0, "band_in_pct": 17.1, "dv_mean": 0.5821},
  "threshold-x-selfish": {"n": 31200, "mean": 19.492, "min": 2.186, "q1": 9.548,
    "median": 17.036, "q3": 33.933, "max": 43.545, "p05": 5.001, "p95": 36.434,
    "coll_pct": 0.0, "band_in_pct": 17.5, "dv_mean": 0.5567},
  "threshold-x-fixedlead": {"n": 31200, "mean": 12.508, "min": 2.186, "q1": 7.788,
    "median": 10.311, "q3": 18.461, "max": 30.076, "p05": 5.001, "p95": 24.938,
    "coll_pct": 0.0, "band_in_pct": 20.8, "dv_mean": 0.7141},
  "selfish-x-selfish": {"n": 31200, "mean": 18.971, "min": 2.186, "q1": 9.362,
    "median": 16.975, "q3": 33.933, "max": 43.545, "p05": 4.782, "p95": 36.434,
    "coll_pct": 0.0, "band_in_pct": 21.3, "dv_mean": 0.5312},
  "selfish-x-fixedlead": {"n": 31200, "mean": 11.82, "min": 2.186, "q1": 6.356,
    "median": 10.236, "q3": 17.071, "max": 30.076, "p05": 4.528, "p95": 24.938,
    "coll_pct": 0.0, "band_in_pct": 25.3, "dv_mean": 0.6806},
  "fixedlead-x-fixedlead": {"n": 31200, "mean": 11.166, "min": 2.186, "q1": 6.657,
    "median": 10.188, "q3": 15.349, "max": 24.726, "p05": 4.782, "p95": 19.489,
    "coll_pct": 0.0, "band_in_pct": 25.0, "dv_mean": 1.0638},
}

# display order + clean labels; POMDP first (left group), then the 6 operator pairs GROUPED:
# the 3 SAME-vs-SAME (self-play) pairs first, then the 3 DIFFERENT-operator (cross-play) pairs.
_POMDP = ["centralized", "sdec", "dec"]
_SAME = ["threshold-x-threshold", "selfish-x-selfish", "fixedlead-x-fixedlead"]
_MIX = ["threshold-x-selfish", "threshold-x-fixedlead", "selfish-x-fixedlead"]
_PAIRS = _SAME + _MIX
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
    common = {"font.family": "serif", "mathtext.fontset": "cm", "axes.titlesize": 30,
              "axes.labelsize": 30, "font.size": 26, "legend.fontsize": 24,
              "xtick.labelsize": 26, "ytick.labelsize": 27, "savefig.dpi": 300}
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
    import matplotlib.patches as mpatches
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=None, help="JSON stats file (from summarize_npz.py). "
                    "If omitted, uses the embedded STATS.")
    ap.add_argument("--tag", default="summary")
    ap.add_argument("--band-legend", default="arrows", choices=["corner", "strip", "arrows"],
                    help="how to show the 3-band key: corner legend / top strip / "
                         "'arrows' = two-line words above each band (default).")
    ap.add_argument("--ymax", type=float, default=None, help="miss-axis cap (km); default auto.")
    ap.add_argument("--no-violin", action="store_true",
                    help="box plot only (drop the reconstructed violin silhouettes).")
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
    # POMDP: blue family. Operator heuristics: two cohesive families so self-play vs cross-play
    # read at a glance -- SAME-vs-SAME in a TEAL/GREEN family, DIFFERENT-operators in a PURPLE
    # family (each shaded light->dark within its group).
    pomdp_col = {"centralized": "#1b4f72", "sdec": "#2874a6", "dec": "#5dade2"}
    same_col = {"threshold-x-threshold": "#7b1e1e", "selfish-x-selfish": "#c0392b",
                "fixedlead-x-fixedlead": "#e8845b"}       # maroon->orange family (self-play)
    mix_col = {"threshold-x-selfish": "#5e3c99", "threshold-x-fixedlead": "#8060c0",
               "selfish-x-fixedlead": "#b39ddb"}          # purple family (cross-play)
    color = {**pomdp_col, **same_col, **mix_col}
    color = {c: color.get(c, "#888888") for c in cats}

    # HORIZONTAL layout: miss on X, methods down Y. Row 0 at TOP -> order reads top-to-bottom.
    rows = list(range(len(cats)))
    ypos = {c: len(cats) - 1 - i for i, c in enumerate(cats)}   # flip so first cat is on top
    # wide landscape ~16:9 (width x height) -- more rectangular than 4:3 so rows spread out
    fig_w = 18.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 9.0 / 16.0))

    xmax_data = max(stats[c]["max"] for c in cats)
    # The visible box/whisker extent (q3 and p95) sets where the data actually ends; the
    # "% in target band" callouts get parked in a CLEAR zone to the right of this so they
    # never sit on top of a wide operator box (which hid some boxes entirely before).
    data_right = max(max(stats[c]["q3"], stats[c].get("p95", stats[c]["q3"])) for c in cats)
    pct_x = data_right + 1.5                     # callout LEFT edge: just past the widest box
    # End the axis just past the callout text so there's no large empty band on the right. The
    # callout ("NN% in target band") spans ~pct_text_w in data units at this font; +0.5 trailing.
    pct_text_w = 11.5
    xhi = args.ymax or (pct_x + pct_text_w + 0.5)
    # bands span the FULL x-range as vertical strips (miss on x now)
    far_start = 10.0
    ax.axvspan(0, 4, color=unsafe, alpha=0.13, lw=0, zorder=0)          # unsafe 0-4 km
    ax.axvspan(4, 7, color=band, alpha=0.16, lw=0, zorder=0)            # safe 4-7 km
    ax.axvspan(far_start, 1e4, color=far, alpha=0.13, lw=0, zorder=0)   # far / over-mitigated
    ax.axvline(1, color="#d32f2f", lw=1.4, ls=":", alpha=0.9, zorder=1) # collision

    usetex = plt.rcParams.get("text.usetex")
    half_w = 0.38
    bw = 0.24 if args.no_violin else 0.16   # fatter boxes when there's no violin around them
    for c in cats:
        s = stats[c]
        col = color[c]
        yc = ypos[c]
        # violin silhouette (reconstructed from percentiles), now horizontal about row yc.
        # Clip to the visible x-range AND drop the near-zero-density tail so a rare far outlier
        # (e.g. a 50 km max) doesn't draw a stray thin line running to the axis edge.
        if not args.no_violin:
            x, dens = _violin_silhouette(s)
            keep = (x <= xhi) & (dens > 0.02)
            x, dens = x[keep], dens[keep]
            if x.size:
                ax.fill_between(x, yc - dens * half_w, yc + dens * half_w,
                                color=col, alpha=0.28, lw=0, zorder=2)
                ax.plot(x, yc - dens * half_w, color=col, lw=1.0, alpha=0.7, zorder=2)
                ax.plot(x, yc + dens * half_w, color=col, lw=1.0, alpha=0.7, zorder=2)
        # EXACT box: q1..q3, median, whiskers p05..p95 (all horizontal). Strong FULL-opacity
        # fill with a darker same-hue edge; whiskers/caps in that darker hue for a cohesive look.
        import matplotlib.colors as mcolors
        rgb = mcolors.to_rgb(col)
        dark = tuple(ch * 0.6 for ch in rgb)      # darker shade of the box color for edges
        box_alpha = 1.0 if args.no_violin else 0.75
        # rounded-corner box. rounding_size is in the box's LOCAL units after mutation_aspect
        # scaling; keep mutation_aspect low so the corner radius stays modest and the ends read as
        # softly rounded, not capsule-shaped.
        ax.add_patch(mpatches.FancyBboxPatch(
            (s["q1"], yc - bw), s["q3"] - s["q1"], 2 * bw,
            boxstyle="round,pad=0,rounding_size=0.10", mutation_aspect=3,
            facecolor=col, alpha=box_alpha, edgecolor=dark, lw=3.0, zorder=3))
        # median: fully-white line, kept INSIDE the box outline (shortened so it doesn't touch
        # the top/bottom edges) so it reads as a clean divider within the rectangle.
        ax.plot([s["median"]] * 2, [yc - bw * 0.78, yc + bw * 0.78], color="white", lw=3.2,
                zorder=5, solid_capstyle="round")
        # whiskers p05..p95 in the darker hue, rounded caps
        for x0, x1 in [(s["p05"], s["q1"]), (s["q3"], s["p95"])]:
            ax.plot([x0, x1], [yc, yc], color=dark, lw=1.6, zorder=3, solid_capstyle="round")
        for xw in (s["p05"], s["p95"]):
            ax.plot([xw] * 2, [yc - bw * 0.42, yc + bw * 0.42], color=dark, lw=2.0, zorder=3,
                    solid_capstyle="round")
        # mean marker: filled diamond, dark edge
        ax.plot(s["mean"], yc, marker="D", ms=7, color="white",
                markeredgecolor=dark, markeredgewidth=1.3, zorder=6)
        # % of rollouts within the target band (4-7 km), printed in green just inside the right
        # spine so each row carries its headline number.
        pct_lbl = (f"{s.get('band_in_pct', 0):.0f}\\%\\ in target band"
                   if usetex else f"{s.get('band_in_pct', 0):.0f}% in target band")
        # placed in the CLEAR zone to the right of the widest box (pct_x) so the callout never
        # covers a box plot. Left-aligned from pct_x so all rows' labels line up.
        ax.text(pct_x, yc, pct_lbl, ha="left", va="center",
                color=band, fontsize=19, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.28", fc="white" if not args.dark else "#111111",
                          ec=band, lw=0.8, alpha=0.92), zorder=6)

    # Group the two blocks: SHADE ONLY the heuristic-operator block (planners stay on white).
    # Group labels sit on the RIGHT, rotated 90 deg, spanning each block's rows.
    n_pomdp = sum(1 for c in cats if c in _POMDP)
    top = len(cats) - 1                       # y of the topmost row
    # small, uniform headroom; the "arrows" (words) style places two-line labels just above the
    # top spine, so give it a touch more room than the corner style.
    y_upper = top + (1.05 if args.band_legend in ("strip", "arrows") else 0.55)
    _group_labels = []
    if 0 < n_pomdp < len(cats):
        div_y = top - n_pomdp + 0.5            # boundary between the two blocks
        ax.axhline(div_y, color=fg, lw=1.4, alpha=0.6, zorder=1)   # divider only (no block shade)
        # rotated group labels just outside the right spine, centered on the ROWS of each block.
        # Drawn in FIGURE coords (via _group_labels below) so bbox_inches="tight" measures their
        # rotated extent reliably -- axes-coord rotated usetex text was being cropped at the edge.
        pomdp_rows = [ypos[c] for c in cats if c in _POMDP]
        op_rows = [ypos[c] for c in cats if c not in _POMDP]
        _group_labels = [
            (sum(pomdp_rows) / len(pomdp_rows), "Optimized planners"),
            (sum(op_rows) / len(op_rows), "Representative operator heuristics"),
        ]

    ax.set_yticks([ypos[c] for c in cats])
    ax.set_yticklabels([_LABEL.get(c, c) for c in cats])
    ax.set_xlabel("Miss distance at TCA (km)", color=fg)
    ax.set_ylim(-0.7, y_upper)
    ax.set_xlim(0, xhi)

    # ---- 3-band key (3 switchable styles) ----
    U, S, F = "$<$4\\,km" if usetex else "<4 km", "4--7\\,km" if usetex else "4-7 km", \
              "$>$10\\,km" if usetex else ">10 km"
    bands = [("Unsafe", U, unsafe, 2.0), ("Safe", S, band, 5.5),
             ("Over-mitigated", F, far, 25.0)]
    style = args.band_legend
    if style == "corner":
        handles = [mpatches.Patch(facecolor=c, edgecolor="none", alpha=0.55,
                                  label=f"{n} ({r})") for n, r, c, _ in bands]
        leg = ax.legend(handles=handles, loc="upper right", title="Miss-distance zones",
                        frameon=True, framealpha=0.95, fontsize=23, title_fontsize=24)
        leg.get_frame().set_edgecolor(fg)
    elif style == "strip":
        # slim colored bar spanning the top, above the axes, labeled left-to-right
        # spread the narrow Unsafe/Safe labels apart and stack them two-line so they don't
        # collide under the strip (single-line "Name (range)" is wider than those bands).
        strip_x = {"Unsafe": 1.2, "Safe": 6.0, "Over-mitigated": (10.0 + xhi) / 2.0}
        for name, rng, c, cx in bands:
            lo, hi = {"Unsafe": (0, 4), "Safe": (4, 7), "Over-mitigated": (10, xhi)}[name]
            ax.axvspan(lo, hi, ymin=1.005, ymax=1.055, color=c, alpha=0.85, lw=0,
                       clip_on=False, zorder=5)
            lbl = (f"{name} ({rng})" if name == "Over-mitigated"
                   else (f"{name}\\\\{rng}" if usetex else f"{name}\n{rng}"))
            ax.text(strip_x[name], y_upper + 0.55, lbl, ha="center", va="bottom", color=fg,
                    fontsize=22, fontweight="bold", multialignment="center", clip_on=False)
    else:  # words above the top spine, centered over each band (no arrows). Two-line label
        # (name over range) so Unsafe/Safe fit side by side without colliding.
        # nudge Unsafe/Safe just enough apart to not collide at the larger font while staying
        # over their own bands (0-4 / 4-7); center Over-mitigated on the wide >10 km band.
        word_x = {"Unsafe": 1.2, "Safe": 6.0, "Over-mitigated": (10.0 + xhi) / 2.0}
        for name, rng, c, cx in bands:
            # Unsafe/Safe stay two-line (tight side by side); the wide band has room for one line.
            if name == "Over-mitigated":
                lbl = f"{name} ({rng})"
            else:
                lbl = f"{name}\\\\{rng}" if usetex else f"{name}\n{rng}"
            ax.text(word_x[name], 1.015, lbl, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", color=c, fontsize=27, fontweight="bold",
                    multialignment="center", clip_on=False)
    ax.grid(True, axis="x", color=grid, alpha=0.4, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(colors=fg)
    for sp in ax.spines.values():
        sp.set_color(fg)
    caption = ("Box = exact quartiles (q1/median/q3), whiskers p05--p95, $\\diamond$ = mean"
               + ("." if args.no_violin
                  else "; violin reconstructed from reported percentiles."))
    fig.text(0.5, -0.02, caption, ha="center", color=fg, fontsize=21)
    fig.tight_layout()
    # Reserve a right margin, then draw the rotated block labels in FIGURE coords inside it.
    # (Axes-coord rotated usetex text was cropped by bbox_inches="tight"; figure text isn't.)
    if _group_labels:
        fig.subplots_adjust(right=0.9)
        pos = ax.get_position()
        lbl_fx = pos.x1 + 0.018
        for y_row, txt in _group_labels:
            # data-y (row) -> figure-y fraction through the axes' data->figure transform
            _, fy = fig.transFigure.inverted().transform(ax.transData.transform((0, y_row)))
            fig.text(lbl_fx, fy, txt, rotation=270, ha="left", va="center",
                     color=fg, fontsize=20, fontweight="bold")

    fig_dir = os.path.join(_SCA, "notes", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    base = os.path.join(fig_dir, f"summary_violin_{args.tag}")
    # bbox_inches="tight" + pad_inches expands the canvas to include the figure-coord group
    # labels on the right and the caption below; pad keeps them off the very edge.
    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(base + ext, dpi=300 if ext == ".png" else None,
                    bbox_inches="tight", pad_inches=0.3, transparent=args.transparent)
    print(f"wrote {base}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
