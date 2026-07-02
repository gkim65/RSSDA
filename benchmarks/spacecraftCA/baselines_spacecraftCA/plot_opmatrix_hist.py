#!/usr/bin/env python
"""plot_opmatrix_hist.py — miss-distance rollout histograms for the operator-vs-operator matrix.

Reads the per-cell .npz dumps written by operator_matrix.py
  notes/results/opmatrix_<tag>/npz/<sc1>-x-<sc2>__obs-<obs>__<conj>.npz
each holding the FULL per-rollout brahe_miss_km (+ dt / dV / n_burns / reward) arrays.

Produces, for a chosen obs fidelity (or one panel per obs), OVERLAID filled histograms of the
brahe miss distance at TCA, one color per strategy PAIR, pooled across all conjunctions. The
4-7 km safe band is shaded so you can read band-occupancy at a glance and compare operator
philosophies. Matches the paper's Computer Modern typography, saves PNG + PDF + SVG, and has a
--dark theme toggle + --transparent flag for slides.

Usage (from benchmarks/spacecraftCA/):
  ../../.venv/bin/python -u baselines_spacecraftCA/plot_opmatrix_hist.py --tag opmatrix50
  # one panel per obs (perfect/tle/frozen), pooled over conjunctions:
  ../../.venv/bin/python -u baselines_spacecraftCA/plot_opmatrix_hist.py --tag opmatrix50 --by-obs
  # restrict to one obs and only the mixed pairs:
  ../../.venv/bin/python -u baselines_spacecraftCA/plot_opmatrix_hist.py --tag opmatrix50 \
      --obs tle --pairs threshold-x-selfish,threshold-x-fixedlead,selfish-x-fixedlead
"""
import argparse
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCA = os.path.dirname(_HERE)


def _setup_typography():
    import shutil
    import matplotlib.pyplot as plt
    common = {
        "font.family": "serif", "mathtext.fontset": "cm",
        "axes.titlesize": 13, "axes.labelsize": 13, "font.size": 12,
        "legend.fontsize": 11, "savefig.dpi": 300,
    }
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

# clean display labels for strategy pairs (sentence case, no underscores)
_PAIR_LABEL = {
    "threshold-x-threshold": "Threshold vs threshold",
    "threshold-x-selfish": "Threshold vs selfish",
    "threshold-x-fixedlead": "Threshold vs fixed-lead",
    "selfish-x-selfish": "Selfish vs selfish",
    "selfish-x-fixedlead": "Selfish vs fixed-lead",
    "fixedlead-x-fixedlead": "Fixed-lead vs fixed-lead",
}
# The obs axis IS the sync-availability story: perfect = sync every stage; tle = sync only at the
# ~8h GS contacts; frozen = no sync (fully decentralized). Label it by what it MEANS, not the flag.
_OBS_LABEL = {
    "perfect": "Sync at all stages",
    "tle": "Sync only at 8\\,h contacts",
    "frozen": "No sync (decentralized)",
}
# stable color per pair so panels/figures stay comparable
_PAIR_ORDER = list(_PAIR_LABEL.keys())
# distinct HATCH per pair (opposite-direction textures cross where distributions overlap, so each
# stays readable through the others -- the plot_rollout_dist.py trick). Kept stable per pair.
_PAIR_HATCH = {
    "threshold-x-threshold": "///", "selfish-x-selfish": "\\\\\\",
    "fixedlead-x-fixedlead": "...",
    "threshold-x-selfish": "///", "threshold-x-fixedlead": "\\\\\\",
    "selfish-x-fixedlead": "...",
}
# the two operator-matrix stories, as separate figures:
_SYMMETRIC = ["threshold-x-threshold", "selfish-x-selfish", "fixedlead-x-fixedlead"]
_MIXED = ["threshold-x-selfish", "threshold-x-fixedlead", "selfish-x-fixedlead"]
_GROUPS = {"symmetric": _SYMMETRIC, "mixed": _MIXED, "all": _PAIR_ORDER}


def _parse_cell(fname):
    """<sc1>-x-<sc2>__obs-<obs>__<conj>.npz -> (pair, obs, conj)."""
    stem = os.path.basename(fname)[:-4]        # drop .npz
    pair_part, rest = stem.split("__obs-", 1)
    obs, conj = rest.split("__", 1)
    return pair_part, obs, conj


def load_cells(tag, out_dir=None):
    """Return list of dicts: {pair, obs, conj, miss(array)}."""
    out_dir = out_dir or os.path.join(_SCA, "notes", "results")
    npz_dir = os.path.join(out_dir, f"opmatrix_{tag}", "npz")
    files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    if not files:
        sys.exit(f"no .npz found in {npz_dir}/ — run operator_matrix.py --tag {tag} first")
    cells = []
    for fp in files:
        pair, obs, conj = _parse_cell(fp)
        z = np.load(fp, allow_pickle=True)
        cells.append({"pair": pair, "obs": obs, "conj": conj,
                      "miss": np.asarray(z["brahe_miss_km"], dtype=float)})
    print(f"[load] {len(cells)} cells from {npz_dir}")
    return cells


def _theme_colors(dark):
    fg = "#f0f0f0" if dark else "#222222"
    band = "#39d353" if dark else "#2e7d32"     # safe-band green, legible on either bg
    grid = "#555555" if dark else "#cccccc"
    return fg, band, grid


def _pair_palette(pairs):
    import matplotlib.pyplot as plt
    # tab10 keyed by canonical pair order -> stable across panels
    cmap = plt.get_cmap("tab10")
    return {p: cmap(_PAIR_ORDER.index(p) % 10) for p in pairs}


def _hist_panel(ax, cells, pairs, palette, fg, band, grid, bins, xmax, annotate_lines=False):
    edges = np.linspace(0, xmax, bins + 1)
    # 4-7 km safe band (the target: clear screening, don't wreck the orbit)
    ax.axvspan(4, 7, color=band, alpha=0.18, lw=0, zorder=0)
    ax.axvline(4, color=band, lw=1.2, ls="--", alpha=0.8, zorder=1)
    ax.axvline(7, color=band, lw=1.2, ls="--", alpha=0.8, zorder=1)
    # collision threshold (miss < 1 km)
    ax.axvline(1, color="#d32f2f", lw=1.4, ls=":", alpha=0.9, zorder=1)
    import matplotlib as mpl
    mpl.rcParams["hatch.linewidth"] = 1.6      # bold enough that the /// \\\ textures read clearly
    for i, p in enumerate(pairs):
        miss = np.concatenate([c["miss"] for c in cells if c["pair"] == p]) \
            if any(c["pair"] == p for c in cells) else np.array([])
        if miss.size == 0:
            continue
        clipped = np.clip(miss, 0, xmax)
        # FRACTION of rollouts per bin (weights sum to 1) -> intuitive "% of rollouts landed here",
        # comparable across pairs even with different n.
        w = np.ones_like(clipped) / clipped.size
        color = palette[p]
        hatch = _PAIR_HATCH.get(p, "///")
        z = 2 + 3 * i
        # THREE layers so overlapping pairs stay individually readable (plot_rollout_dist trick):
        # (1) faint solid fill for mass, (2) FULL-alpha colored hatch on top that cross-textures
        # through overlaps, (3) a crisp outline. Opposite-direction hatches cross in shared bins.
        ax.hist(clipped, bins=edges, weights=w, histtype="stepfilled",
                facecolor=color, alpha=0.16, edgecolor="none", zorder=z)
        ax.hist(clipped, bins=edges, weights=w, histtype="stepfilled",
                facecolor="none", edgecolor=color, hatch=hatch, linewidth=0.0,
                alpha=0.85, zorder=z + 1, label=_PAIR_LABEL.get(p, p))
        ax.hist(clipped, bins=edges, weights=w, histtype="step",
                edgecolor=color, linewidth=1.6, zorder=z + 2)
    if annotate_lines:
        ymax = ax.get_ylim()[1]
        import matplotlib.pyplot as plt
        usetex = plt.rcParams.get("text.usetex")
        coll_txt = "Collision ($<$1\\,km)" if usetex else "Collision ($<$1 km)"
        band_txt = "Safe band 4--7\\,km" if usetex else "Safe band 4-7 km"
        ax.text(1.3, ymax * 0.58, coll_txt, color="#d32f2f",
                rotation=90, va="top", ha="left", fontsize=9)
        # safe-band label: bold, boxed, sitting just above the band so it's unmissable
        ax.text(5.5, ymax * 0.90, band_txt, color=band,
                va="top", ha="center", fontsize=10.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=band, lw=1.0, alpha=0.9))
    ax.set_xlim(0, xmax)
    ax.tick_params(colors=fg)
    for s in ax.spines.values():
        s.set_color(fg)
    ax.grid(True, color=grid, alpha=0.4, lw=0.5)
    ax.set_axisbelow(True)


def _save(fig, name, tag, transparent):
    fig_dir = os.path.join(_SCA, "notes", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    base = os.path.join(fig_dir, f"opmatrix_hist_{name}_{tag}")
    saved = []
    for ext in (".png", ".pdf", ".svg"):
        p = base + ext
        fig.savefig(p, dpi=300 if ext == ".png" else None,
                    bbox_inches="tight", transparent=transparent)
        saved.append(p)
    print("wrote " + ", ".join(saved))


def _make_figure(cells, pairs, obs_list, palette, fg, band, grid, args):
    """One stacked by-obs figure for a given SET of strategy pairs. Returns the Figure."""
    import matplotlib.pyplot as plt
    n = len(obs_list)
    # shorter panels (2.15 in each) per the "y-axis a bit shorter" ask; headroom for the legend.
    fig, axes = plt.subplots(n, 1, figsize=(8.0, 2.15 * n + 1.0), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for i, (ax, obs) in enumerate(zip(axes, obs_list)):
        sub = [c for c in cells if c["obs"] == obs]
        _hist_panel(ax, sub, pairs, palette, fg, band, grid, args.bins, args.xmax,
                    annotate_lines=(i == 0))
        ax.set_ylabel("Fraction of rollouts", color=fg)
        # panel label CENTERED (top-center, ~x=22km) where the data is sparse -> nothing behind it
        ax.text(0.5, 0.90, _OBS_LABEL.get(obs, obs), transform=ax.transAxes,
                color=fg, fontsize=13, va="top", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white" if not args.dark else "#111111",
                          ec=fg, alpha=0.9, lw=0.6))
    axes[-1].set_xlabel("Miss distance at TCA (km)", color=fg)
    handles, labels = axes[0].get_legend_handles_labels()
    ncol = min(3, max(1, len(labels)))
    nrows_leg = int(np.ceil(len(labels) / ncol)) if labels else 1
    fig.legend(handles, labels, loc="upper center", ncol=ncol,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    top = 1.0 - (0.04 + 0.03 * nrows_leg) / (n + 0.4)
    fig.tight_layout(rect=(0, 0, 1, top))
    return fig


def main():
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="operator_matrix run tag (opmatrix_<tag>/)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--obs", default="tle", help="single obs to plot (perfect|tle|frozen). "
                    "Ignored with --by-obs.")
    ap.add_argument("--by-obs", action="store_true",
                    help="one panel per obs fidelity (perfect/tle/frozen), shared axes.")
    ap.add_argument("--group", default="both",
                    choices=["symmetric", "mixed", "both", "all"],
                    help="which strategy pairs: symmetric (self-vs-self), mixed (different "
                         "operators), both (two SEPARATE figures), or all (one combined figure).")
    ap.add_argument("--pairs", default=None,
                    help="explicit comma list of pairs (overrides --group).")
    ap.add_argument("--bins", type=int, default=45)
    ap.add_argument("--xmax", type=float, default=40.0, help="miss-axis clip (km).")
    ap.add_argument("--dark", action="store_true", help="dark-background theme.")
    ap.add_argument("--transparent", action="store_true",
                    help="save with transparent background (drop on any slide/page color).")
    args = ap.parse_args()

    cells = load_cells(args.tag, args.out_dir)
    fg, band, grid = _theme_colors(args.dark)
    if args.dark:
        plt.rcParams.update({"axes.facecolor": "none", "figure.facecolor": "none",
                             "text.color": fg, "axes.labelcolor": fg,
                             "xtick.color": fg, "ytick.color": fg})

    obs_list = ["perfect", "tle", "frozen"] if args.by_obs else [args.obs]
    obs_list = [o for o in obs_list if any(c["obs"] == o for c in cells)]

    # decide which figures to render: explicit --pairs, or one/both of the two groups.
    if args.pairs:
        jobs = [("custom", [p.strip() for p in args.pairs.split(",")])]
    elif args.group == "both":
        jobs = [("symmetric", _SYMMETRIC), ("mixed", _MIXED)]
    else:
        jobs = [(args.group, _GROUPS[args.group])]

    obs_suffix = "byobs" if args.by_obs else args.obs
    for gname, gpairs in jobs:
        pairs = [p for p in gpairs if any(c["pair"] == p for c in cells)]
        if not pairs:
            print(f"[skip] no cells for group '{gname}' pairs {gpairs}")
            continue
        palette = _pair_palette(pairs)
        fig = _make_figure(cells, pairs, obs_list, palette, fg, band, grid, args)
        _save(fig, f"{gname}_{obs_suffix}", args.tag, args.transparent)
        plt.close(fig)


if __name__ == "__main__":
    main()
