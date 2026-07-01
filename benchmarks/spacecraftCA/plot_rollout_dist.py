"""
plot_rollout_dist.py — rebuild per-rollout DISTRIBUTIONS from a sweep's saved raw dumps.

The sweep summary CSV keeps only 8 scalars per cell (mean/min/max miss, collision%, the 4/7 km
band split). When sweep_driver is run with --save-rollouts, each cell ALSO drops one .npz with the
FULL per-rollout arrays (200 brahe miss / dt / dV / n_burns / matrix error) into
  notes/results/rollouts_<tag>/
keyed by filename = the cell's 7-tuple. THIS script globs those .npz, parses the key back into
columns, and gives you (a) a tidy long DataFrame for any custom cut, and (b) ready-made histogram /
violin figures of brahe_miss_km grouped by variant. No sweep re-run needed — the raw data is on disk.

Usage:
  # tidy DataFrame -> CSV (one row per rollout; join/group however you like):
  .venv/bin/python benchmarks/spacecraftCA/plot_rollout_dist.py --tag sweep400_drag --to-csv

  # overlaid histogram of brahe miss by variant (all conjunctions/beliefs pooled):
  .venv/bin/python benchmarks/spacecraftCA/plot_rollout_dist.py --tag sweep400_drag --hist

  # one violin per variant, faceted by angle, for a single belief:
  .venv/bin/python benchmarks/spacecraftCA/plot_rollout_dist.py --tag sweep400_drag \
      --violin --filter init_miss=0.5 init_spread=1.4

  # per-stage burn-rate curve by variant (WHEN each agent burns; needs a post-burn-timing sweep):
  .venv/bin/python benchmarks/spacecraftCA/plot_rollout_dist.py --tag sweep400_drag --burn-timing
"""
import os, sys, glob, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# Typography: match the paper figures (Computer Modern serif). Uses real LaTeX
# if `latex` is on PATH, else matplotlib's mathtext-cm. Mirrors plot_v2_concept
# so labels here share the paper body's serif. Idempotent; call once at import.
# --------------------------------------------------------------------------
def _setup_typography():
    import shutil
    import matplotlib.pyplot as plt
    common = {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "font.size": 11,
        "savefig.dpi": 300,
    }
    if shutil.which("latex"):
        try:
            plt.rcParams.update({"text.usetex": True,
                                 "font.serif": ["Computer Modern Roman"], **common})
            fig = plt.figure(); fig.text(0.5, 0.5, r"$\delta T$"); fig.canvas.draw()
            plt.close(fig)
            print("[typography] using real LaTeX (Computer Modern)")
            return
        except Exception as e:
            plt.rcParams["text.usetex"] = False
            print(f"[typography] LaTeX render failed ({e}); falling back to mathtext-cm")
    plt.rcParams.update({"text.usetex": False,
                         "font.serif": ["CMU Serif", "cmr10", "STIXGeneral", "DejaVu Serif"],
                         **common})
    print("[typography] using mathtext Computer Modern (no usetex)")


_setup_typography()

# Cell-key field order — MUST match _conj_worker._cell_key / sweep_driver.cell_key.
KEY_FIELDS = ["label", "miss_km", "angle_deg", "v_rel_ms", "init_miss", "init_spread", "variant"]
# numeric per-rollout SCALAR arrays stored in each .npz (besides cell_key). These are the
# (n_rollouts,) columns that load_long explodes to one-row-per-rollout. The (n_rollouts,
# N_STAGES) burn matrices (burn_a1/burn_a2) are 2-D and are handled by load_burns instead.
ARRAYS = ["brahe_miss_km", "brahe_dt_km", "matrix_miss_km", "matrix_dt_km",
          "total_dv", "n_burns", "true_term_reward"]


def rollout_dir_for(tag, out_dir=None):
    out_dir = out_dir or os.path.join(_HERE, "notes", "results")
    return os.path.join(out_dir, f"rollouts_{tag}")


def load_long(tag, out_dir=None, filters=None):
    """Glob the tag's .npz dumps -> one tidy long DataFrame (ONE ROW PER ROLLOUT). The cell-key
    columns come from the filename (round-tripped via the stored cell_key array, which is exact);
    the per-rollout arrays are exploded so each rollout is its own row. `filters` is a dict of
    key-field -> value (string-compared) to subset cells before loading."""
    import pandas as pd
    rdir = rollout_dir_for(tag, out_dir)
    files = sorted(glob.glob(os.path.join(rdir, "*.npz")))
    if not files:
        sys.exit(f"no rollout dumps found in {rdir}/ — was the sweep run with --save-rollouts?")
    frames = []
    for fp in files:
        z = np.load(fp, allow_pickle=False)
        key = {k: v for k, v in zip(KEY_FIELDS, [str(x) for x in z["cell_key"]])}
        if filters and any(str(key.get(k)) != str(v) for k, v in filters.items()):
            continue
        n = len(z["brahe_miss_km"])
        d = {k: [v] * n for k, v in key.items()}
        for a in ARRAYS:
            d[a] = z[a] if a in z.files else [np.nan] * n
        frames.append(pd.DataFrame(d))
    if not frames:
        sys.exit(f"no cells matched filters {filters}")
    df = pd.concat(frames, ignore_index=True)
    for c in ("miss_km", "angle_deg", "v_rel_ms", "init_miss", "init_spread"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# pretty axis labels per column (avoids raw underscore-laden field names in the figure).
_COL_LABELS = {
    "brahe_miss_km": "Miss distance at TCA (km)",
    "total_dv": r"Total $\Delta v$ (m/s)",
    "n_burns": "Number of burns",
    "true_term_reward": "Terminal reward",
}


def _tex_safe(s):
    """Escape underscores so raw strings (tags, field names) don't turn into LaTeX subscripts
    when text.usetex is on. No-op when usetex is off, but harmless either way."""
    import matplotlib.pyplot as plt
    return s.replace("_", r"\_") if plt.rcParams.get("text.usetex") else s


def _col_label(col):
    return _COL_LABELS.get(col, _tex_safe(col))


def _save(fig, tag, name, vector=True, transparent=False, exts=(".pdf", ".svg")):
    """Save `rollout_<name>_<tag>.png` plus, when vector=True, crisp vector copies (PDF for the
    paper, SVG for slides/web) at the same base name -- matching the FIGURES.md convention. Set
    transparent=True to drop the figure onto any slide/page color."""
    fig_dir = os.path.join(_HERE, "notes", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    base = os.path.join(fig_dir, f"rollout_{name}_{tag}.png")
    saved = [base]
    fig.savefig(base, dpi=200, bbox_inches="tight", transparent=transparent)
    if vector:
        for ext in exts:
            vpath = os.path.splitext(base)[0] + ext
            fig.savefig(vpath, bbox_inches="tight", transparent=transparent)
            saved.append(vpath)
    print("wrote " + ", ".join(saved))


def plot_hist(df, tag, col="brahe_miss_km", bins=40):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    lo, hi = np.nanpercentile(df[col], [0.5, 99.5])
    edges = np.linspace(lo, hi, bins + 1)
    for v, g in df.groupby("variant"):
        ax.hist(g[col], bins=edges, histtype="step", linewidth=1.8, label=f"{v} (n={len(g)})")
    ax.axvspan(4.0, 7.0, color="green", alpha=0.07, label="Ideal zone at TCA (4--7 km)")
    ax.set_xlabel(_col_label(col)); ax.set_ylabel("Rollouts")
    ax.set_title(f"{_col_label(col)} distribution by variant")
    ax.legend(fontsize=8)
    _save(fig, tag, f"hist_{col}")


def plot_violin(df, tag, col="brahe_miss_km"):
    import matplotlib.pyplot as plt
    variants = sorted(df["variant"].unique())
    data = [df[df["variant"] == v][col].values for v in variants]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.violinplot(data, showmedians=True)
    ax.set_xticks(range(1, len(variants) + 1)); ax.set_xticklabels(variants)
    ax.axhspan(4.0, 7.0, color="green", alpha=0.07)
    ax.set_ylabel(_col_label(col)); ax.set_title(f"{_col_label(col)} by variant")
    _save(fig, tag, f"violin_{col}")
    # (title/labels already start capitalized via _col_label)


# The conj_sweep_spherical_50.json geometries were built at 4 discrete target misses
# (families m1/m2/m5/m10). We facet the miss-shift figure by which family a rollout STARTED
# in, so each starting level's before->after is visible instead of pooling all of them.
INIT_FAMILIES = [1.0, 2.0, 5.0, 10.0]
# color per variant (final histogram); the "initial spread" reference is always grey.
_VARIANT_COLORS = {"centralized": "#1f77b4", "sdec": "#2ca02c", "dec": "#d62728"}
# pretty display names for legends (raw cell-key values -> paper labels).
_VARIANT_LABELS = {"centralized": "Centralized", "sdec": "Semi-Decentralized",
                   "dec": "Decentralized"}


def _variant_label(v):
    return _VARIANT_LABELS.get(v, v)
# per-variant line style + hatch + draw order. Centralized and SDec produce near-identical
# final distributions (SDec with all GS contacts recovers the centralized policy), so they get
# OPPOSITE-DIRECTION hatches ('///' vs '\\\') that cross in the overlap region -- you can see
# both textures where the two coincide instead of one hiding under the other. Dec is left as a
# plain fill (no hatch) so its separate right-shifted mass reads as a solid block. Centralized
# is drawn LAST with a dashed outline on top. (ls, lw, zorder, hatch)
# dec and sdec are SOLID color fills (no hatch); centralized is drawn LAST as a hatch-only
# overlay (blue \\\, clear background) plus a short-dashed blue outline on top -- so where the
# centralized policy coincides with sdec you see the blue hatch riding over the solid green.
# ls uses an explicit (on, off) dash pattern so centralized reads as short '- - -' dashes.
_VARIANT_STYLE = {
    "dec":         dict(ls="-", lw=1.9, zorder=2, hatch=None,     face="self", outline="#7f0000"),
    "sdec":        dict(ls="-", lw=2.4, zorder=3, hatch=None,     face="self", outline="#0b4d0b"),
    "centralized": dict(ls=(0, (4, 3)), lw=1.8, zorder=4, hatch="\\\\\\", face="none", outline="self"),
}
# draw dec, then sdec, then centralized-dashed-on-top (regardless of alphabetical order).
_DRAW_ORDER = ["dec", "sdec", "centralized"]


def _ordered_variants(variants):
    """Order present variants for drawing: known ones by _DRAW_ORDER (so centralized is last /
    on top), any unknown ones appended alphabetically."""
    known = [v for v in _DRAW_ORDER if v in variants]
    rest = sorted(v for v in variants if v not in _DRAW_ORDER)
    return known + rest


def _nearest_family(miss, families=INIT_FAMILIES, tol=0.35):
    """Snap a numeric initial miss to its nearest design family (1/2/5/10 km) within a
    relative tolerance; return None if it doesn't belong to any (so odd cells are dropped
    from the facet rather than silently mislabeled)."""
    if miss is None or not np.isfinite(miss):
        return None
    best = min(families, key=lambda f: abs(miss - f))
    return best if abs(miss - best) <= tol * best else None


def _initial_by_family(json_path, families=INIT_FAMILIES):
    """Return {family: array-of-initial-misses} from a conj-sweep JSON, using the true
    no-maneuver closest-approach miss (conj_initial_miss.initial_misses). Only conjunctions
    that snap to a design family are kept. This is the 'where we started' reference the
    final distributions are drawn against."""
    from conj_initial_miss import initial_misses
    df = initial_misses(json_path)
    out = {f: [] for f in families}
    for m in df["miss_km"].to_numpy():
        fam = _nearest_family(m, families)
        if fam is not None:
            out[fam].append(float(m))
    return {f: np.array(v) for f, v in out.items() if len(v)}


def plot_miss_shift(df, tag, conj_json=None, col="brahe_miss_km", bins=40, families=INIT_FAMILIES):
    """Before->after miss figure: a grid of subplots, columns = initial-miss family
    (1/2/5/10 km), rows = variant. Each subplot draws the INITIAL spread (grey, filled) that
    the family started at, and the variant's FINAL `col` distribution (variant color, filled)
    on top -- so you read 'started here, this method pushed it there'. Collision line (1 km)
    and safe band (4-7 km) marked on every panel.

    The initial reference comes from `conj_json` (recomputed no-maneuver misses via
    conj_initial_miss); if not given, falls back to the cell-key `init_miss`/`miss_km` per
    family as a rough marker line. Families with no rollouts in `df` are dropped, so a
    single-family stand-in tag (e.g. peel_ready at 5 km) renders as one column."""
    import matplotlib.pyplot as plt

    # which families are actually present in the rollout data (snap the cell's miss_km)?
    df = df.copy()
    df["_fam"] = df["miss_km"].map(lambda m: _nearest_family(m, families))
    present_fams = [f for f in families if (df["_fam"] == f).any()]
    if not present_fams:
        # no cell snapped to a family (unusual miss values) -> pool everything into one column
        present_fams = ["all"]
        df["_fam"] = "all"

    variants = sorted(df["variant"].unique())
    init_ref = _initial_by_family(conj_json, families) if conj_json else {}

    # shared x-range across all panels so before/after are visually comparable
    finite = df[col].to_numpy()
    finite = finite[np.isfinite(finite)]
    if conj_json and init_ref:
        finite = np.concatenate([finite] + [v for v in init_ref.values()])
    lo, hi = np.nanpercentile(finite, [0.5, 99.5])
    edges = np.linspace(lo, hi, bins + 1)

    nrow, ncol = len(variants), len(present_fams)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.6 * nrow),
                             sharex=True, sharey="row", squeeze=False)

    for r, v in enumerate(variants):
        for c, fam in enumerate(present_fams):
            ax = axes[r][c]
            sub = df[(df["variant"] == v) & (df["_fam"] == fam)]
            # initial reference for this family (recomputed misses if available)
            if isinstance(fam, float) and fam in init_ref:
                ax.hist(init_ref[fam], bins=edges, color="0.6", alpha=0.55,
                        label=f"Initial (n={len(init_ref[fam])})")
            elif isinstance(fam, float):
                ax.axvline(fam, color="0.5", ls="--", lw=1.2, label="Initial (target)")
            # final distribution for this variant/family
            if len(sub):
                ax.hist(sub[col].to_numpy(), bins=edges, color=_VARIANT_COLORS.get(v, "0.2"),
                        alpha=0.75, label=f"{v} final (n={len(sub)})")
            ax.axvspan(4.0, 7.0, color="green", alpha=0.07)
            if r == 0:
                title = f"Initial Miss Distance: {fam:g} km" if isinstance(fam, float) else "All"
                ax.set_title(title, fontsize=11)
            if c == 0:
                ax.set_ylabel(f"{v}\nrollouts", fontsize=10)
            if r == nrow - 1:
                ax.set_xlabel(_col_label(col), fontsize=10)
            ax.legend(fontsize=6, loc="upper right")

    fig.suptitle(f"Initial vs. final {_col_label(col)} by variant and starting miss",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    _save(fig, tag, "miss_shift")


def _present_families(df, families=INIT_FAMILIES, pool=False):
    """Tag each row of `df` with its starting-miss family (_fam) and return (df, fam_list).
    With pool=True (or when no cell snaps to a design family) everything collapses to a
    single 'all' family so the figure is one column."""
    df = df.copy()
    if pool:
        df["_fam"] = "all"
        return df, ["all"]
    df["_fam"] = df["miss_km"].map(lambda m: _nearest_family(m, families))
    present = [f for f in families if (df["_fam"] == f).any()]
    if not present:
        df["_fam"] = "all"
        return df, ["all"]
    return df, present


def _shared_edges(df, col, conj_json, families, bins):
    """A common set of histogram bin edges across every panel so before/after and
    variant-to-variant are visually comparable. The lower edge is floored at the smallest
    initial-miss family value (with a little margin) so every 'Initial N km' reference line
    sits inside its panel rather than being clipped against the left spine -- the 1 km family
    line in particular was landing right on the edge when the range clipped to the final misses."""
    finite = df[col].to_numpy()
    finite = finite[np.isfinite(finite)]
    init_min = None
    if conj_json:
        ref = _initial_by_family(conj_json, families)
        if ref:
            finite = np.concatenate([finite] + [v for v in ref.values()])
            init_min = min(float(np.min(v)) for v in ref.values())
    lo, hi = np.nanpercentile(finite, [0.5, 99.5])
    if init_min is not None:
        lo = min(lo, init_min - 0.5)     # keep the smallest initial line off the left spine
    return np.linspace(lo, hi, bins + 1)


def plot_miss_shift_overlay(df, tag, conj_json=None, col="brahe_miss_km", bins=40,
                            families=INIT_FAMILIES, pool=False, density=False, alpha=0.45):
    """OVERLAY variant of the miss-shift figure: all three variants drawn on ONE panel per
    starting-miss family, as FILLED semi-transparent histograms with edge outlines, so overlap
    regions blend into a combined color. dec/sdec get solid black outlines; centralized is
    overlaid LAST as a DASHED outline (also alpha-filled) on top -- so you can see it tracing
    over SDec where the two coincide. The initial miss is a dashed vertical line per family.
    With pool=True the four families collapse to a single panel; density=True normalizes each
    variant to unit area; alpha controls fill transparency."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    # thicker hatch lines (default is 1.0) so the /// and \\\ textures read boldly.
    mpl.rcParams["hatch.linewidth"] = 1.8
    df, fams = _present_families(df, families, pool)
    variants = _ordered_variants(df["variant"].unique())
    edges = _shared_edges(df, col, conj_json, families, bins)
    ref = _initial_by_family(conj_json, families) if conj_json else {}

    ncol = len(fams)
    # Panel width ~4.0" (close to the original 4.6", just a bit tighter) so a 4-across row is
    # ~16" wide. Fonts are set LARGE (fs~20) because the figure is scaled down ~2x to fit an
    # 8.5x11 page -- at that reduction the on-page text reads like ~11pt.
    fig_w = 4.0 * ncol + 0.6
    fig_h = 5.0                # taller: extra headroom above the bars for the legend
    fs = 20.0
    fig, axes = plt.subplots(1, ncol, figsize=(fig_w, fig_h), sharey=True, squeeze=False)
    axes = axes[0]
    for c, fam in enumerate(fams):
        ax = axes[c]
        for v in variants:
            sub = df[(df["variant"] == v) & (df["_fam"] == fam)]
            if not len(sub):
                continue
            style = _VARIANT_STYLE.get(v, dict(ls="-", lw=1.6, zorder=2))
            color = _VARIANT_COLORS.get(v, "0.2")
            vals = sub[col].to_numpy()
            # per-variant fill drawn in TWO layers so the background alpha and the hatch alpha are
            # independent: (1) a background fill at low alpha (dec = its own solid color = the
            # overshoot block; sdec = faint white; centralized = clear/skipped), then (2) a
            # hatch-only layer at FULL alpha on top so the colored hatch stays crisp and bold no
            # matter how faint the background is.
            hatch = style.get("hatch")
            facecolor = color if style.get("face") == "self" else style.get("face", "white")
            bg_alpha = style.get("face_alpha", alpha)
            # (1) background fill (skipped for a clear fill). Carries the legend entry when there
            # is no hatch layer to carry it (i.e. dec).
            if facecolor != "none":
                ax.hist(vals, bins=edges, histtype="stepfilled", density=density,
                        facecolor=facecolor, alpha=bg_alpha, edgecolor="none",
                        zorder=style["zorder"],
                        label=(_variant_label(v) if hatch is None else None))
            # (2) hatch-only layer at 0.8 alpha on top -- bold colored hatch (kept independent of
            # the faint background alpha); carries the legend entry.
            if hatch is not None:
                ax.hist(vals, bins=edges, histtype="stepfilled", density=density,
                        facecolor="none", edgecolor=color, hatch=hatch, linewidth=0.0,
                        alpha=0.8, zorder=style["zorder"] + 1, label=_variant_label(v))
            # crisp outer outline (per-variant color): dec maroon, sdec black, centralized its own
            # blue short-dashed line drawn on top so it's distinguishable where it traces over sdec.
            outline_color = color if style.get("outline") == "self" else style.get("outline", "black")
            ax.hist(vals, bins=edges, histtype="step", density=density,
                    edgecolor=outline_color, linewidth=style["lw"], linestyle=style["ls"],
                    zorder=style["zorder"] + 3)
        # initial reference line: where this family's conjunctions STARTED (no collision line --
        # the labeled safe band below is the reference; nothing lands near 1 km anyway).
        if isinstance(fam, float):
            # dashed line at the family's starting miss, annotated INLINE ("N km") near the top of
            # the panel. No legend entry -- a single legend would otherwise show only one family.
            ax.axvline(fam, color="0.4", ls="--", lw=1.4)
            ax.text(fam, 0.80, f"{fam:g} km", transform=ax.get_xaxis_transform(),
                    ha="center", va="center", rotation=90, fontsize=fs - 4.0, color="0.35",
                    style="italic", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
        elif ref:
            for fv in ref:
                ax.axvline(fv, color="0.6", ls="--", lw=1.0)
        # ideal-outcome zone at TCA (labeled once, on the 2nd panel, floating in the band).
        ax.axvspan(4.0, 7.0, color="green", alpha=0.07)
        if c == 1:
            ax.text(5.5, 0.97, "Ideal zone\nat TCA", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=fs - 3.5, color="#0b4d0b",
                    style="italic", zorder=1)
        ax.set_title(f"Initial miss: {fam:g} km" if isinstance(fam, float) else "All starts pooled",
                     fontsize=fs)
        ax.set_xlabel(_col_label(col), fontsize=fs)
        if c == 0:
            ax.set_ylabel("Density" if density else "Rollouts", fontsize=fs)
        ax.tick_params(labelsize=fs - 1.5)
        # raise the y-ceiling to ~1000 (counts) so the legend clears the tallest bars -- same
        # y-scaling, just extra empty headroom at the top. (density mode keeps its auto range.)
        if not density:
            ax.set_ylim(top=1000)
        # legend only on the FIRST panel; the "Ideal zone" label lives on the 2nd panel so the
        # two don't compete.
        if c == 0:
            ax.legend(fontsize=fs - 3.5, loc="upper right", handlelength=1.4,
                      borderpad=0.4, labelspacing=0.3)

    fig.suptitle(f"Final {_col_label(col)} by variant{' (pooled)' if pool else ''}",
                 fontsize=fs + 1.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, tag, "miss_shift_overlay_pooled" if pool else "miss_shift_overlay")


def plot_matrix_error(df, tag, fs=13.0):
    """Planner-vs-truth miss error: how far the DISCRETIZED matrix model's predicted miss
    (matrix_miss_km) diverges from the high-fidelity brahe propagation (brahe_miss_km) that the
    same policy actually achieves. Two panels:
      (left)  scatter matrix vs brahe with the y=x agreement line -- points ABOVE the line are
              rollouts the planner thinks cleared farther than they truly did (optimistic; a
              safety concern). Off-diagonal clusters expose discretization/quantization error.
      (right) histogram of the signed error (matrix - brahe) per variant.
    Needs matrix_miss_km + brahe_miss_km in df (present in every sweep .npz)."""
    import matplotlib.pyplot as plt
    if "matrix_miss_km" not in df or df["matrix_miss_km"].isna().all():
        sys.exit("no matrix_miss_km in these dumps -- cannot plot planner-vs-truth error")

    fig, (axs, axh) = plt.subplots(1, 2, figsize=(11, 4.8))
    err = df["matrix_miss_km"] - df["brahe_miss_km"]
    hi = float(np.nanmax([df["matrix_miss_km"].max(), df["brahe_miss_km"].max()])) * 1.05

    # (left) scatter, colored by variant
    for v in _ordered_variants(df["variant"].unique()):
        g = df[df["variant"] == v]
        axs.scatter(g["brahe_miss_km"], g["matrix_miss_km"], s=10, alpha=0.35,
                    color=_VARIANT_COLORS.get(v, "0.3"), label=_variant_label(v))
    axs.plot([0, hi], [0, hi], "k--", lw=1.2, label="Perfect agreement")
    axs.axvspan(4.0, 7.0, color="green", alpha=0.07)
    axs.set_xlim(0, hi); axs.set_ylim(0, hi)
    axs.set_xlabel("True miss (brahe, km)", fontsize=fs)
    axs.set_ylabel("Planner miss (matrix, km)", fontsize=fs)
    axs.set_title("Planner vs. true miss", fontsize=fs + 1)
    axs.tick_params(labelsize=fs - 1.5)
    axs.legend(fontsize=fs - 3.0, loc="upper left")

    # (right) signed-error histogram per variant
    lo, hip = np.nanpercentile(err, [0.5, 99.5])
    edges = np.linspace(lo, hip, 41)
    for v in _ordered_variants(df["variant"].unique()):
        e = (df[df["variant"] == v]["matrix_miss_km"] - df[df["variant"] == v]["brahe_miss_km"])
        axh.hist(e, bins=edges, histtype="step", linewidth=1.8,
                 color=_VARIANT_COLORS.get(v, "0.3"), label=_variant_label(v))
    axh.axvline(0.0, color="k", ls="--", lw=1.2)
    axh.set_xlabel("Planner error: matrix $-$ brahe (km)", fontsize=fs)
    axh.set_ylabel("Rollouts", fontsize=fs)
    axh.set_title("Optimism $\\rightarrow$", fontsize=fs + 1, loc="right")
    axh.tick_params(labelsize=fs - 1.5)
    axh.legend(fontsize=fs - 3.0, loc="upper right")

    fig.tight_layout()
    _save(fig, tag, "matrix_error")


def load_burns(tag, out_dir=None, filters=None):
    """Glob the tag's .npz dumps and stack the per-stage burn matrices per variant. Unlike
    load_long, this keeps the 2-D (n_rollouts, N_STAGES) shape (no per-rollout explode), then
    column-sums to a (N_STAGES,) "how many rollouts burned at this stage" curve. Returns
    {variant: dict(burns=summed_per_stage, n_rollouts=int, n_stages=int)}. Cells whose .npz
    predate the burn-matrix change (no burn_a1 key) are skipped with a count, so an old sweep
    degrades gracefully instead of crashing."""
    rdir = rollout_dir_for(tag, out_dir)
    files = sorted(glob.glob(os.path.join(rdir, "*.npz")))
    if not files:
        sys.exit(f"no rollout dumps found in {rdir}/ — was the sweep run with --save-rollouts?")
    per_variant, n_skipped, n_stages = {}, 0, None
    for fp in files:
        z = np.load(fp, allow_pickle=False)
        key = {k: v for k, v in zip(KEY_FIELDS, [str(x) for x in z["cell_key"]])}
        if filters and any(str(key.get(k)) != str(v) for k, v in filters.items()):
            continue
        if "burn_a1" not in z.files or "burn_a2" not in z.files:
            n_skipped += 1
            continue
        # an agent burned at a stage iff its action there is nonzero; count rollouts-with-a-burn
        fired = (z["burn_a1"] != 0).astype(int) + (z["burn_a2"] != 0).astype(int)  # (rollouts, stages)
        n_stages = fired.shape[1]
        v = key["variant"]
        acc = per_variant.setdefault(v, dict(burns=np.zeros(n_stages, dtype=float), n_rollouts=0))
        acc["burns"] += fired.sum(axis=0)
        acc["n_rollouts"] += fired.shape[0]
    if n_skipped:
        print(f"  (skipped {n_skipped} cell(s) with no burn matrix — pre-burn-timing dump)")
    if not per_variant:
        sys.exit(f"no cells with burn matrices matched filters {filters} "
                 f"(need a sweep run AFTER the burn-timing change)")
    for acc in per_variant.values():
        acc["n_stages"] = n_stages
    return per_variant


def plot_burn_timing(tag, out_dir=None, filters=None):
    """Per-stage burn-rate curve by variant: fraction of rollouts that fire ANY agent-burn at
    each decision stage. The fewer-sync-contacts -> burns-pushed-later story reads straight off
    this plot (Dec commits earlier/blindly; Cen/SDec wait for a contact then burn)."""
    import matplotlib.pyplot as plt
    per_variant = load_burns(tag, out_dir, filters)
    fig, ax = plt.subplots(figsize=(9, 5))
    for v in sorted(per_variant):
        acc = per_variant[v]
        rate = acc["burns"] / max(acc["n_rollouts"], 1)   # avg agent-burns per rollout at each stage
        ax.plot(range(acc["n_stages"]), rate, marker="o", ms=3, lw=1.8,
                label=f"{v} (n={acc['n_rollouts']})")
    ax.set_xlabel("Decision stage (0 = T-24h start $\\rightarrow$ TCA)")
    ax.set_ylabel("Avg agent-burns per rollout at stage")
    ax.set_title("Burn timing by variant")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, tag, "burn_timing")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="sweep tag (loads notes/results/rollouts_<tag>/)")
    ap.add_argument("--out-dir", default=None, help="override the results dir")
    ap.add_argument("--col", default="brahe_miss_km", help=f"which array to plot ({', '.join(ARRAYS)})")
    ap.add_argument("--filter", nargs="*", default=[],
                    help="subset cells, e.g. --filter variant=sdec init_miss=0.5 angle_deg=0")
    ap.add_argument("--hist", action="store_true", help="overlaid step-histogram by variant")
    ap.add_argument("--violin", action="store_true", help="violin by variant")
    ap.add_argument("--miss-shift", action="store_true",
                    help="before->after grid: rows=variant, cols=initial-miss family "
                         "(1/2/5/10 km). Pass --conj-json for the recomputed initial spread.")
    ap.add_argument("--miss-shift-overlay", action="store_true",
                    help="all variants overlaid on one panel per starting-miss family (step "
                         "outlines); initial miss as a dashed vertical line. The Cen/SDec/Dec "
                         "comparison figure.")
    ap.add_argument("--pool", action="store_true",
                    help="collapse the 4 starting-miss families into a single panel "
                         "(the 3-variants-1-panel pooled view). Applies to --miss-shift-overlay.")
    ap.add_argument("--density", action="store_true",
                    help="normalize histograms to unit area (density) instead of raw counts.")
    ap.add_argument("--alpha", type=float, default=0.45,
                    help="fill transparency for the overlay histograms (default 0.45).")
    ap.add_argument("--conj-json", default=None,
                    help="conj_sweep_*.json to recompute the initial (no-maneuver) miss "
                         "spread from, for the --miss-shift reference histograms.")
    ap.add_argument("--burn-timing", action="store_true",
                    help="per-stage burn-rate curve by variant (from the burn_a1/burn_a2 matrices "
                         "— WHEN each agent burns). Needs a sweep run after the burn-timing change.")
    ap.add_argument("--matrix-error", action="store_true",
                    help="planner-vs-truth miss error: scatter of matrix_miss vs brahe_miss (y=x "
                         "agreement line) + signed-error histogram. Exposes discretization "
                         "optimism (points above y=x = planner thinks it cleared farther than it did).")
    ap.add_argument("--to-csv", action="store_true", help="dump the tidy long DataFrame to CSV")
    args = ap.parse_args()

    filters = dict(kv.split("=", 1) for kv in args.filter) if args.filter else None

    # --burn-timing reads the 2-D burn matrices directly (no per-rollout explode), so it bypasses
    # load_long. Handle it first so it can run on its own without building the long DataFrame.
    if args.burn_timing:
        plot_burn_timing(args.tag, args.out_dir, filters)
        if not (args.hist or args.violin or args.to_csv):
            return

    df = load_long(args.tag, args.out_dir, filters)
    print(f"loaded {len(df)} rollouts across {df.groupby(KEY_FIELDS).ngroups} cells "
          f"({df['variant'].nunique()} variants)")

    if args.to_csv:
        out = os.path.join(_HERE, "notes", "results", f"rollouts_long_{args.tag}.csv")
        df.to_csv(out, index=False)
        print(f"wrote {out}  ({len(df)} rows)")
    if args.hist:
        plot_hist(df, args.tag, args.col)
    if args.violin:
        plot_violin(df, args.tag, args.col)
    if args.miss_shift:
        # fold any --filter into the tag so a belief-subset figure (e.g. init_miss=0.5)
        # doesn't clobber the unfiltered / other-belief one.
        shift_tag = args.tag
        if filters:
            suffix = "_".join(f"{k}{v}" for k, v in sorted(filters.items()))
            shift_tag = f"{args.tag}_{suffix}"
        plot_miss_shift(df, shift_tag, conj_json=args.conj_json, col=args.col)
    if args.miss_shift_overlay:
        shift_tag = args.tag
        if filters:
            suffix = "_".join(f"{k}{v}" for k, v in sorted(filters.items()))
            shift_tag = f"{args.tag}_{suffix}"
        plot_miss_shift_overlay(df, shift_tag, conj_json=args.conj_json, col=args.col,
                                pool=args.pool, density=args.density, alpha=args.alpha)
    if args.matrix_error:
        err_tag = args.tag
        if filters:
            suffix = "_".join(f"{k}{v}" for k, v in sorted(filters.items()))
            err_tag = f"{args.tag}_{suffix}"
        plot_matrix_error(df, err_tag)
    if not (args.to_csv or args.hist or args.violin or args.miss_shift
            or args.miss_shift_overlay or args.matrix_error):
        # default: print per-variant summary so a bare call is still useful
        import pandas as pd
        with pd.option_context("display.width", 120):
            print(df.groupby("variant")[args.col].describe()[["count", "mean", "50%", "min", "max"]])


if __name__ == "__main__":
    main()