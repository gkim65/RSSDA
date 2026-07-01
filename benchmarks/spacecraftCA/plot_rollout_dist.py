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


def _save(fig, tag, name):
    out = os.path.join(_HERE, "notes", "figures", f"rollout_{name}_{tag}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


def plot_hist(df, tag, col="brahe_miss_km", bins=40):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    lo, hi = np.nanpercentile(df[col], [0.5, 99.5])
    edges = np.linspace(lo, hi, bins + 1)
    for v, g in df.groupby("variant"):
        ax.hist(g[col], bins=edges, histtype="step", linewidth=1.8, label=f"{v} (n={len(g)})")
    ax.axvline(1.0, color="k", ls=":", lw=1, label="collision (1 km)")
    ax.axvspan(4.0, 7.0, color="green", alpha=0.07, label="safe band 4-7 km")
    ax.set_xlabel(col); ax.set_ylabel("rollouts"); ax.set_title(f"{col} distribution by variant — {tag}")
    ax.legend(fontsize=8)
    _save(fig, tag, f"hist_{col}")


def plot_violin(df, tag, col="brahe_miss_km"):
    import matplotlib.pyplot as plt
    variants = sorted(df["variant"].unique())
    data = [df[df["variant"] == v][col].values for v in variants]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.violinplot(data, showmedians=True)
    ax.set_xticks(range(1, len(variants) + 1)); ax.set_xticklabels(variants)
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.axhspan(4.0, 7.0, color="green", alpha=0.07)
    ax.set_ylabel(col); ax.set_title(f"{col} by variant — {tag}")
    _save(fig, tag, f"violin_{col}")


# The conj_sweep_spherical_50.json geometries were built at 4 discrete target misses
# (families m1/m2/m5/m10). We facet the miss-shift figure by which family a rollout STARTED
# in, so each starting level's before->after is visible instead of pooling all of them.
INIT_FAMILIES = [1.0, 2.0, 5.0, 10.0]
# color per variant (final histogram); the "initial spread" reference is always grey.
_VARIANT_COLORS = {"centralized": "#1f77b4", "sdec": "#2ca02c", "dec": "#d62728"}


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
                        label=f"initial (n={len(init_ref[fam])})")
            elif isinstance(fam, float):
                ax.axvline(fam, color="0.5", ls="--", lw=1.2, label="initial (target)")
            # final distribution for this variant/family
            if len(sub):
                ax.hist(sub[col].to_numpy(), bins=edges, color=_VARIANT_COLORS.get(v, "0.2"),
                        alpha=0.75, label=f"{v} final (n={len(sub)})")
            ax.axvline(1.0, color="k", ls=":", lw=1)
            ax.axvspan(4.0, 7.0, color="green", alpha=0.07)
            if r == 0:
                title = f"{fam:g} km start" if isinstance(fam, float) else "all"
                ax.set_title(title, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{v}\nrollouts", fontsize=9)
            if r == nrow - 1:
                ax.set_xlabel(col, fontsize=9)
            ax.legend(fontsize=6, loc="upper right")

    fig.suptitle(f"initial vs. final {col} by variant and starting miss — {tag}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    _save(fig, tag, "miss_shift")


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
    ax.set_xlabel("decision stage (0 = T-24h start -> TCA)")
    ax.set_ylabel("avg agent-burns per rollout at stage")
    ax.set_title(f"burn timing by variant — {tag}")
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
    ap.add_argument("--conj-json", default=None,
                    help="conj_sweep_*.json to recompute the initial (no-maneuver) miss "
                         "spread from, for the --miss-shift reference histograms.")
    ap.add_argument("--burn-timing", action="store_true",
                    help="per-stage burn-rate curve by variant (from the burn_a1/burn_a2 matrices "
                         "— WHEN each agent burns). Needs a sweep run after the burn-timing change.")
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
    if not (args.to_csv or args.hist or args.violin or args.miss_shift):
        # default: print per-variant summary so a bare call is still useful
        import pandas as pd
        with pd.option_context("display.width", 120):
            print(df.groupby("variant")[args.col].describe()[["count", "mean", "50%", "min", "max"]])


if __name__ == "__main__":
    main()