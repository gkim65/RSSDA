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
"""
import os, sys, glob, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# Cell-key field order — MUST match _conj_worker._cell_key / sweep_driver.cell_key.
KEY_FIELDS = ["label", "miss_km", "angle_deg", "v_rel_ms", "init_miss", "init_spread", "variant"]
# numeric per-rollout arrays stored in each .npz (besides cell_key)
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
    ap.add_argument("--to-csv", action="store_true", help="dump the tidy long DataFrame to CSV")
    args = ap.parse_args()

    filters = dict(kv.split("=", 1) for kv in args.filter) if args.filter else None
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
    if not (args.to_csv or args.hist or args.violin):
        # default: print per-variant summary so a bare call is still useful
        import pandas as pd
        with pd.option_context("display.width", 120):
            print(df.groupby("variant")[args.col].describe()[["count", "mean", "50%", "min", "max"]])


if __name__ == "__main__":
    main()