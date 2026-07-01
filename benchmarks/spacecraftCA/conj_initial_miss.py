"""
conj_initial_miss.py — compute the TRUE (no-maneuver) closest-approach miss for every
conjunction in a conj-sweep JSON file.

Each entry in a conj_sweep_*.json carries only the two orbit sets (sc1_oe / sc2_oe); the
starting miss distance is NOT stored. This script reconstructs each conjunction via
conjunction_generator.make_conjunction_from_orbits (brahe 3-D closest-approach search
around EPOCH_TCA) and reports the per-conjunction miss plus the distribution across the
whole file. That distribution is the "initial spread" reference for the rollout
miss-shift histograms (where each policy pushed the miss FROM).

Usage:
  # print the per-conjunction table + summary:
  .venv/bin/python benchmarks/spacecraftCA/conj_initial_miss.py \
      --json benchmarks/spacecraftCA/notes/conj_sweep_spherical_50.json

  # dump a tidy CSV (name, miss_km, angle_deg, label, at_tca) for reuse / plotting:
  .venv/bin/python benchmarks/spacecraftCA/conj_initial_miss.py \
      --json benchmarks/spacecraftCA/notes/conj_sweep_spherical_50.json --to-csv

  # import and reuse:
  from conj_initial_miss import initial_misses
  df = initial_misses("...conj_sweep_spherical_50.json")   # pandas DataFrame
"""
import os
import sys
import json
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# model miss-bin edges (matches the discretizer's 10-bin scheme), reused for the summary.
MISS_BIN_EDGES = [0, 0.5, 1, 2, 5, 10, 20, 50, 100, 500, np.inf]


def initial_misses(json_path):
    """Return a tidy DataFrame: one row per conjunction with the true closest-approach
    miss (km) reconstructed from its two orbit sets. Columns:
    name, miss_km, angle_deg, label, at_tca. Rows that fail to reconstruct get miss_km=NaN
    and label='ERR ...'."""
    import pandas as pd
    import conjunction_generator as CG

    with open(json_path) as fh:
        conjs = json.load(fh)

    rows = []
    for c in conjs:
        try:
            conj = CG.make_conjunction_from_orbits(
                np.array(c["sc1_oe"], dtype=float),
                np.array(c["sc2_oe"], dtype=float),
                name=c.get("name"),
            )
            rows.append(dict(name=c.get("name"), miss_km=float(conj.miss_km),
                             angle_deg=float(conj.angle_deg),
                             label=getattr(conj, "label", ""),
                             at_tca=bool(conj.at_tca) if conj.at_tca is not None else None))
        except Exception as e:  # keep the row so the table stays aligned with the file
            rows.append(dict(name=c.get("name"), miss_km=float("nan"),
                             angle_deg=float("nan"), label=f"ERR {e}", at_tca=None))
    return pd.DataFrame(rows)


def summarize(df):
    """Print the per-conjunction table + the initial-miss distribution summary."""
    print(f"{len(df)} conjunctions\n")
    print("per-conjunction true closest-approach miss (km):")
    for r in df.itertuples(index=False):
        flag = "" if (r.at_tca is None or r.at_tca) else "  (NOT at TCA!)"
        print(f"  {str(r.name):28s}  miss={r.miss_km:8.3f} km  "
              f"angle={r.angle_deg:6.1f}  {str(r.label):8s}{flag}")

    m = df["miss_km"].to_numpy()
    m = m[np.isfinite(m)]
    if not len(m):
        print("\n  (no finite misses reconstructed)")
        return
    qs = np.percentile(m, [10, 25, 50, 75, 90])
    print("\n=== initial-miss DISTRIBUTION across the geometries ===")
    print(f"  n            : {len(m)}")
    print(f"  min / max    : {m.min():.3f} / {m.max():.3f} km")
    print(f"  mean / median: {m.mean():.3f} / {np.median(m):.3f} km")
    print("  10/25/50/75/90 pct: " + " / ".join(f"{q:.2f}" for q in qs))
    hist, _ = np.histogram(m, bins=MISS_BIN_EDGES)
    print(f"  binned {MISS_BIN_EDGES[:-1]}+inf): {hist.tolist()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True, help="conj_sweep_*.json to reconstruct")
    ap.add_argument("--to-csv", action="store_true",
                    help="also dump a tidy CSV next to the JSON (…_initial_miss.csv)")
    args = ap.parse_args()

    df = initial_misses(args.json)
    summarize(df)

    if args.to_csv:
        out = os.path.splitext(args.json)[0] + "_initial_miss.csv"
        df.to_csv(out, index=False)
        print(f"\nwrote {out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
