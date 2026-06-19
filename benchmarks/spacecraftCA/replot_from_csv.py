"""
replot_from_csv.py

Regenerate the compare_variants_v2 figures (summary / burn-timing / reward-parts /
action-schedule) from the saved CSVs WITHOUT re-solving. Useful for restyling appendix
figures or rebuilding plots after a tweak to the plotting code.

The plotting functions are pure functions of the row dicts, and write_csv stores those
exact rows, so the CSVs fully reconstruct the plots (see notes/SCENARIO_KNOBS.md).

Usage:
  python replot_from_csv.py --tag gs_spread6_25stage
  python replot_from_csv.py --tag <tag> --out-dir notes/results --fig-dir notes/figures
"""
import os
import sys
import csv
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from compare_variants import (plot_summary, plot_burn_timing, plot_action_schedule)
from compare_variants_v2 import plot_reward_parts

# Columns to coerce to float/int when reading back (everything else stays str).
_FLOAT_COLS = {"expected_return", "collision_prob", "expected_dv_ms",
               "mean_agent_burns", "burn_stage_rate", "value", "action_prob"}
_INT_COLS = {"init_bin", "stage", "joint_action", "expected_syncs"}


def _read_rows(path):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for k in list(row.keys()):
                if k in _FLOAT_COLS:
                    row[k] = float(row[k])
                elif k in _INT_COLS:
                    row[k] = int(float(row[k]))
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "notes", "results"))
    ap.add_argument("--fig-dir", default=os.path.join(_HERE, "notes", "figures"))
    args = ap.parse_args()

    jobs = [
        (f"variant_expected_{args.tag}.csv", plot_summary, "summary"),
        (f"variant_burn_timing_{args.tag}.csv", plot_burn_timing, "burn_timing"),
        (f"variant_reward_parts_{args.tag}.csv", plot_reward_parts, "reward_parts"),
        (f"variant_action_by_stage_{args.tag}.csv", plot_action_schedule, "action_schedule"),
    ]
    any_done = False
    for fname, fn, label in jobs:
        rows = _read_rows(os.path.join(args.out_dir, fname))
        if not rows:
            print(f"  SKIP {label}: missing/empty {fname}")
            continue
        out = fn(rows, args.fig_dir, args.tag)
        print(f"  {label:16} -> {out}")
        any_done = True
    if not any_done:
        print(f"No CSVs found for tag '{args.tag}' in {args.out_dir}")
        sys.exit(1)


if __name__ == "__main__":
    main()
