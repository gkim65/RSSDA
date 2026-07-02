#!/usr/bin/env python3
"""Recover per-stage maneuver (burn) timing from cached peel rollout .npz files.

Run this ON THE CLUSTER, pointed at the rollout dir the peel run dumped (the one
passed to peel_contacts.py --rollout-dir / --save-dir). Each cell .npz stores the
per-stage joint action for every rollout as:

    burn_a1  shape (n_rollouts, n_stages)  int8   SC1 action per stage
    burn_a2  shape (n_rollouts, n_stages)  int8   SC2 action per stage

Action codes: 0 = WAIT, 1 = +dV along-track, 2 = -dV along-track.
A stage is a MANEUVER stage (for a spacecraft) iff its action != 0.

The .npz filename (not cell_key) carries the subset, e.g.
    head_on__5.0__0.0001__0.0__0.5__1.4__sdec__c1-2.npz
                                                    ^^^^ subset tag: contacts 1 and 2
"__centralized__" and "__dec__" cells are the rails; sdec cells carry "__c<..>" or
(for the full set) no subset tag.

Outputs:
  * a human-readable per-cell table (which stages burned, and how consistently
    across rollouts), and
  * (with --json OUT) a JSON map
        { label: { subset_key: {
            "burn_stages": [...],           # stages burned by SC1 or SC2 (>= --frac)
            "sc1": [...], "sc2": [...],      # burned stages per spacecraft (>= --frac)
            "sc1_rate": [...], "sc2_rate": [...],  # per-stage burn FREQUENCY (0..1), len n_stages
            "n_stages": N } } }
    plot_peel_heatmap.py reads sc1_rate/sc2_rate to draw a hashed-red overlay whose
    ALPHA scales with burn frequency, with SC1 vs SC2 in different hatch directions.

Usage on the cluster:
    python3 inspect_burn_stages.py --dir /path/to/rollouts_peel_ready
    python3 inspect_burn_stages.py --dir /path/to/rollouts_peel_ready \
        --label head_on --json burn_stages_head_on.json --frac 0.5
"""
import argparse
import glob
import json
import os

import numpy as np

_ACT = {0: "WAIT", 1: "+dV", 2: "-dV"}


def _parse_name(path):
    """Split a rollout .npz filename into (label, subset_key).

    Filenames look like:
      <label>__<miss>__<coll>__<perp>__<init_miss>__<init_spread>__<variant>[__<subset>].npz
    We only need the leading label and the trailing variant/subset tag.
    """
    stem = os.path.basename(path)
    if stem.endswith(".npz"):
        stem = stem[:-4]
    parts = stem.split("__")
    label = parts[0] if parts else stem
    # variant is field index 6; anything after it is the subset tag (e.g. "c1-2").
    variant = parts[6] if len(parts) > 6 else ""
    subset = parts[7] if len(parts) > 7 else ""
    if variant == "centralized":
        subset_key = "__centralized__"
    elif variant == "dec":
        subset_key = "__dec__"
    elif subset:
        subset_key = subset            # e.g. "c1-2"
    else:
        subset_key = "sdec_full"       # sdec, no subset tag => full available set
    return label, subset_key


def _burn_stages(a1, a2, frac):
    """Given per-stage action arrays (n_rollouts, n_stages), return the stages
    where a burn happens in at least `frac` of rollouts, split by spacecraft.

    A cell is 'burned' if the action != 0 (WAIT). frac=0 => any rollout burned;
    frac=1 => every rollout burned there. frac=0.5 => majority."""
    a1 = np.asarray(a1)
    a2 = np.asarray(a2)
    n_roll, n_stages = a1.shape
    sc1_rate = (a1 != 0).mean(axis=0)          # fraction of rollouts burning at each stage
    sc2_rate = (a2 != 0).mean(axis=0)
    sc1 = [int(s) for s in range(n_stages) if sc1_rate[s] >= frac and sc1_rate[s] > 0]
    sc2 = [int(s) for s in range(n_stages) if sc2_rate[s] >= frac and sc2_rate[s] > 0]
    both = sorted(set(sc1) | set(sc2))
    return both, sc1, sc2, sc1_rate, sc2_rate, n_stages, n_roll


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True,
                    help="Rollout dir with per-cell .npz files (the peel run's --rollout-dir).")
    ap.add_argument("--label", default=None,
                    help="Only inspect this conjunction label (else all found).")
    ap.add_argument("--frac", type=float, default=0.5,
                    help="A stage counts as a burn stage if >= this fraction of rollouts burn "
                         "there (0 = any, 0.5 = majority, 1 = all). Default 0.5.")
    ap.add_argument("--json", default=None,
                    help="Write the burn-stage map to this JSON path (for the figure overlay).")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.npz")))
    if not files:
        raise SystemExit(f"No .npz files in {args.dir}")

    out = {}
    print(f"[scan] {len(files)} .npz cells in {args.dir}  (burn if >= {args.frac:.0%} of rollouts)\n")
    for f in files:
        label, subset_key = _parse_name(f)
        if args.label and label != args.label:
            continue
        d = np.load(f, allow_pickle=True)
        if "burn_a1" not in d.files or "burn_a2" not in d.files:
            print(f"[skip] {os.path.basename(f)}: no burn_a1/burn_a2 arrays")
            continue
        both, sc1, sc2, r1, r2, n_stages, n_roll = _burn_stages(d["burn_a1"], d["burn_a2"], args.frac)
        out.setdefault(label, {})[subset_key] = {
            "burn_stages": both, "sc1": sc1, "sc2": sc2,
            "sc1_rate": [round(float(x), 4) for x in r1],
            "sc2_rate": [round(float(x), 4) for x in r2],
            "n_stages": int(n_stages),
        }
        print(f"{label:12s} {subset_key:16s} n_stages={n_stages:2d} rollouts={n_roll:3d}")
        print(f"    burn stages (SC1 or SC2, >= {args.frac:.0%}): {both}")
        print(f"      SC1: {sc1}   rates={[f'{x:.2f}' for x in r1]}")
        print(f"      SC2: {sc2}   rates={[f'{x:.2f}' for x in r2]}")
        # Show the most-common action per burned stage from rollout 0 as a sanity peek.
        a1_0 = np.asarray(d["burn_a1"])[0]
        a2_0 = np.asarray(d["burn_a2"])[0]
        peek = [f"s{s}:SC1={_ACT.get(int(a1_0[s]))}/SC2={_ACT.get(int(a2_0[s]))}"
                for s in both]
        if peek:
            print(f"      rollout[0] actions: {'  '.join(peek)}")
        print()

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"[saved] burn-stage map -> {args.json}  "
              f"({sum(len(v) for v in out.values())} cells across {len(out)} label(s))")


if __name__ == "__main__":
    main()
