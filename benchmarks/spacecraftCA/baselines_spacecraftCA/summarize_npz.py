#!/usr/bin/env python
"""summarize_npz.py — walk rollout .npz dumps and print paste-ready summary stats per category.

Handles BOTH .npz layouts, auto-detected per file:
  * BASELINE (operator_matrix.py):   opmatrix_<tag>/npz/<sc1>-x-<sc2>__obs-<obs>__<conj>.npz
        -> category = the strategy pair "<sc1>-x-<sc2>"  (obs kept as a sub-key)
  * POMDP (sweep --save-rollouts):   rollouts_<tag>/<...cell_key...>.npz  (cell_key ends in variant)
        -> category = the variant (centralized / sdec / dec)
Every .npz carries the per-rollout arrays brahe_miss_km / total_dv / n_burns / true_term_reward.

Groups all rollouts by (category [, obs]) POOLED across conjunctions, and prints for each group the
stats a violin/box plot needs: n, mean, std, min, q1, median, q3, max, p05, p95, plus collision %
(<1km) and 4-7 km band split. Output is (a) a readable table and (b) a JSON block you can paste back.

USAGE (run on the cluster, point at either dump dir):
  python summarize_npz.py --dir notes/results/opmatrix_opmatrix50/npz         # baselines
  python summarize_npz.py --dir notes/results/rollouts_sweep50                 # POMDP variants
  # combine several dirs into one table (baselines + POMDP together):
  python summarize_npz.py --dir notes/results/rollouts_sweep50 notes/results/opmatrix_opmatrix50/npz
  # split baselines by obs fidelity (default pools all obs together):
  python summarize_npz.py --dir .../opmatrix_opmatrix50/npz --by-obs
  # filter to one obs, or one conjunction substring:
  python summarize_npz.py --dir .../npz --obs tle --conj sweep_003
  # write the JSON to a file to download:
  python summarize_npz.py --dir ... --json-out stats_sweep50.json
"""
import argparse
import glob
import json
import os

import numpy as np

# canonical display order so pasted stats line up across runs
_ORDER = ["centralized", "sdec", "dec",
          "threshold-x-threshold", "threshold-x-selfish", "threshold-x-fixedlead",
          "selfish-x-selfish", "selfish-x-fixedlead", "fixedlead-x-fixedlead"]


def _parse_baseline(fname):
    """<sc1>-x-<sc2>__obs-<obs>__<conj>.npz -> (pair, obs, conj) or None if not this layout."""
    stem = os.path.basename(fname)[:-4]
    if "__obs-" not in stem:
        return None
    pair_part, rest = stem.split("__obs-", 1)
    if "__" not in rest:
        return None
    obs, conj = rest.split("__", 1)
    return pair_part, obs, conj


def _category_and_obs(fp, z):
    """Return (category, obs, conj) for either layout. obs is '' when not applicable (POMDP)."""
    b = _parse_baseline(fp)
    if b is not None:
        return b[0], b[1], b[2]
    # POMDP: cell_key = [label, miss, angle, vrel, init_miss, init_spread, variant]
    if "cell_key" in z.files:
        key = [str(x) for x in z["cell_key"]]
        variant = key[-1] if key else "unknown"
        conj = key[0] if key else ""
        return variant, "", conj
    # fallback: scalar sc1/sc2/obs if present
    if "sc1" in z.files and "sc2" in z.files:
        pair = f"{z['sc1']}-x-{z['sc2']}"
        obs = str(z["obs"]) if "obs" in z.files else ""
        conj = str(z["conj"]) if "conj" in z.files else ""
        return pair, obs, conj
    return os.path.basename(fp)[:-4], "", ""


def _stats(miss, dv, nb, rew):
    m = np.asarray(miss, dtype=float)
    n = m.size
    def q(a, p):
        return float(np.percentile(a, p)) if a.size else float("nan")
    return {
        "n": int(n),
        "mean": round(float(m.mean()), 3),
        "std": round(float(m.std()), 3),
        "min": round(float(m.min()), 3),
        "q1": round(q(m, 25), 3),
        "median": round(q(m, 50), 3),
        "q3": round(q(m, 75), 3),
        "max": round(float(m.max()), 3),
        "p05": round(q(m, 5), 3),
        "p95": round(q(m, 95), 3),
        "coll_pct": round(float((m < 1).mean() * 100.0), 3),
        "band_below4_pct": round(float((m < 4).mean() * 100.0), 1),
        "band_in_pct": round(float(((m >= 4) & (m <= 7)).mean() * 100.0), 1),
        "band_over7_pct": round(float((m > 7).mean() * 100.0), 1),
        "dv_mean": round(float(np.mean(dv)), 4) if len(dv) else None,
        "nburns_mean": round(float(np.mean(nb)), 3) if len(nb) else None,
        "reward_mean": round(float(np.mean(rew)), 3) if len(rew) else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", nargs="+", required=True,
                    help="one or more dirs of .npz dumps (globbed recursively).")
    ap.add_argument("--by-obs", action="store_true",
                    help="keep obs fidelity as a separate group key (baselines only).")
    ap.add_argument("--obs", default=None, help="filter to this obs (perfect|tle|frozen).")
    ap.add_argument("--conj", default=None, help="only cells whose conj name CONTAINS this string.")
    ap.add_argument("--json-out", default=None, help="also write the JSON block to this path.")
    args = ap.parse_args()

    files = []
    for d in args.dir:
        files += glob.glob(os.path.join(d, "**", "*.npz"), recursive=True)
        files += glob.glob(os.path.join(d, "*.npz"))
    files = sorted(set(files))
    if not files:
        raise SystemExit(f"no .npz found under {args.dir}")

    # accumulate arrays per group key
    groups = {}   # key -> dict of lists
    n_cells = 0
    for fp in files:
        try:
            z = np.load(fp, allow_pickle=True)
        except Exception as e:
            print(f"[skip] {os.path.basename(fp)}: {e}")
            continue
        if "brahe_miss_km" not in z.files:
            continue
        cat, obs, conj = _category_and_obs(fp, z)
        if args.obs and obs and obs != args.obs:
            continue
        if args.conj and args.conj not in conj:
            continue
        key = f"{cat} [{obs}]" if (args.by_obs and obs) else cat
        g = groups.setdefault(key, {"miss": [], "dv": [], "nb": [], "rew": []})
        g["miss"].append(np.asarray(z["brahe_miss_km"], dtype=float))
        if "total_dv" in z.files:
            g["dv"].append(np.asarray(z["total_dv"], dtype=float))
        if "n_burns" in z.files:
            g["nb"].append(np.asarray(z["n_burns"], dtype=float))
        if "true_term_reward" in z.files:
            g["rew"].append(np.asarray(z["true_term_reward"], dtype=float))
        n_cells += 1

    summary = {}
    for key, g in groups.items():
        miss = np.concatenate(g["miss"]) if g["miss"] else np.array([])
        dv = np.concatenate(g["dv"]) if g["dv"] else np.array([])
        nb = np.concatenate(g["nb"]) if g["nb"] else np.array([])
        rew = np.concatenate(g["rew"]) if g["rew"] else np.array([])
        summary[key] = _stats(miss, dv, nb, rew)

    def sort_key(k):
        base = k.split(" [")[0]
        return (_ORDER.index(base) if base in _ORDER else 99, k)
    ordered = sorted(summary, key=sort_key)

    # readable table
    print(f"\n{n_cells} cells across {len(groups)} groups; {len(files)} .npz scanned\n")
    hdr = ["category", "n", "mean", "med", "q1", "q3", "min", "max",
           "coll%", "in-band%", "dV", "burns"]
    print("  ".join(f"{h:>10}" if i else f"{h:<26}" for i, h in enumerate(hdr)))
    print("-" * 132)
    for k in ordered:
        s = summary[k]
        row = [f"{k:<26}", f"{s['n']:>10}", f"{s['mean']:>10}", f"{s['median']:>10}",
               f"{s['q1']:>10}", f"{s['q3']:>10}", f"{s['min']:>10}", f"{s['max']:>10}",
               f"{s['coll_pct']:>10}", f"{s['band_in_pct']:>10}",
               f"{s['dv_mean']:>10}", f"{s['nburns_mean']:>10}"]
        print("  ".join(row))

    # paste-ready JSON (ordered)
    out = {k: summary[k] for k in ordered}
    print("\n===== PASTE THIS JSON BACK =====")
    print(json.dumps(out, indent=2))
    print("===== END JSON =====")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[wrote] {args.json_out}")


if __name__ == "__main__":
    main()
