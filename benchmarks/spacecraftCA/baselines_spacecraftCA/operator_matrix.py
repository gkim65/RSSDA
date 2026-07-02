#!/usr/bin/env python
"""operator_matrix.py — the operator-vs-operator (mixed per-craft) baseline matrix.

WHY THIS EXISTS (sweep_driver can't do it): sweep_driver's worker hardcodes a single
--strategy for BOTH craft, so it cannot run the MIXED per-craft pairings the paper wants
("what happens when a threshold operator flies against a fixed-lead operator?"). This driver
loops the UNORDERED strategy pairs x obs-fidelity x conjunction grid, calling baseline_b1.py
per cell with --strategy-sc1/--strategy-sc2, and converts each per-rollout CSV to a .npz.

REUSES the canonical machinery verbatim:
  * conjunctions_from_file / _conj_scenario_file from sweep_driver (same geometry + YAML handoff
    the real sweep uses; each conjunction's stage count rebuilds correctly in its own subprocess).
  * baseline_b1.py as the executor (its own MC-200 brahe loop, mixed-strategy support).

BELIEF: --init-miss "true" (per-conjunction: each cell's belief center = that conjunction's OWN
miss_km) with a fixed --init-spread sigma. Matches the sweep's `true` sentinel semantics.

GRID (default): 3 representative operator TYPES {threshold, selfish, fixedlead} -> 6 UNORDERED
pairs (3 self + 3 mixed) x 3 obs {perfect, tle, frozen} x 52 conjunctions x MC-200.
The 'selfish' slot uses --selfish-model (default obsaware, the SDec-POMDP foil).

OUTPUT (notes/results/opmatrix_<tag>/):
  * opmatrix_<tag>.csv       — one tidy SUMMARY row per cell (band split, coll%, dV, miss stats)
  * npz/<cell>.npz           — the FULL per-rollout arrays for that cell (brahe_miss, dt, dV,
                               n_burns, term_reward) so histograms/CDFs rebuild post-hoc.
RESUMABLE: a cell whose .npz already exists is skipped. Ctrl-C leaves a valid partial set.
PARALLEL: --jobs N runs up to N cells concurrently (each is an independent baseline_b1 process).

Run (from benchmarks/spacecraftCA/):
  ../../.venv/bin/python -u baselines_spacecraftCA/operator_matrix.py \
      --conj-file notes/conj_sweep_spherical_50.json \
      --init-spread 1.4 --rollouts 200 --jobs 8 --tag opmatrix50
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCA = os.path.dirname(_HERE)          # benchmarks/spacecraftCA
sys.path.insert(0, _SCA)

import sweep_driver as SW              # conjunctions_from_file, _conj_scenario_file

_B1 = os.path.join(_HERE, "baseline_b1.py")
_PY = sys.executable

# The 3 representative operator TYPES and their 6 UNORDERED pairs (self + mixed).
DEFAULT_TYPES = ["threshold", "selfish", "fixedlead"]
DEFAULT_OBS = ["perfect", "tle", "frozen"]


def unordered_pairs(types):
    pairs = []
    for i, a in enumerate(types):
        for b in types[i:]:
            pairs.append((a, b))
    return pairs


def _cell_tag(pair, obs, conj_name):
    s1, s2 = pair
    safe = conj_name.replace("/", "_")
    return f"{s1}-x-{s2}__obs-{obs}__{safe}"


def run_cell(job):
    """One (pair, obs, conjunction) cell: call baseline_b1 MC, convert CSV -> npz, return summary."""
    pair = job["pair"]
    obs = job["obs"]
    conj_name = job["conj_name"]
    scenario = job["scenario"]
    npz_path = job["npz_path"]
    csv_tmp = job["csv_tmp"]
    s1, s2 = pair

    cmd = [_PY, "-u", _B1,
           "--scenario-config", scenario,
           "--strategy-sc1", s1, "--strategy-sc2", s2,
           "--selfish-model", job["selfish_model"],
           "--policy", job["policy"],
           "--other-obs", obs,
           "--variant", "sdec", "--mode", "mc",
           "--rollouts", str(job["rollouts"]),
           "--init-miss", str(job["init_miss"]),
           "--init-spread", str(job["init_spread"]),
           "--perp", str(job["perp"]),
           "--backend", job["backend"],
           "--csv", csv_tmp]
    proc = subprocess.run(cmd, cwd=_SCA, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"cell": job["cell_tag"], "error": f"rc={proc.returncode}: {proc.stderr.strip()[-300:]}"}
    if not os.path.exists(csv_tmp):
        return {"cell": job["cell_tag"], "error": f"no CSV: {proc.stdout.strip()[-200:]}"}

    # per-rollout CSV -> arrays
    cols = {"brahe_miss_km": [], "brahe_dt_km": [], "total_dv": [],
            "n_burns": [], "true_term_reward": []}
    with open(csv_tmp, newline="") as f:
        for r in csv.DictReader(f):
            for k in cols:
                cols[k].append(float(r[k]))
    miss = np.asarray(cols["brahe_miss_km"], dtype=float)
    dv = np.asarray(cols["total_dv"], dtype=float)
    reward = np.asarray(cols["true_term_reward"], dtype=float)

    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez(npz_path,
             brahe_miss_km=miss,
             brahe_dt_km=np.asarray(cols["brahe_dt_km"], dtype=float),
             total_dv=dv,
             n_burns=np.asarray(cols["n_burns"], dtype=float),
             true_term_reward=reward,
             sc1=s1, sc2=s2, obs=obs, conj=conj_name,
             init_miss=job["init_miss"], init_spread=job["init_spread"], perp=job["perp"])
    os.remove(csv_tmp)

    n = len(miss)
    below = int((miss < 4).sum())
    inband = int(((miss >= 4) & (miss <= 7)).sum())
    over = int((miss > 7).sum())
    coll = float((miss < 1).mean() * 100.0)
    return {
        "cell": job["cell_tag"], "error": "",
        "sc1": s1, "sc2": s2, "obs": obs, "conj": conj_name,
        "init_miss": round(job["init_miss"], 4), "init_spread": job["init_spread"],
        "perp_km": round(job["perp"], 4),
        "n": n, "coll_pct": round(coll, 3),
        "miss_mean": round(float(miss.mean()), 3),
        "miss_min": round(float(miss.min()), 3),
        "miss_max": round(float(miss.max()), 3),
        "band_below4": below, "band_in": inband, "band_over7": over,
        "band_in_pct": round(inband / n * 100.0, 1),
        "dv_mean": round(float(dv.mean()), 4),
        "reward_mean": round(float(reward.mean()), 3),
    }


ROW_FIELDS = ["cell", "sc1", "sc2", "obs", "conj", "init_miss", "init_spread", "perp_km",
              "n", "coll_pct", "miss_mean", "miss_min", "miss_max",
              "band_below4", "band_in", "band_over7", "band_in_pct",
              "dv_mean", "reward_mean", "error"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conj-file", required=True, help="JSON conjunction set (e.g. conj_sweep_spherical_50.json)")
    ap.add_argument("--types", default=",".join(DEFAULT_TYPES),
                    help="operator TYPES to pair (unordered). Default: threshold,selfish,fixedlead")
    ap.add_argument("--obs", default=",".join(DEFAULT_OBS),
                    help="other-obs fidelities. Default: perfect,tle,frozen")
    ap.add_argument("--selfish-model", default="obsaware",
                    help="model used wherever 'selfish' appears in a pair (default obsaware)")
    ap.add_argument("--policy", default="conservative",
                    help="threshold sub-policy used wherever 'threshold' appears (default conservative)")
    ap.add_argument("--init-spread", type=float, default=1.4, help="belief sigma (km)")
    ap.add_argument("--rollouts", type=int, default=200)
    ap.add_argument("--backend", default="numerical", choices=["numerical", "keplerian", "drag"])
    ap.add_argument("--jobs", type=int, default=1, help="cells to run concurrently")
    ap.add_argument("--tag", default="opmatrix")
    ap.add_argument("--out-dir", default=os.path.join(_SCA, "notes", "results"))
    ap.add_argument("--limit-conj", type=int, default=None,
                    help="debug: only run the first N conjunctions")
    args = ap.parse_args()

    SW.initialize_eop()
    conjs = SW.conjunctions_from_file(args.conj_file)
    if args.limit_conj:
        conjs = conjs[:args.limit_conj]

    types = [t.strip() for t in args.types.split(",") if t.strip()]
    obs_list = [o.strip() for o in args.obs.split(",") if o.strip()]
    pairs = unordered_pairs(types)

    run_root = os.path.join(args.out_dir, f"opmatrix_{args.tag}")
    npz_dir = os.path.join(run_root, "npz")
    tmp_dir = os.path.join(run_root, "_tmp")
    scen_dir = os.path.join(run_root, "_scenarios")
    for d in (run_root, npz_dir, tmp_dir, scen_dir):
        os.makedirs(d, exist_ok=True)
    csv_path = os.path.join(run_root, f"opmatrix_{args.tag}.csv")

    # one scenario YAML per conjunction (reused across all its pairs/obs)
    conj_scen = {}
    for c in conjs:
        wd = os.path.join(scen_dir, c.label if hasattr(c, "label") else "c")
        # give each conjunction a UNIQUE workdir (labels repeat across the sweep)
        wd = os.path.join(scen_dir, getattr(c, "name", None) or f"{c.label}_{id(c)}")
        os.makedirs(wd, exist_ok=True)
        conj_scen[id(c)] = SW._conj_scenario_file(c, wd, args.backend)

    # build the cell job list, skipping cells whose npz already exists (resume)
    jobs = []
    for c in conjs:
        cname = getattr(c, "name", None) or c.label
        for pair in pairs:
            for obs in obs_list:
                cell_tag = _cell_tag(pair, obs, cname)
                npz_path = os.path.join(npz_dir, cell_tag + ".npz")
                if os.path.exists(npz_path):
                    continue
                jobs.append({
                    "pair": pair, "obs": obs, "conj_name": cname,
                    "scenario": conj_scen[id(c)],
                    "npz_path": npz_path,
                    "csv_tmp": os.path.join(tmp_dir, cell_tag + ".csv"),
                    "cell_tag": cell_tag,
                    "selfish_model": args.selfish_model, "policy": args.policy,
                    "init_miss": float(c.miss_km), "init_spread": args.init_spread,
                    "perp": float(c.perp_km), "backend": args.backend,
                    "rollouts": args.rollouts,
                })

    total_cells = len(conjs) * len(pairs) * len(obs_list)
    print(f"[operator_matrix] {len(conjs)} conj x {len(pairs)} pairs x {len(obs_list)} obs "
          f"= {total_cells} cells; {len(jobs)} to run ({total_cells - len(jobs)} cached)")
    print(f"[operator_matrix] pairs: {pairs}")
    print(f"[operator_matrix] out -> {run_root}")

    rows = []
    # resume: load any existing summary rows so the CSV stays complete
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))

    def _flush():
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ROW_FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in ROW_FIELDS})

    done = 0
    if args.jobs <= 1:
        for j in jobs:
            r = run_cell(j)
            rows.append(r)
            done += 1
            _flush()
            _log_row(r, done, len(jobs))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(run_cell, j): j for j in jobs}
            for fut in as_completed(futs):
                r = fut.result()
                rows.append(r)
                done += 1
                _flush()
                _log_row(r, done, len(jobs))

    _flush()
    print(f"[operator_matrix] DONE. summary -> {csv_path}")
    print(f"[operator_matrix] per-cell arrays -> {npz_dir}")


def _log_row(r, done, total):
    if r.get("error"):
        print(f"  [{done}/{total}] ERR {r['cell']}: {r['error']}")
    else:
        print(f"  [{done}/{total}] {r['sc1']}x{r['sc2']} obs={r['obs']} {r['conj']}: "
              f"miss={r['miss_mean']}km in-band={r['band_in_pct']}% coll={r['coll_pct']}% "
              f"dV={r['dv_mean']}")


if __name__ == "__main__":
    main()
