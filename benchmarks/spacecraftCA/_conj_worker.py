"""
_conj_worker.py — the per-CONJUNCTION sweep child (one fresh subprocess per conjunction).

You do NOT run this directly — sweep_driver.py spawns it. It is the body of the driver's
per-conjunction loop, run in its own process so this conjunction's stage grid (N_STAGES) freezes
correctly at model import (the value-binding the sweep has always relied on; you cannot solve two
different-stage-count conjunctions in one process).

WHY THIS EXISTS (matrix reuse + the right subprocess unit)
----------------------------------------------------------
The T/O/R matrices depend ONLY on the conjunction (orbit pair / grid cadence / propagator /
contact-stage subset / reward) — NOT on the initial belief. init_b is the ONLY per-belief input.
The OLD sweep_driver spawned ONE subprocess PER CELL (conjunction x belief x variant), so it paid
the ~22s matrix build for EVERY belief even though only init_b changed.

This worker is the natural unit AFTER matrix reuse: ONE conjunction = ONE process that
  1. applies that conjunction's scenario ONCE (so N_STAGES freezes for this orbit pair), then
  2. builds T/O/R ONCE per (variant, contact-subset) via compare_variants_v2.build_matrices, and
  3. loops this conjunction's beliefs x variants, building init_b per belief and SOLVING by
     REUSING the same matrices (CV.solve_policy / CV.solve_and_eval with matrices=...), then
     brahe-validates IN-PROCESS via rollout_v2.run_mc (the same vectorized engine; no extra
     subprocess). ALL THREE variants are brahe-validated — dec flies its RS-MAA* policy through
     run_mc via rollout_v2.DecPolicySource (CV.solve_dec_policy gives the matrices + dec_full).

So the ~22s build is paid ONCE per conjunction, not once per belief. Nothing here reimplements
the solver, matrices, or rollout — it reuses the canonical machinery.

CONTRACT (how sweep_driver drives it)
-------------------------------------
  --scenario-config <yaml>   the conjunction's scenario (orbit pair / grid / backend / perp /
                             belief / optional contact subset). Applied PRE-import by
                             _cli_bootstrap_scenario (the ONE config handoff; NO env vars).
  --job <json>               { label, miss_km, angle_deg, perp_km, dt0_km, v_rel_ms,
                               beliefs:[[init_miss,init_spread],...], variants:[...],
                               baselines:[...], rollouts, backend, contacts(opt comma-str),
                               b1_opts{...}, done_keys:[[...cell_key...],...] }
  --shard <csv>              the shard CSV this child writes (ROW_FIELDS schema). The parent
                             merges shards into the master sweep CSV. The child appends a row as
                             EACH cell completes (a killed child still leaves resumable progress).

Exits 0 even if individual cells error (each cell row carries its own status/reason — fail-soft
per cell). A hard import/scenario failure exits non-zero (the parent records the whole
conjunction as errored).
"""
import os, sys, csv, json, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_BENCH)
for p in (_ROOT, _BENCH, _HERE):
    sys.path.insert(0, p)

# Apply THIS conjunction's scenario BEFORE importing any model module (import-order discipline:
# N_STAGES / N_STATES freeze by value at model import). _cli_bootstrap_scenario reads
# --scenario-config from argv. NO env vars.
from scenario_config import _cli_bootstrap_scenario
_SCENARIO = _cli_bootstrap_scenario(sys.argv)

from brahe import initialize_eop
import numpy as np

# The canonical solve + rollout machinery — REUSED, not reimplemented.
import compare_variants_v2 as CV
import rollout_v2 as RV
import spacecraft_transition_v2 as TV
import spacecraft_discretizer_v2 as D
import spacecraft_matrices as M

# Band edges — keep in lockstep with sweep_driver / rollout_v2.
SAFE_LO_KM = 4.0
SAFE_HI_KM = 7.0
COLLISION_KM = 1.0

# Same row schema as sweep_driver.ROW_FIELDS — MUST stay in lockstep (duplicated so the child
# stays import-light; sweep_driver merges these rows verbatim).
ROW_FIELDS = [
    "label", "miss_km", "angle_deg", "perp_km", "dt0_km", "v_rel_ms",
    "n_stages", "n_contacts",
    "init_miss", "init_spread",
    "variant",
    "expected_return", "collision_prob_matrix", "dv_ms", "deviation", "syncs",
    "brahe_coll_pct", "brahe_miss_mean", "brahe_miss_min", "brahe_miss_max",
    "band_below_pct", "band_in_pct", "band_above_pct", "n_rollouts",
    "status", "reason",
]


def _band_stats(miss_list):
    """Band split + collision% from a list of brahe miss values (same math as sweep_driver)."""
    m = np.array([float(x) for x in miss_list])
    return dict(
        brahe_coll_pct=100.0 * float(np.mean(m < COLLISION_KM)),
        brahe_miss_mean=float(m.mean()), brahe_miss_min=float(m.min()),
        brahe_miss_max=float(m.max()),
        band_below_pct=100.0 * float(np.mean(m < SAFE_LO_KM)),
        band_in_pct=100.0 * float(np.mean((m >= SAFE_LO_KM) & (m <= SAFE_HI_KM))),
        band_above_pct=100.0 * float(np.mean(m > SAFE_HI_KM)),
        n_rollouts=int(m.size),
    )


def _dump_rollouts(rollout_dir, row, results):
    """Tee the FULL per-rollout arrays for this cell to one .npz BEFORE _band_stats collapses
    them to 8 scalars. The filename IS the cell key (the same 7-tuple that keys the summary CSV),
    so each dump joins back to its summary row by construction — no separate index. Reorganizing
    these into histograms later is a glob + parse-filename, with NO change to the sweep.

    Keeps every per-rollout field run_mc returns that's useful post-hoc: the real (brahe) outcome,
    the matrix-vs-brahe error, the maneuver effort flown, AND the per-stage burn matrices
    (burn_a1/burn_a2: (n_rollouts, N_STAGES) signed action per agent -> WHEN each burn fired).
    Overwritten cleanly on a cell retry."""
    if not rollout_dir:
        return
    os.makedirs(rollout_dir, exist_ok=True)
    key = _cell_key(row["label"], row["miss_km"], row["angle_deg"], row["v_rel_ms"],
                    row["init_miss"], row["init_spread"], row["variant"])
    fname = "__".join(str(k) for k in key)
    # Disambiguate same-variant runs that differ ONLY by contact subset (peel solves many SDec
    # subsets; the cell_key ends in `variant`, so without this they'd all overwrite one ...sdec.npz).
    csig = row.get("contacts")
    if csig not in (None, "", "ALL"):
        fname += "__c" + str(csig).replace(",", "-")
    fname = fname.replace("/", "_").replace(":", "-") + ".npz"

    def col(name):
        return np.array([float(r[name]) for r in results])

    # Per-stage burn matrices: (n_rollouts, N_STAGES) signed action per agent, scattered
    # from each rollout's `burns` list of (stage, a1, a2). 0=WAIT, 1=+dV, 2=-dV (the _ACT
    # encoding). This keeps WHEN each burn fired (not just the n_burns count), so a "burns
    # over time" histogram is a column-sum -- the fewer-contacts-push-burns-later story.
    burn_a1 = np.zeros((len(results), D.N_STAGES), dtype=np.int8)
    burn_a2 = np.zeros((len(results), D.N_STAGES), dtype=np.int8)
    for i, r in enumerate(results):
        for (stage, a1, a2) in r["burns"]:
            burn_a1[i, stage] = a1
            burn_a2[i, stage] = a2

    np.savez(os.path.join(rollout_dir, fname),
             cell_key=np.array([str(k) for k in key]),
             brahe_miss_km=col("brahe_miss_km"), brahe_dt_km=col("brahe_dt_km"),
             matrix_miss_km=col("matrix_miss_km"), matrix_dt_km=col("matrix_dt_km"),
             total_dv=col("total_dv"), n_burns=col("n_burns"),
             true_term_reward=col("true_term_reward"),
             burn_a1=burn_a1, burn_a2=burn_a2)


def _append_row(shard_path, row):
    """Append ONE row to the shard CSV (header once). Append-as-you-go so a killed child still
    leaves the cells it finished — parent merge + cell_key dedup makes it resumable."""
    new = not os.path.exists(shard_path)
    with open(shard_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in ROW_FIELDS})


def _cell_key(label, miss, angle, vrel, init_miss, init_spread, variant):
    """MUST match sweep_driver.cell_key (resume dedup key)."""
    return (str(label), round(float(miss), 4), round(float(angle), 4),
            round(float(vrel), 4), round(float(init_miss), 4),
            round(float(init_spread), 4), str(variant))


def run_b1_subprocess(job, init_miss, init_spread, variant, shard_dir):
    """B1 operator baseline: a separate hand-coded heuristic with its own knobs (no solve), kept
    as a subprocess to baseline_b1.py so its many flags aren't reimplemented here. Returns
    (band_dict, err). NOT the matrix-reuse hot path (B1 doesn't solve)."""
    import subprocess
    b1 = job.get("b1_opts", {})
    csv_path = os.path.join(shard_dir, f"b1_{init_miss}_{init_spread}_{variant}.csv")
    cmd = [sys.executable, "-u",
           os.path.join(_HERE, "baselines_spacecraftCA", "baseline_b1.py"),
           "--scenario-config", job["scenario_config"],
           "--strategy", b1.get("b1_strategy", "threshold"),
           "--policy", b1.get("b1_policy", "conservative"),
           "--other-obs", b1.get("b1_other_obs", "tle"),
           "--variant", "sdec", "--mode", "mc", "--rollouts", str(job["rollouts"]),
           "--init-miss", str(init_miss), "--init-spread", str(init_spread),
           "--perp", str(job["perp_km"]), "--backend", job["backend"], "--csv", csv_path]
    if job.get("contacts"):
        cmd += ["--contact-stages", job["contacts"]]
    proc = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True)
    if proc.returncode != 0:
        return None, f"b1 rc={proc.returncode}: {proc.stderr.strip()[-400:]}"
    if not os.path.exists(csv_path):
        return None, f"b1 produced no CSV: {proc.stdout.strip()[-300:]}"
    miss = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            miss.append(float(r["brahe_miss_km"]))
    if not miss:
        return None, "b1 CSV empty"
    return _band_stats(miss), None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario-config", required=True)  # consumed pre-import; declared for argparse
    ap.add_argument("--job", required=True, help="JSON file: this conjunction's sweep job")
    ap.add_argument("--shard", required=True, help="shard CSV this child writes")
    ap.add_argument("--rollout-dir", default=None,
                    help="if set, dump the FULL per-rollout arrays for each cell to one .npz here "
                         "(filename = cell key). None => only the 8 summary scalars in the row.")
    args = ap.parse_args()

    initialize_eop()
    with open(args.job) as f:
        job = json.load(f)
    job["scenario_config"] = args.scenario_config   # so the B1 child can re-point at it

    shard_dir = os.path.dirname(os.path.abspath(args.shard))
    os.makedirs(shard_dir, exist_ok=True)

    perp = float(job["perp_km"])
    # perp is a CONJUNCTION property (fixed across this conjunction's beliefs). Set the module
    # globals the canonical solve + rollout read, ONCE.
    CV.PERP_KM = perp
    RV.PERP_KM = perp

    # contact subset for SDec (already resolved by the parent to a comma-string of THIS
    # conjunction's own available stages, or None for the full set). Applied to the live global
    # the O-matrix builder + sync_states read.
    contacts = job.get("contacts")
    if contacts:
        M.set_contact_stages([int(s) for s in contacts.split(",") if s.strip() != ""])

    n_stages = D.N_STAGES
    n_contacts = len(M.get_contact_stages())
    beliefs = job["beliefs"]
    variants = list(job.get("variants", []))
    variants += [f"b1:{b}" for b in job.get("baselines", [])]
    rollouts = int(job["rollouts"])
    done = {tuple(k) for k in job.get("done_keys", [])}

    base_ident = dict(
        label=job["label"], miss_km=job["miss_km"], angle_deg=job["angle_deg"],
        perp_km=perp, dt0_km=job["dt0_km"], v_rel_ms=job["v_rel_ms"],
        n_stages=n_stages, n_contacts=n_contacts,
        contacts=contacts,        # explicit subset string (None=full); used to keep .npz names unique
    )

    # --- MATRIX REUSE: build T/O/R ONCE per (variant, contact-subset), reuse across beliefs.
    #     Centralized and SDec each get their own matrices (O differs by sync stages); both are
    #     reused across ALL of this conjunction's beliefs. Dec uses RS-MAA* (matrices rebuilt
    #     inside solve_and_eval's rsmaa path per belief — RS-MAA* has no reuse seam here, but it
    #     is the cheap variant). Built lazily so a single-variant sweep doesn't pay for others.
    matrices = {}

    def get_matrices(variant):
        if variant not in matrices:
            print(f"  [build] T/O/R for {variant} (once per conjunction) ...", flush=True)
            matrices[variant] = CV.build_matrices(variant)
        return matrices[variant]

    rsmaa_cfg = dict(CV.RSMAA_DEFAULTS)

    for (init_miss, init_spread) in beliefs:
        init_miss = float(init_miss); init_spread = float(init_spread)
        # init_b is the ONLY per-belief input. Built per belief; matrices are NOT rebuilt.
        init_b, _flagged, _eff = TV.build_init_b_danger(
            init_miss, init_spread, perp, sign_mode="both")
        for variant in variants:
            key = _cell_key(job["label"], job["miss_km"], job["angle_deg"], job["v_rel_ms"],
                            init_miss, init_spread, variant)
            if key in done:
                continue
            row = dict(base_ident, init_miss=init_miss, init_spread=init_spread, variant=variant)
            print(f"  CELL {job['label']:<11} im={init_miss} is={init_spread} | {variant} ...",
                  flush=True)
            try:
                if variant.startswith("b1"):
                    band, err = run_b1_subprocess(job, init_miss, init_spread, variant, shard_dir)
                    if err:
                        row.update(status="rollout_error", reason=err)
                    else:
                        row.update(status="ok", reason="", syncs=0, expected_return="",
                                   collision_prob_matrix="", dv_ms="", deviation="", **band)
                elif variant == "dec":
                    # Dec: RS-MAA* solve ONCE (reuse the dec matrices across beliefs), eval stats
                    # from that result, THEN brahe-validate via DecPolicySource (same vectorized
                    # run_mc as Cen/SDec — all three endpoints brahe-validated, apples-to-apples).
                    mats = get_matrices("dec")
                    T, O, R, perp_used, dec_full = CV.solve_dec_policy(
                        init_b, rsmaa_cfg=rsmaa_cfg, matrices=mats)
                    er, cp, burns, comp, _td, _am = CV.expected_rsmaa_return_v2(
                        T, O, R, dec_full, init_b)
                    row.update(
                        expected_return=er, collision_prob_matrix=cp,
                        dv_ms=float(burns.sum()) * float(M.DV_MAGNITUDE), syncs=0,
                        deviation=float(comp.get("deviation", 0.0)))
                    dec_src = RV.DecPolicySource(dec_full, TV.N_OBS_AGENT)
                    results = RV.run_mc(T, O, R, perp_used, None, dec_full, init_b,
                                        TV.N_OBS_AGENT, rollouts, seed=0, policy_source=dec_src)
                    _dump_rollouts(args.rollout_dir, row, results)
                    row.update(status="ok", reason="",
                               **_band_stats([r["brahe_miss_km"] for r in results]))
                else:
                    # POMDP centralized/sdec: solve REUSING the conjunction's matrices, then
                    # brahe-validate IN-PROCESS via the same vectorized run_mc.
                    mats = get_matrices(variant)
                    T, O, R, perp_used, sdec, full = CV.solve_policy(
                        variant, init_b, matrices=mats)
                    # expected-eval (return / collision / deviation) on the SAME matrices.
                    er, cp, burns, comp, _td, _am, _ = CV.solve_and_eval(
                        variant, init_b, matrices=mats)
                    syncs = D.N_STAGES if variant == "centralized" else len(M.get_contact_stages())
                    row.update(
                        expected_return=er, collision_prob_matrix=cp,
                        dv_ms=float(burns.sum()) * float(M.DV_MAGNITUDE), syncs=syncs,
                        deviation=float(comp.get("deviation", 0.0)))
                    results = RV.run_mc(T, O, R, perp_used, sdec, full, init_b,
                                        TV.N_OBS_AGENT, rollouts, seed=0)
                    _dump_rollouts(args.rollout_dir, row, results)
                    row.update(status="ok", reason="",
                               **_band_stats([r["brahe_miss_km"] for r in results]))
            except Exception as e:
                import traceback
                row.update(status="solve_error",
                           reason=f"{type(e).__name__}: {e} | {traceback.format_exc()[-300:]}")
            er = row.get("expected_return", "")
            bm = row.get("brahe_miss_mean", "")
            inb = row.get("band_in_pct", "")
            print(f"      -> [{row['status']}] return={er} brahe_mean={bm} "
                  f"in-band={inb}  {row.get('reason','')[:60]}", flush=True)
            _append_row(args.shard, row)   # persist as EACH cell completes (resumable)

    print(f"  [done] conjunction {job['label']} -> {args.shard}", flush=True)


if __name__ == "__main__":
    main()
