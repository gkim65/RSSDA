"""
sweep_driver.py — conjunction-generator -> solve -> brahe-rollout -> tidy CSV.

WHAT THIS DOES (the measuring stick for the sync-value story)
-------------------------------------------------------------
For each (conjunction x initial-belief setting) it solves all three POMDP variants
(Centralized / SDec / Dec) AND runs the B1 operator baselines, brahe-validates each
(Monte-Carlo ~200), and writes ONE tidy row per (conjunction x belief x variant) with:

  conjunction params  : label / miss_km / angle_deg / perp_km / dt0_km / v_rel_ms
  belief setting       : init_miss / init_spread
  solve result         : expected_return / collision_prob (matrix) / dV / deviation / syncs
  brahe validation     : brahe miss mean/min/max, 4-7 km band split (<4 / in / >7), brahe coll%

so the per-sweep CSV is the wandb seam (reuses rollout_v2's per-rollout CSV conventions).

WHY SUBPROCESS-PER-CONJUNCTION + MATRIX REUSE (the CRITICAL structure)
---------------------------------------------------------------------
N_STAGES / STAGE_EPOCHS / N_STATES_TOTAL are computed ONCE at import in
spacecraft_stage_grid and frozen across discretizer_v2 / transition_v2 / matrices. A
conjunction whose orbit changes the stage COUNT (head-on=25, oblique/cross=26) therefore
CANNOT be re-solved correctly in the same process. So each conjunction runs in a FRESH
subprocess, handed its scenario via a per-conjunction YAML (--scenario-config; NO env vars).

The SUBPROCESS UNIT is "one conjunction + its FULL belief x variant grid" (_conj_worker.py),
NOT one cell. The T/O/R matrices depend only on the conjunction (orbit / grid / contacts /
reward) — NOT on the belief — so the worker builds them ONCE per (variant, contact-subset) and
REUSES them across all of that conjunction's beliefs (the ~22s build is paid once per
conjunction, not once per cell). The worker reuses compare_variants_v2's solve + build and
rollout_v2's run_mc verbatim — this driver only orchestrates conjunctions + collects rows.

PARALLELISM: --jobs N runs up to N conjunction-children CONCURRENTLY (a bounded Popen pool).
Conjunctions share NO in-process state (each is its own process), so concurrency is safe; the
ONLY shared resource is the output CSV, which we never write from two processes at once — each
child writes its OWN shard CSV and the PARENT merges shards into the master CSV as children
complete (serialized single writer).

RESUMABLE: the master CSV is the cache. Each cell is keyed by
  (label, miss_km, angle_deg, v_rel_ms, init_miss, init_spread, variant)
Completed keys are skipped (the parent tells each child which of its cells are already done), so
a re-run — or a Ctrl-C'd sweep — only fills the holes. Children append rows as cells complete, so
even a killed child leaves a valid partial shard the next run picks up.

INFEASIBLE conjunctions are SKIPPED with their reason logged (never crash).

Usage:
  # small validation + belief probe on the head-on cell (the default):
  .venv/bin/python -u benchmarks/spacecraftCA/sweep_driver.py --probe \
      --rollouts 200 --backend numerical --tag probe1

  # explicit conjunction grid + belief grid, 4 conjunctions in parallel:
  .venv/bin/python -u benchmarks/spacecraftCA/sweep_driver.py \
      --miss 5 --angles 0,45,90 --init-miss 0.5,3,5,8 --init-spread 1.4,3.0 \
      --variants centralized,sdec,dec --baselines b1 --rollouts 200 --jobs 4 --tag sweepA
"""
import os, sys, csv, json, time, argparse, subprocess, tempfile, shutil

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_BENCH)
for p in (_ROOT, _BENCH, _HERE):
    sys.path.insert(0, p)

# --coarse: a SPEED PRESET for quick machinery-verification runs. Sets a coarse base stage
# cadence (~4h) + a 2h contact-merge threshold so N_STAGES drops 25 -> ~6 (states 4726 -> ~700,
# solve ~6s vs ~6min full). NO env vars: the preset is written into each child's per-conjunction
# scenario YAML (the --scenario-config handoff). UNSET => the default ~2h / 25-stage grid.
_COARSE_GRID = (dict(hour_grid_h=[24, 20, 16, 12, 8, 4], merge_threshold_h=2.0)
                if "--coarse" in sys.argv else {})

# Only the generator is imported in THIS (parent) process. It DOES import the model modules
# (transition_v2 / matrices) for its geometry helpers, which freezes a default stage grid in
# the PARENT — that's fine: the parent never solves. Every solve/rollout runs in a FRESH child
# process (so its N_STAGES is correct for that conjunction), handed its scenario via a YAML
# config file (--scenario-config), NOT env vars.
from brahe import initialize_eop
import numpy as np
import yaml
import conjunction_generator as CG

PY = sys.executable
WORKER_SCRIPT = os.path.join(_HERE, "_conj_worker.py")   # one child per conjunction

# Safe-band edges — keep in lockstep with rollout_v2 (SAFE_LO_KM / SAFE_HI_KM) so the band
# split this driver computes matches the harness everywhere else.
SAFE_LO_KM = 4.0
SAFE_HI_KM = 7.0
COLLISION_KM = 1.0          # < this at TCA = collision (D.COLLISION_THRESHOLD_KM)

# Output row schema (one row per conjunction x belief x variant). Order = CSV column order.
ROW_FIELDS = [
    # --- conjunction identity ---
    "label", "miss_km", "angle_deg", "perp_km", "dt0_km", "v_rel_ms",
    "n_stages", "n_contacts",
    # --- belief setting ---
    "init_miss", "init_spread",
    # --- variant ---
    "variant",
    # --- solve result (from compare_variants_v2 / baseline) ---
    "expected_return", "collision_prob_matrix", "dv_ms", "deviation", "syncs",
    # --- brahe validation (from the MC rollout) ---
    "brahe_coll_pct", "brahe_miss_mean", "brahe_miss_min", "brahe_miss_max",
    "band_below_pct", "band_in_pct", "band_above_pct", "n_rollouts",
    # --- bookkeeping ---
    "status",            # ok / infeasible / solve_error / rollout_error
    "reason",            # infeasibility reason or error note
]


# ---------------------------------------------------------------------------
# Cell key + cache
# ---------------------------------------------------------------------------

def cell_key(label, miss, angle, vrel, init_miss, init_spread, variant):
    return (str(label), round(float(miss), 4), round(float(angle), 4),
            round(float(vrel), 4), round(float(init_miss), 4),
            round(float(init_spread), 4), str(variant))


def load_done_keys(csv_path):
    """Keys already in the output CSV -> skip them (resumable). Only rows with status=ok
    count as done; infeasible/error rows are also kept (no point re-running an infeasible
    conjunction, but a transient solve_error should be retryable -> NOT marked done)."""
    done = set()
    rows = []
    if not os.path.exists(csv_path):
        return done, rows
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
            if r.get("status") in ("ok", "infeasible"):
                done.add(cell_key(r["label"], r["miss_km"], r["angle_deg"], r["v_rel_ms"],
                                  r["init_miss"], r["init_spread"], r["variant"]))
    return done, rows


def write_rows(csv_path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in ROW_FIELDS})


def _wandb_log_row(row, wb, extra_config=None, name_suffix=""):
    """Log ONE sweep cell (conjunction x belief x variant) as a wandb run, mirroring the tidy
    ROW_FIELDS the CSV holds. FAIL-SOFT: any wandb error is warned + swallowed so the sweep never
    aborts (CSV is the source of truth). `wb` = dict(project, entity, mode) or None to skip.

    extra_config: extra (usually non-numeric) fields to record in the run's CONFIG — e.g.
    peel_contacts passes {contacts, subset_name, n_contacts} so WHICH contacts were kept is logged
    (strings can't be wandb.log metrics, so they MUST ride in config). name_suffix disambiguates
    the run name when many rows share the same conj/belief/variant (e.g. peel subsets)."""
    if not wb:
        return
    try:
        import wandb
        name = f"{row.get('label','')}_m{row.get('miss_km','')}_a{row.get('angle_deg','')}_" \
               f"im{row.get('init_miss','')}_is{row.get('init_spread','')}_{row.get('variant','')}" \
               f"{name_suffix}"
        config = {k: row.get(k) for k in
                  ("label", "miss_km", "angle_deg", "perp_km", "v_rel_ms",
                   "init_miss", "init_spread", "variant")}
        if extra_config:
            config.update(extra_config)
        r = wandb.init(project=wb["project"], entity=wb["entity"] or None, mode=wb["mode"],
                       name=name, group=wb.get("tag"), reinit=True, config=config)
        # numeric metrics (skip blanks / non-numeric so B1's empty solver fields don't crash)
        for k in ("expected_return", "collision_prob_matrix", "dv_ms", "syncs", "deviation",
                  "brahe_coll_pct", "brahe_miss_mean", "brahe_miss_min", "brahe_miss_max",
                  "band_below_pct", "band_in_pct", "band_above_pct", "n_stages", "n_contacts"):
            v = row.get(k, "")
            try:
                if v != "" and v is not None:
                    wandb.log({k: float(v)})
            except (TypeError, ValueError):
                pass
        wandb.summary["status"] = row.get("status", "")
        r.finish()
    except Exception as e:
        print(f"      [wandb] log skipped ({type(e).__name__}: {e}); CSV row intact.", flush=True)


# ---------------------------------------------------------------------------
# Per-conjunction subprocess plumbing — write the conjunction's scenario YAML + sweep job,
# spawn ONE _conj_worker child that handles the whole belief x variant grid (matrix reuse).
# ---------------------------------------------------------------------------

def _conj_scenario_file(conj, workdir, backend, contacts=None,
                        obs_fidelity=None, obs_sigma=None,
                        sdec_memory_limit_gb=None, sdec_iter_limit=None):
    """Write this conjunction's SCENARIO as a YAML config the child consumes via
    --scenario-config (the ONE config handoff; replaces the SPACECRAFT_CONJ_GRID + env model).
    Carries the orbit pair (so the child's grid rebuilds for THIS conjunction's stage count),
    the propagator backend, this conjunction's perp (fixed across its beliefs), the coarse-grid
    preset if any, and the contact subset if any. init_b is built PER BELIEF inside the worker,
    so the YAML carries only perp here (not init_miss/spread — those vary per cell)."""
    sc1 = np.asarray(conj.sc1_oe, dtype=float).tolist()
    sc2 = np.asarray(CG.conjunction_sc2_oe(conj), dtype=float).tolist()
    grid = dict(propagator=backend, **_COARSE_GRID)
    cfg = {
        "conjunction": {"sc1_oe": sc1, "sc2_oe": sc2},
        "grid": grid,
        "belief": {"perp": float(conj.perp_km)},
    }
    if contacts is not None:
        cfg["contacts"] = {"stages": [int(s) for s in contacts.split(",") if s.strip() != ""]}
    # obs-fidelity + solve knobs (same across the sweep; the worker reads them via build_scenario).
    obs = {}
    if obs_fidelity is not None:
        obs["fidelity"] = obs_fidelity
    if obs_sigma is not None:
        obs["sigma"] = float(obs_sigma)
    if obs:
        cfg["obs"] = obs
    solve = {}
    if sdec_memory_limit_gb is not None:
        solve["sdec_memory_limit_gb"] = float(sdec_memory_limit_gb)
    if sdec_iter_limit is not None:
        solve["sdec_iter_limit"] = int(sdec_iter_limit)
    if solve:
        cfg["solve"] = solve
    path = os.path.join(workdir, "conj_scenario.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


def resolve_contacts(conj, n_contacts):
    """Resolve a SDec contact subset FOR THIS CONJUNCTION (geometry-robust speed lever).

    The available contact INDICES differ per conjunction (head-on [1,3,5,...] vs oblique
    [1,4,6,...]) and the child HARD-ERRORS on a stage that isn't available, so a fixed index
    list can't be reused across geometries. So we take a COUNT (n_contacts) and pick that many
    of THIS conjunction's OWN available contacts. Fewer sync points => a much smaller belief
    tree => faster SDec solve (the peel-down showed even 2 well-placed contacts ~= the full set).

    n_contacts is the number of FIRST (earliest) available contacts to keep — matches the user's
    'first 2 contacts' speed knob. NOTE: only affects SDec; Centralized always syncs at all
    stages (so skip Centralized when you want this to actually speed things up). Returns a
    comma-string for --contact-stages, or None (full default set)."""
    if n_contacts is None:
        return None
    _, avail = CG.conjunction_contacts(conj)          # orbit-derived contacts for THIS conj
    avail = sorted(int(a) for a in avail)
    chosen = avail[:int(n_contacts)] if n_contacts > 0 else []
    return ",".join(str(c) for c in chosen)


def spawn_conjunction(conj, beliefs, variants, baselines, backend, rollouts, b1_opts,
                      n_contacts, done_keys, shard_path, rollout_dir=None, solve_obs=None,
                      contacts=None):
    """Launch ONE _conj_worker child for this conjunction's WHOLE belief x variant grid.

    Writes the per-conjunction scenario YAML (--scenario-config) + a sweep-job JSON (the belief
    grid, variants, rollouts, the SDec contact subset, the already-done cell keys to skip), then
    starts the child as a Popen. The child builds T/O/R ONCE and reuses them across beliefs,
    appending each cell's row to `shard_path` as it completes. Returns (Popen, workdir, job_label)
    so the caller's pool can wait on it, drain its shard, and clean up. NON-blocking (for --jobs).

    contacts: EXPLICIT contact subset override (comma-string of stages, or "" for none). Default
    None => derive first-N from n_contacts via resolve_contacts (the sweep's behavior). peel_contacts
    passes an explicit arbitrary subset here so it doesn't have to monkeypatch resolve_contacts."""
    workdir = tempfile.mkdtemp(prefix="sweep_conj_")
    if contacts is None:
        contacts = resolve_contacts(conj, n_contacts)  # comma-string subset or None (full set)
    scen = _conj_scenario_file(conj, workdir, backend, contacts, **(solve_obs or {}))
    job = {
        "label": conj.name or conj.label, "miss_km": conj.miss_km,
        "angle_deg": conj.angle_deg, "perp_km": conj.perp_km, "dt0_km": conj.dt0_km,
        "v_rel_ms": conj.v_rel_ms,
        "beliefs": [[float(im), float(sp)] for (im, sp) in beliefs],
        "variants": list(variants), "baselines": list(baselines),
        "rollouts": int(rollouts), "backend": backend,
        "contacts": contacts, "b1_opts": b1_opts,
        "done_keys": [list(k) for k in done_keys],   # cells already in the master CSV -> skip
    }
    job_path = os.path.join(workdir, "job.json")
    with open(job_path, "w") as f:
        json.dump(job, f)
    cmd = [PY, "-u", WORKER_SCRIPT, "--scenario-config", scen,
           "--job", job_path, "--shard", shard_path]
    if rollout_dir:
        cmd += ["--rollout-dir", rollout_dir]
    proc = subprocess.Popen(cmd, cwd=_HERE)
    return proc, workdir, (conj.name or conj.label)


def drain_shard(shard_path, done, rows, csv_path, wb):
    """Merge any NEW rows from a finished child's shard into the master CSV (single writer = the
    parent), skipping cells already present (cell_key dedup), then wandb-log each new row. Returns
    the count of new rows merged. Safe to call repeatedly; only un-merged cells are added."""
    if not os.path.exists(shard_path):
        return 0
    n_new = 0
    with open(shard_path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                k = cell_key(r["label"], r["miss_km"], r["angle_deg"], r["v_rel_ms"],
                             r["init_miss"], r["init_spread"], r["variant"])
            except (KeyError, TypeError, ValueError):
                # a partial last line from a child mid-write (live drain) — skip; the next
                # drain (or the child's completion) picks it up once fully flushed.
                continue
            if k in done:
                continue
            rows.append(r)
            done.add(k)
            n_new += 1
            er = r.get("expected_return", "")
            bm = r.get("brahe_miss_mean", "")
            inb = r.get("band_in_pct", "")
            print(f"      <- [{r.get('status','')}] {r['label']} im={r['init_miss']} "
                  f"is={r['init_spread']} {r['variant']}: return={er} brahe_mean={bm} "
                  f"in-band={inb}", flush=True)
            write_rows(csv_path, rows)   # persist the master after EACH new cell (resumable)
            _wandb_log_row(r, wb)        # wandb on top (fail-soft; CSV is source of truth)
    return n_new


# ---------------------------------------------------------------------------
# Conjunction set builders
# ---------------------------------------------------------------------------

def conjunctions_from_file(path):
    """Hand-in a reviewable JSON set of conjunction specs (the user's 'one set of JSONs I give
    in' model). Each entry is EITHER geometry-first {miss_km, angle_deg[, v_rel_ms, name]} OR
    orbit-first {sc1_oe, sc2_oe[, name]} (real catalogued/TLE orbit pairs). The file IS the
    reviewable sweep artifact (decouples the conjunction set from the driver; good for the
    appendix 'defensible conjunction set' figure + reproducible wandb runs)."""
    with open(path) as f:
        specs = json.load(f)
    if isinstance(specs, dict):                 # allow {"conjunctions": [...]} or a bare list
        specs = specs.get("conjunctions", specs)
    out = []
    for s in specs:
        name = s.get("name")
        if "sc1_oe" in s and "sc2_oe" in s:     # orbit-first: real orbit pair
            out.append(CG.make_conjunction_from_orbits(
                np.asarray(s["sc1_oe"], float), np.asarray(s["sc2_oe"], float), name=name))
        else:                                   # geometry-first: (miss, angle[, v_rel])
            out.append(CG.make_conjunction(
                float(s["miss_km"]), float(s["angle_deg"]),
                float(s.get("v_rel_ms", CG.DEFAULT_V_REL_MS)), name=name))
    return out


def conjunctions_to_specs(conjs):
    """Dump conjunctions to plain geometry-first specs (round-trippable via --conj-file)."""
    return [dict(name=c.name or c.label, miss_km=float(c.miss_km),
                 angle_deg=float(c.angle_deg), v_rel_ms=float(c.v_rel_ms))
            for c in conjs]


def build_conjunctions(args):
    """Return a list of Conjunction. Precedence: --conj-file (hand-in set) > --probe
    (validation set) > the inline --miss/--angles/--v-rel cross-product.

    --probe = the validation set: the head-on cell (reproduces the headline) plus one oblique +
    one cross-track so the per-conjunction grid rebuild (N_STAGES 25->26) is exercised."""
    if args.conj_file:
        return conjunctions_from_file(args.conj_file)
    if args.probe:
        return [
            CG.make_conjunction(0.5, 0.0, name="head_on"),     # the headline cell (25 stages)
            CG.make_conjunction(5.0, 45.0, name="oblique"),    # 26 stages — exercises rebuild
            CG.make_conjunction(5.0, 90.0, name="cross_track"),# 26 stages
        ]
    miss = [float(x) for x in args.miss.split(",") if x.strip()]
    angles = [float(x) for x in args.angles.split(",") if x.strip()]
    vrels = ([float(x) for x in args.v_rel.split(",") if x.strip()]
             if args.v_rel else [CG.DEFAULT_V_REL_MS])
    out = []
    for m in miss:
        for a in angles:
            for v in vrels:
                out.append(CG.make_conjunction(m, a, v))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", action="store_true",
                    help="validation set: head-on (headline) + oblique + cross-track, "
                         "swept over the belief grid. Ignores --miss/--angles/--v-rel.")
    ap.add_argument("--conj-file", default=None,
                    help="JSON list of conjunction specs to run (hand-in set; overrides "
                         "--probe/--miss/--angles). Each entry: geometry-first "
                         "{miss_km,angle_deg[,v_rel_ms,name]} OR orbit-first {sc1_oe,sc2_oe[,name]}.")
    ap.add_argument("--dump-conjunctions", default=None,
                    help="write the conjunction set this run WOULD use to this JSON path and "
                         "exit (inspect/edit/re-feed via --conj-file). No solving.")
    ap.add_argument("--miss", default="5", help="comma list of total miss (km)")
    ap.add_argument("--angles", default="0", help="comma list of geometry angles (deg)")
    ap.add_argument("--v-rel", default=None, help="comma list of along-track v_rel (m/s)")
    ap.add_argument("--init-miss", default="0.5,3,4,5,8",
                    help="comma list of belief DANGER centers (km); or 'true' to center each "
                         "conjunction's belief on its own true miss (spread still from --init-spread)")
    ap.add_argument("--init-spread", default="1.4,3.0",
                    help="comma list of belief UNCERTAINTY half-widths (km)")
    ap.add_argument("--variants", default="centralized,sdec,dec",
                    help="comma subset of centralized,sdec,dec")
    ap.add_argument("--baselines", default="",
                    help="comma subset of b1 (operator floor). Empty = no baselines.")
    ap.add_argument("--b1-strategy", default="threshold")
    ap.add_argument("--b1-policy", default="conservative")
    ap.add_argument("--b1-other-obs", default="tle")
    ap.add_argument("--coarse", action="store_true",
                    help="SPEED PRESET for quick machinery checks: ~4h base cadence + 2h merge "
                         "=> N_STAGES ~6 (solve ~6s vs ~6min full 25-stage). NOT for production "
                         "(coarse grid burns differently => returns won't match the 25-stage "
                         "headline). Pre-scanned at import; see top of file.")
    ap.add_argument("--rollouts", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=1,
                    help="run up to N conjunction-children CONCURRENTLY (default 1 = sequential). "
                         "Each conjunction is its own process (no shared in-process state), so "
                         "concurrency is safe; children write per-conjunction shards the parent "
                         "merges into the master CSV (single writer). Ctrl-C leaves a valid "
                         "resumable CSV. Set to ~#cores for a multi-conjunction sweep.")
    ap.add_argument("--n-contacts", type=int, default=None,
                    help="SDec SPEED LEVER: keep only the first N of EACH conjunction's available "
                         "contacts (geometry-robust; fewer sync points => smaller belief tree => "
                         "faster SDec solve). Only affects SDec (Centralized syncs at all stages, "
                         "Dec at none). Use with --variants sdec to actually go faster.")
    ap.add_argument("--backend", default="numerical",
                    choices=["numerical", "keplerian", "drag"])
    # --- obs-fidelity + solve knobs (written into each conjunction's scenario YAML; the worker
    #     picks them up via build_scenario, same as the orbit/grid). Defaults = the anchor. ---
    ap.add_argument("--obs-fidelity", default=None,
                    help="SDec sync obs fidelity: perfect|gps|tle|asymmetric (obs.fidelity).")
    ap.add_argument("--obs-sigma", default=None,
                    help="raw SDec sync obs sigma km (obs.sigma); overrides the named fidelity.")
    ap.add_argument("--sdec-memory-limit-gb", type=float, default=None,
                    help="RS-SDA* memory ceiling GB (solve.sdec_memory_limit_gb); raise on a "
                         "big-memory box; <=0 => no limit. Default = solver default (16).")
    ap.add_argument("--sdec-iter-limit", type=int, default=None,
                    help="RS-SDA* TI2 budget (solve.sdec_iter_limit); a hung cell bails with a "
                         "partial return instead of running forever.")
    ap.add_argument("--tag", default="sweep")
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "notes", "results"))
    ap.add_argument("--save-rollouts", action="store_true",
                    help="ALSO dump the FULL per-rollout arrays (200 brahe miss / dt / dV / "
                         "n_burns / matrix-error per cell + per-stage burn matrices burn_a1/burn_a2 "
                         "= WHEN each agent burned) to notes/results/rollouts_<tag>/, one "
                         ".npz per cell keyed by the cell's 7-tuple. The summary CSV is unchanged; "
                         "these let you rebuild histograms/percentiles/CDFs post-hoc "
                         "(plot_rollout_dist.py). Survives shard cleanup. A few MB per sweep.")
    # wandb on top of the CSV (CSV stays the source of truth). One wandb run PER CELL
    # (conjunction x belief x variant), logging the same tidy ROW_FIELDS. FAIL-SOFT: a wandb
    # auth/network error never aborts the sweep. Mirrors main.py's wandb seam + default entity.
    ap.add_argument("--wandb", action="store_true", help="log each cell row to wandb (CSV-on-top).")
    ap.add_argument("--wandb-project", default="spacecraftCA")
    ap.add_argument("--wandb-entity", default="kmeans_gsopt",
                    help="wandb account/team (set 2026-06-24). Pass '' for your default login.")
    ap.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--keep-infeasible", action="store_true",
                    help="record infeasible conjunctions as skipped rows (default: skip silently "
                         "after logging).")
    args = ap.parse_args()

    initialize_eop()
    conjs = build_conjunctions(args)

    if args.dump_conjunctions:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_conjunctions)) or ".",
                    exist_ok=True)
        with open(args.dump_conjunctions, "w") as f:
            json.dump(conjunctions_to_specs(conjs), f, indent=2)
        print(f"wrote {len(conjs)} conjunction specs -> {args.dump_conjunctions}")
        for c in conjs:
            print(" ", repr(c))
        return

    # --init-miss "true" (sentinel) => seed each conjunction's belief DANGER center at ITS OWN
    # true miss (conj.miss_km) rather than a fixed global list. The belief UNCERTAINTY (sigma /
    # half-width) still comes entirely from --init-spread, unchanged; only the center moves.
    # Otherwise it's a numeric list of global belief centers applied identically to every
    # conjunction (legacy behavior, bit-for-bit).
    true_belief = args.init_miss.strip().lower() == "true"
    init_misses = ([] if true_belief
                   else [float(x) for x in args.init_miss.split(",") if x.strip()])
    spreads = [float(x) for x in args.init_spread.split(",") if x.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    variants += [f"b1:{b.strip()}" for b in args.baselines.split(",") if b.strip()]
    b1_opts = dict(b1_strategy=args.b1_strategy, b1_policy=args.b1_policy,
                   b1_other_obs=args.b1_other_obs)

    csv_path = os.path.join(args.out_dir, f"sweep_{args.tag}.csv")
    wb = (dict(project=args.wandb_project, entity=args.wandb_entity, mode=args.wandb_mode,
               tag=args.tag) if args.wandb else None)
    done, rows = load_done_keys(csv_path)
    beliefs = [(im, sp) for im in init_misses for sp in spreads]

    def _beliefs_for(conj):
        """Belief (init_miss, init_spread) list for ONE conjunction. Numeric --init-miss => the
        global list (identical per conjunction). --init-miss true => center on this conjunction's
        own true miss, one belief per spread. Spread (uncertainty) is unchanged either way."""
        if true_belief:
            return [(conj.miss_km, sp) for sp in spreads]
        return beliefs

    n_beliefs = len(spreads) if true_belief else len(beliefs)
    print(f"=== sweep_driver: {len(conjs)} conjunctions x {n_beliefs} beliefs x "
          f"{len(variants)} variants  (backend={args.backend}, rollouts={args.rollouts}, "
          f"jobs={args.jobs}"
          f"{', init_miss=TRUE (per-conj true miss)' if true_belief else ''})")
    print(f"    output: {csv_path}  ({len(done)} cells already done -> skipped)")

    # --- partition conjunctions: feasible-with-work vs infeasible (recorded, never crash) ---
    n_skip_done = n_infeasible = 0
    pending = []                     # feasible conjunctions that still have un-done cells
    for conj in conjs:
        lbl = conj.name or conj.label
        if not conj.feasible:
            n_infeasible += 1
            print(f"  SKIP infeasible {lbl} (miss={conj.miss_km} angle={conj.angle_deg}): "
                  f"{conj.reason}")
            if args.keep_infeasible:
                for (im, sp) in _beliefs_for(conj):
                    for var in variants:
                        k = cell_key(lbl, conj.miss_km, conj.angle_deg, conj.v_rel_ms, im, sp, var)
                        if k in done:
                            continue
                        rows.append(dict(
                            label=lbl, miss_km=conj.miss_km, angle_deg=conj.angle_deg,
                            perp_km=conj.perp_km, dt0_km=conj.dt0_km, v_rel_ms=conj.v_rel_ms,
                            init_miss=im, init_spread=sp, variant=var,
                            status="infeasible", reason=conj.reason))
                        done.add(k)
                write_rows(csv_path, rows)
            continue
        # which of THIS conjunction's cells are already done (passed to the child to skip)
        conj_beliefs = _beliefs_for(conj)
        conj_done = {cell_key(lbl, conj.miss_km, conj.angle_deg, conj.v_rel_ms, im, sp, var)
                     for (im, sp) in conj_beliefs for var in variants
                     if cell_key(lbl, conj.miss_km, conj.angle_deg, conj.v_rel_ms, im, sp, var)
                     in done}
        n_total = len(conj_beliefs) * len(variants)
        if len(conj_done) >= n_total:
            n_skip_done += n_total
            continue
        n_skip_done += len(conj_done)
        pending.append((conj, conj_done))

    # --- bounded process pool: up to --jobs conjunction-children run concurrently. Each child
    #     handles its FULL belief x variant grid (matrices built once, reused across beliefs) and
    #     writes its own shard CSV; the parent drains finished shards into the master CSV (single
    #     writer) + wandb. KeyboardInterrupt drains what's done and exits with a valid CSV. ---
    shard_dir = os.path.join(args.out_dir, f"_shards_{args.tag}")
    os.makedirs(shard_dir, exist_ok=True)
    # permanent (survives shard cleanup) raw-rollout dir, one .npz per cell — only if requested.
    rollout_dir = None
    if args.save_rollouts:
        rollout_dir = os.path.join(args.out_dir, f"rollouts_{args.tag}")
        os.makedirs(rollout_dir, exist_ok=True)
        print(f"    saving full per-rollout arrays -> {rollout_dir}/ (one .npz per cell)")
    n_jobs = max(1, int(args.jobs))
    n_new = 0
    running = {}                      # proc -> (workdir, label, shard_path)
    queue = list(pending)

    # obs-fidelity + solve knobs applied to EVERY conjunction's scenario YAML (same across the sweep).
    solve_obs = {"obs_fidelity": args.obs_fidelity, "obs_sigma": args.obs_sigma,
                 "sdec_memory_limit_gb": args.sdec_memory_limit_gb,
                 "sdec_iter_limit": args.sdec_iter_limit}

    def _launch(conj, conj_done):
        shard_path = os.path.join(shard_dir, f"{(conj.name or conj.label)}_"
                                  f"m{conj.miss_km}_a{int(conj.angle_deg)}.csv")
        conj_beliefs = _beliefs_for(conj)
        proc, workdir, label = spawn_conjunction(
            conj, conj_beliefs, variants, args.baselines.split(",") if args.baselines else [],
            args.backend, args.rollouts, b1_opts, args.n_contacts, conj_done, shard_path,
            rollout_dir=rollout_dir, solve_obs=solve_obs)
        running[proc] = (workdir, label, shard_path)
        print(f"  LAUNCH {label:<11} miss={conj.miss_km} a={conj.angle_deg:.0f} "
              f"({len(conj_beliefs)*len(variants)-len(conj_done)} cells) [pid {proc.pid}]",
              flush=True)

    try:
        while queue or running:
            while queue and len(running) < n_jobs:
                conj, conj_done = queue.pop(0)
                _launch(conj, conj_done)
            # poll the running set; drain + reap any that finished
            done_procs = [p for p in running if p.poll() is not None]
            if not done_procs:
                # also drain shards of still-running children so progress + wandb stream live
                for p, (wd, lbl, shard) in running.items():
                    n_new += drain_shard(shard, done, rows, csv_path, wb)
                time.sleep(0.5)
                continue
            for p in done_procs:
                wd, lbl, shard = running.pop(p)
                n_new += drain_shard(shard, done, rows, csv_path, wb)   # merge its rows
                if p.returncode != 0:
                    print(f"  [warn] child {lbl} exited rc={p.returncode}; "
                          f"its completed cells (if any) were merged.", flush=True)
                shutil.rmtree(wd, ignore_errors=True)
    except KeyboardInterrupt:
        print("\n  [interrupt] draining finished cells, terminating children "
              "(CSV stays resumable) ...", flush=True)
        for p, (wd, lbl, shard) in list(running.items()):
            n_new += drain_shard(shard, done, rows, csv_path, wb)
            p.terminate()
        for p, (wd, lbl, shard) in list(running.items()):
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()
            n_new += drain_shard(shard, done, rows, csv_path, wb)
            shutil.rmtree(wd, ignore_errors=True)

    print(f"\n=== done: {n_new} new cells, {n_skip_done} skipped(done), "
          f"{n_infeasible} infeasible conjunctions. CSV -> {csv_path}")


if __name__ == "__main__":
    main()
