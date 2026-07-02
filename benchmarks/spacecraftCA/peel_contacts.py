#!/usr/bin/env python
"""peel_contacts.py — contact-subset search per conjunction ("how few syncs match Centralized?").

Finds, per conjunction, a (locally) MINIMAL set of GS sync contacts whose SDec return still
matches Centralized (within --tol). Two modes:

ADAPTIVE (default, --mode adaptive) — the smart peel-down:
  0. RAILS   solve Centralized (all syncs) + Dec (none) once; Centralized return is the bar.
  1. WINDOW  read the Centralized BURN CENTROID stage from its rollout .npz, keep the HALF of the
             contacts nearest the burn ("contacts near the maneuver"); fall back to the full set
             if that half can't match (or to an even-spread half if Centralized never burns).
  2. GREEDY  drop single contacts that don't matter until none can be removed -> minimal set.
  This operationalizes the "contacts near the maneuver are the most helpful" hypothesis: the cut
  is CENTERED on where Centralized actually burns, then greedily refined. The greedy phase can
  still end up spanning the early/late halves, so a split minimal set stays findable.

GRID (--mode grid) — the older static sweep:
  HALVE  full -> half -> quarter -> ... contacts, spread evenly (count ladder, no feedback).
  PLACE  at small k: early (first k) / late (last k). near_burn is disabled in grid mode (adaptive
         mode owns burn timing).

Both modes reuse sweep_driver's canonical spawn/drain path verbatim (same worker, same matrices,
same brahe MC), only choosing the contact subset itself.

Output: notes/results/peel_<tag>.csv, one row per (conjunction x belief x subset), carrying the
usual columns PLUS `subset_name` and `contacts` so subsets don't dedup-collide. Adaptive subset
names: __centralized__ / __dec__ (rails), window_*, greedy_drop*, minimal.

Example (adaptive):
  .venv/bin/python -u benchmarks/spacecraftCA/peel_contacts.py \
    --conj-file benchmarks/spacecraftCA/notes/conj_cases_spherical.json \
    --init-miss 0.5 --init-spread 1.4 --rollouts 200 --backend numerical \
    --tol 5 --tag peel_cases
"""
import os, sys, csv, json, argparse, tempfile, shutil
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import sweep_driver as SD          # reuse spawn_conjunction / drain_shard / conj loaders
import conjunction_generator as CG

# Run-config solve knobs (obs fidelity + RS-SDA* iter/memory limits), identical for every subset
# and conjunction, so they live module-level — main() fills this once from the CLI and every
# _solve_subset spawn forwards it as spawn_conjunction(solve_obs=...). The iter/memory limits are
# the CLUSTER SAFETY VALVE: a pathological greedy cell that hits the memory ceiling ERRORS out
# (status=solve_error, blank return) which _passes reads as a FAIL => greedy KEEPS that contact
# (conservative: never drops a contact on an unfinished solve). Set the iter-limit GENEROUS so a
# *partial* (status=ok) bail effectively never happens; only true hangs get caught.
_SOLVE_OBS = {}


# ---------------------------------------------------------------------------
# Subset construction
# ---------------------------------------------------------------------------
def halving_counts(n_avail):
    """full, half, quarter, ... down to 1 (the bisection-on-count ladder)."""
    counts, k = [], n_avail
    while k >= 1:
        counts.append(k)
        if k == 1:
            break
        k = max(1, k // 2)
    return counts                                   # e.g. 16 -> [16, 8, 4, 2, 1]


def spread_subset(avail, k):
    """k contacts evenly spaced across the available window (indices into avail)."""
    if k >= len(avail):
        return list(avail)
    idx = np.linspace(0, len(avail) - 1, k).round().astype(int)
    return [avail[i] for i in sorted(set(idx.tolist()))]


def near_stage_subset(avail, k, center_stage):
    """k available contacts CLOSEST to center_stage (the burn stage) — the 'near the maneuver' set."""
    order = sorted(avail, key=lambda s: (abs(s - center_stage), s))
    return sorted(order[:k])


def build_subsets(avail, place_counts, burn_stage):
    """Return {subset_name: [stage,...]} for one conjunction.

    HALVE ladder uses spread placement (the unbiased 'where are they' baseline). PLACE adds
    early/late/near_burn at each requested small count to contrast timing at matched count."""
    avail = sorted(int(a) for a in avail)
    subsets = {}
    for c in halving_counts(len(avail)):
        subsets[f"halve_spread_c{c}"] = spread_subset(avail, c)
    for k in place_counts:
        if k > len(avail):
            continue
        subsets[f"place_early_c{k}"] = avail[:k]
        subsets[f"place_late_c{k}"] = avail[-k:]
        if burn_stage is not None:
            subsets[f"place_nearburn_c{k}"] = near_stage_subset(avail, k, burn_stage)
    # de-dup IDENTICAL stage lists (e.g. spread_c2 == early_c2 sometimes) but KEEP distinct names
    # so the CSV records every strategy label even when they coincide.
    return subsets


import glob


# ---------------------------------------------------------------------------
# Burn stage from the Centralized rollout (.npz the worker dumps when rollout_dir is set)
# ---------------------------------------------------------------------------
def burn_centroid_from_npz(npz_dir):
    """Mass-weighted CENTROID stage of the Centralized policy's burns, read from the worker's
    per-cell .npz (burn_a1/burn_a2 = (n_rollouts, N_STAGES) signed action per agent; nonzero =
    a burn fired at that stage). The centroid is sum(stage * burns_at_stage) / total_burns over
    BOTH agents and ALL rollouts, rounded to an int stage. Returns None if there are no burns or
    no .npz (caller then falls back to an even-spread half). One conjunction => one centralized
    cell, so we just take the first .npz in the dir (the centralized rail dumps exactly one)."""
    files = glob.glob(os.path.join(npz_dir, "*.npz"))
    if not files:
        return None
    z = np.load(files[0])
    if "burn_a1" not in z:
        return None
    mass = (z["burn_a1"] != 0).sum(axis=0) + (z["burn_a2"] != 0).sum(axis=0)   # per-stage count
    total = mass.sum()
    if total == 0:
        return None
    stages = np.arange(len(mass))
    return int(round(float((stages * mass).sum() / total)))


# ---------------------------------------------------------------------------
# Solve ONE subset (the shared spawn/drain primitive used by BOTH grid + adaptive)
# ---------------------------------------------------------------------------
def _solve_subset(conj, beliefs, name, stages, backend, rollouts, b1_opts, rollout_dir=None):
    """Spawn the worker for one explicit contact subset, return (rows, primary_row).

    `rows` are the raw worker rows (already tagged subset_name+contacts); `primary_row` is the
    single row for the variant this subset actually ran (sdec for normal subsets, but centralized
    for the __centralized__ rail and dec for __dec__) — the adaptive search reads expected_return
    off it, so it MUST match the run variant (else the centralized rail's return reads as None and
    the search wrongly skips the conjunction). The rails are the special names '__centralized__'
    (all stages, variant centralized) and '__dec__' (no syncs, variant dec).

    rollout_dir: if given, the worker dumps its per-cell .npz (incl. burn_a1/burn_a2 = WHEN each
    burn fired) there — read for the Centralized burn centroid AND, when --save-rollouts is on, the
    full per-rollout brahe distribution for every subset."""
    variants = ["sdec"]
    contacts = ",".join(str(s) for s in stages)
    if name == "__centralized__":
        variants, contacts = ["centralized"], None
    elif name == "__dec__":
        variants, contacts = ["dec"], None
    shard_dir = tempfile.mkdtemp(prefix="peel_shard_")
    shard = os.path.join(shard_dir, "shard.csv")
    # Inject our EXPLICIT contact subset via spawn_conjunction's `contacts=` override (None => the
    # rails: full set for centralized, none for dec). No monkeypatching of resolve_contacts.
    proc, workdir, _ = SD.spawn_conjunction(
        conj, beliefs, variants, [], backend, rollouts, b1_opts,
        0, set(), shard, rollout_dir=rollout_dir, solve_obs=dict(_SOLVE_OBS), contacts=contacts,
    )
    proc.wait()
    out = []
    if os.path.exists(shard):
        with open(shard, newline="") as f:
            for r in csv.DictReader(f):
                r["subset_name"] = name
                r["contacts"] = contacts or "ALL"
                out.append(r)
    shutil.rmtree(workdir, ignore_errors=True)
    shutil.rmtree(shard_dir, ignore_errors=True)
    primary_row = next((r for r in out if r.get("variant") == variants[0]), None)
    return out, primary_row


def _record(out, rows, csv_path, wb, is_final=False):
    """Persist a freshly-solved subset's worker rows: extend the master list, rewrite the CSV, and
    (if wb) log each row to wandb REUSING sweep_driver._wandb_log_row.

    Peel-specific info rides extra_config (strings can't be wandb metrics): `contacts` /
    `subset_name` (WHICH contacts were kept) + `is_final` (True for the rails + the minimal set,
    False for the intermediate window / greedy-drop candidates) so the wandb UI can filter to the
    3 headline runs per conjunction OR expand to the full search trace. subset_name is the run
    name_suffix so same-conj/belief/variant subsets get DISTINCT runs instead of colliding. (return,
    collision_prob, brahe_*, n_contacts already log as numeric metrics via _wandb_log_row.)"""
    rows.extend(out)
    _write(csv_path, rows)
    if not wb:
        return
    for r in out:
        SD._wandb_log_row(
            r, wb,
            extra_config={"contacts": r.get("contacts"), "subset_name": r.get("subset_name"),
                          "is_final": is_final},
            name_suffix=f"_{r.get('subset_name', '')}",
        )


def _return_of(row):
    """expected_return as float, or None if missing/blank/non-numeric (e.g. infeasible solve)."""
    try:
        return float(row.get("expected_return", "")) if row else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Run one conjunction across all subsets (sequential; one worker per subset)
# ---------------------------------------------------------------------------
def run_conjunction(conj, beliefs, subsets, backend, rollouts, csv_path, done, rows, b1_opts,
                    wb=None, save_dir=None):
    """GRID mode: for each named subset, solve it, tag rows with subset_name + contacts, record
    (CSV + wandb). Rails (__centralized__/__dec__) are tagged is_final for the headline filter."""
    for name, stages in subsets.items():
        key = (str(conj.name or conj.label), name)
        if key in done:
            print(f"  skip {conj.name} :: {name} (done)")
            continue
        out, _ = _solve_subset(conj, beliefs, name, stages, backend, rollouts, b1_opts,
                               rollout_dir=save_dir)
        _record(out, rows, csv_path, wb, is_final=name.startswith("__"))
        done.add(key)
        contacts = out[0]["contacts"] if out else "?"
        print(f"  {conj.name} :: {name:22s} contacts={contacts} -> {len(out)} row(s)")

# ---------------------------------------------------------------------------
# ADAPTIVE search: shrink the contact set until it stops matching Centralized
# ---------------------------------------------------------------------------
def _passes(sdec_row, cen_return, tol):
    """Does this SDec subset still 'match' Centralized? PASS iff its expected_return is within
    `tol` of the Centralized rail. Higher return = better, so the only way to fail is to fall
    BELOW the rail by more than tol; a subset that scores >= the rail trivially passes. A missing
    return (infeasible / solve error) is a FAIL (can't keep a set we couldn't even solve)."""
    er = _return_of(sdec_row)
    if er is None or cen_return is None:
        return False
    return er >= cen_return - tol


def _solve_and_record(conj, beliefs, name, stages, backend, rollouts, b1_opts,
                      csv_path, rows, cache, wb, rollout_dir=None, is_final=False):
    """Solve a subset ONCE (memoized on the frozenset of stages so the bisection + greedy phases
    never re-pay a solve for the same contact set), record its rows (CSV + wandb via _record),
    return its sdec_row.

    rollout_dir: forwarded to the worker so it dumps the per-cell .npz (burn timing + full
    per-rollout brahe arrays). Used for the Centralized rail (to read the burn centroid) and, when
    --save-rollouts is on, for every subset. Memoized rows skip the re-solve (and its dump).
    is_final: tag the wandb run as a headline (rails + minimal) vs an intermediate candidate."""
    fz = frozenset(stages)
    if fz in cache:
        return cache[fz]
    out, sdec = _solve_subset(conj, beliefs, name, stages, backend, rollouts, b1_opts,
                              rollout_dir=rollout_dir)
    _record(out, rows, csv_path, wb, is_final=is_final)
    cache[fz] = sdec
    er = _return_of(sdec)
    print(f"  {conj.name} :: {name:24s} stages={stages} return={er}")
    return sdec


def run_conjunction_adaptive(conj, beliefs, avail, backend, rollouts, csv_path, rows, b1_opts,
                             tol, wb=None, save_dir=None):
    """Peel contacts down to a (locally) minimal set that still MATCHES Centralized, using the
    Centralized burn location to choose WHERE to cut first (the physically-motivated window).

    Pipeline per conjunction:
      0. RAILS   — solve Centralized (all stages, dumping its burn .npz) + Dec (none) once.
                   Centralized's return is the bar; a subset PASSES iff its return is within `tol`
                   of it (see _passes). Read the burn CENTROID stage from the Centralized .npz.
      1. WINDOW  — keep the HALF of the available contacts CLOSEST to the burn centroid (the
                   'contacts near the maneuver' set; if the burn is early you get the early half,
                   late->late half, middle->middle band). Solve it. If it passes it becomes the
                   working set; if it fails we fall back to the FULL set (never start worse).
                   Falls back to an even-spread half if Centralized logged no burns.
      2. GREEDY  — from the working set, drop each single contact; keep any drop that still passes
                   and restart; stop when no single drop survives. This non-bisecting fine phase
                   finds the locally-minimal set (and CAN end up spanning the early/late halves).
    Every solved subset is one CSV row (and a wandb run if `wb`); minimal is re-tagged 'minimal'.

    wb: sweep_driver wandb dict (or None). save_dir: persistent --save-rollouts dir (or None) where
    the worker dumps full per-rollout .npz for EVERY subset. The Centralized burn centroid is read
    from save_dir when set, else from a throwaway temp dir."""
    cache = {}
    # Centralized needs a rollout dir for its burn .npz; reuse the persistent save_dir if saving
    # rollouts, else a temp dir we delete after reading the centroid.
    burn_dir = save_dir or tempfile.mkdtemp(prefix="peel_burn_")

    # --- 0. rails (centralized dumps its burn .npz into burn_dir) ---
    cen = _solve_and_record(conj, beliefs, "__centralized__", avail, backend, rollouts,
                            b1_opts, csv_path, rows, cache, wb, rollout_dir=burn_dir, is_final=True)
    _solve_and_record(conj, beliefs, "__dec__", [], backend, rollouts, b1_opts,
                      csv_path, rows, cache, wb, rollout_dir=save_dir, is_final=True)
    cen_return = _return_of(cen)
    if cen_return is None:
        print(f"  {conj.name}: centralized infeasible -> skip adaptive")
        if save_dir is None:
            shutil.rmtree(burn_dir, ignore_errors=True)
        return
    burn = burn_centroid_from_npz(burn_dir)
    if save_dir is None:
        shutil.rmtree(burn_dir, ignore_errors=True)
    print(f"  {conj.name}: centralized return={cen_return:.3f}  floor={cen_return - tol:.3f}  "
          f"burn_centroid_stage={burn}")

    # --- 1. burn-centered window (half the contacts nearest the burn; spread-half if no burn) ---
    half = (len(avail) + 1) // 2
    if burn is not None:
        window = near_stage_subset(avail, half, burn)        # k nearest the burn centroid
        wname = f"window_nearburn_c{len(window)}"
    else:
        window = spread_subset(avail, half)
        wname = f"window_spread_c{len(window)}"
    sd_win = _solve_and_record(conj, beliefs, wname, window, backend, rollouts, b1_opts,
                               csv_path, rows, cache, wb, rollout_dir=save_dir)
    if _passes(sd_win, cen_return, tol):
        best = window
        print(f"  {conj.name}: window {window} PASSES -> greedy from window")
    else:
        best = list(avail)
        print(f"  {conj.name}: window {window} fails tol -> greedy from FULL set")

    # --- 2. greedy single-removal (non-bisecting fine phase -> locally minimal) ---
    improved = True
    while improved and len(best) > 1:
        improved = False
        for drop in list(best):                      # try removing each single contact
            cand = [s for s in best if s != drop]
            sd = _solve_and_record(conj, beliefs, f"greedy_drop{drop}", cand, backend, rollouts,
                                   b1_opts, csv_path, rows, cache, wb, rollout_dir=save_dir)
            if _passes(sd, cen_return, tol):
                best = cand
                improved = True
                break                                # restart the scan on the smaller set

    # --- final: re-tag the minimal set so it's trivially queryable in CSV + wandb (is_final) ---
    out, _ = _solve_subset(conj, beliefs, "minimal", best, backend, rollouts, b1_opts,
                           rollout_dir=save_dir)
    _record(out, rows, csv_path, wb, is_final=True)
    print(f"  {conj.name}: MINIMAL = {best} ({len(best)} contacts), floor={cen_return - tol:.3f}")


def _write(path, rows):
    if not rows:
        return
    cols = list(rows[0].keys())
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _load_done(path):
    done, rows = set(), []
    if not os.path.exists(path):
        return done, rows
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
            if r.get("status") in ("ok", "infeasible"):
                done.add((r.get("label", ""), r.get("subset_name", "")))
    return done, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conj-file", required=True)
    ap.add_argument("--init-miss", default="0.5",
                    help="belief center(s): comma list of km, OR 'true' to center each "
                         "conjunction's belief on its own true miss (conj.miss_km).")
    ap.add_argument("--init-spread", default="1.4")
    ap.add_argument("--rollouts", type=int, default=200)
    ap.add_argument("--backend", default="numerical", choices=["numerical", "keplerian", "drag"])
    ap.add_argument("--place-counts", default="2,3", help="small counts k for early/late/near-burn")
    ap.add_argument("--no-rails", action="store_true", help="skip centralized/dec rails")
    ap.add_argument("--coarse", action="store_true",
                    help="SPEED PRESET (machinery checks): ~4h cadence + 2h merge => N_STAGES ~6, "
                         "solve ~6s vs ~6min (sub-minute peel). Reuses sweep_driver._COARSE_GRID "
                         "(must also pass --coarse so the worker's scenario matches). NOT for "
                         "production: coarse grid burns differently, returns won't match the "
                         "25-stage headline.")
    ap.add_argument("--mode", default="adaptive", choices=["adaptive", "grid"],
                    help="adaptive=burn-centered window + greedy peel-down (default); "
                         "grid=static halving/placement subsets")
    ap.add_argument("--tol", type=float, default=0.001,
                    help="adaptive: a subset PASSES if its expected_return is within this many "
                         "units BELOW the Centralized rail (smaller = stricter, keeps more contacts). "
                         "Default 0.001 = 'must match Centralized exactly' (tiny slack absorbs "
                         "floating-point noise; a larger tol lets the peel over-shrink toward Dec).")
    ap.add_argument("--tag", default="peel")
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "notes", "results"))
    # --- RS-SDA* solve limits (CLUSTER SAFETY VALVE; mirror sweep_driver). Hitting the memory
    #     ceiling ERRORS the cell (blank return) => _passes FAIL => greedy KEEPS that contact
    #     (conservative). Set iter-limit GENEROUS so a partial bail ~never fires. ---
    ap.add_argument("--sdec-iter-limit", type=int, default=50000,
                    help="RS-SDA* TI2 budget per solve (solve.sdec_iter_limit). Generous default so "
                         "it only catches genuine hangs; a bail reads as FAIL => contact kept.")
    ap.add_argument("--sdec-memory-limit-gb", type=float, default=None,
                    help="RS-SDA* memory ceiling GB (solve.sdec_memory_limit_gb); raise on a "
                         "big-memory box. Hitting it errors the cell (contact kept, job continues).")
    ap.add_argument("--obs-fidelity", default=None,
                    help="SDec sync obs fidelity: perfect|gps|tle|asymmetric (obs.fidelity).")
    ap.add_argument("--obs-sigma", default=None,
                    help="raw SDec sync obs sigma km (obs.sigma); overrides the named fidelity.")
    # --- tracking (mirrors sweep_driver): CSV is always written; these add wandb + raw rollouts ---
    ap.add_argument("--save-rollouts", action="store_true",
                    help="ALSO dump full per-rollout arrays (brahe miss/dt/dV + per-stage burn "
                         "matrices) for EVERY subset to <out-dir>/rollouts_<tag>/, one .npz per "
                         "cell (same format as sweep_driver --save-rollouts).")
    ap.add_argument("--wandb", action="store_true",
                    help="log each subset as a wandb run (reuses sweep_driver._wandb_log_row); the "
                         "kept `contacts`/`subset_name`/`is_final` ride the run config so you can "
                         "filter to rails+minimal or expand to the full peel trace.")
    ap.add_argument("--wandb-project", default="spacecraftCA")
    ap.add_argument("--wandb-entity", default="kmeans_gsopt")
    ap.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    args = ap.parse_args()

    # solve knobs forwarded to every worker spawn (see _SOLVE_OBS). Only non-None keys are sent so
    # _conj_scenario_file leaves unset knobs at the solver defaults.
    global _SOLVE_OBS
    _SOLVE_OBS = {k: v for k, v in dict(
        sdec_iter_limit=args.sdec_iter_limit,
        sdec_memory_limit_gb=args.sdec_memory_limit_gb,
        obs_fidelity=args.obs_fidelity,
        obs_sigma=args.obs_sigma,
    ).items() if v is not None}

    conjs = SD.conjunctions_from_file(args.conj_file)
    # --init-miss "true" (sentinel, mirrors sweep_driver): seed each conjunction's belief center
    # at ITS OWN true miss (conj.miss_km) rather than a fixed global list. Spread is unchanged.
    true_belief = args.init_miss.strip().lower() == "true"
    spreads = [float(x) for x in args.init_spread.split(",") if x.strip()]
    beliefs = ([] if true_belief
               else [(float(im), sp)
                     for im in (float(x) for x in args.init_miss.split(",") if x.strip())
                     for sp in spreads])

    def _beliefs_for(conj):
        """Per-conjunction belief list. Numeric --init-miss => the global list (same for every
        conjunction). --init-miss true => center on this conjunction's own true miss, one belief
        per spread (uncertainty from --init-spread, unchanged)."""
        return [(conj.miss_km, sp) for sp in spreads] if true_belief else beliefs
    place_counts = [int(x) for x in args.place_counts.split(",") if x.strip()]
    csv_path = os.path.join(args.out_dir, f"peel_{args.tag}.csv")
    done, rows = _load_done(csv_path)
    b1_opts = dict(b1_strategy="threshold", b1_policy="conservative", b1_other_obs="tle")

    wb = (dict(project=args.wandb_project, entity=args.wandb_entity, mode=args.wandb_mode,
               tag=args.tag) if args.wandb else None)
    save_dir = None
    if args.save_rollouts:
        save_dir = os.path.join(args.out_dir, f"rollouts_{args.tag}")
        os.makedirs(save_dir, exist_ok=True)
        print(f"    saving full per-rollout arrays -> {save_dir}/ (one .npz per subset cell)")

    n_belief = len(spreads) if true_belief else len(beliefs)
    print(f"peel_contacts[{args.mode}]: {len(conjs)} conj x {n_belief} belief, "
          f"backend={args.backend}{' +wandb' if wb else ''}"
          f"{', init_miss=TRUE (per-conj true miss)' if true_belief else ''} -> {csv_path}")
    # --coarse: peel reads available contacts itself (CG.conjunction_contacts), so it MUST use the
    # SAME coarse grid the worker's solve will use (SD._COARSE_GRID, auto-populated from --coarse in
    # argv) — else peel would pick fine-grid contacts the 6-stage worker doesn't have (hard-error).
    grid_kw = SD._COARSE_GRID if args.coarse else {}
    for conj in conjs:
        _, avail = CG.conjunction_contacts(conj, **grid_kw)
        avail = sorted(int(a) for a in avail)
        if args.mode == "adaptive":
            print(f"\n[{conj.name}] avail({len(avail)})={avail}  tol={args.tol}")
            run_conjunction_adaptive(conj, _beliefs_for(conj), avail, args.backend, args.rollouts,
                                     csv_path, rows, b1_opts, args.tol, wb=wb, save_dir=save_dir)
        else:
            # grid mode: static halving + early/late placement. near_burn (place_nearburn_c*) is
            # skipped because build_subsets gets burn_stage=None — adaptive mode owns burn timing.
            subsets = build_subsets(avail, place_counts, None)
            if not args.no_rails:
                subsets = {"__centralized__": avail, "__dec__": [], **subsets}
            print(f"\n[{conj.name}] avail({len(avail)})={avail} subsets={list(subsets)}")
            run_conjunction(conj, _beliefs_for(conj), subsets, args.backend, args.rollouts,
                            csv_path, done, rows, b1_opts, wb=wb, save_dir=save_dir)
    print(f"\ndone -> {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
