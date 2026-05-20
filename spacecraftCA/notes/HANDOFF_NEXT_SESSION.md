# Handoff — Spacecraft CA SDec-POMDP (Next Session)

Read these two files first before doing anything:
  spacecraftCA/notes/SESSION_NOTES.md  — full design decisions, history, results, run order
  spacecraftCA/notes/report/formulation.tex — complete POMDP formulation

The venv is set up: run everything with `.venv/bin/python spacecraftCA/<file>.py`

Always do tasks step by step and pause if you hit issues.

---

## WHERE WE ARE (end of session 2026-05-20)

Three-variant matrix architecture is working end-to-end. All three variants
(centralized, sdec, dec) build, solve, and run rollouts. Dec OOMs at σ=0.50 with
spread belief (expected — key paper result). SDec works but never syncs.

Committed baseline: commit 022f66d (sessions 1-4). Session 6 changes are uncommitted
(check `git status` — modified: spacecraft_matrices.py, spacecraft_simulator.py,
spacecraft_discretizer.py, sdec_spacecraft.py, notes/).

Repo structure:
  spacecraft_discretizer.py   — 6 miss-distance bins × 16 stages + 1 sink = 97 states
  spacecraft_matrices.py      — T, O, R built with Brahe (3 variants, stochastic σ=0.50)
  sdec_spacecraft.py          — RS-SDA* wrapper; infers action space from T shape
  spacecraft_simulator.py     — closed-loop rollout; --compare mode; force_sync flag
  plot_policy.py              — (miss_bin × stage) policy grid visualization
  plot_conjunction_geometry.py — static PNG + GIF of RTN approach trajectory
  examples/plot_deviation_vs_burn_time.py — deviation analysis plot

Notes:
  notes/SESSION_NOTES.md      — full history and run order
  notes/report/formulation.tex/.pdf  — 11-page technical reference
  notes/slides/slides.tex/.pdf       — 21-page Beamer deck
  notes/figures/                     — all generated plots

Current model parameters (session 6):
  dv = 0.5 m/s
  EXEC_NOISE_SIGMA = 0.50  (±50% burn execution noise, 1-sigma)
  GPS_SIGMA_KM = 0.1  (centralized / SDec-sync obs precision)
  TLE_SIGMA_KM = 3.0  (dec and between-contact obs noise)
  REWARD_COLLISION = -10000, REWARD_HIGH = -1000, REWARD_MOD = -100
  REWARD_MANEUVER = -10 (flat, same all stages — this is what we're changing next)
  SYNC_COST = -0.5, SYNC_OUTSIDE_COST = -0.05
  16 stages: 10 ~2h grid + 6 GS contacts merged; CONTACT_STAGES = [3,5,7,9,11,13]
  6 miss bins: [0,1), [1,5), [5,20), [20,100), [100,500), [500+) km
  Spread initial belief: uniform over bins 0, 1, 2

Session 6 comparison (5 rollouts, spread belief, sigma=0.50):
  Centralized    bin 0:  0% coll, 19km miss,  1.0 m/s dv, 6.0 syncs
  SDec           bin 0:  0% coll,  8km miss,  1.0 m/s dv, 0.0 syncs
  Dec            bin 0:  OOM (7GB+, no belief collapse without sync states)
  Centralized    bin 1:  0% coll, 129km miss, 0.5 m/s dv, 6.0 syncs
  SDec           bin 1:  0% coll,   9km miss, 1.0 m/s dv, 0.0 syncs
  Dec            bin 1:  OOM

---

## IMMEDIATE TASK 1: Stage-Dependent Maneuver Cost (highest priority)

**Goal:** incentivize the policy to wait for a GS contact before committing a burn,
so that SDec actually uses sync to refine belief before acting.

**Root cause of SDec never syncing:** policy burns at stage 0 (T-24h, before any
contact). The first contact is at stage 1 (T-23.39h). After the stage-0 burn is
committed, syncing at stage 1 can't change the decision — the trajectory is set.

**Fix:** add `stage_maneuver_cost(stage)` in spacecraft_matrices.py that makes
early burns expensive and late burns cheap:

```python
REWARD_MANEUVER       = -10.0   # base cost (stage 0, T-24h)
REWARD_MANEUVER_LATE  =  -2.0   # cost at last stage (T-1h)

def stage_maneuver_cost(stage: int) -> float:
    frac = stage / max(N_STAGES - 1, 1)
    return REWARD_MANEUVER + frac * (REWARD_MANEUVER_LATE - REWARD_MANEUVER)
```

Then replace flat `r += REWARD_MANEUVER` in `build_matrices` with:
  `r += stage_maneuver_cost(k)`

Rebuild all three variants (`--variant all --force`), re-run comparison.
Check if SDec policy now waits for first contact before burning.
Expected: centralized still burns once (gets GPS obs first), SDec syncs at least once.

**Important:** CONTACT_STAGES in spacecraft_matrices.py must be verified against the
actual 16-stage schedule. The first contact is at `_GS_TIMES_H = 23.39h` which maps
to stage index 3 in the merged 16-stage list (not stage 1). Verify with:
```python
print(CONTACT_STAGES)  # should be [3, 5, 7, 9, 11, 13] or similar
print([_ALL_TIMES_H[i] for i in CONTACT_STAGES])  # should print GS contact hours
```

---

## IMMEDIATE TASK 2: Sigma Sweep

Sweep EXEC_NOISE_SIGMA ∈ {0.05, 0.15, 0.30, 0.50} and TLE_SIGMA_KM ∈ {1, 3, 5, 10}.
For each, rebuild centralized only (fastest), run 20 rollouts, report miss distance spread.
Goal: find minimum sigma where policies diverge across variants.

Dec OOM threshold: find the sigma / spread combination where Dec first runs out of memory.
At σ=0.50 with spread belief it OOMs at 7GB. At σ=0.15 it was tractable (near-deterministic).

---

## IMMEDIATE TASK 3: Ensure SDec Has Meaningful Policy

After implementing stage-dependent cost, verify that SDec:
1. Syncs at least once in rollouts
2. Burns at a *different* stage than centralized (not stage 0)
3. Uses fewer burns than Dec (more efficient due to coordination)

If SDec still doesn't sync after stage-dependent cost, consider:
- Moving stage 0 to T-23h so the first contact is actually BEFORE stage 0
- Or: reorder stages so first decision is after first contact
- Or: add a small "sync bonus" (negative SYNC_COST, i.e., positive reward for syncing
  when it helps reduce uncertainty before a burn)

---

## FUTURE TODOS (lower priority)

- **Deviation tracking**: expand state to (miss_bin, SC1_dev_bin, SC2_dev_bin, stage) —
  577 states. Longer-term path to meaningful SDec differentiation without relying purely
  on timing incentives. See Session 5 notes for bin thresholds and implementation plan.

- **Full comparison table**: 100 rollouts with all 3 variants (blocked by Dec OOM).
  Run centralized + SDec at 100 rollouts, skip Dec or reduce σ for Dec.

- **Scenario 2**: asymmetric control — SC2 uncooperative (never maneuvers).
  T_asym[a, s, s'] = T_full[(a1*3 + 0), s, s'] where a1 = a // 3.

- **Scenario 3 sweep**: c_sync sweep now has structural support (SYNC_COST in R).
  Run sweep over SYNC_COST ∈ {0, -0.1, -0.5, -1.0, -5.0} and compare sync count vs. miss.

- **Observation noise sweep**: compare variants at TLE_SIGMA_KM ∈ {1, 3, 5, 10} km.

- **`split_joint_action` bug**: module-level `N_ACT_AGENT = 3` is wrong for SDec's
  6-per-agent actions. Fix by passing `n_act_agent` to the function or using
  `decode_agent_action` everywhere instead.

- **`--fixed-init` trajectory tree**: needs updating for new 3-variant structure.

---

## ALWAYS

Update SESSION_NOTES.md with any new design decisions (append new session section).
Update formulation.tex and slides.tex after major changes.
Recompile PDFs: cd spacecraftCA/notes/report && pdflatex formulation.tex
               cd spacecraftCA/notes/slides && pdflatex slides.tex
