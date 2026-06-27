# Spacecraft CA SDec-POMDP — V2 (config-first) how-to-run

> **This is the V2 pipeline.** The top-level `README.md` describes the OLD v1 pipeline — ignore
> it for v2 work. Everything here runs through ONE config surface (`conf/config.yaml`), no
> environment variables. Landed 2026-06-24. See `notes/SCENARIO_KNOBS.md` for the full
> knob→config-path map and `notes/session-log/2026-06-24.md` for the design.

## The pattern

```
cfg  ->  build_scenario(cfg)  ->  solve(scenario)
```

- `conf/config.yaml` is the ONE file you edit. It has labeled subgroups: **conjunction** (the
  orbit pair), **grid** (propagator + decision cadence), **contacts** (the SDec sync lever),
  **belief** (danger × uncertainty), **reward**, **solve** (which variants), **run** (output),
  **wandb**.
- `scenario_config.build_scenario(cfg)` populates the model before it loads (the "keystone").
- `main.py` is the runner: it resolves the config, solves with the canonical solver
  (`compare_variants_v2`, reused — not reimplemented), writes the CSVs, and optionally logs to
  wandb. **CSVs are always the source of truth; wandb is a layer on top.**

All commands below assume you run from the repo root with the project venv:

```bash
V=.venv/bin/python ; CA=benchmarks/spacecraftCA
```

---

## 1. Single run (no wandb — CSV only)

```bash
# full-res head-on reference conjunction, all 3 variants
$V $CA/main.py

# the REGRESSION ANCHOR (machinery check): Cen = SDec = -7.8300 return, 0% collision, ~6 min
$V $CA/main.py +preset=verify

# quick machinery check (~seconds): coarse ~4h grid, N_STAGES~6 (NOT production numbers)
$V $CA/main.py +preset=coarse
```

### Override any knob on the command line (Hydra dotpath)

```bash
$V $CA/main.py belief.init_miss=0.5 belief.init_spread=1.4 run.tag=danger_run
$V $CA/main.py reward.man_cost=-3 reward.disp_k=0.5
$V $CA/main.py grid.propagator=drag                       # J2+drag+SRP (experiments)
$V $CA/main.py 'grid.hour_grid_h=[24,20,16,12,8,4]' grid.merge_threshold_h=2.0
$V $CA/main.py 'solve.variants=[centralized,sdec]'        # quote brackets in zsh
$V $CA/main.py 'contacts.stages=[1,9,13,15]'              # SDec contact subset (ablation)
$V $CA/main.py run.rollout=true run.rollouts=200          # also brahe-validate the policy
```

### Add the B1 operator floor (the brahe-MC comparison baseline)

`baseline.enabled=true` runs the **B1 operator-heuristic floor** alongside the POMDP solve —
a separate hand-coded per-craft heuristic (NO solver), reusing `baseline_b1.py` through the
same brahe engine. It is **additive and off by default** (the anchor is POMDP-only). B1 and the
POMDP are independent policy sets, so you can run either, both, or neither:

```bash
$V $CA/main.py baseline.enabled=true run.rollout=true     # POMDP + B1, both flown through brahe
$V $CA/main.py 'solve.variants=[]' baseline.enabled=true  # baseline-ONLY (skip the POMDP solve)
$V $CA/main.py baseline.enabled=true baseline.strategy=firereturn baseline.other_obs=frozen
```

B1 writes its per-rollout CSV to `notes/results/b1_<run.tag>.csv` and prints the SAME 4-7km band
stats (collision% / in-band / miss / dV / burns) as the POMDP brahe rollout — so the comparison
is brahe-MC vs brahe-MC on one axis (turn on `run.rollout=true` to get the POMDP side too). The
in-process result is byte-identical to the standalone `baseline_b1.py --mode mc`. All knobs
(`baseline.strategy/policy/selfish_model/other_obs/tle_sigma/variant/rollouts/seed`) are in
`conf/config.yaml`; see `notes/SCENARIO_KNOBS.md`.

Outputs land in `notes/results/variant_*_<run.tag>.csv` (figures in `notes/figures/` if
`run.figures=true`). Regenerate figures from a CSV without re-solving:
`$V $CA/replot_from_csv.py --tag <tag>`.

---

## 2. A real (orbit-first) conjunction

`conf/config.yaml` takes the conjunction as an explicit orbit pair (KOE in degrees,
`[a, e, i, RAAN, omega, M]`, a in metres). `null/null` => the default reference scenario.

```yaml
conjunction:
  sc1_oe: [6928136.3, 0.001, 55.0, 20.0, 0.0,   0.0]
  sc2_oe: [6900842.3, 0.003, 55.0, 20.0, 180.0, 180.0]
```

The stage grid (N_STAGES, GS contacts) is rebuilt from THIS orbit pair automatically. The solver
assumes the pair is valid — deciding *which* conjunctions are admissible and *generating* the
pairs is a separate concern (`conjunction_generator.py` + `sweep_driver.py`, §4).

---

## 3. Single run WITH wandb

```bash
$V $CA/main.py wandb.enabled=true wandb.project=spacecraftCA run.tag=exp1
# offline (no network; sync later with `wandb sync`):
$V $CA/main.py wandb.enabled=true wandb.mode=offline run.tag=exp1
```

wandb logs the fully-resolved config dict, a set of **summary scalars** (the sweep AXES, so runs
group/sort in the UI), and **six tables** — one per figure, so every paper figure is reproducible
PURELY from the wandb run (the CSVs are still the source of truth).

Summary scalars: `n_stages`, `n_contacts`, `contact_stages`, `init_miss`, `init_spread`, `perp`,
`propagator`, `man_cost`, `disp_k`, `flagged`, `eff_miss`.

Tables (kept on separate axes so nothing is apples-to-oranges):
- `variant_summary` — POMDP **matrix-expectation** (expected_return / collision_prob / dv_ms /
  syncs). Solved variants only.
- `reward_parts` — per-variant maneuver / deviation / risk / displace → reward-decomposition figure.
- `burn_timing` — per-variant per-stage mean agent burns → burn-timing figure.
- `action_schedule` — per-variant per-stage dominant joint action + prob → policy figure.
- `brahe_summary` — **brahe-MC** band stats + maneuvers + fidelity (collision% / in-band% / miss /
  dV / burns / ≤2-burn% / term-reward-gap / dt-err) for every policy actually flown: POMDP variants
  under `run.rollout=true` AND B1 under `baseline.enabled=true`, on ONE axis = the direct compare.
- `brahe_rollouts` — per-rollout brahe/matrix miss + dV + burns + terminal rewards + dt-err →
  the histogram / fuel-distribution / model-fidelity figures (re-plottable from the table).

### Plot every figure FROM a wandb run

`plot_from_wandb.py` is the wandb twin of `replot_from_csv.py`: it pulls the logged tables and
feeds them to the SAME plot functions, so the figures are byte-for-byte the CSV ones (the tables
carry the full row dicts). Works on a local run dir (offline) or a remote run id (`wandb.Api`):

```bash
$V $CA/plot_from_wandb.py --run kmeans_gsopt/spacecraftCA/<run_id>   # remote, via API
$V $CA/plot_from_wandb.py --run <run_id>                             # local: searched under ./wandb
$V $CA/plot_from_wandb.py --run wandb/run-<ts>-<id> --tag myfig      # explicit local dir
```

Renders summary / reward_parts / burn_timing / action_schedule and a per-policy START/END miss
histogram (centralized / sdec / B1) into `notes/figures/`. (CSV path unchanged:
`$V $CA/replot_from_csv.py --tag <tag>`.)

---

## 4. Sweeps

### (a) Hydra multirun — sweep a knob, one process per cell

```bash
$V $CA/main.py -m belief.init_miss=0.5,3,5                      # 1-D belief sweep
$V $CA/main.py -m belief.init_miss=0.5,3 'contacts.stages=[1,9],[1,9,13,15],null'  # 2-D
$V $CA/main.py -m wandb.enabled=true belief.init_miss=0.5,3,5   # logged sweep
```

Each `-m` combination is a fresh process (correct N_STAGES per cell; this is the right pattern).

### (b) Conjunction-set sweep (the suite path) — `sweep_driver.py`

Sweeps many conjunctions × beliefs × variants, brahe-validates each (Monte-Carlo), and writes ONE
tidy row per (conjunction × belief × variant) to a resumable CSV.

**The subprocess unit is ONE conjunction + its whole belief × variant grid** (`_conj_worker.py`),
not one cell. Each conjunction runs in a fresh subprocess handed its scenario via a YAML
`--scenario-config` (no env vars). Two things make a sweep fast:

- **Matrix reuse.** T/O/R depend only on the conjunction (orbit / grid / contacts / reward) —
  *not* on the belief (`init_b` is the only per-belief input). The worker builds T/O/R **once per
  conjunction** and reuses them across all of that conjunction's beliefs, so the ~22 s matrix
  build is paid once per conjunction instead of once per cell.
- **`--jobs N`** runs up to N conjunction-children **concurrently** (default 1 = sequential). Each
  child writes its own shard CSV; the parent merges shards into the master CSV (single writer), so
  Ctrl-C always leaves a valid resumable CSV. Set `--jobs` to ~#cores for a multi-conjunction
  sweep. Reuse + parallelism change only SPEED, never the numbers.

```bash
# validation probe (head-on + oblique + cross-track), coarse for speed:
$V $CA/sweep_driver.py --coarse --probe --rollouts 20 --tag probe

# explicit geometry sweep (generator builds the orbit pairs), 4 conjunctions at a time:
$V $CA/sweep_driver.py --miss 5 --angles 0,45,90 --init-miss 0.5,3,5 \
    --variants centralized,sdec,dec --baselines b1 --rollouts 200 --jobs 4 --tag suiteA

# hand-in conjunction set (reviewable JSON of orbit pairs or geometry specs):
$V $CA/sweep_driver.py --conj-file my_suite.json --init-miss 0.5,3 --variants centralized,sdec --tag suiteB
```

Re-run with the same `--tag` to resume — done cells are skipped (the parent tells each child which
of its cells are already in the master CSV).

> The conjunction TEST SUITE (which conjunctions, coverage figure, a clean conjunction-set config
> artifact, B1 folded into config) is the next piece of design work — see `notes/todo.md`.

---

## 5. Direct CLIs (debug / diagnostics)

The per-variant CLIs still exist and are thin pass-throughs to the SAME config surface (every flag
routes through a `Scenario`). Use them for quick debugging — especially `rollout_v2 --trace`, the
per-stage brahe-vs-matrix diagnostic the project requires.

```bash
# solve one variant directly:
$V $CA/compare_variants_v2.py --variants sdec --init-miss 0.5 --init-spread 1.4 \
    --backend numerical --no-figures --tag dbg

# brahe rollout validation of the SOLVED policy (ALWAYS --trace per the every-stage rule):
$V $CA/rollout_v2.py --variant sdec --mode point --init-miss 0.5 --init-spread 1.4 \
    --backend numerical --trace
$V $CA/rollout_v2.py --variant sdec --mode mc --rollouts 200 --init-miss 0.5 --init-spread 1.4

# a YAML can supply the scenario base; explicit flags override it:
$V $CA/compare_variants_v2.py --scenario-config conf/preset/verify.yaml --tag dbg
```

---

## Presets (`conf/preset/`, swapped in with `+preset=NAME`)

| preset | what | use |
|---|---|---|
| `+preset=verify` | full-res head-on reference + 0.5/1.4 belief | the REGRESSION ANCHOR (Cen=SDec=-7.83/0%); run after any change |
| `+preset=coarse` | ~4h cadence + 2h merge => N_STAGES~6 | quick machinery checks (seconds), NOT production numbers |

## Dependencies

Pinned in the root `requirements.txt`: `hydra-core==1.3.2`, `omegaconf==2.3.0`,
`wandb==0.18.7`, `pyyaml==6.0.2` (this is a `uv` venv — use
`VIRTUAL_ENV=$PWD/.venv uv pip install ...`, not `python -m pip`).
