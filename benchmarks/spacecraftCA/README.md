# SpacecraftCA Benchmark

`spacecraftCA` is a spacecraft conjunction-assessment benchmark for comparing
centralized, semi-decentralized, and fully decentralized multiagent planning.
The current model focuses on automatic information sharing at synchronization
points; costly or optional communication is out of scope.

## Current Formulation

- State: `(miss_bin, dev1_bin, dev2_bin, stage)`.
- Miss-distance bins: `[0,0.5)`, `[0.5,1)`, `[1,2)`, `[2,5)`, `[5,10)`,
  `[10,20)`, `[20,50)`, `[50,100)`, `[100,500)`, `[500,inf)` km.
- Deviation bins: `NEG`, `NOM`, `POS` for each spacecraft.
- Stages: 16 decision stages, including 6 ground-station contact stages.
- State count: `10 * 3 * 3 * 16 + 1 sink = 1441`.
- Per-spacecraft actions: `WAIT`, `+dV_T`, `-dV_T`.
- Joint action encoding: `joint_action = a1 + 3 * a2`, with SC1 as the
  low-order factor and SC2 as the high-order factor.

Synchronization semantics:

- Centralized: shared history and centralized belief at every stage.
- SDec: shared history and centralized belief only at ground-station contact
  stages `[1, 9, 10, 11, 13, 15]`.
- Decentralized: no synchronization.

Observation semantics:

- At synchronization stages, the joint observation perfectly reveals the shared
  miss bin and both spacecraft deviation bins.
- Away from synchronization stages, each spacecraft observes only its own
  deviation bin plus a null miss symbol.
- Fully decentralized policies therefore receive no private miss-distance or
  other-spacecraft observation; the Dec RS-MAA* adapter uses the resulting
  three-symbol local observation factor.

Transitions use Brahe-backed representative orbital propagation, then perturb
the underlying continuous RTN relative position before re-binning the miss
distance. The current stochastic transition constants are:

- `PROCESS_DRIFT_SIGMA_KM_PER_SQRT_H = 0.15`
- `EXEC_NOISE_SIGMA = 0.50`
- `TRANSITION_NOISE_N_SAMPLES = 101`

Terminal risk rewards by miss bin are:

`[-10000, -10000, -3000, -1000, -300, -100, -25, -5, 0, 0]`

Collision probability is reported as probability mass below 1 km, so it sums
terminal bins 0 and 1.

## Solvers

`compare_variants.py` uses approximate RS-SDA* for Centralized and SDec
policies. Fully decentralized policies default to approximate RS-MAA*, which is
the cleaner solver for a no-sync Dec-POMDP comparison. The `--dec-solver rssda`
option remains available only as a diagnostic.

The main current comparison uses fixed offline policies. A companion SDec run
uses TI1 with interleaved replanning at synchronization nodes.

## Regenerate Matrices

Run from the repository root:

```powershell
.venv\Scripts\python.exe -u benchmarks\spacecraftCA\spacecraft_matrices.py --force --variant all
```

## Regenerate Current Results

Fixed Centralized/SDec/Dec comparison:

```powershell
.venv\Scripts\python.exe -u benchmarks\spacecraftCA\compare_variants.py --variants centralized,sdec,dec --solver-modes fixed --eval-mode expected --eval-bins 0,1,2 --belief-bins 0,1,2 --iter-limit 10000 --tag refined_drift_main
```

SDec TI1 interleaved-replanning companion:

```powershell
.venv\Scripts\python.exe -u benchmarks\spacecraftCA\compare_variants.py --variants sdec --solver-modes interleaved --eval-mode expected --eval-bins 0,1,2 --belief-bins 0,1,2 --iter-limit 10000 --tag refined_drift_main_sdec_ti1
```

## Current Artifacts

Current CSV outputs live in `notes/results`:

- `variant_expected_refined_drift_main.csv`
- `variant_action_by_stage_refined_drift_main.csv`
- `variant_burn_timing_refined_drift_main.csv`
- `variant_expected_refined_drift_main_sdec_ti1.csv`
- `variant_action_by_stage_refined_drift_main_sdec_ti1.csv`
- `variant_burn_timing_refined_drift_main_sdec_ti1.csv`

Current figures live in `notes/figures`:

- `variant_comparison_refined_drift_main.png`
- `burn_timing_refined_drift_main.png`
- `action_schedule_refined_drift_main.png`
- `variant_comparison_refined_drift_main_sdec_ti1.png`
- `burn_timing_refined_drift_main_sdec_ti1.png`
- `action_schedule_refined_drift_main_sdec_ti1.png`

Approximate expected returns for the fixed comparison are:

- Centralized: `-22.29`
- SDec: `-22.44`
- Decentralized: `-165.96`

The large Decentralized gap is primarily terminal risk from lack of any
off-sync miss-distance information, not maneuver cost.
