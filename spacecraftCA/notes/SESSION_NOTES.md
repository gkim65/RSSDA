# Spacecraft CA SDec-POMDP — Session Notes
_Last updated: 2026-05-03 (Session 3)_

---

## What We're Building

A **spacecraft-to-spacecraft collision avoidance SDec-POMDP** solved with RSSDA (RS-SDA*).

Two active spacecraft in a conjunction scenario (~24h to TCA), each controlled by an operator
with intermittent ground station contact. Ground station contact windows = synchronization
triggers in RSSDA.

Three planned scenarios (Scenario 1 first):
1. **Cooperative shared operator** — same operator, 1 GS, ~6 joint contacts
2. **Asymmetric control** — one controllable SC, one with fixed/reactive policy
3. **Costly communication** — sync events incur reward penalty

Paper: SDec-POMDP + RS-SDA* (Alhusseini et al., AAMAS 2026)
Conference target: AAS/AIAA Astrodynamics 2026

---

## Current Repo Structure

```
RSSDA/
├── RSSDA.py                         ← core RS-SDA* solver (3196 lines, do not modify)
├── benchmarks/sdec_mars.py          ← closest analog — follow this pattern
├── pyproject.toml                   ← uv venv, Python 3.12, deps: numpy numba psutil pandas brahe scipy matplotlib
└── spacecraftCA/
    ├── spacecraft_discretizer.py    ← DONE: miss-distance state (37 states)
    ├── spacecraft_matrices.py       ← DONE: Brahe offline T, O, R with back-propagation
    ├── sdec_spacecraft.py           ← DONE: RS-SDA* solver wrapper + solve() helper
    ├── spacecraft_simulator.py      ← DONE: RS-SDA* + greedy rollout, --compare mode
    ├── scenario3_sweep.py           ← DONE: Scenario 3 c_sync sweep + plot
    ├── plot_conjunction_geometry.py ← DONE: static PNG + GIF of RTN approach trajectory
    ├── notes/
    │   ├── SESSION_NOTES.md         ← this file
    │   ├── formulation.tex/.pdf     ← full POMDP formulation (compile with pdflatex)
    │   ├── slides.tex/.pdf          ← Beamer slides (compile with pdflatex)
    │   ├── conjunction_geometry.png ← static plot of radial-miss trajectory
    │   ├── conjunction_geometry.gif ← animated approach
    │   └── SemiDecentralized 2026 AAS_AIAA Astrodynamics Conference Abstract (1).pdf
    └── examples/
        ├── test_contacts.py         ← DONE: GS contact windows via Brahe
        └── test_conjunction_geometry.py  ← DONE: RTN trajectory, conjunction generation
```

Run with: `.venv/bin/python spacecraftCA/<file>.py`

---

## Key Design Decisions (Settled)

### State Space — FINAL

**State = (miss_distance_bin, stage)**

`miss_distance` = predicted miss distance at TCA in km, discretized into 6 bins:

| Bin | Range     | Interpretation |
|-----|-----------|----------------|
| 0   | [0, 1) km    | Collision zone |
| 1   | [1, 5) km    | High risk |
| 2   | [5, 20) km   | Moderate risk |
| 3   | [20, 100) km | Low risk |
| 4   | [100, 500) km | Nominal approach |
| 5   | [500+) km    | Safe |

**Total: 6 bins × 6 stages + 1 sink = 37 states**
Flat index: `s = stage * N_MISS + miss_bin`. Sink = index 36.

### Conjunction Geometry — radial-miss (SETTLED)

**Radial-miss**: SC2 approaches SC1 from behind (along-track), with miss distance in the
radial (R) direction at TCA.

At TCA:
- `δr_RTN = [d_miss, 0, 0]` — radial offset = miss distance
- `δv_RTN = [0, -V_REL_MS, 0]` — along-track closing at V_REL_MS = 15 m/s
- `δN ≈ 0` throughout

**Why NOT crossing geometry:** In a crossing conjunction the miss axis is dN (cross-track),
which collapses to a single bin. No along-track burn shifts dN. The radial-miss geometry
puts the collision axis in dR, which is directly the state variable, and along-track burns
shift dR predictably.

### Back-Propagation for Representative States — CRITICAL

**Wrong approach (old code):** Place SC2 at `δr_RTN = [d_miss, 0, 0]` at the *stage epoch*.
This gives completely wrong TCA miss (thousands of km off) because the radial offset evolves
under Keplerian dynamics over 23h.

**Correct approach (current code):**
1. Place SC2 at TCA with `δr_RTN = [d_miss, 0, 0]`, `δv_RTN = [0, -V_REL, 0]`
2. Back-propagate SC2 from EPOCH_TCA to STAGE_EPOCHS[k] using Brahe
3. The resulting initial condition produces exactly `d_miss` at TCA when forward-propagated
   → round-trip error < 1 m across all 6 bins and all 6 stages

This is implemented in `spacecraft_matrices.py`:
- `place_sc2_at_tca(miss_bin)` → SC2 ECI at TCA
- `back_prop_sc2_to_stage(sc2_tca, stage)` → SC2 ECI at stage epoch
Both simulator and matrix builder use this convention.

### DV Magnitude — Configurable

`DV_MAGNITUDE = 0.5` m/s (default). Configurable via `--dv` CLI flag.
Cache stores the dv it was built with; `sdec_spacecraft.py` auto-detects mismatches
and triggers a rebuild.

**dv sweep results (radial-miss, back-prop geometry):**
- 0.05 m/s: SC1-only burn from bin 0 at stage 3 → bin 1 (3.5 km). Some differentiation.
- 0.5 m/s: SC1-only from bin 0 at stage 3 → bin 3 (34 km). Rich differentiation bins 0-3.
- 1.0 m/s: SC1-only from bin 0 at stage 3 → bin 3 (69 km). Very large; spans bins 0-4.

0.5 m/s is the recommended default for now.

### Actions

Per agent: `WAIT=0`, `+dV_T=1` (prograde), `-dV_T=2` (retrograde) — 3 each → 9 joint.
Delta-v applied impulsively in along-track direction in ECI.

Joint action index: `a = a1 * 3 + a2`.

**Physical insight on joint actions:**
- `(+SC1, WAIT)` or `(WAIT, +SC2)`: one spacecraft burns → relative miss changes
- `(+SC1, +SC2)`: both burn same direction → relative miss barely changes (cancels)
- `(+SC1, -SC2)`: opposite burns → largest miss increase (additive effect)

### Transition Model

Built offline with Brahe. For each (miss_bin, stage, joint_action):
1. Back-propagate SC2 from TCA to stage epoch
2. Apply maneuvers to SC1 and SC2
3. Brahe-propagate BOTH from STAGE_EPOCHS[k] to EPOCH_TCA
4. Measure miss distance → bin it → T[a, s, s']

**T is currently deterministic** (T[a, s, s'] = 1.0 for one s'). This is correct given
bin-center representative states. Stochastic transitions would require sampling over the
initial state distribution within each bin.

### Reward

```
R(s, a) = R_collision(s) + R_maneuver(a)    [Scenario 1/2]

R_collision = -1000  if miss_bin == 0 AND stage == N_STAGES-1  (terminal, last stage only)
R_maneuver  = -10 per agent that maneuvers
```

Note: Collision penalty is on the *current* state at the terminal stage — this fires
correctly because the bin-0 representative trajectory stays in bin 0 all the way to TCA
under WAIT. A maneuver earlier in the horizon must shift the miss out of bin 0 to avoid it.

### Observations

Each agent observes a noisy miss distance estimate during ground contact:
- Gaussian noise: `N(d_miss_true, OBS_SIGMA_KM²)`, `OBS_SIGMA_KM = 50 km`
- Discretized into same 6 miss bins
- Joint obs: `o1 * N_MISS + o2`, 36 total

### Communications in State — NOT INCLUDED

Communication is encoded via `sync_states` in SDecPOMDPModel. For Scenario 1, all 36
non-sink states are sync triggers (all stages have joint contact). This is the standard
pattern across all RS-SDA* benchmarks.

---

## Architecture (Three Components)

```
OFFLINE (once, ~2 min)
  Brahe + spacecraft_matrices.py  →  T, O, R, init_b  →  spacecraft_matrices_cache.npz

POLICY COMPUTATION (once, ~seconds)
  .npz cache  →  sdec_spacecraft.py + RS-SDA*  →  policy π(belief) → action

SIMULATION (per scenario, evaluation only)
  spacecraft_simulator.py + policy  →  miss distance, dv, sync count metrics
```

---

## Contact Windows (Scenario 1)

GS: lon=-76.8, lat=39.0 (Goddard), 10° min elevation
Orbit: 550km, 55° inclination, 24h planning horizon

Decision epochs (SC1 contact midpoints):
- Stage 0: T-23.39h   (2025-06-01 00:36 UTC)
- Stage 1: T-9.48h    (2025-06-01 14:31 UTC)
- Stage 2: T-7.83h    (2025-06-01 16:10 UTC)
- Stage 3: T-6.17h    (2025-06-01 17:49 UTC)
- Stage 4: T-2.80h    (2025-06-01 21:12 UTC)
- Stage 5: T-1.13h    (2025-06-01 22:52 UTC)
TCA:       2025-06-02 00:00 UTC

All 6 contacts are simultaneous for Scenario 1 → `CONTACT_STAGES = [0,1,2,3,4,5]`
→ all 36 non-sink states are sync triggers

---

## Verification Checklist (run after any matrix rebuild)

```bash
.venv/bin/python spacecraftCA/spacecraft_discretizer.py
# Expect: all 36 round-trips OK, bin centers printed

.venv/bin/python spacecraftCA/spacecraft_matrices.py --force
# Expect:
#   T row sums: min=1.0 max=1.0 bad_rows=0
#   O row sums: min=1.0 max=1.0 bad_rows=0
#   Collision states R <= -999: 9 / 9
#   Stage-4 transitions: multiple distinct next bins per action (not all same bin)

.venv/bin/python spacecraftCA/sdec_spacecraft.py
# Expect: runs without error, prints optimal value (0.0 for init belief on safe bins)

.venv/bin/python spacecraftCA/spacecraft_simulator.py --rollouts 20 --init-miss-bin 0
# Expect: 100% collision rate (greedy WAIT policy, bin-0 init)

.venv/bin/python spacecraftCA/spacecraft_simulator.py --rollouts 20 --init-miss-bin 4
# Expect: 0% collision rate, miss stays in bin 4
```

---

## Three Scenarios

**Scenario 1** — Cooperative shared operator (CURRENT)
- Both SC, one operator, one GS (Goddard)
- All 6 contacts joint → always centralized
- Vary: ground network density (global 6-12 GS vs regional 1-3 GS)
- Baselines: centralized (perfect info), decentralized (no sync), SDec (ours)

**Scenario 2** — Asymmetric control (TODO)
- SC1 controllable, SC2 follows fixed third-party policy
- SC2 variants: uncooperative (never maneuvers), reactive (maneuvers if miss < θ)
- TODO: how to encode SC2's fixed policy in T matrix

**Scenario 3** — Costly communication (TODO)
- Same contacts as Scenario 1, but sync costs c_sync in reward
- Sweep c_sync, measure optimal sync count and miss distance

---

## Session 3 Accomplishments (2026-05-03/04)

### RS-SDA* Policy Wired into Simulator — DONE

`spacecraft_simulator.py` now uses the full RS-SDA* policy from `multi_agent_astar`.
Key implementation details (mirroring `sdec_mars.py`):

- Policy structure: `policy[step][0]` = dec `[agent][history_idx]→action`,
  `policy[step][1]` = cen `[cluster_idx]→[joint_action]`
- Centralization check: `cen_dists_map[step]` maps `belief_id → cluster_idx`
- Belief navigation: `sdec.get_terminal(belief_idx, joint_act)` → sparse list
  `(obs_id, prob, next_belief_id)`; match on actual `obs_joint` to advance belief
- Observation history: `clustering[step][agent][history_idx][obs]` → `next_history_idx`
  (decentralized); `clustering_cen[step][agent][c_ptr][obs]` (centralized)
- **Sentinel actions**: policy returns `-4` for unexplored branches → clamp to WAIT (0)
- **Critical**: initial belief must be set to the specific test bin before solving,
  not the default safe/nominal init_b from the cache.

New modes:
```bash
python spacecraft_simulator.py --policy rssda --init-miss-bin 0   # RS-SDA* from bin 0
python spacecraft_simulator.py --compare --rollouts 100            # 3-variant table
```

`sdec_spacecraft.py` gains a `solve()` helper function that builds model, runs RS-SDA*,
and returns `(value, sdec, full_result, T, O, R, init_b, contact_stages)`.

### Scenario 1: 100-Rollout Comparison Results

All three variants for Scenario 1 (all-stages joint contact), bins 0 and 1:

```
Scenario               Init bin   Coll%  Mean miss(km)  Mean dv(m/s)  Mean syncs
Centralized            0            0.0        127.751        0.5000         6.0
Decentralized          0            0.0        128.006        0.5000         0.0
SDec (contacts)        0            0.0        127.751        0.5000         6.0
Centralized            1            0.0          2.953        0.0000         6.0
Decentralized          1            0.0          2.947        0.0000         0.0
SDec (contacts)        1            0.0          2.953        0.0000         6.0
```

**Interpretation:**
- Bin 0 → all variants maneuver to bin 4 (~128 km) with exactly 1 burn (0.5 m/s).
  Policy value = -10 (one maneuver cost). The policy knows to act at the first stage.
- Bin 1 → all variants WAIT. Policy value = 0 (miss of 1-5 km is outside bin 0 at TCA,
  so no collision penalty fires). This reveals a **gap**: bin 1 (1-5 km) is "high risk"
  by human standards but the POMDP collision reward only fires at bin 0 (< 1 km).
  Consider adding R_collision for bin 1 at terminal or tightening the bin threshold.
- Centralized = SDec for Scenario 1 (expected: all stages sync → identical policies).
- Decentralized achieves same results: for these deterministic dynamics, independent
  agents reach the same decision without coordination.

### Scenario 3: Costly Comms Sweep — Script Done

`scenario3_sweep.py` implements the c_sync sweep:
- Loads cached T, O, R; applies `R[:, sync_stage_states] -= c_sync` post-build
- Re-solves with RS-SDA* for each c_sync value
- Runs 100 rollouts, records collision_rate, mean_miss, mean_dv, mean_syncs
- Generates plots to `notes/scenario3_sweep_bin*.png`

```bash
python scenario3_sweep.py --c-max 200 --c-step 20 --rollouts 100 --init-miss-bin 0
```

Script is ready but the sweep hasn't been run for the full 200 range yet.
Run it and report results in the next session.

### LaTeX / Slides Updates — DONE

- `formulation.tex` and `slides.tex`: replaced ASCII verbatim figure with
  `\includegraphics{conjunction_geometry.png}` — both compile cleanly (pdflatex).
- `graphicx` package added to both files.

---

## Session 4 Accomplishments (2026-05-05)

### Completed
- **Bin-1 terminal penalty** — added `R = -200` at terminal stage for bin 1 (1-5 km).
  Policy now correctly maneuvers from bin 1 instead of WAITing.
- **Observation noise** — reduced `OBS_SIGMA_KM` from 50 km → 5 km (physically realistic
  for operational CDM-quality tracking).
- **Scenario 3 sweep run** — ran `c_sync ∈ [0, 200]`, 100 rollouts. Result: sync count
  stuck at 6 regardless of cost. Root cause: sync is mandatory (all states are
  `sync_states`), not optional. Requires structural fix (see below).
- **Policy visualization** — `plot_policy.py` generates a (miss_bin × stage) action grid
  solved per cell. Saved to `notes/figures/policy_grid.png`.
- **Repo cleanup** — `.gitignore` added; `notes/` reorganized into `report/`, `slides/`,
  `figures/` subfolders; LaTeX build artifacts removed.
- **Stochastic T explored and reverted** — decided against it: bin widths are large enough
  that the bin center dominates; stochastic T only matters when dv effect ≈ bin width.

### Updated Results (Scenario 1, after bin-1 fix, σ=5 km)
```
Policy           Init bin  Coll%  Mean miss(km)  Mean dv(m/s)  Mean syncs
Centralized      0          0.0   127.751        0.5000        6.0
Decentralized    0          0.0   128.006        0.5000        0.0
SDec (contacts)  0          0.0   127.751        0.5000        6.0
Centralized      1          0.0   127.895        0.5000        6.0
Decentralized    1          0.0   128.090        0.5000        0.0
SDec (contacts)  1          0.0   127.895        0.5000        6.0
```

---

## Script Run Order

Run all scripts from repo root with `.venv/bin/python`:

```bash
# 1. (Re)build T, O, R matrices — ~2 min, required after any change to spacecraft_matrices.py
.venv/bin/python spacecraftCA/spacecraft_matrices.py --force

# 2. Verify matrices are valid
.venv/bin/python spacecraftCA/spacecraft_discretizer.py

# 3. Solve RS-SDA* and print optimal value
.venv/bin/python spacecraftCA/sdec_spacecraft.py

# 4. Run Scenario 1 comparison (3 variants × 2 init bins × 100 rollouts)
.venv/bin/python spacecraftCA/spacecraft_simulator.py --compare --rollouts 100

# 5. Run from a specific bin with a specific policy
.venv/bin/python spacecraftCA/spacecraft_simulator.py --policy rssda --init-miss-bin 0 --rollouts 50

# 6. Scenario 3: costly comms sweep
.venv/bin/python spacecraftCA/scenario3_sweep.py --c-max 200 --c-step 20 --rollouts 100 --init-miss-bin 0

# 7. Policy visualization (6×6 grid, one solve per cell — ~10 min)
.venv/bin/python spacecraftCA/plot_policy.py --out spacecraftCA/notes/figures/policy_grid.png

# 8. Conjunction geometry plot/GIF
.venv/bin/python spacecraftCA/plot_conjunction_geometry.py

# Recompile LaTeX (from the relevant subfolder)
cd spacecraftCA/notes/report && pdflatex formulation.tex
cd spacecraftCA/notes/slides && pdflatex slides.tex
```

---

## Immediate Next Steps

1. **Scenario 3 structural fix** — sync must be a *choice*, not automatic. Options:
   - Use `sync_actions`: define a "communicate" joint action that triggers centralization
   - Use `sync_observations`: certain observations trigger centralization
   - Requires rethinking how contact windows translate to RSSDA sync mechanism

2. **Scenario 2: asymmetric control** — SC2 uncooperative (never maneuvers).
   Encode by setting SC2's action to always WAIT in T:
   `T_asym[a, s, s'] = T_full[(a1*3 + 0), s, s']` where `a1 = a // 3`.
   Contact stages for SC2: none (or `[0,1,2]` only).

3. **Reward shaping** — replace flat thresholds with continuous function:
   `R_collision(d) = -1000 * exp(-d / d_safe)`. Makes bin-1 penalty principled.

4. **Trajectory deviation cost** — penalize total ΔV or semi-major axis change
   to model fuel cost more realistically.

5. **Finer bins** — 8 bins: split bin 0 into [0,0.1) and [0.1,1); split bin 1
   into [1,2) and [2,5). Requires rebuilding discretizer + matrices.

6. **Observation noise sweep** — compare variants at σ ∈ {1, 5, 20} km.

7. **Multi-dv sweep** — compare policy quality at dv = 0.1, 0.5, 1.0 m/s.

---

## What Did NOT Work (Important History)

**RTN position bins (dT, dR)** — three attempts, all failed:
- Original crossing conjunction: miss axis is dN (collapsed), collision reward never fires
- Added finer dT bins: still no differentiation
- Fixed reward to propagate to TCA: miss still in dN
- Switched to radial-miss geometry: better, but bin centers at km scale →
  smallest miss from any bin center was 569 km (3 orders of magnitude above 1 km threshold)
- Root cause: bin centers and collision threshold differ by 3 orders of magnitude

**RTN placement at stage epoch** — placing SC2 at `δr_RTN = [d_miss, 0, 0]` at the
*stage* epoch gives completely wrong TCA miss (thousands of km off). Fixed by
back-propagating SC2 from TCA instead (verified < 1 m round-trip error).

---

## RSSDA API Reference

```python
from RSSDA import SDecPOMDP, SDecPOMDPModel, RSSDAConfig, int_tuple

model = SDecPOMDPModel(
    nagents=2,
    nstates=N_STATES_TOTAL,      # 37 (36 + 1 sink)
    nactions=N_JOINT_ACTIONS,    # 9
    nobs=N_JOINT_OBS,            # 36
    transitions=T.flatten().tolist(),
    obs=O.flatten().tolist(),
    rewards=R.flatten().tolist(),
    init_beliefs=init_b.tolist(),
    nacts_factor=[3, 3],
    nobs_factor=[6, 6],
    sync_states=sync_states,     # list of state indices
    sync_actions=[],
    sync_observations=[]
)

config = RSSDAConfig(
    maxh=6,
    algorithm="approximate",
    TI1=True, TI2=True, TI3=True, TI4=True,
    iter_limit=2000, rec_limit=2, max_clusters=20, heuristic_type="HYBRID",
)
sdec = SDecPOMDP(model, config)
value, policy, *_ = sdec.multi_agent_astar(6)
```

---

## Brahe API Reference

```python
import brahe
brahe.initialize_eop()   # REQUIRED first

# Propagation
prop = NumericalOrbitPropagator(epoch, eci_state,
    NumericalPropagationConfig.default(), ForceModelConfig.two_body())
prop.propagate_to(t)
state = prop.current_state()[:6]  # [x,y,z,vx,vy,vz] m, m/s

# RTN (returns [dR, dT, dN, dVR, dVT, dVN] in meters/m/s)
rtn = brahe.state_eci_to_rtn(sc1_eci, sc2_eci)
eci = brahe.state_rtn_to_eci(sc1_eci, rtn_rel)

# Orbital elements → ECI
state = brahe.state_koe_to_eci(oe_deg, AngleFormat.DEGREES)

# Epoch arithmetic
e2 - e1       # float seconds
e + 3600.0    # advance by seconds
e - 3600.0    # go back by seconds (back-propagation)
```
