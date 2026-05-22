# Handoff — Spacecraft CA SDec-POMDP (Next Session)

Read these two files first before doing anything:
  spacecraftCA/notes/SESSION_NOTES.md  — full design decisions, history, results, run order
  spacecraftCA/notes/report/formulation.tex — complete POMDP formulation

The venv is set up: run everything with `.venv/bin/python spacecraftCA/<file>.py`

Always do tasks step by step and pause if you hit issues.

---

## WHERE WE ARE (end of session 2026-05-21)

Three-variant architecture is correct and working. Sync counting is fixed. All three
variants use 9 joint actions. Session 7 changes are uncommitted (check `git status`).

Key architectural facts:
- sync_states is correct for GS contacts (state-based, not action-based)
- O matrix uses diagonal GPS obs at sync contacts (shared belief after sync)
- Dec uses independent TLE outer product obs at contacts (no sharing)
- iter_limit=2000 is fine; sync counting uses at_contact not cen_dists_map
- Centralized = SDec right now because both get same contact_stages as sync_states

Session 7 results (5 rollouts, spread belief, σ=0.50):
  Centralized:  0% coll, 128km miss, 0.5 m/s dv, 1 sync  (waits for first contact)
  SDec:         0% coll, 128km miss, 0.5 m/s dv, 1 sync  (identical to centralized)
  Dec:          0% coll, 130km miss, 5.5 m/s dv, 0 syncs (11x more fuel)

Current model parameters:
  dv=0.5 m/s, EXEC_NOISE_SIGMA=0.50, GPS_SIGMA_KM=0.1, TLE_SIGMA_KM=3.0
  REWARD_COLLISION=-10000, REWARD_HIGH=-1000, REWARD_MOD=-100, REWARD_MANEUVER=-10
  16 stages (10 regular + 6 GS contacts), CONTACT_STAGES=[1,9,10,11,13,15]
  iter_limit=10000, spread initial belief over bins 0-2

---

## IMMEDIATE TASK 1: Subset-Contacts SDec Ablation (highest priority)

This is the actual paper experiment. Right now Centralized = SDec because both use
all 6 contacts as sync_states. The interesting question: what happens when SDec only
has access to a subset?

Setup in spacecraft_simulator.py, variant_specs:
```python
variant_specs = [
    ("Centralized",       "centralized", contact_stages),        # all 6 contacts
    ("SDec (3 contacts)", "sdec",        contact_stages[3:]),    # last 3 only: [11,13,15]
    ("SDec (1 contact)",  "sdec",        contact_stages[-1:]),   # last 1 only: [15]
    ("Decentralized",     "dec",         []),                    # none
]
```

Expected: SDec with fewer contacts should use more dv (burns earlier with less info)
and possibly miss more — showing the value of each additional contact window.
This demonstrates why SDec exists: "how much communication is actually needed?"

Note: all SDec variants load the same "sdec" matrix cache (same O matrix). Only the
sync_states set passed to build_model differs between them.

---

## IMMEDIATE TASK 2: Stage-Dependent Maneuver Cost

Still unimplemented. Add to spacecraft_matrices.py:

```python
REWARD_MANEUVER_LATE = -2.0   # cost at last stage (T-1h)

def stage_maneuver_cost(stage: int) -> float:
    frac = stage / max(N_STAGES - 1, 1)
    return REWARD_MANEUVER + frac * (REWARD_MANEUVER_LATE - REWARD_MANEUVER)
```

Replace flat `r += REWARD_MANEUVER` with `r += stage_maneuver_cost(k)` in build_matrices.
Rebuild all variants (--variant all --force), re-run comparison.
May interact with the subset-contacts ablation — do Task 1 first to establish baseline.

---

## IMMEDIATE TASK 3: 100-Rollout Full Comparison

Dec no longer OOMs (9 actions vs old 36). Run full comparison:
  .venv/bin/python spacecraftCA/spacecraft_simulator.py --compare --rollouts 100

---

## FUTURE TODOS

- **Sigma sweep**: EXEC_NOISE_SIGMA ∈ {0.15, 0.30, 0.50} and TLE_SIGMA_KM ∈ {1, 3, 5}
  to characterize when policies diverge.

- **Deviation tracking**: expand state to (miss_bin, SC1_dev_bin, SC2_dev_bin, stage)
  = 577 states. Longer-term path for richer policy differentiation. See Session 5 notes.

- **TODO(future code)**: when Mahdi adds joint state+action conditional sync to RS-SDA*,
  re-introduce per-agent sync_flag (N_ACT_AGENT_SDEC=6) and wire sync_actions into
  SDecPOMDPModel. See spacecraft_matrices.py TODO comment at N_ACT_AGENT definition.

---

## SYNC MECHANISM REFERENCE (read this before touching sync logic)

sync_states = proportional belief split, not mandatory all-or-nothing:
  belief_split_by_id(d_next):
    prob_cen = sum of belief mass on sync_states
    prob_dec = 1 - prob_cen
    → splits belief into c_id (centralized part) and d_id (decentralized part)
    → value = (1-prob_dec)*V_cen(c_id) + prob_dec*V_dec(d_id)

At contacts all stage-k states (6 miss bins) are sync_states → if belief lands on
a contact stage after an action, prob_cen≈1 → full centralization.

cen_dists_map[step] = list of belief IDs RSSDA planned as centralized at that step.
In rollout: is_cen = (current_belief_idx in cen_dists_map[step]).
sync_count increments when is_cen=True AND at_contact=True.

iter_limit: RS-SDA* stops after this many node expansions. 2000 is fine — policy
value converges well before that. Do NOT use cen_dists_map to count syncs; use
at_contact instead (see sync counting note above).

---

## ALWAYS

Update SESSION_NOTES.md with any new design decisions (append new session section).
Update formulation.tex and slides.tex after major changes.
Recompile PDFs: cd spacecraftCA/notes/report && pdflatex formulation.tex
               cd spacecraftCA/notes/slides && pdflatex slides.tex
