"""
baseline_b1.py — operator-heuristic FLOOR baselines (the non-POMDP comparison family).

WHY: the POMDP variants (Centralized / SDec / Dec) are only interesting if they BEAT a
sensible operator heuristic. This module is that floor — a FAMILY of independent, per-craft
operator strategies (NO joint planning). The paper story: "the POMDP recovers X km / Y%
safety over operators who act sensibly but cannot coordinate their plans."

FAIRNESS (important): the POMDP is CHARGED a displacement cost for ending far from nominal
(convex displacement_cost ramp) — that is WHY it threads a tight 5-7 km landing instead of
overshooting. So a fair operator floor must ALSO (a) aim for a safe BAND, not max push, and
(b) RETURN toward nominal after the conjunction passes (counter-burn), paying dV/displacement
like the POMDP. Otherwise we'd compare a POMDP-that-pays-to-stay-close against an operator
that abandons station-keeping (unfair to the POMDP, flatters the floor). Both behaviors are
documented operator practice (LITERATURE_CA_THRESHOLDS.md: "clear the threshold and stop";
"design return maneuvers to recover the original position").

STRATEGY FAMILY (per-craft, assignable independently => mixed-strategy runs fall out free):
  THRESHOLD  (--strategy threshold, B1): burn when believed miss < own threshold, sized to
             land in the safe band (defer to the latest stage whose single-burn lever lands
             near the target), then return-to-nominal counter-burn once danger has passed.
             Sub-policies: conservative(5km) / aggressive(2km) / asymmetric(SC1 defers) /
             poc(P(collision)>1e-4, the GSOC/GISTDA go/no-go threshold).
  SELFISH    (--strategy selfish, B1a): each operator OPTIMIZES its own burn lever/timing to
             land ITSELF in-band given its current believed dt — an optimizer, not a fixed
             threshold. Crucially it has NO comms, so it must MODEL the other craft:
               blind     : assume the other does nothing (each clears alone => double push).
               polite    : assume the other shares the load 50/50 (each does half).
               obsaware  : re-read the SHARED dt at each contact (which ALREADY reflects the
                           other's burns) and RE-PLAN the remaining clearance. Implicit
                           coordination through the OBSERVATION channel, no explicit comms —
                           the realistic middle ground, and the closest foil to the SDec POMDP.
             KEY METRIC: how often do two selfish optimizers STILL collide / leave the band?
             (= the cost of not coordinating, even with smart agents.)
  FIXEDLEAD  (--strategy fixedlead, B1b): always burn at a fixed time-to-go (~CDM decision
             window), one burn sized to clear, then return. Realistic ops cadence.

Mixed strategies: --strategy-sc1 / --strategy-sc2 override the shared --strategy per craft.

TWO BELIEF ESTIMATES (report BOTH + the gap):
  pomdp : Bayes-filtered belief over the SAME T/O matrices (isolates POMDP PLANNING value).
  raw   : threshold on the most recent OBSERVED miss, no filtering (the naive operator).

REUSE: imports rollout_v2's harness wholesale — geometry helpers (signed_dt_and_miss_at_tca,
root_sc2_eci), summarize (collision %, 4-7 km band, fuel, deviation, terminal reward), CSV,
histograms — so every baseline produces rows directly comparable to the POMDP variants. Only
the ACTION SOURCE differs (a strategy object instead of the solved-policy decode).

Usage:
  .venv/bin/python -u benchmarks/spacecraftCA/baselines_spacecraftCA/baseline_b1.py \
      --strategy threshold --policy conservative --belief pomdp --mode mc --rollouts 200 \
      --init-miss 0.5 --init-spread 1.4 --backend numerical --csv b1_cons --hist b1_cons
  .venv/bin/python -u .../baseline_b1.py --strategy selfish --selfish-model obsaware \
      --mode point --trace --init-miss 0.5 --init-spread 1.4 --backend numerical
  # mixed: SC1 fixed-lead, SC2 selfish-blind
  .venv/bin/python -u .../baseline_b1.py --strategy-sc1 fixedlead --strategy-sc2 selfish \
      --selfish-model blind --mode mc --rollouts 200 ...
"""
import os, sys, argparse
import numpy as np

# This file lives in benchmarks/spacecraftCA/baselines_spacecraftCA/. Its sibling model
# modules (rollout_v2, spacecraft_*) live ONE level up in benchmarks/spacecraftCA/, so add
# that parent (_SCA) to the path in addition to the usual bench/root.
_HERE = os.path.dirname(os.path.abspath(__file__))      # .../spacecraftCA/baselines_spacecraftCA
_SCA = os.path.dirname(_HERE)                            # .../spacecraftCA  (siblings live here)
_BENCH = os.path.dirname(_SCA)
_ROOT = os.path.dirname(_BENCH)
for p in (_ROOT, _BENCH, _SCA, _HERE):
    sys.path.insert(0, p)

# --backend must be honored BEFORE the model modules import (stage grid + N_STAGES are
# computed at import time from the backend). Pre-scan argv, same as rollout_v2 / CV.
for _i, _a in enumerate(sys.argv):
    if _a == "--backend" and _i + 1 < len(sys.argv):
        os.environ["SPACECRAFT_PROPAGATOR"] = sys.argv[_i + 1].lower()
    elif _a.startswith("--backend="):
        os.environ["SPACECRAFT_PROPAGATOR"] = _a.split("=", 1)[1].lower()

from brahe import initialize_eop

import spacecraft_discretizer_v2 as D
import spacecraft_transition_v2 as TV
import spacecraft_matrices as M

# Reuse rollout_v2's VECTORIZED brahe engine + harness: B1 supplies a PolicySource and the
# engine does all rollout/propagation/CSV/hist/summary. (Geometry + the rollout loop live in
# rollout_v2; B1 only needs the reporting helpers + the safe-band constants here.)
import rollout_v2 as RV
from rollout_v2 import summarize, save_csv, SAFE_LO_KM, SAFE_HI_KM

# PoC operational go/no-go threshold (GSOC/GISTDA): maneuver when P(collision) > 1e-4.
POC_THRESHOLD = 1e-4
ASYMMETRIC_DEFER_KM = 0.0          # asymmetric passive craft: threshold 0 => never triggers
# Target landing the operators aim for: the MIDDLE of the safe band. A sensible operator
# clears the screening threshold with margin but does not over-mitigate. Used to SIZE/TIME a
# single burn (target-band) instead of firing the full early lever.
TARGET_MISS_KM = 0.5 * (SAFE_LO_KM + SAFE_HI_KM)   # 5.5 km (centre of [4,7])
# Fixed-lead operator's decision time-to-go (h): the documented CDM/decision window (~T-7h).
FIXEDLEAD_T2GO_H = 7.0
N_AGENT = 2


# ---------------------------------------------------------------------------
# Lever table (km of dt produced at TCA by ONE 0.5 m/s burn applied at each stage).
# The operators use this as KNOWLEDGE OF THEIR OWN DYNAMICS — choosing when/how to burn to
# land in the band is a lookup against their own lever schedule, not joint planning. Built
# once per conjunction at model build (the same levers the matrices auto-generate the grid from).
# ---------------------------------------------------------------------------
_LEVER_KM = None        # np.ndarray[N_STAGES], signed dt at TCA per +dV unit burn at stage k


def _build_lever_table(rate_at):
    global _LEVER_KM
    _LEVER_KM = np.array([float(rate_at[k]) * TV.stage_t2go_h(k) for k in range(D.N_STAGES)])
    return _LEVER_KM


def lever_at(stage):
    return float(_LEVER_KM[stage])


# ---------------------------------------------------------------------------
# Per-agent estimator: each craft independently estimates the (shared) along-track dt.
# ---------------------------------------------------------------------------
# B1 has NO communication, so an agent refines its dt estimate only from ITS OWN local
# observation (a non-null dt symbol arrives only at a sync/contact). Both belief modes consume
# the SAME joint observation rows the POMDP uses; the difference is purely how each agent
# CONVERTS the obs into a believed dt/miss and what its strategy does with it.

def _decode_local_dt_obs_km(obs, obs_agent_size, agent):
    """This agent's observed dt-at-TCA (km, from the observed dt_bin centre), or None if
    null/off-sync. The observation reflects the COMBINED effect of both craft's burns."""
    o_local = obs % obs_agent_size if agent == 0 else obs // obs_agent_size
    dt_obs = o_local % TV.N_DT_OBS
    if dt_obs == TV.DT_NULL_OBS:
        return None
    return float(D.dt_bin_center_km(int(dt_obs)))


def _dt_tables(perp_km):
    centers = np.array([D.dt_bin_center_km(i) for i in range(D.N_DT)])
    misses = np.array([D.miss_km_from_dt(c, perp_km) for c in centers])
    return centers, misses


class AgentEstimator:
    """Per-agent estimate of the (shared) along-track dt AT TCA. Honest operator model:

      believed_dt = observed_baseline + own_committed_shift

    where `observed_baseline` is the LAST observed dt-at-TCA (which already reflects EVERYONE'S
    burns up to that observation), and `own_committed_shift` is the dt-at-TCA this craft has
    added with ITS OWN burns SINCE that observation — projected with the BRAHE lever table (the
    operator knows its own commanded maneuvers and its own dynamics; no matrix/transition model
    needed for self-knowledge). At a contact the observation RESETS the baseline to the truth
    (combined effect of both craft) and the own-committed-shift zeroes — so the gap an obsaware
    operator sees between its expected dt and the observed dt is exactly the OTHER craft's
    contribution (implicit, comms-free coordination).

    belief mode:
      pomdp : observed_baseline is the exact observed dt-at-TCA (perfect dt readout at a contact).
      raw   : same baseline, but believed_miss reported from the binned observation centre (the
              cruder 'naive operator' readout). The own-burn projection is identical; the gap
              pomdp-vs-raw is purely the readout granularity of the observation.
    """

    def __init__(self, mode, init_dt_km, perp_km):
        self.mode = mode
        self.perp = perp_km
        self.centers, self.misses = _dt_tables(perp_km)
        self.observed_baseline_km = float(init_dt_km)   # best guess before any contact
        self.own_committed_shift_km = 0.0               # dt-at-TCA from own burns since last obs
        self.got_obs_this_stage = False

    def believed_dt_km(self):
        return self.observed_baseline_km + self.own_committed_shift_km

    def believed_miss_km(self):
        dt = self.believed_dt_km()
        if self.mode == "raw":
            dt = D.dt_bin_center_km(D.dt_to_bin(dt))     # crude binned readout
        return D.miss_km_from_dt(dt, self.perp)

    def collision_mass(self):
        """P(collision) proxy for the PoC rule. Operators have a point estimate, not a full
        distribution, so this is a soft indicator: 1 if believed miss is inside the collision
        floor, else a small exponential tail (so PoC>1e-4 fires a touch before exact contact)."""
        miss = self.believed_miss_km()
        if miss <= D.COLLISION_THRESHOLD_KM:
            return 1.0
        # crude PoC ~ exp(-(miss-floor)^2 / 2σ^2) with σ ~ 1 km screening; > 1e-4 out to ~3.9 km
        return float(np.exp(-0.5 * (miss - D.COLLISION_THRESHOLD_KM) ** 2))

    def commit_burn(self, action, stage):
        """Fold this craft's OWN commanded burn into its committed dt-at-TCA shift via the BRAHE
        lever (action 1=+dV -> dt by +lever; 2=-dV -> -lever). lever_at(stage) is signed (the
        dt a +dV burn at this stage produces by TCA)."""
        if action == 0:
            return
        sign = +1.0 if action == 1 else -1.0
        self.own_committed_shift_km += sign * lever_at(stage)

    def observe(self, observed_dt_km):
        """Reset the baseline to a fresh observation (the combined truth) and zero the own-shift
        accumulator (the observation already contains it). None => off-contact, no info."""
        self.got_obs_this_stage = (observed_dt_km is not None)
        if observed_dt_km is None:
            return
        self.observed_baseline_km = float(observed_dt_km)
        self.own_committed_shift_km = 0.0


# ---------------------------------------------------------------------------
# Direction helper (shared by all strategies)
# ---------------------------------------------------------------------------
# SIGN convention (rollout_v2 header): a +dV burn (action 1) drives dt NEGATIVE. Both craft's
# burns act on the SAME relative dt (vdev1+vdev2 BOTH add to drift), so two craft burning in
# OPPOSITE directions CANCEL. Operators therefore share a deterministic default avoidance
# direction (they have no comms, but use the same rule) so independent burns ADD, not cancel.

def clear_direction(believed_dt_km):
    """Direction (1=+dV, 2=-dV) that pushes |dt| AWAY from the collision line. On the line
    (dt~0) both craft default to -dV (push dt positive) so independent burns add."""
    if believed_dt_km < -0.05:
        return 1            # already negative -> push more negative -> +dV
    return 2                # positive or on-line -> push positive -> -dV


def _stage_for_target(believed_dt_km, target_miss_km=TARGET_MISS_KM, early_stages=0):
    """Latest stage whose ONE burn lever moves |dt| by ~ (target_miss - |believed dt|), i.e.
    lands the clearance near `target_miss` with a single late (small-lever) burn. Returns
    (stage, direction). `early_stages` shifts the fire stage EARLIER by that many grid steps
    (a bigger lever / time buffer — the cautious-early overcompensation). If even the earliest
    lever is too small to reach the target, fire at stage 0 (max lever)."""
    need = max(target_miss_km - abs(believed_dt_km), 0.0)
    direction = clear_direction(believed_dt_km)
    best = 0
    for k in range(D.N_STAGES - 1):          # exclude terminal (no transition); |lever| shrinks with k
        if abs(lever_at(k)) >= need:
            best = k
    best = max(0, best - max(0, early_stages))
    return best, direction


# ---------------------------------------------------------------------------
# Operator strategies (one per craft; decide(stage, est) -> action 0/1/2)
# ---------------------------------------------------------------------------
class OperatorStrategy:
    """Base: a per-craft reactive operator. Holds whether it has burned (avoidance) and
    whether it has returned (counter-burn), so it issues at most one of each. Subclasses
    implement want_avoidance_now() (when/whether to fire the clearing burn)."""

    def __init__(self, agent):
        self.agent = agent
        self.avoided = False           # has issued the avoidance burn
        self.returned = False          # has issued the return-to-nominal counter-burn
        self.avoid_dir = 0             # direction used for avoidance (return is the opposite)
        self.danger_seen = False       # believed miss was once below the act threshold

    # --- avoidance: subclass decides timing/sizing; default never (override) ---
    def want_avoidance_now(self, stage, est):
        return False

    # --- return-to-nominal: shared reactive rule ---
    def want_return_now(self, stage, est):
        """After avoiding, once danger has clearly passed (believed miss comfortably in/above
        the band) counter-burn ONCE to null vdev and drift back toward nominal — paying dV /
        displacement like the POMDP. Must leave time for the counter to take effect, so only
        before the last couple of decision stages."""
        if not self.avoided or self.returned:
            return False
        if stage >= D.N_STAGES - 2:
            return False
        return est.believed_miss_km() >= SAFE_LO_KM

    def decide(self, stage, est):
        if not self.avoided and self.want_avoidance_now(stage, est):
            self.avoided = True
            self.avoid_dir = clear_direction(est.believed_dt_km())
            return self.avoid_dir
        if self.want_return_now(stage, est):
            self.returned = True
            return 1 if self.avoid_dir == 2 else 2     # opposite of the avoidance burn
        return 0


class ThresholdStrategy(OperatorStrategy):
    """B1: burn when believed miss < threshold (or PoC>1e-4), but defer the burn to a LATE
    small-lever stage so it lands in the band (target-band sizing) rather than max-pushing."""

    def __init__(self, agent, threshold_km, poc=False):
        super().__init__(agent)
        self.threshold_km = threshold_km
        self.poc = poc

    def _triggered(self, est):
        if self.threshold_km is not None and self.threshold_km <= 0.0:
            return False                                  # defer (asymmetric passive craft)
        if self.poc:
            return est.collision_mass() > POC_THRESHOLD
        return est.believed_miss_km() < self.threshold_km

    def want_avoidance_now(self, stage, est):
        if not self._triggered(est):
            return False
        self.danger_seen = True
        # Target-band: only FIRE at/after the stage whose single-burn lever lands near target.
        # Before that, WAIT (lever too big -> would overshoot).
        fire_stage, _ = _stage_for_target(est.believed_dt_km())
        return stage >= fire_stage


class FixedLeadStrategy(OperatorStrategy):
    """B1b: burn ONCE at a fixed time-to-go (~CDM decision window), sized by the target-band
    rule, regardless of threshold — then return. Realistic ops cadence (act on schedule)."""

    def __init__(self, agent, t2go_h=FIXEDLEAD_T2GO_H):
        super().__init__(agent)
        self.t2go_h = t2go_h
        # the decision stage = the stage closest to (but not after) the lead time-to-go
        self.fire_stage = self._stage_at_t2go(t2go_h)

    @staticmethod
    def _stage_at_t2go(t2go_h):
        best = 0
        for k in range(D.N_STAGES - 1):
            if TV.stage_t2go_h(k) >= t2go_h:
                best = k
        return best

    def want_avoidance_now(self, stage, est):
        # Only bother if the conjunction is actually dangerous (believed miss inside the band
        # floor); a fixed-lead operator still doesn't burn for a clearly-safe pass.
        if est.believed_miss_km() >= SAFE_LO_KM:
            return False
        return stage >= self.fire_stage


# Cautious overcompensation knobs. cautious-margin aims PAST the band (extra clearance, in case
# the other craft doesn't move); cautious-early fires this many grid steps EARLIER (bigger lever
# / time buffer). Two cautious operators both overcompensate -> double overshoot (the sharp
# uncoordinated-cost floor). Tunable; chosen to clearly separate from blind without being absurd.
CAUTIOUS_MARGIN_TARGET_KM = SAFE_HI_KM + 2.0    # aim ~2 km past the band's upper edge
CAUTIOUS_EARLY_STAGES = 3                        # fire 3 grid steps earlier (bigger lever)


class SelfishOptimizerStrategy(OperatorStrategy):
    """B1a: each contact, compute the burn lever/timing that lands THIS craft in the band given
    its current believed dt and a MODEL of the other craft (NO comms). Other-craft models:
      blind          : other does nothing -> clear fully alone, aim for band CENTRE.
      cautious-margin: other MIGHT not move -> overcompensate on DISTANCE (aim past the band).
      cautious-early : other MIGHT not move -> overcompensate on TIMING (fire earlier, bigger
                       lever). Both cautious flavours: two of them -> overshoot (uncoordinated).
      obsaware       : re-read the SHARED dt each contact (already reflects the other's burns)
                       and re-plan the REMAINING clearance from the observed dt; back off if a
                       fresh obs shows the pass already cleared. Implicit coordination via the
                       observation channel, no explicit comms — the closest foil to SDec POMDP.
    """

    def __init__(self, agent, other_model="obsaware"):
        super().__init__(agent)
        self.other_model = other_model

    def _plan(self, est):
        """(fire_stage, target_km) for this model given the current believed dt."""
        dt = est.believed_dt_km()
        if self.other_model == "cautious-margin":
            return _stage_for_target(dt, target_miss_km=CAUTIOUS_MARGIN_TARGET_KM)[0], \
                   CAUTIOUS_MARGIN_TARGET_KM
        if self.other_model == "cautious-early":
            return _stage_for_target(dt, early_stages=CAUTIOUS_EARLY_STAGES)[0], TARGET_MISS_KM
        # blind / obsaware: aim for band centre, no early shift
        return _stage_for_target(dt)[0], TARGET_MISS_KM

    def want_avoidance_now(self, stage, est):
        if est.believed_miss_km() >= SAFE_LO_KM:
            return False                                 # already clear of the band floor
        if self.other_model == "obsaware":
            # Re-plan only on a FRESH observation (can't tell what the other did otherwise);
            # but if we're running out of small-lever stages, fire anyway.
            fire_stage, _ = self._plan(est)
            # hold between contacts (can't tell what the other did without a fresh obs),
            # unless we're already at/past the small-lever fire window (then commit).
            if not est.got_obs_this_stage and stage > 0 and stage < fire_stage:
                return False
            return stage >= fire_stage
        fire_stage, _ = self._plan(est)
        return stage >= fire_stage


def make_strategy(name, agent, policy=None, selfish_model="obsaware"):
    """Factory: build a per-craft strategy. `policy` only matters for threshold."""
    if name == "threshold":
        thr_map = {"conservative": 5.0, "aggressive": 2.0,
                   "asymmetric": (ASYMMETRIC_DEFER_KM if agent == 0 else 5.0),
                   "poc": None}
        if policy == "poc":
            return ThresholdStrategy(agent, threshold_km=None, poc=True)
        return ThresholdStrategy(agent, threshold_km=thr_map[policy])
    if name == "fixedlead":
        return FixedLeadStrategy(agent)
    if name == "selfish":
        return SelfishOptimizerStrategy(agent, other_model=selfish_model)
    raise ValueError(f"unknown strategy {name}")


# ---------------------------------------------------------------------------
# Build T/O/R + per-agent init dt belief (NO solve — operators have no policy to solve).
# ---------------------------------------------------------------------------

def build_model(variant, perp_km):
    rate_at, perp_meas, _ = TV.compute_gain_table_and_perp(perp_km, 0.0)
    T, O = TV.build_T_O(rate_at, variant)
    R = TV.build_R(perp_meas)
    _build_lever_table(rate_at)            # operators' knowledge of their own dynamics
    return T, O, R, perp_meas


def init_dt_km_from_b(init_b):
    """The operators' prior dt-at-TCA = the |dt| EXPECTATION over the init belief's stage-0
    support (a CDM-style nominal estimate). Magnitude only (the spread is symmetric in sign;
    the operator's avoidance direction is resolved by clear_direction at burn time)."""
    num = 0.0
    den = 0.0
    for s in np.flatnonzero(init_b[:D.N_STATES]):
        dt_bin, _, _, stage = D.index_to_state(int(s))
        if stage == 0:
            num += init_b[s] * abs(D.dt_bin_center_km(dt_bin))
            den += init_b[s]
    return (num / den) if den > 0 else 0.0


# ---------------------------------------------------------------------------
# B1 policy source — plugs the operator heuristic into rollout_v2's VECTORIZED brahe engine.
# ---------------------------------------------------------------------------
# This is the whole point of the refactor: B1 does NOT re-implement the rollout loop. It builds
# a rollout_v2.PolicySource that holds per-trajectory operator estimators + strategies, and feeds
# it to RV.run_mc / RV.run_point. The engine handles all brahe propagation (vectorized, lockstep
# in MC), burn execution, the matrix transition, and the CSV/hist/summary reporting; B1 only
# supplies (a) the action each node and (b) how each operator folds its own burn / the realized
# observation into its estimate. The POMDP belief/oh passed by the engine are IGNORED (operators
# have no POMDP belief); B1 carries its own estimator state keyed by trajectory index.

class B1PolicySource(RV.PolicySource):
    """Per-craft operator heuristic as a rollout_v2 PolicySource (see module docstring)."""

    def __init__(self, init_dt_km, perp_km, obs_agent_size, strat_names, policy,
                 selfish_model, belief_mode):
        self.init_dt_km = init_dt_km
        self.perp_km = perp_km
        self.obs_agent_size = obs_agent_size
        self.strat_names = strat_names
        self.policy = policy
        self.selfish_model = selfish_model
        self.belief_mode = belief_mode
        self._est = {}        # traj i -> [AgentEstimator x2]
        self._strat = {}      # traj i -> [OperatorStrategy x2]
        self._last_miss = {}  # traj i -> (b1_miss, b2_miss) captured at action() for the trace

    def reset_traj(self, i, init_state):
        # operators' prior dt = the conjunction's nominal dt-at-TCA (CDM-style); contacts refine.
        self._est[i] = [AgentEstimator(self.belief_mode, self.init_dt_km, self.perp_km)
                        for _ in range(N_AGENT)]
        self._strat[i] = [make_strategy(self.strat_names[a], a, policy=self.policy,
                                        selfish_model=self.selfish_model)
                          for a in range(N_AGENT)]
        self._last_miss[i] = (self._est[i][0].believed_miss_km(),
                              self._est[i][1].believed_miss_km())

    def action(self, i, step, belief_i, oh_i):
        est, strat = self._est[i], self._strat[i]
        # operators see their PRE-burn believed miss (record for the trace) then decide + commit
        self._last_miss[i] = (est[0].believed_miss_km(), est[1].believed_miss_km())
        a1 = strat[0].decide(step, est[0])
        a2 = strat[1].decide(step, est[1])
        est[0].commit_burn(a1, step)      # each operator folds its OWN burn into its own estimate
        est[1].commit_burn(a2, step)
        joint_act = a1 + TV.N_ACT_AGENT * a2     # SC1 low, SC2 high (CLAUDE.md invariant)
        # is_cen=False, c_ptr=-1: operators never "sync" (no joint coordination); contacts only
        # deliver observations, handled in update(). sync_count stays 0 for B1 (correct).
        return joint_act, a1, a2, False, -1

    def update(self, i, step, joint_act, is_cen, c_ptr, belief_i, oh_i, next_state, obs):
        if obs is not None:
            for a in range(N_AGENT):
                self._est[i][a].observe(
                    _decode_local_dt_obs_km(obs, self.obs_agent_size, a))
        return belief_i, oh_i          # POMDP belief/oh unused by B1; pass through untouched

    def trace_header(self):
        return f" {'b1_miss':>8} {'b2_miss':>8}"

    def trace_cols(self, i):
        m1, m2 = self._last_miss[i]
        return f" {m1:>8.2f} {m2:>8.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["threshold", "selfish", "fixedlead"],
                    default="threshold", help="shared per-craft strategy (unless overridden).")
    ap.add_argument("--strategy-sc1", choices=["threshold", "selfish", "fixedlead"],
                    default=None, help="override strategy for SC1 (mixed-strategy runs).")
    ap.add_argument("--strategy-sc2", choices=["threshold", "selfish", "fixedlead"],
                    default=None, help="override strategy for SC2 (mixed-strategy runs).")
    ap.add_argument("--policy", choices=["conservative", "aggressive", "asymmetric", "poc"],
                    default="conservative", help="threshold strategy's sub-policy.")
    ap.add_argument("--selfish-model",
                    choices=["blind", "cautious-margin", "cautious-early", "obsaware"],
                    default="obsaware", help="selfish strategy's model of the OTHER craft.")
    ap.add_argument("--belief", choices=["pomdp", "raw"], default="pomdp")
    ap.add_argument("--variant", choices=["centralized", "sdec", "dec"], default="sdec",
                    help="controls only the OBS model (which stages feed shared dt); no solver.")
    ap.add_argument("--mode", choices=["point", "mc"], default="point")
    ap.add_argument("--init-miss", type=float, default=0.5)
    ap.add_argument("--init-spread", type=float, default=1.4)
    ap.add_argument("--perp", type=float, default=0.0)
    ap.add_argument("--contact-stages", type=str, default=None)
    ap.add_argument("--rollouts", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default=None, choices=["numerical", "keplerian", "drag"])
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--hist", type=str, default=None)
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--from-csv", type=str, default=None)
    args = ap.parse_args()

    s1 = args.strategy_sc1 or args.strategy
    s2 = args.strategy_sc2 or args.strategy
    strat_names = [s1, s2]
    strat_tag = s1 if s1 == s2 else f"{s1}X{s2}"
    if "selfish" in strat_names:
        strat_tag += f"-{args.selfish_model}"
    if "threshold" in strat_names:
        strat_tag += f"-{args.policy}"
    label = f"B1[{strat_tag}]-{args.belief}"

    def _tagged_path(val, subdir, prefix, ext):
        if os.sep in val or val.endswith(ext):
            return val
        d = os.path.join(_SCA, "notes", subdir)
        os.makedirs(d, exist_ok=True)
        return os.path.join(
            d, f"{prefix}_{val}_{strat_tag}_{args.belief}_{args.variant}_{args.mode}{ext}")

    if args.from_csv is not None:
        results = RV.load_csv(args.from_csv)
        fig_path = _tagged_path(args.hist, "figures", "b1_hist", ".png") \
            if args.hist is not None else None
        summarize(results, label, args.mode, fig_path=fig_path)
        return

    initialize_eop()
    if args.contact_stages is not None:
        stages = [int(s) for s in args.contact_stages.split(",") if s.strip() != ""]
        M.set_contact_stages(stages)
    RV.CV.PERP_KM = args.perp
    print(f"  backend={M._SG.PROPAGATOR_BACKEND}  N_STAGES={D.N_STAGES}  "
          f"contacts={M.get_contact_stages()}")
    print(f"  strategy: SC1={s1} SC2={s2}  policy={args.policy} selfish_model={args.selfish_model}"
          f"  belief={args.belief}")

    init_b, flagged, eff_miss = TV.build_init_b_danger(
        args.init_miss, args.init_spread, args.perp, sign_mode="both")
    print(f"  init: miss={args.init_miss} spread={args.init_spread} perp={args.perp} "
          f"-> eff_miss={eff_miss:.3f} km  flagged={flagged}")

    T, O, R, perp = build_model(args.variant, args.perp)
    init_dt_km = init_dt_km_from_b(init_b)
    obs_agent_size = TV.N_OBS_AGENT
    print(f"  built T/O/R [{args.variant}]  perp_meas={perp:.3f}  target_miss={TARGET_MISS_KM} km"
          f"  init_dt={init_dt_km:.3f} km")

    # The heuristic is piped into rollout_v2's VECTORIZED brahe engine via a PolicySource;
    # B1 does not run its own rollout loop. sdec/full are unused by the source (operators have
    # no POMDP belief) -> pass None. A fresh source per call resets per-trajectory estimators.
    def _source():
        return B1PolicySource(init_dt_km, perp, obs_agent_size, strat_names,
                              args.policy, args.selfish_model, args.belief)

    if args.mode == "point":
        results = RV.run_point(T, O, R, perp, None, None, init_b, obs_agent_size,
                               trace=args.trace, policy_source=_source())
    else:
        if args.trace:
            print("  [--trace in mc mode] one point trace first (per the per-stage rule):")
            RV.run_point(T, O, R, perp, None, None, init_b, obs_agent_size,
                         trace=True, policy_source=_source())
        results = RV.run_mc(T, O, R, perp, None, None, init_b, obs_agent_size,
                            args.rollouts, seed=args.seed, policy_source=_source())

    if args.csv is not None:
        save_csv(results, _tagged_path(args.csv, "results", "b1", ".csv"))
    fig_path = _tagged_path(args.hist, "figures", "b1_hist", ".png") \
        if args.hist is not None else None
    summarize(results, label, args.mode, fig_path=fig_path)


if __name__ == "__main__":
    main()
