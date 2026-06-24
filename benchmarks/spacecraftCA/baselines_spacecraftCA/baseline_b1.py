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
  THRESHOLD  (--strategy threshold, B1): burn when believed miss < own threshold, DEFER to the
             latest small-lever stage so one burn lands in the band (target-band), then a
             return counter-burn once danger has passed. Sub-policies (--policy): conservative
             (5km) / aggressive(2km) / asymmetric(SC1 defers) / poc(P(coll)>1e-4 GSOC/GISTDA).
  FIRERETURN (--strategy firereturn): the OPPOSITE timing — fire IMMEDIATELY on threshold cross
             (big early lever, over-clears far), then trim back toward the band centre. Out-and-
             back. Tests acting-early-then-trimming vs acting-late-once.
  SELFISH    (--strategy selfish, B1a): each operator OPTIMIZES its own burn to land ITSELF
             in-band, with a MODEL of the other craft (--selfish-model): blind / cautious-margin
             / cautious-early / obsaware. KEY METRIC: how often do two selfish optimizers STILL
             collide / leave the band (the cost of not coordinating).
  FIXEDLEAD  (--strategy fixedlead, B1b): burn at a fixed time-to-go (~CDM window), then return.

Mixed strategies: --strategy-sc1 / --strategy-sc2 override the shared --strategy per craft.

BELIEF: a CONTINUOUS Gaussian Kalman filter over signed dt-at-TCA (belief_filter.py) — NO binning
(baselines don't solve the POMDP). The operator thresholds the belief MEAN. The OTHER-craft
observation fidelity is set by --other-obs {perfect, tle, frozen} (--tle-sigma sets the TLE noise);
own state is always known via own burns. See belief_filter.py.

REUSE: pipes a PolicySource into rollout_v2's VECTORIZED brahe engine (summarize / CSV / hist /
band stats), so every baseline produces rows directly comparable to the POMDP variants. Only the
ACTION SOURCE differs (a strategy object instead of the solved-policy decode).

Usage:
  .venv/bin/python -u benchmarks/spacecraftCA/baselines_spacecraftCA/baseline_b1.py \
      --strategy threshold --policy conservative --other-obs tle --mode mc --rollouts 200 \
      --init-miss 0.5 --init-spread 1.4 --backend numerical --csv b1_cons --hist b1_cons
  .venv/bin/python -u .../baseline_b1.py --strategy firereturn --other-obs tle \
      --mode point --trace --init-miss 0.5 --init-spread 1.4 --backend numerical
  # mixed: SC1 fixed-lead, SC2 selfish-blind
  .venv/bin/python -u .../baseline_b1.py --strategy-sc1 fixedlead --strategy-sc2 selfish \
      --selfish-model blind --other-obs tle --mode mc --rollouts 200 ...
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

# The scenario MUST be applied BEFORE the model modules import (stage grid + N_STAGES derive
# from it). Same config-first bootstrap as rollout_v2 / compare_variants_v2 — NO env vars.
from scenario_config import _cli_bootstrap_scenario
_SCENARIO = _cli_bootstrap_scenario(sys.argv)

from brahe import initialize_eop

import spacecraft_discretizer_v2 as D
import spacecraft_transition_v2 as TV
import spacecraft_matrices as M
import belief_filter as BF          # the SHARED discrete Bayes filter (B1 + diagnostic + POMDP-style)

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
    """The COMBINED dt-at-TCA (km, from the observed dt_bin centre) carried by this agent's local
    observation, or None if null/off-sync. NOTE: this is the JOINT outcome (both craft's burns).
    The estimator must NOT treat it as a free self-measurement of the joint state at a GS pass —
    see AgentEstimator. It is used only to (a) refresh OWN drift at the craft's own GS contact and
    (b) feed the slow TLE channel for the OTHER craft."""
    o_local = obs % obs_agent_size if agent == 0 else obs // obs_agent_size
    dt_obs = o_local % TV.N_DT_OBS
    if dt_obs == TV.DT_NULL_OBS:
        return None
    return float(D.dt_bin_center_km(int(dt_obs)))


# TLE cadence + refresh-stage alignment now live in belief_filter (BF.TLE_CADENCE_H,
# BF.tle_refresh_stages) so B1 and the diagnostic share the same TLE clock.
tle_refresh_stages = BF.tle_refresh_stages
TLE_CADENCE_H = BF.TLE_CADENCE_H


class AgentEstimator:
    """Per-agent belief over the (shared) along-track dt AT TCA — a thin wrapper over the shared
    CONTINUOUS Gaussian filter (belief_filter.BeliefFilter). The operator thresholds the belief MEAN.
    NO binning anywhere (the baselines don't solve the POMDP, so they don't use the dt grid). See
    belief_filter.py for the asymmetric observation model:

      own GS contact -> NO collapse (own state already known; tells you nothing about the other craft).
      TLE epoch (~8h) -> PARTIAL collapse (noisy Kalman update on the OTHER craft, sigma grows w/ age).
      perfect sync    -> full collapse (idealized; --other-obs perfect).
      frozen          -> never observe the other; belief only widens (predict) + own-burn shifts.
    """

    def __init__(self, init_dt_mean_km, init_dt_std_km, perp_km, other_obs="tle",
                 tle_sigma_base=BF.TLE_SIGMA_BASE_KM, tle_stages=None):
        self.perp = perp_km
        self.filter = BF.BeliefFilter(init_dt_mean_km, init_dt_std_km, perp_km, other_obs=other_obs,
                                      tle_sigma_base=tle_sigma_base, tle_stages=tle_stages)

    def believed_dt_km(self):
        return self.filter.mean_dt_km()

    def believed_miss_km(self):
        return self.filter.mean_miss_km()

    def collision_mass(self):
        """P(miss < collision floor) under the Gaussian belief (used by the PoC-style rule)."""
        return self.filter.mass_below_miss(D.COLLISION_THRESHOLD_KM)

    @property
    def got_tle_this_stage(self):
        """Whether the LAST observe() sharpened the belief (a TLE/perfect collapse) — the only new
        info about the other craft. obsaware keys off this to decide when to re-plan."""
        return self.filter.collapsed_this_stage

    def commit_burn(self, action, stage):
        """OWN burn (known): shift the belief mean by the signed brahe lever at this stage."""
        self.filter.commit_burn(action, lever_at(stage))

    def observe(self, combined_dt_km, stage):
        """Predict (coast widening) then Kalman-condition on this stage's obs. combined_dt_km = the
        COMBINED dt from a contact obs, or None off-contact."""
        self.filter.predict(stage)
        self.filter.observe(stage, combined_dt_km)


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


class FireReturnStrategy(OperatorStrategy):
    """B1 'fire-now then trim': the OPPOSITE timing philosophy to ThresholdStrategy. Fires the
    avoidance burn IMMEDIATELY when believed miss crosses the threshold (big early lever -> over-
    clears far out), then a RETURN burn that trims back only as far as needed to land near the band
    centre (clear the threshold and stop, don't over-displace). Out-and-back. Contrast with target-
    band (one deferred burn). Tests whether acting EARLY-then-trimming beats acting LATE-once."""

    def __init__(self, agent, threshold_km=5.0):
        super().__init__(agent)
        self.threshold_km = threshold_km

    def want_avoidance_now(self, stage, est):
        # fire the moment the belief says we're inside the act line (no deferral)
        return est.believed_miss_km() < self.threshold_km

    def want_return_now(self, stage, est):
        """Trim back toward the band: fire the counter only if it brings |dt| CLOSER to the band
        centre while staying clear of the 4 km floor. (Overrides the base full-counter return.)"""
        if not self.avoided or self.returned or stage >= D.N_STAGES - 1:
            return False
        dt = est.believed_dt_km()
        ret_dir = 1 if dt > 0 else 2                    # opposite of how we pushed
        rsign = +1.0 if ret_dir == 1 else -1.0
        after = dt + rsign * lever_at(stage)
        return abs(after) >= SAFE_LO_KM and \
            abs(abs(after) - TARGET_MISS_KM) < abs(abs(dt) - TARGET_MISS_KM)


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
      obsaware       : re-plan the REMAINING clearance whenever a TLE refresh updates the other
                       craft's estimate (the only honest channel for the other craft, ~8h cadence);
                       back off if the updated estimate shows the pass already cleared. Implicit,
                       SLOW coordination via the TLE channel only — NOT a free per-pass readout.
                       (Under --other-obs frozen there is no TLE channel, so obsaware degenerates
                       to a one-shot blind-style plan off the screening nominal.)
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
            # The ONLY new info about the other craft is a TLE refresh (got_tle_this_stage); a GS
            # pass alone tells the operator nothing new about the other craft under the honest obs
            # model. So hold until a TLE epoch updates the other estimate, unless we've reached the
            # small-lever fire window (then commit with whatever we know — can't wait forever).
            fire_stage, _ = self._plan(est)
            if not est.got_tle_this_stage and stage > 0 and stage < fire_stage:
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
    if name == "firereturn":
        return FireReturnStrategy(agent)
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

    def __init__(self, init_dt_mean, init_dt_std, perp_km, obs_agent_size, strat_names, policy,
                 selfish_model, other_obs="tle", tle_sigma_base=BF.TLE_SIGMA_BASE_KM):
        self.init_dt_mean = init_dt_mean      # prior MEAN signed dt (CDM nominal) for the Gaussian filter
        self.init_dt_std = init_dt_std        # prior STD (km) — the operator's initial uncertainty
        self.perp_km = perp_km
        self.obs_agent_size = obs_agent_size
        self.strat_names = strat_names
        self.policy = policy
        self.selfish_model = selfish_model
        self.other_obs = other_obs
        self.tle_sigma_base = tle_sigma_base
        self.tle_stages = tle_refresh_stages()    # build-time TLE epochs aligned to the stage grid
        self._est = {}        # traj i -> [AgentEstimator x2]
        self._strat = {}      # traj i -> [OperatorStrategy x2]
        self._last_miss = {}  # traj i -> (b1_miss, b2_miss) captured at action() for the trace

    def reset_traj(self, i, init_state):
        # operators' prior = a Gaussian about the conjunction's nominal dt (CDM-style); the continuous
        # filter then refines it per the --other-obs model (own contacts no-collapse, TLE partial).
        self._est[i] = [AgentEstimator(self.init_dt_mean, self.init_dt_std, self.perp_km,
                                       other_obs=self.other_obs, tle_sigma_base=self.tle_sigma_base,
                                       tle_stages=self.tle_stages)
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
        # call observe EVERY stage (even off-contact: obs=None) so the per-stage obs/TLE flags
        # reset; the estimator decides whether there is anything to refresh.
        for a in range(N_AGENT):
            combined = (_decode_local_dt_obs_km(obs, self.obs_agent_size, a)
                        if obs is not None else None)
            self._est[i][a].observe(combined, step)
        return belief_i, oh_i          # POMDP belief/oh unused by B1; pass through untouched

    def trace_header(self):
        return f" {'b1_miss':>8} {'b2_miss':>8}"

    def trace_cols(self, i):
        m1, m2 = self._last_miss[i]
        return f" {m1:>8.2f} {m2:>8.2f}"


def main():
    ap = argparse.ArgumentParser()
    # scenario knobs already applied pre-import by _cli_bootstrap_scenario; declared so argparse
    # accepts them (the config surface; values consumed before model import).
    ap.add_argument("--scenario-config", default=None,
                    help="YAML scenario config (the ONE config surface). NO env vars.")
    ap.add_argument("--man-cost", type=float, default=None)
    ap.add_argument("--disp-k", default=None)
    ap.add_argument("--hour-grid", default=None)
    ap.add_argument("--merge-threshold", type=float, default=None)
    ap.add_argument("--strategy", choices=["threshold", "selfish", "fixedlead", "firereturn"],
                    default="threshold", help="shared per-craft strategy (unless overridden).")
    ap.add_argument("--strategy-sc1", choices=["threshold", "selfish", "fixedlead", "firereturn"],
                    default=None, help="override strategy for SC1 (mixed-strategy runs).")
    ap.add_argument("--strategy-sc2", choices=["threshold", "selfish", "fixedlead", "firereturn"],
                    default=None, help="override strategy for SC2 (mixed-strategy runs).")
    ap.add_argument("--policy", choices=["conservative", "aggressive", "asymmetric", "poc"],
                    default="conservative", help="threshold strategy's sub-policy.")
    ap.add_argument("--selfish-model",
                    choices=["blind", "cautious-margin", "cautious-early", "obsaware"],
                    default="obsaware", help="selfish strategy's model of the OTHER craft.")
    ap.add_argument("--other-obs", choices=["perfect", "tle", "frozen"], default="tle",
                    help="OTHER-craft observation model (the decentralized fidelity axis). "
                         "perfect=contact reveals the combined dt exactly (belief collapses to a "
                         "point; idealized). tle=other craft refreshes on a slow ~8h TLE clock with "
                         "NOISY along-track sigma (realistic, asymmetric self-vs-other). frozen=other "
                         "craft NEVER refreshes (strictest self-knowledge-only floor). Own state is "
                         "always tracked exactly via own burns. See belief_filter.py.")
    ap.add_argument("--tle-sigma", type=float, default=BF.TLE_SIGMA_BASE_KM,
                    help="base TLE along-track sigma (km) for the OTHER craft under --other-obs tle; "
                         "grows with fix age toward ~5 km (aged-TLE literature). Sweepable.")
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
    strat_tag += f"-{args.other_obs}"
    label = f"B1[{strat_tag}]"

    def _tagged_path(val, subdir, prefix, ext):
        if os.sep in val or val.endswith(ext):
            return val
        d = os.path.join(_SCA, "notes", subdir)
        os.makedirs(d, exist_ok=True)
        return os.path.join(
            d, f"{prefix}_{val}_{strat_tag}_{args.variant}_{args.mode}{ext}")

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
    print(f"  strategy: SC1={s1} SC2={s2}  policy={args.policy} selfish_model={args.selfish_model}")

    init_b, flagged, eff_miss = TV.build_init_b_danger(
        args.init_miss, args.init_spread, args.perp, sign_mode="both")
    print(f"  init: miss={args.init_miss} spread={args.init_spread} perp={args.perp} "
          f"-> eff_miss={eff_miss:.3f} km  flagged={flagged}")

    T, O, R, perp = build_model(args.variant, args.perp)
    # continuous-Gaussian prior for the operators' filter (no binning): mean = nominal signed dt,
    # std = the screening spread. (init_b is still built for the brahe seed-state sampling in run_mc.)
    init_dt_mean, init_dt_std = BF.prior_from_init(args.init_miss, args.init_spread, args.perp)
    obs_agent_size = TV.N_OBS_AGENT
    print(f"  built T/O/R [{args.variant}]  perp_meas={perp:.3f}  target_miss={TARGET_MISS_KM} km")
    print(f"  prior dt N({init_dt_mean:.2f},{init_dt_std:.2f})  other-obs={args.other_obs}  "
          f"tle-sigma={args.tle_sigma}  TLE@{tle_refresh_stages()} (every {TLE_CADENCE_H:.0f}h)")

    # The heuristic is piped into rollout_v2's VECTORIZED brahe engine via a PolicySource;
    # B1 does not run its own rollout loop. sdec/full are unused by the source (operators have
    # no POMDP belief) -> pass None. A fresh source per call resets per-trajectory estimators.
    def _source():
        return B1PolicySource(init_dt_mean, init_dt_std, perp, obs_agent_size, strat_names,
                              args.policy, args.selfish_model,
                              other_obs=args.other_obs, tle_sigma_base=args.tle_sigma)

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
