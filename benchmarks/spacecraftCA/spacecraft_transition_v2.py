"""
spacecraft_transition_v2.py

v2 transition model for the reduced relative-motion state
(dT_bin, vdev1, vdev2, stage). See notes/MODEL_DEFINITION.md.

This module builds:
  - per-stage brahe DRIFT GAIN table  gain[k]  (km of dT per unit vdev per stage step)
  - per-conjunction perp (sideways standoff, km), build-time constant
  - the dt transition: dt_next = dt + (vdev1+vdev2)*gain[k]   [+ stochastic spread]
  - the T matrix over v2 states

Kept separate from spacecraft_matrices.py (v1) so the v1 model stays runnable for
A/B and so CHECKPOINT 2 (T vs brahe) is verifiable in isolation before O/R wiring.
"""

import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from spacecraft_matrices import (
    sc1_eci_at_tca, apply_maneuver, propagate_batch_to,
    STAGE_EPOCHS, EPOCH_TCA, DV_MAGNITUDE,
    PROCESS_DRIFT_SIGMA_KM_PER_SQRT_H,
    TRANSITION_NOISE_N_SAMPLES, ACT_WAIT, CONTACT_STAGES,
    state_eci_to_rtn, state_rtn_to_eci,
)
import spacecraft_discretizer_v2 as D

N_STAGES = D.N_STAGES
N_ACT_AGENT = 3
N_JOINT_ACTIONS = N_ACT_AGENT ** 2

# ---------------------------------------------------------------------------
# Observation space (v2) — mirrors v1 structure with dt-bin replacing miss-bin
# and vdev replacing dev. Per agent: (dt_bin_obs OR null, own vdev_bin).
# ---------------------------------------------------------------------------
DT_NULL_OBS  = D.N_DT                 # "no new dt observation" off-sync
N_DT_OBS     = D.N_DT + 1             # dt bins + null
N_OBS_AGENT  = N_DT_OBS * D.N_VDEV    # local obs = (dt_obs_or_null, own vdev_bin)
N_JOINT_OBS  = N_OBS_AGENT ** 2

# Reward constants. (v1 used REWARD_MANEUVER=-10, but at -10/burn vs -1/deviation-stage
# a single burn cost as much as 10 off-nominal stages, biasing the policy toward NOT
# maneuvering. Lowered to -1 (2026-06-10 cont.) so the burn/deviation/risk tradeoff is
# balanced and the policy will actually maneuver when warranted.)
REWARD_MANEUVER  =  -1.0   # per agent-burn
REWARD_DEVIATION =  -1.0   # per stage per agent off-nominal (vdev != NOM)
REWARD_STEP      =   0.0

# ---------------------------------------------------------------------------
# Terminal reward = TWO OPPOSING RAMPS in miss/dT space (v2 redesign).
# See notes/MODEL_DEFINITION.md §7 and LITERATURE_CA_THRESHOLDS.md.
# Replaces v1's monotone graded-risk table (which rewarded distance to 100 km and
# made the policy over-mitigate to ~45 km). Literature objective: MINIMIZE dV to
# CLEAR a threshold, not maximize distance. Optimal sits in the valley between:
#   (1) RISK ramp  — decays from the collision floor (<1km, PoC>=1e-4) to ~0 by the
#       screening-volume clearance (~5 km, PoC<1e-4 for LEO); no reward past it.
#   (2) DISPLACEMENT ramp — grows with |dT_TCA| beyond a station-keeping tube
#       half-width (return-to-orbit cost-to-go proxy, §5b); 0 while in-slot.
# Both knees are physically citable; only the relative WEIGHT is a free knob,
# calibrated so collision ALWAYS dominates.
# ---------------------------------------------------------------------------

# --- RISK ramp parameters (keyed to operational screening thresholds, km) ---
RISK_COLLISION_KM   = 1.0      # < this: PoC >= 1e-4 act/no-act line -> max penalty
RISK_CLEARED_KM     = 5.0      # >= this: screening volume cleared -> 0 risk
RISK_MAX_PENALTY    = -10000.0 # penalty at miss = 0 (collision); dominates everything

# --- DISPLACEMENT ramp parameters (keyed to station-keeping tube, km) ---
DISP_TUBE_HALFWIDTH_KM = 5.0   # within this |dT|: effectively in-slot, no return owed
DISP_COST_PER_KM       = 0.5   # return-cost slope beyond the tube (free WEIGHT knob)

# Collision threshold used for the collision-probability metric (unchanged).
# (RISK_COLLISION_KM is the reward floor; the binary collision flag uses D's threshold.)


def risk_ramp_reward(miss_km: float) -> float:
    """
    Risk penalty as a SOFT ramp decaying from RISK_MAX_PENALTY at miss=0 to 0 at
    RISK_CLEARED_KM. Flat (max penalty) below RISK_COLLISION_KM (the PoC>=1e-4 floor),
    then smoothly to 0 by the screening-volume clearance distance. Smooth (cosine)
    interpolation avoids the exact-tie cliff of a hard threshold.
    """
    if miss_km <= RISK_COLLISION_KM:
        return RISK_MAX_PENALTY
    if miss_km >= RISK_CLEARED_KM:
        return 0.0
    # smoothstep from 1 (at collision km) to 0 (at cleared km)
    frac = (miss_km - RISK_COLLISION_KM) / (RISK_CLEARED_KM - RISK_COLLISION_KM)
    smooth = 0.5 * (1.0 + np.cos(np.pi * frac))   # 1 -> 0, zero-slope at both ends
    return RISK_MAX_PENALTY * smooth


def displacement_cost(dt_km: float) -> float:
    """
    Return-to-orbit cost-to-go proxy (negative): 0 within the station-keeping tube
    half-width, then grows linearly with |dT| beyond it. Makes far-field drift COST
    something so the policy clears the threshold and stops (no over-mitigation).
    """
    excess = max(abs(dt_km) - DISP_TUBE_HALFWIDTH_KM, 0.0)
    return -DISP_COST_PER_KM * excess


def terminal_reward(dt_km: float, miss_km: float) -> float:
    """Combined terminal reward = risk ramp (by quadrature miss) + displacement ramp (by |dT|)."""
    return risk_ramp_reward(miss_km) + displacement_cost(dt_km)


def local_obs_index(dt_obs: int, vdev_bin: int) -> int:
    return dt_obs + N_DT_OBS * vdev_bin


def joint_obs_index(o1: int, o2: int) -> int:
    return o1 + N_OBS_AGENT * o2


def _independent_joint_obs(p1, p2):
    pj = np.zeros(N_JOINT_OBS, dtype=np.float64)
    for o1 in range(N_OBS_AGENT):
        if p1[o1] <= 0: continue
        for o2 in range(N_OBS_AGENT):
            if p2[o2] > 0:
                pj[joint_obs_index(o1, o2)] = p1[o1] * p2[o2]
    return pj


def perfect_shared_obs(dt_bin: int, vdev1: int, vdev2: int) -> np.ndarray:
    """Synchronized: both agents see the shared dt_bin plus each own vdev."""
    pj = np.zeros(N_JOINT_OBS, dtype=np.float64)
    o1 = local_obs_index(dt_bin, vdev1)
    o2 = local_obs_index(dt_bin, vdev2)
    pj[joint_obs_index(o1, o2)] = 1.0
    return pj


def private_vdev_only_obs(vdev1: int, vdev2: int) -> np.ndarray:
    """Off-sync: each agent sees only its own vdev (null dt symbol)."""
    p1 = np.zeros(N_OBS_AGENT); p1[local_obs_index(DT_NULL_OBS, vdev1)] = 1.0
    p2 = np.zeros(N_OBS_AGENT); p2[local_obs_index(DT_NULL_OBS, vdev2)] = 1.0
    return _independent_joint_obs(p1, p2)


# Maneuver-execution error as a fraction of commanded dV. This is the PROPORTIONAL-
# MAGNITUDE term of the standard Gates execution-error model (used operationally, e.g.
# DART TCMs); a low single-digit % is representative (chem looser, electric tighter).
# 2% is a representative value -> SWEEP for sensitivity. See notes/LITERATURE_CA_
# THRESHOLDS.md ("Maneuver execution error"). (We model only proportional magnitude;
# Gates' pointing/cross-track terms are dropped since along-track burns act in dT.)
# Brahe-measured effect: dT spread = frac * |lever_arm(k)| — large early (~2.5 km at
# T-24h @2%), small late (~0.16 km), since an early burn error has longer to amplify.
# (Replaces v1's 50% multiplicative-on-offset noise, ~25x too large for the dT rep.)
EXEC_DV_ERROR_FRAC = 0.02


def split_joint_action(a: int):
    """(a1, a2) per-agent burns; matches v1/RSSDA product order a = a1 + 3*a2."""
    return a % N_ACT_AGENT, a // N_ACT_AGENT


def _action_vdev_delta(action: int) -> int:
    """+dV (1) -> +1, -dV (2) -> -1, WAIT (0) -> 0."""
    if action == ACT_WAIT:
        return 0
    return +1 if action == 1 else -1


def stage_t2go_h(k: int) -> float:
    return (EPOCH_TCA - STAGE_EPOCHS[k]) / 3600.0


# ---------------------------------------------------------------------------
# Per-stage brahe drift-gain table + perp (build-time constants per conjunction)
# ---------------------------------------------------------------------------

def make_sc2_rel_state_at_tca(perp_km: float, dt0_km: float,
                              v_rel_ms: float = 15.0) -> np.ndarray:
    """
    Build SC2's RTN relative state at TCA for a conjunction with a given
    along-track offset dt0 and sideways standoff perp (placed in radial here;
    cross-track gives the same dT dynamics since the burn acts in T regardless).
      dT = dt0 (along-track),  perp = radial standoff,  closing in along-track.
    """
    return np.array([perp_km * 1e3, dt0_km * 1e3, 0.0, 0.0, -v_rel_ms, 0.0])


def compute_gain_table_and_perp(perp_km: float = 0.0, dt0_km: float = 0.0,
                                dv: float = None):
    """
    Per-stage along-track drift RATE from brahe, and the conjunction perp.

    A unit along-track burn does not add a fixed dT; it sets a constant drift RATE
    (km/h) that then accumulates over the remaining time-to-go. We measure the rate
    so the transition can advance dt incrementally per step:
        rate_at[k] = lever_arm(k) / t2go(k)        # km/h per unit vdev, burn at stage k
        dt_step    = rate * (t2go(k) - t2go(k+1))  # per-step increment
    where lever_arm(k) = dT-at-TCA produced by one unit burn applied at stage k.
    (Applying the full lever arm per step instead — the naive reading of "gain" —
    double-counts and blows dt up; this rate form reproduces brahe, see CHECKPOINT 2.)

    Returns (rate_at: np.ndarray[N_STAGES] km/h per unit vdev, perp_km, dt0_km).
    """
    if dv is None:
        dv = DV_MAGNITUDE
    sc1 = sc1_eci_at_tca()
    sc2_tca = np.array(state_rtn_to_eci(sc1, make_sc2_rel_state_at_tca(perp_km, dt0_km)))

    def rtn_T_at_tca(sc2_at_stage_eci, k):
        tca = propagate_batch_to([STAGE_EPOCHS[k]], [sc2_at_stage_eci], EPOCH_TCA)[0]
        return np.array(state_eci_to_rtn(sc1, tca))[1] / 1e3   # along-track km

    rate_at = np.zeros(N_STAGES, dtype=np.float64)
    nom_rtn = np.array(state_eci_to_rtn(sc1, sc2_tca))
    perp_meas = float(np.hypot(nom_rtn[0], nom_rtn[2]) / 1e3)

    for k in range(N_STAGES):
        sc2_s = propagate_batch_to([EPOCH_TCA], [sc2_tca], STAGE_EPOCHS[k])[0]
        dT_nom = rtn_T_at_tca(sc2_s, k)
        sc2_burn = apply_maneuver(sc2_s, 1, dv=dv)          # +dV unit burn
        dT_burn = rtn_T_at_tca(sc2_burn, k)
        lever = dT_burn - dT_nom                            # total dT by TCA, km
        t2go = stage_t2go_h(k)
        rate_at[k] = lever / t2go if t2go > 0 else 0.0      # km/h per unit vdev
    return rate_at, perp_meas, dt0_km


# ---------------------------------------------------------------------------
# dt transition (deterministic core + stochastic spread)
# ---------------------------------------------------------------------------

_rng = np.random.default_rng(0)
_DRIFT_STD_NORMAL = _rng.standard_normal(TRANSITION_NOISE_N_SAMPLES)
_EXEC_STD_NORMAL = _rng.standard_normal(TRANSITION_NOISE_N_SAMPLES)


def step_hours(stage: int) -> float:
    """Hours of coast between stage and stage+1."""
    return max(stage_t2go_h(stage) - stage_t2go_h(stage + 1), 0.0)


def next_dt_distribution(dt_km: float, vdev_sum: int, mean_rate: float,
                         stage: int, n_new_burns: int, lever_k: float) -> np.ndarray:
    """
    Distribution over next dt bins.

    vdev_sum (= vdev1+vdev2 in {-2..+2}) is a net drift RATE in units of mean_rate
    (km/h per unit vdev). The per-step dt increment is rate * step_hours:
        dt_next_mean = dt + vdev_sum * mean_rate * step_hours(stage)

    Stochastic spread (added to dt before re-binning):
      - process drift: PROCESS_DRIFT_SIGMA * sqrt(step_hours)
      - execution noise: ADDITIVE, sigma = EXEC_DV_ERROR_FRAC * |lever_k| per NEW
        burn this step. lever_k = dT a unit burn at stage k produces by TCA
        (= mean_rate * t2go(k)). This is the brahe-measured structure (big early,
        small late); independent burns add in quadrature -> sqrt(n_new_burns).
    """
    dt_step_h = step_hours(stage)
    drift_sigma = PROCESS_DRIFT_SIGMA_KM_PER_SQRT_H * np.sqrt(dt_step_h)
    exec_sigma = (EXEC_DV_ERROR_FRAC * abs(lever_k) * np.sqrt(n_new_burns)
                  if n_new_burns > 0 else 0.0)
    dt_mean = dt_km + vdev_sum * mean_rate * dt_step_h
    counts = np.zeros(D.N_DT, dtype=np.float64)
    for i in range(TRANSITION_NOISE_N_SAMPLES):
        x = dt_mean + drift_sigma * _DRIFT_STD_NORMAL[i] + exec_sigma * _EXEC_STD_NORMAL[i]
        counts[D.dt_to_bin(x)] += 1.0
    return counts / counts.sum()


# ---------------------------------------------------------------------------
# T matrix
# ---------------------------------------------------------------------------

def variant_sync_next(variant: str, next_stage: int) -> bool:
    """Whether the variant has synchronized info arriving at next_stage."""
    if variant == "centralized":
        return True
    if variant == "sdec":
        return next_stage in CONTACT_STAGES
    return False  # dec


def build_T_O(rate_at: np.ndarray, variant: str, verbose: bool = False):
    """
    Build T[a,s,s'] and O[a,s',o] over (dt_bin, vdev1, vdev2, stage).
    `rate_at` mean -> single representative drift rate (Markovian in vdev count).
    Observations: sync stages reveal shared dt_bin + both vdev; off-sync reveal
    only each agent's own vdev. Terminal stage -> sink.
    """
    mean_rate = float(np.mean(rate_at))
    T = np.zeros((N_JOINT_ACTIONS, D.N_STATES_TOTAL, D.N_STATES_TOTAL), dtype=np.float64)
    O = np.zeros((N_JOINT_ACTIONS, D.N_STATES_TOTAL, N_JOINT_OBS), dtype=np.float64)
    for k in range(N_STAGES):
        lever_k = mean_rate * stage_t2go_h(k)
        for dt_bin in range(D.N_DT):
            dt_c = D.dt_bin_center_km(dt_bin)
            for v1 in range(D.N_VDEV):
                for v2 in range(D.N_VDEV):
                    s = D.state_index(dt_bin, v1, v2, k)
                    for a in range(N_JOINT_ACTIONS):
                        a1, a2 = split_joint_action(a)
                        if k == N_STAGES - 1:
                            T[a, s, D.SINK_STATE] = 1.0
                            continue
                        nv1 = D.update_vdev_bin(v1, a1)
                        nv2 = D.update_vdev_bin(v2, a2)
                        vdev_sum = D.vdev_value(nv1) + D.vdev_value(nv2)
                        n_new_burns = (a1 != ACT_WAIT) + (a2 != ACT_WAIT)
                        sync_next = variant_sync_next(variant, k + 1)
                        dist = next_dt_distribution(dt_c, vdev_sum, mean_rate, k,
                                                    n_new_burns, lever_k)
                        for nb, p in enumerate(dist):
                            if p > 0.0:
                                sp = D.state_index(nb, nv1, nv2, k + 1)
                                T[a, s, sp] += p
                                if sync_next:
                                    pj = perfect_shared_obs(nb, nv1, nv2)
                                else:
                                    pj = private_vdev_only_obs(nv1, nv2)
                                O[a, sp, :] += p * pj
        if verbose:
            print(f"  ... v2 T/O stage {k}/{N_STAGES-1} built")
    for a in range(N_JOINT_ACTIONS):
        T[a, D.SINK_STATE, D.SINK_STATE] = 1.0
        O[a, D.SINK_STATE, :] = 1.0 / N_JOINT_OBS
    # normalize O rows
    for a in range(N_JOINT_ACTIONS):
        for sp in range(D.N_STATES_TOTAL):
            rs = O[a, sp, :].sum()
            O[a, sp, :] = (O[a, sp, :] / rs) if rs > 1e-12 else (1.0 / N_JOINT_OBS)
    return T, O


def build_R(perp_km: float) -> np.ndarray:
    """
    Reward R[a, s] (v2 two-ramp terminal reward):
      maneuver cost per burn, per-stage deviation penalty (vdev != NOM),
      and at the final stage the combined terminal reward = RISK ramp (by quadrature
      miss) + DISPLACEMENT ramp (by |dT_TCA|). See terminal_reward() / §7.
    """
    R = np.zeros((N_JOINT_ACTIONS, D.N_STATES_TOTAL), dtype=np.float64)
    for k in range(N_STAGES):
        for dt_bin in range(D.N_DT):
            dt_c = D.dt_bin_center_km(dt_bin)
            miss = D.miss_km_from_dt(dt_c, perp_km)
            for v1 in range(D.N_VDEV):
                for v2 in range(D.N_VDEV):
                    s = D.state_index(dt_bin, v1, v2, k)
                    dev_pen = REWARD_DEVIATION * ((v1 != D.VDEV_ZERO) + (v2 != D.VDEV_ZERO))
                    term = terminal_reward(dt_c, miss) if k == N_STAGES - 1 else 0.0
                    for a in range(N_JOINT_ACTIONS):
                        a1, a2 = split_joint_action(a)
                        man = (REWARD_MANEUVER if a1 != ACT_WAIT else 0.0) + \
                              (REWARD_MANEUVER if a2 != ACT_WAIT else 0.0)
                        R[a, s] = REWARD_STEP + dev_pen + man + term
    # sink: zero
    return R


def build_init_b(dt0_km: float) -> np.ndarray:
    """Point-mass initial belief: nominal vdev, stage 0, dt bin from dt0."""
    init_b = np.zeros(D.N_STATES_TOTAL, dtype=np.float64)
    init_b[D.state_index(D.dt_to_bin(dt0_km), D.VDEV_ZERO, D.VDEV_ZERO, 0)] = 1.0
    return init_b


def build_init_b_spread(miss_bins_km, sign_mode="both") -> np.ndarray:
    """
    v1-matched SPREAD initial belief over uncertain along-track magnitude, to give
    sync something to resolve (v1 spread over unsigned miss bins; here over |dt|).

    miss_bins_km : iterable of candidate |dt| magnitudes (km), e.g. [50, 200] for
                   "far/uncertain" like v1's two farthest bins.
    sign_mode    : "both" -> split each magnitude over +dt and -dt (don't know which
                   side; faithful to v1's unsigned-miss ambiguity, exercises signed
                   dynamics); "pos" -> +dt only (known side).
    All mass at stage 0, vdev = NOM.
    """
    b = np.zeros(D.N_STATES_TOTAL, dtype=np.float64)
    for m in miss_bins_km:
        signs = [+1.0, -1.0] if sign_mode == "both" else [+1.0]
        for s in signs:
            b[D.state_index(D.dt_to_bin(s * m), D.VDEV_ZERO, D.VDEV_ZERO, 0)] += 1.0
    total = b.sum()
    if total <= 0:
        return build_init_b(0.0)
    return b / total


def build_matrices_v2(variant: str, perp_km: float = 0.0, dt0_km: float = 2.0,
                      dv: float = None, init_b=None, verbose: bool = True):
    """Build T, O, R, init_b for the v2 reduced-state model. Mirrors v1 build_matrices.
    If init_b is given (e.g. a spread belief), it overrides the default point-mass."""
    if verbose:
        print(f"Building v2 matrices [{variant}] perp={perp_km} dt0={dt0_km} ...")
    rate_at, perp_meas, _ = compute_gain_table_and_perp(perp_km, dt0_km, dv)
    T, O = build_T_O(rate_at, variant, verbose=verbose)
    R = build_R(perp_meas)
    if init_b is None:
        init_b = build_init_b(dt0_km)
    if verbose:
        print(f"  v2 matrices built: {D.N_STATES_TOTAL} states, {N_JOINT_ACTIONS} acts, "
              f"{N_JOINT_OBS} obs; perp_meas={perp_meas:.3f} km")
    return T, O, R, init_b


if __name__ == "__main__":
    gain, perp, dt0 = compute_gain_table_and_perp(perp_km=0.0, dt0_km=2.0)
    print("Per-stage brahe gain table (km of dT per +1 vdev unit):")
    for k in range(N_STAGES):
        print(f"  stage {k:>2}  t2go={stage_t2go_h(k):>6.2f}h  gain={gain[k]:>8.3f} km/unit")
    print(f"perp(measured) = {perp:.4f} km")
