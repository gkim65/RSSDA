"""
spacecraft_matrices.py

Builds T, O, R matrices for the spacecraft CA SDec-POMDP.
State = (miss_distance_bin, sc1_deviation_bin, sc2_deviation_bin, stage).
Brahe propagates representative conjunction states to TCA offline.

Usage:
  python spacecraft_matrices.py           # build and cache
  python spacecraft_matrices.py --verify  # load and verify
  python spacecraft_matrices.py --force   # force rebuild
"""

import os
import sys
import argparse
import numpy as np
import multiprocessing as mp
from math import erf

from brahe import (
    Epoch, AngleFormat, R_EARTH,
    initialize_eop,
    state_koe_to_eci, state_rtn_to_eci, state_eci_to_rtn,
    NumericalOrbitPropagator, NumericalPropagationConfig, ForceModelConfig,
    KeplerianPropagator,
    par_propagate_to,
)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from spacecraft_discretizer import (
    N_MISS, N_DEV, DEV_ZERO, N_STAGES, N_STATES, N_STATES_TOTAL, SINK_STATE,
    MISS_EDGES_KM,
    miss_to_bin, bin_center_km,
    state_index, index_to_state, update_dev_bin,
    sync_trigger_states, collision_bin_indices,
)


# ---------------------------------------------------------------------------
# Problem constants
# ---------------------------------------------------------------------------

N_BURN_AGENT    = 3                    # WAIT, +dV_T, -dV_T
MISS_NULL_OBS   = N_MISS               # no new miss observation off-sync
N_MISS_OBS      = N_MISS + 1           # miss bins + null
N_OBS_AGENT     = N_MISS_OBS * N_DEV   # local obs = (miss_obs_or_null, own dev_bin)
N_JOINT_OBS     = N_OBS_AGENT ** 2

# All three variants use the same 9-action space (3 burns per agent).
# Sync is state-triggered in RSSDA via sync_states, not encoded in actions.
# A future joint state+action sync model would need an explicit per-agent
# sync flag and corresponding sync_actions wiring in SDecPOMDPModel.
N_ACT_AGENT     = N_BURN_AGENT         # 3
N_JOINT_ACTIONS = N_ACT_AGENT ** 2    # 9

DV_MAGNITUDE    = 0.5                 # m/s  (good bin differentiation; use --dv to sweep)
V_REL_MS        = 15.0                # along-track closing speed at TCA (m/s)

# SC1 orbital elements at TCA [a, e, i, RAAN, omega, M] (deg)
SC1_OE_AT_TCA = np.array([
    R_EARTH + 550e3,   # semi-major axis (m)
    0.001,             # eccentricity
    55.0,              # inclination (deg)
    20.0,              # RAAN (deg)
    0.0,               # argument of perigee (deg)
    0.0,               # mean anomaly (deg)
])
EPOCH_TCA = Epoch(2025, 6, 2, 0, 0, 0.0)

# 16 stages: 10 ~2h grid points merged with 6 GS contact windows.
# T-8h and T-6h dropped since T-7.83h and T-6.17h GS contacts already cover those slots.
# Contact stages are sync/centralization points; others are always-decentralized.
_GS_TIMES_H = [23.39, 9.48, 7.83, 6.17, 2.80, 1.13]  # hours before TCA
_HOUR_GRID_H = [24, 22, 20, 18, 16, 14, 12, 10, 4, 2]  # ~2h grid, T-8 and T-6 dropped
_ALL_TIMES_H = sorted(
    set([float(h) for h in _HOUR_GRID_H] + _GS_TIMES_H),
    reverse=True,
)  # 16 values, descending (T-24h first)

STAGE_T_BEFORE_TCA_SEC = [h * 3600.0 for h in _ALL_TIMES_H]
STAGE_EPOCHS = [EPOCH_TCA - dt for dt in STAGE_T_BEFORE_TCA_SEC]

# Indices of stages that are GS contact windows (sync/centralization triggers).
# SINGLE SOURCE OF TRUTH: every consumer (this module's v1 builder, the v2 builder in
# spacecraft_transition_v2, and compare_variants_v2's SDec sync_states) must read THIS
# global. To override it (e.g. the Scenario-1 contact-timing ablation) call
# set_contact_stages() — do NOT rebind the name, which would orphan callers that
# imported a copy. Mutating in place keeps existing list bindings consistent.
CONTACT_STAGES = [i for i, h in enumerate(_ALL_TIMES_H) if h in _GS_TIMES_H]


def set_contact_stages(stages):
    """Override the global GS contact-stage list IN PLACE (so any module that holds a
    binding to this list — historically spacecraft_transition_v2 — sees the change).
    Pass an iterable of stage indices (subset of range(N_STAGES)); [] = no contacts."""
    stages = sorted({int(s) for s in stages})
    CONTACT_STAGES[:] = stages
    return CONTACT_STAGES


def get_contact_stages():
    return list(CONTACT_STAGES)

# Reward constants
REWARD_COLLISION = -10000.0  # miss < 1 km at TCA
REWARD_HIGH      =  -1000.0  # retained name for legacy scripts
REWARD_MOD       =   -100.0  # retained name for legacy scripts
REWARD_MANEUVER  = -10.0
REWARD_DEVIATION =  -1.0     # per stage per spacecraft outside nominal dev bin
REWARD_STEP      =  0.0
REWARD_MODEL_VERSION = "refined_terminal_risk_v1"
REWARD_TERMINAL_RISK_BY_BIN = np.array([
    -10000.0,  # [0, 0.5) km
    -10000.0,  # [0.5, 1) km
     -3000.0,  # [1, 2) km
     -1000.0,  # [2, 5) km
      -300.0,  # [5, 10) km
      -100.0,  # [10, 20) km
       -25.0,  # [20, 50) km
        -5.0,  # [50, 100) km
         0.0,  # [100, 500) km
         0.0,  # [500, inf) km
], dtype=np.float64)
if len(REWARD_TERMINAL_RISK_BY_BIN) != N_MISS:
    raise ValueError("Terminal risk reward table must match N_MISS.")


def terminal_risk_reward(miss_bin: int) -> float:
    """Terminal conjunction-risk reward for a miss-distance bin."""
    return float(REWARD_TERMINAL_RISK_BY_BIN[miss_bin])


def collision_probability_from_bin_probs(bin_probs: np.ndarray) -> float:
    """Probability mass in bins below the 1 km collision threshold."""
    return float(sum(float(bin_probs[i]) for i in collision_bin_indices()))

# Observation noise constants retained for simulator/future noisy-contact variants.
# Current clean semantics: sync stages share a perfect joint history/state
# observation; off-sync stages reveal only each agent's own deviation bin and a
# null miss symbol.
OBS_SIGMA_KM  = 5.0   # legacy symmetric sigma (kept for greedy/fallback)
TLE_SIGMA_KM  = 3.0   # 1-sigma TLE position uncertainty used in asymmetric obs model
GPS_SIGMA_KM  = 0.1   # 1-sigma GPS position uncertainty (own spacecraft) â€” near-negligible

# Stochastic transition model.
#
# We do not drift the compressed miss_bin label directly. Instead, matrix
# generation computes the nominal continuous RTN relative position at TCA,
# perturbs that underlying vector, and then re-bins the resulting miss distance.
# Process drift applies during coast and burn. Burn execution noise is layered on
# top when at least one spacecraft maneuvers.
TRANSITION_MODEL_VERSION = "rtn_process_drift_v1"
PROCESS_DRIFT_SIGMA_KM_PER_SQRT_H = 0.15
EXEC_NOISE_SIGMA = 0.50
TRANSITION_NOISE_N_SAMPLES = 101

CACHE_PATH_TEMPLATE = os.path.join(_HERE, "spacecraft_matrices_cache_{variant}.npz")

def cache_path(variant: str) -> str:
    return CACHE_PATH_TEMPLATE.format(variant=variant)

CACHE_PATH = cache_path("sdec")  # legacy default

def variant_sync_stages(variant: str) -> list:
    """Stages where the variant is synchronized."""
    if variant == "centralized":
        return list(range(N_STAGES))
    if variant == "sdec":
        return list(CONTACT_STAGES)
    if variant == "dec":
        return []
    raise ValueError(f"Unknown variant: {variant}")

ACT_WAIT = 0
ACT_POS  = 1   # +dV_T prograde
ACT_NEG  = 2   # -dV_T retrograde

# RSSDA product-order encoding:
#   joint action      a = a1 + N_ACT_AGENT * a2
#   joint observation o = o1 + N_OBS_AGENT * o2
# where agent 1/SC1 is the low-order factor and agent 2/SC2 is the high-order
# factor. This matches RSSDA.a_prod/o_prod and the other benchmark drivers.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_joint_action(a: int):
    """Returns (a1, a2) per-agent burn actions (each 0=WAIT, 1=POS, 2=NEG)."""
    return a % N_ACT_AGENT, a // N_ACT_AGENT

def joint_obs_index(o1: int, o2: int) -> int:
    """Flat joint observation index using RSSDA product order."""
    return o1 + N_OBS_AGENT * o2

def local_obs_index(miss_obs: int, dev_obs: int) -> int:
    """Per-agent observation index for (miss bin or null, own deviation)."""
    return miss_obs + N_MISS_OBS * dev_obs

def independent_joint_obs_distribution(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Private independent observations for the two agents."""
    p_joint = np.zeros(N_JOINT_OBS, dtype=np.float64)
    for o1 in range(N_OBS_AGENT):
        for o2 in range(N_OBS_AGENT):
            p_joint[joint_obs_index(o1, o2)] = p1[o1] * p2[o2]
    return p_joint

def shared_joint_obs_distribution(p: np.ndarray) -> np.ndarray:
    """
    Shared observation after synchronization.

    The diagonal support enforces identical local observation histories after
    sync: P(o1=o, o2=o) = p[o].
    """
    p_joint = np.zeros(N_JOINT_OBS, dtype=np.float64)
    for o in range(N_OBS_AGENT):
        p_joint[joint_obs_index(o, o)] = p[o]
    return p_joint

def local_obs_distribution(miss_dist: np.ndarray, dev_bin: int) -> np.ndarray:
    """Distribution over one agent's local observation space."""
    p = np.zeros(N_OBS_AGENT, dtype=np.float64)
    for miss_obs, prob in enumerate(miss_dist):
        if prob > 0.0:
            p[local_obs_index(miss_obs, dev_bin)] = prob
    return p

def joint_private_obs_distribution(miss_dist: np.ndarray, dev1_bin: int, dev2_bin: int) -> np.ndarray:
    """Independent private local observations."""
    p1 = local_obs_distribution(miss_dist, dev1_bin)
    p2 = local_obs_distribution(miss_dist, dev2_bin)
    return independent_joint_obs_distribution(p1, p2)

def local_dev_only_obs_distribution(dev_bin: int) -> np.ndarray:
    """Deterministic off-sync observation: null miss symbol plus own deviation."""
    p = np.zeros(N_OBS_AGENT, dtype=np.float64)
    p[local_obs_index(MISS_NULL_OBS, dev_bin)] = 1.0
    return p

def joint_private_dev_only_obs_distribution(dev1_bin: int, dev2_bin: int) -> np.ndarray:
    """Off-sync private observations: each agent sees only its own deviation."""
    p1 = local_dev_only_obs_distribution(dev1_bin)
    p2 = local_dev_only_obs_distribution(dev2_bin)
    return independent_joint_obs_distribution(p1, p2)

def perfect_shared_obs_for_state(miss_bin: int, dev1_bin: int, dev2_bin: int) -> np.ndarray:
    """
    Perfect synchronized observation.

    Agent 1's component reports (miss_bin, dev1_bin), agent 2's component
    reports (miss_bin, dev2_bin). Since the branch is centralized after sync,
    the joint observation reveals both local deviations.
    """
    p_joint = np.zeros(N_JOINT_OBS, dtype=np.float64)
    o1 = local_obs_index(miss_bin, dev1_bin)
    o2 = local_obs_index(miss_bin, dev2_bin)
    p_joint[joint_obs_index(o1, o2)] = 1.0
    return p_joint

# Propagator backend for matrix construction.
#   "numerical" : NumericalOrbitPropagator with ForceModelConfig.two_body()
#                 (integrates two-body; this is the historical baseline)
#   "keplerian" : KeplerianPropagator (closed-form two-body; brahe docs note
#                 two_body() is "equivalent to Keplerian propagation"). Much
#                 faster over long horizons and free of integration error.
# Both are PURE TWO-BODY (no J2/drag). Switching backends does not change the
# physics model, only how the same two-body motion is computed.
PROPAGATOR_BACKEND = "numerical"

# Drag-backend physical parameters [mass(kg), drag_area(m^2), Cd, srp_area(m^2), Cr].
# Used only when PROPAGATOR_BACKEND == "drag" (leo_default force model with
# J2+drag+SRP+third-body). Literature-default smallsat values.
DRAG_PARAMS = np.array([150.0, 1.0, 2.2, 1.0, 1.3])
_SW_INITIALIZED = False

def _ensure_sw():
    global _SW_INITIALIZED
    if not _SW_INITIALIZED:
        from brahe import initialize_sw
        initialize_sw()
        _SW_INITIALIZED = True

def make_prop(epoch: Epoch, eci: np.ndarray):
    if PROPAGATOR_BACKEND == "keplerian":
        return KeplerianPropagator.from_eci(epoch, np.asarray(eci, dtype=float), 60.0)
    if PROPAGATOR_BACKEND == "drag":
        _ensure_sw()
        return NumericalOrbitPropagator(
            epoch, eci,
            NumericalPropagationConfig.default(),
            ForceModelConfig.leo_default(),
            DRAG_PARAMS,
        )
    return NumericalOrbitPropagator(
        epoch, eci,
        NumericalPropagationConfig.default(),
        ForceModelConfig.two_body()
    )

def propagate_batch_to(epochs0, ecis, target: Epoch):
    """
    Propagate many states (each from its own start epoch) to a common target.

    numerical backend: builds NumericalOrbitPropagator objects and runs a single
        par_propagate_to (parallel) call -- requires identical target epoch.
    keplerian backend: uses the closed-form KeplerianPropagator.state(target);
        par_propagate_to / propagate_to are NOT used because brahe's Keplerian
        propagate_to steps in step_size increments and does not land exactly on
        the target epoch (it composes incorrectly across legs). .state(epoch) is
        the exact analytic state and round-trips to <1 m.

    Returns a list of 6-vectors aligned with `ecis`.
    """
    if PROPAGATOR_BACKEND == "keplerian":
        out = []
        for e0, eci in zip(epochs0, ecis):
            p = KeplerianPropagator.from_eci(e0, np.asarray(eci, dtype=float), 60.0)
            out.append(np.array(p.state(target)[:6]))
        return out
    props = [make_prop(e0, eci) for e0, eci in zip(epochs0, ecis)]
    par_propagate_to(props, target)
    return [np.array(p.current_state()[:6]) for p in props]

def apply_maneuver(eci: np.ndarray, action: int, dv: float = None) -> np.ndarray:
    """Apply impulsive along-track delta-v in ECI. Uses DV_MAGNITUDE if dv not given."""
    if action == ACT_WAIT:
        return eci.copy()
    if dv is None:
        dv = DV_MAGNITUDE
    r, v = eci[:3], eci[3:]
    r_hat = r / np.linalg.norm(r)
    n_hat = np.cross(r, v)
    n_hat /= np.linalg.norm(n_hat)
    t_hat = np.cross(n_hat, r_hat)
    sign  = +1.0 if action == ACT_POS else -1.0
    new_eci = eci.copy()
    new_eci[3:] = v + sign * dv * t_hat
    return new_eci

def propagate(epoch_start: Epoch, eci: np.ndarray, epoch_end: Epoch) -> np.ndarray:
    """Propagate ECI state from epoch_start to epoch_end, return 6-vector."""
    return propagate_batch_to([epoch_start], [eci], epoch_end)[0]

def sc1_eci_at_tca() -> np.ndarray:
    return np.array(state_koe_to_eci(SC1_OE_AT_TCA, AngleFormat.DEGREES))

def place_sc2_at_tca(miss_bin: int) -> np.ndarray:
    """Place SC2 at TCA using the bin center miss distance."""
    return place_sc2_at_tca_km(bin_center_km(miss_bin))

def place_sc2_at_tca_km(miss_km: float) -> np.ndarray:
    """
    Place SC2 at TCA with a specific miss distance in RTN:
      Î´r_RTN = [miss_km*1e3, 0, 0]  (radial offset = miss distance)
      Î´v_RTN = [0, -V_REL_MS, 0]    (along-track closing speed)
    Returns SC2 ECI state AT TCA.
    """
    sc1_tca = sc1_eci_at_tca()
    d_m = miss_km * 1e3
    rtn_rel = np.array([d_m, 0.0, 0.0, 0.0, -V_REL_MS, 0.0])
    return np.array(state_rtn_to_eci(sc1_tca, rtn_rel))


def back_prop_sc2_to_stage(sc2_tca: np.ndarray, stage: int) -> np.ndarray:
    """Back-propagate SC2 from TCA to the given stage epoch."""
    return propagate(EPOCH_TCA, sc2_tca, STAGE_EPOCHS[stage])


# ---------------------------------------------------------------------------
# Observation model
# ---------------------------------------------------------------------------

def obs_distribution(miss_km: float, sigma: float = None) -> np.ndarray:
    """
    Probability over N_MISS miss-distance bins for a given true miss distance.
    Gaussian noise: N(miss_km, sigma^2), integrated over MISS_EDGES_KM bins.
    Defaults to OBS_SIGMA_KM if sigma not specified.
    """
    if sigma is None:
        sigma = OBS_SIGMA_KM
    edges = MISS_EDGES_KM
    probs = np.zeros(N_MISS)
    for i in range(N_MISS):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        cdf_lo = 0.5 * (1 + erf((lo - miss_km) / (sigma * np.sqrt(2))))
        if hi == float('inf'):
            cdf_hi = 1.0
        else:
            cdf_hi = 0.5 * (1 + erf((hi - miss_km) / (sigma * np.sqrt(2))))
        probs[i] = max(0.0, cdf_hi - cdf_lo)
    total = probs.sum()
    if total < 1e-12:
        nearest = miss_to_bin(miss_km)
        probs[nearest] = 1.0
    else:
        probs /= total
    return probs


# ---------------------------------------------------------------------------
# Stochastic transition helper
# ---------------------------------------------------------------------------

# Fixed process/execution samples drawn once â€” reproducible across builds.
_RNG_TRANSITION = np.random.default_rng(0)
_EXEC_EPS = _RNG_TRANSITION.normal(
    0.0, EXEC_NOISE_SIGMA, TRANSITION_NOISE_N_SAMPLES
)
_DRIFT_STD_NORMAL = _RNG_TRANSITION.normal(
    0.0, 1.0, (TRANSITION_NOISE_N_SAMPLES, 3)
)

def transition_interval_hours(stage: int) -> float:
    """Elapsed time from this decision stage to the next decision stage."""
    if stage >= N_STAGES - 1:
        return 0.0
    return abs(
        STAGE_T_BEFORE_TCA_SEC[stage] - STAGE_T_BEFORE_TCA_SEC[stage + 1]
    ) / 3600.0

def stochastic_next_bin_dist(rtn_pos_tca_km: np.ndarray, any_burn: bool,
                             stage: int) -> np.ndarray:
    """
    Return probability distribution over next miss bins.

    The input is the nominal continuous RTN relative-position vector at TCA,
    not a miss-bin label. We add modest stochastic process drift to that vector
    for every transition. If a burn occurred, the existing maneuver-execution
    uncertainty is also applied as a multiplicative perturbation to the nominal
    vector before re-binning.
    """
    rtn_pos_tca_km = np.asarray(rtn_pos_tca_km, dtype=np.float64)
    drift_sigma = (
        PROCESS_DRIFT_SIGMA_KM_PER_SQRT_H
        * np.sqrt(max(transition_interval_hours(stage), 0.0))
    )
    counts = np.zeros(N_MISS, dtype=np.float64)
    for i in range(TRANSITION_NOISE_N_SAMPLES):
        pos_km = rtn_pos_tca_km.copy()
        if any_burn:
            pos_km *= max(0.0, 1.0 + _EXEC_EPS[i])
        pos_km += drift_sigma * _DRIFT_STD_NORMAL[i]
        counts[miss_to_bin(float(np.linalg.norm(pos_km)))] += 1.0
    return counts / counts.sum()


# ---------------------------------------------------------------------------
# Matrix generation
# ---------------------------------------------------------------------------

def build_matrices(verbose: bool = True, dv_magnitude: float = None,
                   variant: str = "sdec"):
    """
    Build T, O, R, init_b for the spacecraft CA SDec-POMDP.

    variant semantics:
      centralized: shared perfect observation/history at every stage
      sdec:        automatic shared perfect observation/history at contact stages
      dec:         no synchronization; off-sync observations are own-deviation only

    Returns T, O, R, init_b (numpy float64).
    """
    if dv_magnitude is None:
        dv_magnitude = DV_MAGNITUDE
    assert variant in ("centralized", "sdec", "dec"), f"Unknown variant: {variant}"

    n_act_agent  = N_ACT_AGENT
    n_joint_acts = N_JOINT_ACTIONS

    if verbose:
        print(f"Building spacecraft CA matrices [{variant}] "
              f"(miss + 3-bin per-agent deviation state)...")
        print(f"  States: {N_STATES_TOTAL} ({N_STATES} + 1 sink), "
              f"Actions: {n_joint_acts}, Obs: {N_JOINT_OBS}")
        print(f"  dv_magnitude = {dv_magnitude} m/s")
        print(f"  transition_model = {TRANSITION_MODEL_VERSION}")
        print(f"  drift_sigma = {PROCESS_DRIFT_SIGMA_KM_PER_SQRT_H} km/sqrt(h), "
              f"exec_noise_sigma = {EXEC_NOISE_SIGMA}, "
              f"samples = {TRANSITION_NOISE_N_SAMPLES}")

    T = np.zeros((n_joint_acts, N_STATES_TOTAL, N_STATES_TOTAL), dtype=np.float64)
    O = np.zeros((n_joint_acts, N_STATES_TOTAL, N_JOINT_OBS),    dtype=np.float64)
    R = np.zeros((n_joint_acts, N_STATES_TOTAL),                  dtype=np.float64)

    # Back-propagate SC1 and SC2 reference states from TCA to each stage epoch.
    sc1_tca = sc1_eci_at_tca()
    sc2_tca_per_bin = [place_sc2_at_tca(mb) for mb in range(N_MISS)]

    # Back-propagation: per stage, propagate (SC1 + N_MISS SC2s) from TCA to the
    # stage epoch. Each stage has a distinct target epoch, so it is its own batch.
    sc1_at_stage = []
    sc2_at_stage = [[] for _ in range(N_MISS)]
    for k in range(N_STAGES):
        ecis = [sc1_tca] + [sc2_tca_per_bin[mb] for mb in range(N_MISS)]
        epochs0 = [EPOCH_TCA] * len(ecis)
        states = propagate_batch_to(epochs0, ecis, STAGE_EPOCHS[k])
        sc1_at_stage.append(states[0])
        for mb in range(N_MISS):
            sc2_at_stage[mb].append(states[1 + mb])

    if verbose:
        print("  Reference states back-propagated to all stages (parallel).")

    # Forward-propagate every post-burn state to TCA.
    #
    # The post-burn TCA state is far less varied than the full (stage, miss_bin,
    # joint_action) grid suggests:
    #   - SC1's post-burn trajectory depends only on (stage, burn1) -- not on the
    #     miss bin (which only places SC2) nor on SC2's burn.
    #   - SC2's post-burn trajectory depends only on (stage, miss_bin, burn2).
    # So per stage there are just N_BURN unique SC1 props and N_MISS*N_BURN unique
    # SC2 props (33 for the current model), versus N_MISS*2*N_BURN**2 = 180 if we
    # naively propagated every combination. All forward props share the same target
    # epoch (EPOCH_TCA), so the numerical backend runs them in one parallel call.
    sc1_epochs0, sc1_ecis, sc1_key_list = [], [], []   # keys: (stage, burn1)
    sc2_epochs0, sc2_ecis, sc2_key_list = [], [], []   # keys: (stage, miss_bin, burn2)

    for k in range(N_STAGES):
        for burn1 in range(N_BURN_AGENT):
            sc1_epochs0.append(STAGE_EPOCHS[k])
            sc1_ecis.append(apply_maneuver(sc1_at_stage[k], burn1, dv=dv_magnitude))
            sc1_key_list.append((k, burn1))
        for mb in range(N_MISS):
            for burn2 in range(N_BURN_AGENT):
                sc2_epochs0.append(STAGE_EPOCHS[k])
                sc2_ecis.append(apply_maneuver(sc2_at_stage[mb][k], burn2, dv=dv_magnitude))
                sc2_key_list.append((k, mb, burn2))

    sc1_states = propagate_batch_to(sc1_epochs0, sc1_ecis, EPOCH_TCA)
    sc2_states = propagate_batch_to(sc2_epochs0, sc2_ecis, EPOCH_TCA)

    sc1_tca_prop = dict(zip(sc1_key_list, sc1_states))   # (k, burn1) -> state
    sc2_tca_prop = dict(zip(sc2_key_list, sc2_states))   # (k, mb, burn2) -> state

    if verbose:
        print(f"  Forward-propagated {len(sc1_states) + len(sc2_states)} "
              f"unique post-burn states to TCA [{PROPAGATOR_BACKEND}].")

    # Now compute T, O, R from the propagated states.
    for k in range(N_STAGES):
        for mb in range(N_MISS):
            for dev1 in range(N_DEV):
                for dev2 in range(N_DEV):
                    s = state_index(mb, dev1, dev2, k)

                    for a in range(n_joint_acts):
                        burn1, burn2 = split_joint_action(a)

                        rtn_tca = np.array(state_eci_to_rtn(
                            sc1_tca_prop[(k, burn1)],
                            sc2_tca_prop[(k, mb, burn2)]))
                        rtn_pos_tca_km = rtn_tca[:3] / 1e3
                        any_burn = (burn1 != ACT_WAIT) or (burn2 != ACT_WAIT)

                        r = REWARD_STEP
                        r += REWARD_DEVIATION * ((dev1 != DEV_ZERO) + (dev2 != DEV_ZERO))
                        if burn1 != ACT_WAIT: r += REWARD_MANEUVER
                        if burn2 != ACT_WAIT: r += REWARD_MANEUVER
                        if k == N_STAGES - 1:
                            r += terminal_risk_reward(mb)
                        R[a, s] = r

                        if k < N_STAGES - 1:
                            next_stage = k + 1
                            next_dev1 = update_dev_bin(dev1, burn1)
                            next_dev2 = update_dev_bin(dev2, burn2)
                            sync_next = (
                                variant == "centralized" or
                                (variant == "sdec" and next_stage in CONTACT_STAGES)
                            )
                            next_bin_dist = stochastic_next_bin_dist(
                                rtn_pos_tca_km, any_burn, k
                            )
                            for nb, prob in enumerate(next_bin_dist):
                                if prob > 0.0:
                                    sp = state_index(nb, next_dev1, next_dev2, next_stage)
                                    T[a, s, sp] += prob
                                    if sync_next:
                                        p_joint = perfect_shared_obs_for_state(nb, next_dev1, next_dev2)
                                    else:
                                        p_joint = joint_private_dev_only_obs_distribution(next_dev1, next_dev2)
                                    O[a, sp, :] += prob * p_joint
                        else:
                            T[a, s, SINK_STATE] = 1.0
        if verbose:
            print(f"  ... stage {k}/{N_STAGES - 1} matrices built")

    # Sink state: absorbing
    for a in range(n_joint_acts):
        T[a, SINK_STATE, SINK_STATE] = 1.0
        O[a, SINK_STATE, :] = 1.0 / N_JOINT_OBS
        R[a, SINK_STATE] = 0.0

    # Normalize O rows
    for a in range(n_joint_acts):
        for sp in range(N_STATES_TOTAL):
            row_sum = O[a, sp, :].sum()
            if row_sum < 1e-12:
                O[a, sp, :] = 1.0 / N_JOINT_OBS
            else:
                O[a, sp, :] /= row_sum

    # Initial belief: uniform over the two farthest stage-0 miss bins.
    init_b = np.zeros(N_STATES_TOTAL, dtype=np.float64)
    for mb in [N_MISS - 2, N_MISS - 1]:
        init_b[state_index(mb, DEV_ZERO, DEV_ZERO, 0)] = 1.0
    init_b /= init_b.sum()

    if verbose:
        print(f"  Done. T nonzero: {(T > 0).sum()}, O nonzero: {(O > 0).sum()}")
        print(f"  Initial belief: {(init_b > 0).sum()} states with nonzero prob")

    return T, O, R, init_b


def save_matrices(T, O, R, init_b, variant: str = "sdec",
                  dv_magnitude: float = None):
    if dv_magnitude is None:
        dv_magnitude = DV_MAGNITUDE
    path = cache_path(variant)
    np.savez_compressed(path, T=T, O=O, R=R, init_b=init_b,
                        contact_stages=np.array(variant_sync_stages(variant)),
                        dv_magnitude=np.float64(dv_magnitude),
                        variant=np.array(variant),
                        state_encoding=np.array("miss_dev1_dev2_stage"),
                        action_encoding=np.array("a1_plus_n_a2"),
                        observation_encoding=np.array("o1_plus_n_o2"),
                        local_observation_encoding=np.array("miss_or_null_plus_nmissobs_dev"),
                        reward_model=np.array(REWARD_MODEL_VERSION),
                        terminal_risk_reward_by_bin=REWARD_TERMINAL_RISK_BY_BIN,
                        transition_model=np.array(TRANSITION_MODEL_VERSION),
                        process_drift_sigma_km_per_sqrt_h=np.float64(
                            PROCESS_DRIFT_SIGMA_KM_PER_SQRT_H
                        ),
                        exec_noise_sigma=np.float64(EXEC_NOISE_SIGMA),
                        transition_noise_n_samples=np.int64(
                            TRANSITION_NOISE_N_SAMPLES
                        ),
                        obs_semantics=np.array(
                            "sync_perfect_state__offsync_private_own_dev_only"
                        ))
    print(f"Saved to {path}  (variant={variant}, dv={dv_magnitude} m/s)")

def load_matrices(variant: str = "sdec"):
    path = cache_path(variant)
    data = np.load(path, allow_pickle=True)
    T = data['T']
    O = data['O']
    R = data['R']
    init_b = data['init_b']
    expected = (N_JOINT_ACTIONS, N_STATES_TOTAL, N_STATES_TOTAL)
    if T.shape != expected or O.shape[:2] != expected[:2] or R.shape != expected[:2]:
        raise ValueError(
            f"Stale spacecraft matrix cache at {path}: expected T {expected}, "
            f"O (*,{N_JOINT_OBS}), R {expected[:2]} for the current state encoding; "
            f"got T {T.shape}, O {O.shape}, R {R.shape}. Rebuild with --force."
        )
    if O.shape[2] != N_JOINT_OBS or init_b.shape[0] != N_STATES_TOTAL:
        raise ValueError(
            f"Stale spacecraft matrix cache at {path}: expected {N_JOINT_OBS} "
            f"joint observations and {N_STATES_TOTAL} belief entries; "
            f"got O {O.shape}, init_b {init_b.shape}. Rebuild with --force."
        )
    model_version = str(data['transition_model']) if 'transition_model' in data else ""
    if model_version != TRANSITION_MODEL_VERSION:
        raise ValueError(
            f"Stale spacecraft matrix cache at {path}: expected transition model "
            f"{TRANSITION_MODEL_VERSION!r}, got {model_version!r}. "
            f"Rebuild with --force."
        )
    cached_drift = (
        float(data['process_drift_sigma_km_per_sqrt_h'])
        if 'process_drift_sigma_km_per_sqrt_h' in data
        else None
    )
    if cached_drift is None or abs(cached_drift - PROCESS_DRIFT_SIGMA_KM_PER_SQRT_H) > 1e-12:
        raise ValueError(
            f"Stale spacecraft matrix cache at {path}: expected drift sigma "
            f"{PROCESS_DRIFT_SIGMA_KM_PER_SQRT_H}, got {cached_drift}. "
            f"Rebuild with --force."
        )
    cached_exec = float(data['exec_noise_sigma']) if 'exec_noise_sigma' in data else None
    cached_samples = (
        int(data['transition_noise_n_samples'])
        if 'transition_noise_n_samples' in data
        else None
    )
    if cached_exec is None or abs(cached_exec - EXEC_NOISE_SIGMA) > 1e-12:
        raise ValueError(
            f"Stale spacecraft matrix cache at {path}: expected exec noise "
            f"{EXEC_NOISE_SIGMA}, got {cached_exec}. Rebuild with --force."
        )
    if cached_samples != TRANSITION_NOISE_N_SAMPLES:
        raise ValueError(
            f"Stale spacecraft matrix cache at {path}: expected "
            f"{TRANSITION_NOISE_N_SAMPLES} transition samples, got "
            f"{cached_samples}. Rebuild with --force."
        )
    reward_model = str(data['reward_model']) if 'reward_model' in data else ""
    if reward_model != REWARD_MODEL_VERSION:
        raise ValueError(
            f"Stale spacecraft matrix cache at {path}: expected reward model "
            f"{REWARD_MODEL_VERSION!r}, got {reward_model!r}. "
            f"Rebuild with --force."
        )
    cached_terminal_rewards = (
        np.asarray(data['terminal_risk_reward_by_bin'], dtype=np.float64)
        if 'terminal_risk_reward_by_bin' in data
        else None
    )
    if (
        cached_terminal_rewards is None or
        cached_terminal_rewards.shape != REWARD_TERMINAL_RISK_BY_BIN.shape or
        not np.allclose(cached_terminal_rewards, REWARD_TERMINAL_RISK_BY_BIN)
    ):
        raise ValueError(
            f"Stale spacecraft matrix cache at {path}: terminal risk reward "
            "table changed. Rebuild with --force."
        )
    contact_stages = [int(x) for x in data['contact_stages']]
    dv = float(data['dv_magnitude']) if 'dv_magnitude' in data else DV_MAGNITUDE
    return T, O, R, init_b, contact_stages, dv


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_matrices(T, O, R, init_b, contact_stages):
    print("\n=== Matrix Verification ===")

    # T rows sum to 1
    row_sums = T.sum(axis=2)
    bad_T = np.abs(row_sums - 1.0) > 1e-6
    print(f"  T row sums: min={row_sums.min():.6f} max={row_sums.max():.6f} "
          f"bad_rows={bad_T.sum()}")

    # O rows sum to 1
    row_sums_O = O.sum(axis=2)
    bad_O = np.abs(row_sums_O - 1.0) > 1e-6
    print(f"  O row sums: min={row_sums_O.min():.6f} max={row_sums_O.max():.6f} "
          f"bad_rows={bad_O.sum()}")

    print(f"  init_b sum: {init_b.sum():.6f}")
    print(f"  R range: [{R.min():.1f}, {R.max():.1f}]")

    print("\n  Terminal risk reward by miss bin:")
    for mb, reward in enumerate(REWARD_TERMINAL_RISK_BY_BIN):
        print(f"    bin {mb}: {reward:.1f}")

    # Collision reward must fire for all final-stage bins below 1 km.
    collision_r = []
    collision_bins = collision_bin_indices()
    for a in range(N_JOINT_ACTIONS):
        for mb in collision_bins:
            for dev1 in range(N_DEV):
                for dev2 in range(N_DEV):
                    s = state_index(mb, dev1, dev2, N_STAGES - 1)
                    collision_r.append(R[a, s])
    print(f"\n  Collision states (bins={collision_bins}, final stage) reward range: "
          f"[{min(collision_r):.1f}, {max(collision_r):.1f}]")
    n_collision = sum(1 for r in collision_r if r <= REWARD_COLLISION)
    print(f"  States with R <= {REWARD_COLLISION:.0f}: {n_collision} / {len(collision_r)}")
    assert n_collision > 0, "FAIL: collision reward never fires!"

    high_r = []
    for a in range(N_JOINT_ACTIONS):
        for mb in range(N_MISS):
            if mb in collision_bins:
                continue
            if terminal_risk_reward(mb) >= 0.0:
                continue
            for dev1 in range(N_DEV):
                for dev2 in range(N_DEV):
                    s = state_index(mb, dev1, dev2, N_STAGES - 1)
                    high_r.append(R[a, s])
    if high_r:
        print(f"  Non-collision risk states reward range: "
              f"[{min(high_r):.1f}, {max(high_r):.1f}]")

    # Keep an explicit bin-0 check because this catches accidental threshold drift.
    bin0_r = []
    for a in range(N_JOINT_ACTIONS):
        for dev1 in range(N_DEV):
            for dev2 in range(N_DEV):
                s = state_index(0, dev1, dev2, N_STAGES - 1)
                bin0_r.append(R[a, s])
    assert min(bin0_r) <= REWARD_COLLISION, "FAIL: bin-0 collision reward missing!"

    # Example transitions for a mid-horizon stage â€” check action differentiation
    mid_stage = N_STAGES // 2
    print(f"\n  Example transitions (stage {mid_stage}, miss_bins 0-2):")
    burn_names = {0: 'WAIT', 1: '+dVT', 2: '-dVT'}
    for mb in range(3):
        s = state_index(mb, DEV_ZERO, DEV_ZERO, mid_stage)
        next_bins_per_action = {}
        for a in range(N_JOINT_ACTIONS):
            a1, a2 = split_joint_action(a)
            label = f"({burn_names[a1]},{burn_names[a2]})"
            next_states = np.where(T[a, s, :] > 0)[0]
            bins = {}
            for sp in next_states:
                if sp < N_STATES:
                    next_mb, next_dev1, next_dev2, _ = index_to_state(sp)
                    key = (next_mb, next_dev1, next_dev2)
                    bins[key] = bins.get(key, 0.0) + float(T[a, s, sp])
            next_bins_per_action[label] = [
                (key, round(prob, 3))
                for key, prob in sorted(bins.items(), key=lambda item: item[1], reverse=True)
            ]
        print(f"    miss_bin={mb}, stage={mid_stage} -> next (miss,dev1,dev2): "
              f"{next_bins_per_action}")

    sync_states = sync_trigger_states(contact_stages)
    print(f"\n  Sync trigger states: {len(sync_states)} / {N_STATES_TOTAL}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify",  action="store_true")
    parser.add_argument("--force",   action="store_true")
    parser.add_argument("--dv", type=float, default=None,
                        help="Delta-v magnitude in m/s (default: 0.5)")
    parser.add_argument("--variant", default=None,
                        help="centralized | sdec | dec | all (default: all)")
    args = parser.parse_args()

    initialize_eop()

    variants_to_build = (
        ["centralized", "sdec", "dec"] if args.variant in (None, "all")
        else [args.variant]
    )

    for v in variants_to_build:
        p = cache_path(v)
        if args.verify:
            print(f"Loading [{v}] from cache: {p}")
            T, O, R, init_b, contact_stages, dv = load_matrices(v)
            print(f"  dv_magnitude in cache: {dv} m/s")
        else:
            dv = args.dv if args.dv is not None else DV_MAGNITUDE
            if os.path.exists(p) and not args.force:
                print(f"Cache exists at {p}. Use --force to rebuild.")
                T, O, R, init_b, contact_stages, dv = load_matrices(v)
            else:
                T, O, R, init_b = build_matrices(verbose=True, dv_magnitude=dv, variant=v)
                contact_stages = variant_sync_stages(v)
                save_matrices(T, O, R, init_b, variant=v, dv_magnitude=dv)
        verify_matrices(T, O, R, init_b, contact_stages)
