"""
spacecraft_matrices.py

Builds T, O, R matrices for the spacecraft CA SDec-POMDP (miss-distance state).
State = (miss_distance_bin, stage). Brahe propagates to TCA offline.

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

from brahe import (
    Epoch, AngleFormat, R_EARTH,
    initialize_eop,
    state_koe_to_eci, state_rtn_to_eci, state_eci_to_rtn,
    NumericalOrbitPropagator, NumericalPropagationConfig, ForceModelConfig,
    par_propagate_to,
)
from scipy.special import erf

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from spacecraft_discretizer import (
    N_MISS, N_STAGES, N_STATES, N_STATES_TOTAL, SINK_STATE,
    MISS_EDGES_KM,
    miss_to_bin, bin_center_km,
    state_index, index_to_state,
    sync_trigger_states,
)


# ---------------------------------------------------------------------------
# Problem constants
# ---------------------------------------------------------------------------

N_BURN_AGENT    = 3                    # WAIT, +dV_T, -dV_T
N_OBS_AGENT     = N_MISS              # 6 observation bins per agent
N_JOINT_OBS     = N_OBS_AGENT ** 2    # 36

# All three variants use the same 9-action space (3 burns per agent).
# Sync is state-triggered in RSSDA via sync_states — not encoded in actions.
# TODO(future): when Mahdi adds joint state+action conditional sync to RS-SDA*,
#   re-introduce per-agent sync_flag here (N_ACT_AGENT_SDEC = N_BURN_AGENT * 2 = 6)
#   and wire sync_actions into SDecPOMDPModel. Until then, sync is purely state-based.
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

# Indices of stages that are GS contact windows (sync/centralization triggers)
CONTACT_STAGES = [i for i, h in enumerate(_ALL_TIMES_H) if h in _GS_TIMES_H]

# Reward constants
REWARD_COLLISION = -10000.0  # bin 0: <1km at TCA
REWARD_HIGH      =  -1000.0  # bin 1: 1-5km at TCA
REWARD_MOD       =   -100.0  # bin 2: 5-20km at TCA
REWARD_MANEUVER  = -10.0
REWARD_STEP      =  0.0

# Observation noise model: GPS-own / TLE-other
# Each agent knows its own position to GPS precision (~100m) — negligible.
# Uncertainty is driven by TLE precision on the opposing spacecraft (~3-5 km 1-sigma).
# OBS_SIGMA_KM is used for symmetric fallback; per-agent obs uses TLE_SIGMA_KM.
OBS_SIGMA_KM  = 5.0   # legacy symmetric sigma (kept for greedy/fallback)
TLE_SIGMA_KM  = 3.0   # 1-sigma TLE position uncertainty used in asymmetric obs model
GPS_SIGMA_KM  = 0.1   # 1-sigma GPS position uncertainty (own spacecraft) — near-negligible

# Stochastic transition: burn execution noise ε ~ N(0, EXEC_NOISE_SIGMA²)
# Applied multiplicatively: miss_noisy = miss_deterministic * (1 + ε)
# Only applies when at least one agent maneuvers (WAIT is deterministic).
EXEC_NOISE_SIGMA = 0.50   # ±50% execution noise (1-sigma)
EXEC_NOISE_N_SAMPLES = 50 # number of ε samples to average over

CACHE_PATH_TEMPLATE = os.path.join(_HERE, "spacecraft_matrices_cache_{variant}.npz")

def cache_path(variant: str) -> str:
    return CACHE_PATH_TEMPLATE.format(variant=variant)

CACHE_PATH = cache_path("sdec")  # legacy default

ACT_WAIT = 0
ACT_POS  = 1   # +dV_T prograde
ACT_NEG  = 2   # -dV_T retrograde

# Joint action encoding: a = a1 * N_ACT_AGENT + a2
# a1 = a // N_ACT_AGENT, a2 = a % N_ACT_AGENT  (each 0=WAIT, 1=POS, 2=NEG)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_joint_action(a: int):
    """Returns (a1, a2) per-agent burn actions (each 0=WAIT, 1=POS, 2=NEG)."""
    return a // N_ACT_AGENT, a % N_ACT_AGENT

def make_prop(epoch: Epoch, eci: np.ndarray) -> NumericalOrbitPropagator:
    return NumericalOrbitPropagator(
        epoch, eci,
        NumericalPropagationConfig.default(),
        ForceModelConfig.two_body()
    )

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
    prop = make_prop(epoch_start, eci)
    prop.propagate_to(epoch_end)
    return np.array(prop.current_state()[:6])

def sc1_eci_at_tca() -> np.ndarray:
    return np.array(state_koe_to_eci(SC1_OE_AT_TCA, AngleFormat.DEGREES))

def place_sc2_at_tca(miss_bin: int) -> np.ndarray:
    """Place SC2 at TCA using the bin center miss distance."""
    return place_sc2_at_tca_km(bin_center_km(miss_bin))

def place_sc2_at_tca_km(miss_km: float) -> np.ndarray:
    """
    Place SC2 at TCA with a specific miss distance in RTN:
      δr_RTN = [miss_km*1e3, 0, 0]  (radial offset = miss distance)
      δv_RTN = [0, -V_REL_MS, 0]    (along-track closing speed)
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
    Probability over N_OBS_AGENT miss-distance bins for a given true miss distance.
    Gaussian noise: N(miss_km, sigma^2), integrated over MISS_EDGES_KM bins.
    Defaults to OBS_SIGMA_KM if sigma not specified.
    """
    if sigma is None:
        sigma = OBS_SIGMA_KM
    edges = MISS_EDGES_KM   # [0, 1, 5, 20, 100, 500, inf]
    probs = np.zeros(N_OBS_AGENT)
    for i in range(N_OBS_AGENT):
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

# Fixed epsilon samples drawn once — reproducible across builds.
_RNG_EXEC = np.random.default_rng(0)
_EXEC_EPS = _RNG_EXEC.normal(0.0, EXEC_NOISE_SIGMA, EXEC_NOISE_N_SAMPLES)

def stochastic_next_bin_dist(miss_tca_km: float, any_burn: bool) -> np.ndarray:
    """
    Return probability distribution over N_MISS bins for the next miss distance.
    If any_burn is False (pure WAIT), transition is deterministic (no execution noise).
    Otherwise samples miss_tca_km * (1 + ε) for each pre-drawn ε and averages bins.
    """
    if not any_burn:
        dist = np.zeros(N_MISS, dtype=np.float64)
        dist[miss_to_bin(miss_tca_km)] = 1.0
        return dist
    counts = np.zeros(N_MISS, dtype=np.float64)
    for eps in _EXEC_EPS:
        noisy = max(0.0, miss_tca_km * (1.0 + eps))
        counts[miss_to_bin(noisy)] += 1.0
    return counts / counts.sum()


# ---------------------------------------------------------------------------
# Matrix generation
# ---------------------------------------------------------------------------

def build_matrices(verbose: bool = True, dv_magnitude: float = None,
                   variant: str = "sdec"):
    """
    Build T, O, R, init_b for the spacecraft CA SDec-POMDP.

    variant: "centralized" | "sdec" | "dec"
      centralized: 9 actions (burns), GPS obs forced at contacts
      sdec:       18 actions (burn × sync), GPS obs if both choose sync at contact
      dec:         9 actions (burns), TLE obs at contacts (no GPS sharing)

    Returns T, O, R, init_b (numpy float64).
    """
    if dv_magnitude is None:
        dv_magnitude = DV_MAGNITUDE
    assert variant in ("centralized", "sdec", "dec"), f"Unknown variant: {variant}"

    n_act_agent  = N_ACT_AGENT
    n_joint_acts = N_JOINT_ACTIONS

    if verbose:
        print(f"Building spacecraft CA matrices [{variant}] (miss-distance state)...")
        print(f"  States: {N_STATES_TOTAL} ({N_STATES} + 1 sink), "
              f"Actions: {n_joint_acts}, Obs: {N_JOINT_OBS}")
        print(f"  dv_magnitude = {dv_magnitude} m/s, exec_noise_sigma = {EXEC_NOISE_SIGMA}")

    T = np.zeros((n_joint_acts, N_STATES_TOTAL, N_STATES_TOTAL), dtype=np.float64)
    O = np.zeros((n_joint_acts, N_STATES_TOTAL, N_JOINT_OBS),    dtype=np.float64)
    R = np.zeros((n_joint_acts, N_STATES_TOTAL),                  dtype=np.float64)

    # Back-propagate SC1 and SC2 reference states from TCA to each stage epoch.
    sc1_tca = sc1_eci_at_tca()
    sc2_tca_per_bin = [place_sc2_at_tca(mb) for mb in range(N_MISS)]

    # Batch back-propagation: propagate all (SC1 + N_MISS SC2s) to each stage at once.
    sc1_at_stage = []
    sc2_at_stage = [[] for _ in range(N_MISS)]
    for k in range(N_STAGES):
        props = [make_prop(EPOCH_TCA, sc1_tca)] + \
                [make_prop(EPOCH_TCA, sc2_tca_per_bin[mb]) for mb in range(N_MISS)]
        par_propagate_to(props, STAGE_EPOCHS[k])
        sc1_at_stage.append(np.array(props[0].current_state()[:6]))
        for mb in range(N_MISS):
            sc2_at_stage[mb].append(np.array(props[1 + mb].current_state()[:6]))

    if verbose:
        print("  Reference states back-propagated to all stages (parallel).")

    # Process one stage at a time, batching all forward propagations with par_propagate_to.
    for k in range(N_STAGES):
        # For each (miss_bin, joint_action) we need sc1_post and sc2_post propagated to TCA.
        # Also need sc1_no_burn and sc2_no_burn per miss_bin for the obs model.
        # Total propagators per stage: N_MISS * (2*N_JOINT_ACTIONS + 2) forward props.

        # Build propagator list and an index map to retrieve results.
        prop_list = []
        # index_map entries: (miss_bin, action_or_None, 'sc1'/'sc2'/'sc1nb'/'sc2nb')
        index_map = []

        for mb in range(N_MISS):
            sc1_eci = sc1_at_stage[k]
            sc2_eci = sc2_at_stage[mb][k]

            # No-burn propagators for asymmetric obs model (one pair per miss_bin)
            prop_list.append(make_prop(STAGE_EPOCHS[k], sc1_eci))
            index_map.append((mb, None, None, 'sc1nb'))
            prop_list.append(make_prop(STAGE_EPOCHS[k], sc2_eci))
            index_map.append((mb, None, None, 'sc2nb'))

            # Propagate unique burn combos only (9, not 36) — sync flag is irrelevant for dynamics
            for burn1 in range(N_BURN_AGENT):
                for burn2 in range(N_BURN_AGENT):
                    prop_list.append(make_prop(STAGE_EPOCHS[k], apply_maneuver(sc1_eci, burn1, dv=dv_magnitude)))
                    index_map.append((mb, burn1, burn2, 'sc1'))
                    prop_list.append(make_prop(STAGE_EPOCHS[k], apply_maneuver(sc2_eci, burn2, dv=dv_magnitude)))
                    index_map.append((mb, burn1, burn2, 'sc2'))

        # Single parallel propagation call for all propagators at this stage
        par_propagate_to(prop_list, EPOCH_TCA)

        # Unpack results into lookup dicts
        sc1_tca_prop  = {}   # (mb, burn1, burn2) -> state
        sc2_tca_prop  = {}
        sc1_no_burn   = {}   # mb -> state
        sc2_no_burn   = {}

        for idx, (mb, burn1, burn2, role) in enumerate(index_map):
            state = np.array(prop_list[idx].current_state()[:6])
            if role == 'sc1nb':
                sc1_no_burn[mb] = state
            elif role == 'sc2nb':
                sc2_no_burn[mb] = state
            elif role == 'sc1':
                sc1_tca_prop[(mb, burn1, burn2)] = state
            else:
                sc2_tca_prop[(mb, burn1, burn2)] = state

        # Now compute T, O, R from the propagated states
        for mb in range(N_MISS):
            s = state_index(mb, k)

            for a in range(n_joint_acts):
                burn1, burn2 = split_joint_action(a)

                rtn_tca = np.array(state_eci_to_rtn(
                    sc1_tca_prop[(mb, burn1, burn2)],
                    sc2_tca_prop[(mb, burn1, burn2)]))
                miss_tca_km = np.linalg.norm(rtn_tca[:3]) / 1e3
                any_burn = (burn1 != ACT_WAIT) or (burn2 != ACT_WAIT)
                at_contact = (k + 1) in CONTACT_STAGES

                # Reward
                r = REWARD_STEP
                if burn1 != ACT_WAIT: r += REWARD_MANEUVER
                if burn2 != ACT_WAIT: r += REWARD_MANEUVER
                if k == N_STAGES - 1:
                    if mb == 0: r += REWARD_COLLISION
                    elif mb == 1: r += REWARD_HIGH
                    elif mb == 2: r += REWARD_MOD
                R[a, s] = r

                # Observation model at next stage
                if k < N_STAGES - 1:
                    next_bin_dist = stochastic_next_bin_dist(miss_tca_km, any_burn)
                    for nb, prob in enumerate(next_bin_dist):
                        if prob > 0.0:
                            T[a, s, state_index(nb, k + 1)] += prob

                    if (variant in ("centralized", "sdec")) and at_contact:
                        # Sync contact: agents share one GPS measurement → identical obs.
                        # Joint obs distribution is diagonal: P(o1=o, o2=o) = p_GPS[o].
                        # This ensures both agents end up with the same belief after sync.
                        p_gps = obs_distribution(miss_tca_km, sigma=GPS_SIGMA_KM)
                        p_joint = np.diag(p_gps).flatten()
                    elif variant == "dec" and at_contact:
                        # Dec: independent TLE obs per agent (ground catalog, no sharing)
                        p1 = obs_distribution(miss_tca_km, sigma=TLE_SIGMA_KM)
                        p2 = obs_distribution(miss_tca_km, sigma=TLE_SIGMA_KM)
                        p_joint = np.outer(p1, p2).flatten()
                    else:
                        # Non-contact: asymmetric TLE obs (own burn known, other SC via TLE)
                        rtn_sc1_view = np.array(state_eci_to_rtn(
                            sc1_tca_prop[(mb, burn1, burn2)], sc2_no_burn[mb]))
                        miss_sc1_view = np.linalg.norm(rtn_sc1_view[:3]) / 1e3
                        rtn_sc2_view = np.array(state_eci_to_rtn(
                            sc1_no_burn[mb], sc2_tca_prop[(mb, burn1, burn2)]))
                        miss_sc2_view = np.linalg.norm(rtn_sc2_view[:3]) / 1e3
                        p1 = obs_distribution(miss_sc1_view, sigma=TLE_SIGMA_KM)
                        p2 = obs_distribution(miss_sc2_view, sigma=TLE_SIGMA_KM)
                        p_joint = np.outer(p1, p2).flatten()

                    for nb, prob in enumerate(next_bin_dist):
                        if prob > 0.0:
                            O[a, state_index(nb, k + 1), :] += prob * p_joint
                else:
                    T[a, s, SINK_STATE] = 1.0

        if verbose:
            print(f"  ... stage {k}/{N_STAGES - 1} done  "
                  f"({N_MISS * (2 * N_BURN_AGENT**2 + 2)} props batched)")

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

    # Initial belief: uniform over stage-0 states with miss bin 4-5 (far/safe)
    init_b = np.zeros(N_STATES_TOTAL, dtype=np.float64)
    for mb in [4, 5]:
        init_b[state_index(mb, 0)] = 1.0
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
                        contact_stages=np.array(CONTACT_STAGES),
                        dv_magnitude=np.float64(dv_magnitude),
                        variant=np.array(variant))
    print(f"Saved to {path}  (variant={variant}, dv={dv_magnitude} m/s)")

def load_matrices(variant: str = "sdec"):
    path = cache_path(variant)
    data = np.load(path, allow_pickle=True)
    contact_stages = list(data['contact_stages'])
    dv = float(data['dv_magnitude']) if 'dv_magnitude' in data else DV_MAGNITUDE
    return data['T'], data['O'], data['R'], data['init_b'], contact_stages, dv


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

    # Collision reward must fire for (miss_bin=0, stage=5) states
    collision_r = []
    for a in range(N_JOINT_ACTIONS):
        for mb in range(N_MISS):
            for k in range(N_STAGES):
                s = state_index(mb, k)
                if mb == 0 and k == N_STAGES - 1:
                    collision_r.append(R[a, s])
    print(f"\n  Collision states (bin=0, final stage) reward range: "
          f"[{min(collision_r):.1f}, {max(collision_r):.1f}]")
    n_collision = sum(1 for r in collision_r if r <= -9999)
    print(f"  States with R <= -9999: {n_collision} / {len(collision_r)}")
    assert n_collision > 0, "FAIL: collision reward never fires!"

    # Example transitions for a mid-horizon stage — check action differentiation
    mid_stage = N_STAGES // 2
    print(f"\n  Example transitions (stage {mid_stage}, miss_bins 0-2):")
    burn_names = {0: 'WAIT', 1: '+dVT', 2: '-dVT'}
    for mb in range(3):
        s = state_index(mb, mid_stage)
        next_bins_per_action = {}
        for a in range(N_JOINT_ACTIONS):
            a1, a2 = split_joint_action(a)
            label = f"({burn_names[a1]},{burn_names[a2]})"
            next_states = np.where(T[a, s, :] > 0)[0]
            for sp in next_states:
                if sp < N_STATES:
                    next_mb, _ = index_to_state(sp)
                    next_bins_per_action[label] = next_mb
        print(f"    miss_bin={mb}, stage={mid_stage} -> next miss bins: {next_bins_per_action}")

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
                contact_stages = CONTACT_STAGES
                save_matrices(T, O, R, init_b, variant=v, dv_magnitude=dv)
        verify_matrices(T, O, R, init_b, contact_stages)
