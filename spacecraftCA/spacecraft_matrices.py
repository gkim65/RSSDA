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

from brahe import (
    Epoch, AngleFormat, R_EARTH,
    initialize_eop,
    state_koe_to_eci, state_rtn_to_eci, state_eci_to_rtn,
    NumericalOrbitPropagator, NumericalPropagationConfig, ForceModelConfig,
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

N_ACT_AGENT     = 3                    # WAIT, +dV_T, -dV_T
N_OBS_AGENT     = N_MISS              # 6 observation bins per agent (same as miss bins)
N_JOINT_ACTIONS = N_ACT_AGENT ** 2    # 9
N_JOINT_OBS     = N_OBS_AGENT ** 2    # 36

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

STAGE_T_BEFORE_TCA_SEC = [
    23.39 * 3600,
     9.48 * 3600,
     7.83 * 3600,
     6.17 * 3600,
     2.80 * 3600,
     1.13 * 3600,
]
STAGE_EPOCHS = [EPOCH_TCA - dt for dt in STAGE_T_BEFORE_TCA_SEC]

CONTACT_STAGES = list(range(N_STAGES))

# Reward constants
REWARD_COLLISION = -1000.0
REWARD_MANEUVER  = -10.0
REWARD_STEP      =  0.0

# Observation noise (sigma in km, integrated over miss distance bins)
OBS_SIGMA_KM = 5.0

CACHE_PATH = os.path.join(_HERE, "spacecraft_matrices_cache.npz")

ACT_WAIT = 0
ACT_POS  = 1   # +dV_T prograde
ACT_NEG  = 2   # -dV_T retrograde


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_joint_action(a: int):
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

def obs_distribution(miss_km: float) -> np.ndarray:
    """
    Probability over N_OBS_AGENT miss-distance bins for a given true miss distance.
    Gaussian noise: N(miss_km, OBS_SIGMA_KM^2), integrated over MISS_EDGES_KM bins.
    """
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
# Matrix generation
# ---------------------------------------------------------------------------

def build_matrices(verbose: bool = True, dv_magnitude: float = None):
    """
    Build T, O, R, init_b for the spacecraft CA SDec-POMDP.

    Args:
      dv_magnitude: delta-v per maneuver in m/s. Defaults to DV_MAGNITUDE (0.5 m/s).

    Returns T, O, R, init_b (numpy float64).
      T shape: (N_JOINT_ACTIONS, N_STATES_TOTAL, N_STATES_TOTAL)
      O shape: (N_JOINT_ACTIONS, N_STATES_TOTAL, N_JOINT_OBS)
      R shape: (N_JOINT_ACTIONS, N_STATES_TOTAL)
    """
    if dv_magnitude is None:
        dv_magnitude = DV_MAGNITUDE
    if verbose:
        print("Building spacecraft CA matrices (miss-distance state)...")
        print(f"  States: {N_STATES_TOTAL} ({N_STATES} + 1 sink), "
              f"Actions: {N_JOINT_ACTIONS}, Obs: {N_JOINT_OBS}")
        print(f"  dv_magnitude = {dv_magnitude} m/s")

    T = np.zeros((N_JOINT_ACTIONS, N_STATES_TOTAL, N_STATES_TOTAL), dtype=np.float64)
    O = np.zeros((N_JOINT_ACTIONS, N_STATES_TOTAL, N_JOINT_OBS),    dtype=np.float64)
    R = np.zeros((N_JOINT_ACTIONS, N_STATES_TOTAL),                  dtype=np.float64)

    # Pre-compute SC1 and SC2 ECI at each stage epoch by back-propagating from TCA.
    # SC2 is placed at TCA with the target miss distance in RTN, then back-propagated.
    # This ensures the WAIT trajectory produces the correct miss at TCA.
    sc1_at_stage = []
    for k in range(N_STAGES):
        sc1_k = propagate(EPOCH_TCA, sc1_eci_at_tca(), STAGE_EPOCHS[k])
        sc1_at_stage.append(sc1_k)

    if verbose:
        print("  SC1/SC2 reference states computed (back-propagated from TCA).")

    # SC2 at TCA for each miss bin (back-propagated to each stage inside the loop)
    sc2_tca_per_bin = [place_sc2_at_tca(mb) for mb in range(N_MISS)]

    for s in range(N_STATES):
        miss_bin, k = index_to_state(s)

        sc1_eci = sc1_at_stage[k]
        sc2_eci = back_prop_sc2_to_stage(sc2_tca_per_bin[miss_bin], k)

        for a in range(N_JOINT_ACTIONS):
            a1, a2 = split_joint_action(a)

            sc1_post = apply_maneuver(sc1_eci, a1, dv=dv_magnitude)
            sc2_post = apply_maneuver(sc2_eci, a2, dv=dv_magnitude)

            sc1_tca_prop = propagate(STAGE_EPOCHS[k], sc1_post, EPOCH_TCA)
            sc2_tca_prop = propagate(STAGE_EPOCHS[k], sc2_post, EPOCH_TCA)

            rtn_tca = np.array(state_eci_to_rtn(sc1_tca_prop, sc2_tca_prop))
            miss_tca_km = np.linalg.norm(rtn_tca[:3]) / 1e3
            next_miss_bin = miss_to_bin(miss_tca_km)

            # --- Reward ---
            r = REWARD_STEP
            if a1 != ACT_WAIT:
                r += REWARD_MANEUVER
            if a2 != ACT_WAIT:
                r += REWARD_MANEUVER
            # Collision penalty at terminal stage: full for bin-0, partial for bin-1
            if miss_bin == 0 and k == N_STAGES - 1:
                r += REWARD_COLLISION
            elif miss_bin == 1 and k == N_STAGES - 1:
                r += -200.0
            R[a, s] = r

            # --- Transition ---
            if k < N_STAGES - 1:
                s_next = state_index(next_miss_bin, k + 1)
                T[a, s, s_next] = 1.0

                p_obs = obs_distribution(miss_tca_km)
                p_joint = np.outer(p_obs, p_obs).flatten()
                O[a, s_next, :] = p_joint
            else:
                T[a, s, SINK_STATE] = 1.0

        if verbose and (s + 1) % 6 == 0:
            print(f"  ... {s + 1}/{N_STATES} states processed (stage {k})")

    # Sink state: absorbing
    for a in range(N_JOINT_ACTIONS):
        T[a, SINK_STATE, SINK_STATE] = 1.0
        O[a, SINK_STATE, :] = 1.0 / N_JOINT_OBS
        R[a, SINK_STATE] = 0.0

    # Normalize O rows (some s_next may have been set multiple times with same value;
    # rows that were never set get uniform)
    for a in range(N_JOINT_ACTIONS):
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


def save_matrices(T, O, R, init_b, path: str = CACHE_PATH, dv_magnitude: float = None):
    if dv_magnitude is None:
        dv_magnitude = DV_MAGNITUDE
    np.savez_compressed(path, T=T, O=O, R=R, init_b=init_b,
                        contact_stages=np.array(CONTACT_STAGES),
                        dv_magnitude=np.float64(dv_magnitude))
    print(f"Saved to {path}  (dv={dv_magnitude} m/s)")

def load_matrices(path: str = CACHE_PATH):
    data = np.load(path)
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
    print(f"\n  Collision states (bin=0, stage=5) reward range: "
          f"[{min(collision_r):.1f}, {max(collision_r):.1f}]")
    n_collision = sum(1 for r in collision_r if r <= -999)
    print(f"  States with R <= -999: {n_collision} / {len(collision_r)}")
    assert n_collision > 0, "FAIL: collision reward never fires!"

    # Example transitions for stage 4 states — check action differentiation
    print("\n  Example transitions (stage 4, miss_bins 0-2):")
    act_names = {0: 'WAIT', 1: '+dVT', 2: '-dVT'}
    for mb in range(3):
        s = state_index(mb, 4)
        next_bins_per_action = {}
        for a in range(N_JOINT_ACTIONS):
            a1, a2 = split_joint_action(a)
            next_states = np.where(T[a, s, :] > 0)[0]
            for sp in next_states:
                if sp < N_STATES:
                    next_mb, _ = index_to_state(sp)
                    next_bins_per_action[f"({act_names[a1]},{act_names[a2]})"] = next_mb
        print(f"    miss_bin={mb}, stage=4 -> next miss bins: {next_bins_per_action}")

    sync_states = sync_trigger_states(contact_stages)
    print(f"\n  Sync trigger states: {len(sync_states)} / {N_STATES_TOTAL}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--force",  action="store_true")
    parser.add_argument("--dv", type=float, default=None,
                        help="Delta-v magnitude in m/s (default: 0.5)")
    args = parser.parse_args()

    initialize_eop()

    if args.verify:
        print(f"Loading from cache: {CACHE_PATH}")
        T, O, R, init_b, contact_stages, dv = load_matrices()
        print(f"  dv_magnitude in cache: {dv} m/s")
    else:
        dv = args.dv if args.dv is not None else DV_MAGNITUDE
        if os.path.exists(CACHE_PATH) and not args.force:
            print(f"Cache exists at {CACHE_PATH}. Use --force to rebuild.")
            T, O, R, init_b, contact_stages, dv = load_matrices()
        else:
            T, O, R, init_b = build_matrices(verbose=True, dv_magnitude=dv)
            contact_stages = CONTACT_STAGES
            save_matrices(T, O, R, init_b, dv_magnitude=dv)

    verify_matrices(T, O, R, init_b, contact_stages)
