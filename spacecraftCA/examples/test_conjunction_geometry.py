"""
test_conjunction_geometry.py

Port of genConjunctions.jl logic to Python/brahe.

Approach (same as Julia version):
  1. Define spacecraft ECI state at TCA via orbital elements
  2. Define SC2 (debris/second spacecraft) relative to SC1 at TCA in RTN frame
  3. Propagate both backwards TCA_HOURS to get T=0 initial conditions
  4. Sample RTN relative state over planning horizon -> informs bin design

Run this to understand what RTN ranges we need to cover with our discretization.
"""

import numpy as np
import brahe
from brahe import (
    Epoch, AngleFormat, R_EARTH,
    initialize_eop,
    state_koe_to_eci, state_rtn_to_eci, state_eci_to_rtn,
    NumericalOrbitPropagator, NumericalPropagationConfig, ForceModelConfig,
)


# ---------------------------------------------------------------------------
# Scenario parameters (mirroring Julia POMDP defaults)
# ---------------------------------------------------------------------------

TCA_HOURS = 24.0                   # planning horizon before TCA (hours)
R_ALT     = 550e3                  # orbit altitude (m)
INCL      = 55.0                   # inclination (deg)

# At TCA: SC2 is rMag meters away from SC1, with relative velocity vMag m/s
R_MAG = 500.0                      # miss distance at TCA (m) -- start large for visibility
V_MAG = 15.0                       # relative speed at TCA (m/s) -- typical LEO crossing

CONJUNCTION_TYPE = "crossing"      # "head-on", "overtaking", "crossing"

# SC1 orbital elements at TCA [a, e, i, RAAN, argp, M] degrees
SC1_OE_AT_TCA = np.array([
    R_EARTH + R_ALT,   # semi-major axis (m)
    0.001,             # eccentricity
    INCL,              # inclination (deg)
    20.0,              # RAAN (deg)
    0.0,               # argument of perigee (deg)
    0.0,               # mean anomaly at TCA (deg)
])

EPOCH_TCA = Epoch(2025, 6, 2, 0, 0, 0.0)   # TCA epoch


# ---------------------------------------------------------------------------
# Conjunction generation (ported from genConjunctions.jl)
# ---------------------------------------------------------------------------

def generate_sc1_eci_at_tca(oe_deg: np.ndarray) -> np.ndarray:
    """Convert orbital elements (degrees) to ECI state at TCA."""
    return np.array(state_koe_to_eci(oe_deg, AngleFormat.DEGREES))


def generate_sc2_rtn_at_tca(r_mag: float, v_mag: float,
                              conjunction_type: str, seed: int = 42) -> np.ndarray:
    """
    Generate SC2 relative state in RTN frame at TCA.
    Matches Julia generate_tca_relative logic.
    """
    rng = np.random.default_rng(seed)
    noise_scale = r_mag * 0.01

    # Relative position: offset in RTN depending on conjunction type
    if conjunction_type == "head-on":
        r_rel = np.array([rng.standard_normal()*noise_scale, -r_mag, rng.standard_normal()*noise_scale])
    elif conjunction_type == "overtaking":
        r_rel = np.array([rng.standard_normal()*noise_scale,  r_mag, rng.standard_normal()*noise_scale])
    elif conjunction_type == "crossing":
        r_rel = np.array([rng.standard_normal()*noise_scale, rng.standard_normal()*noise_scale, r_mag])
    else:
        r_rel = r_mag * (2*rng.random(3) - 1)

    # Relative velocity perpendicular to r_rel
    tmp = rng.standard_normal(3)
    v_rel = tmp - (np.dot(tmp, r_rel) / np.dot(r_rel, r_rel)) * r_rel
    v_rel = v_rel / np.linalg.norm(v_rel) * v_mag

    return np.concatenate([r_rel, v_rel])


def make_propagator(epoch: Epoch, eci_state: np.ndarray,
                    two_body: bool = True) -> NumericalOrbitPropagator:
    """Build a NumericalOrbitPropagator from ECI state."""
    prop_config = NumericalPropagationConfig.default()
    force_config = ForceModelConfig.two_body() if two_body else ForceModelConfig.default()
    return NumericalOrbitPropagator(epoch, eci_state, prop_config, force_config)


def generate_conjunction(tca_hours: float = TCA_HOURS,
                         r_mag: float = R_MAG,
                         v_mag: float = V_MAG,
                         conjunction_type: str = CONJUNCTION_TYPE,
                         seed: int = 42,
                         two_body: bool = True):
    """
    Generate initial conditions for a conjunction scenario.

    Returns:
        epoch_start:   Epoch at T=0 (tca_hours before TCA)
        epoch_tca:     Epoch at TCA
        sc1_eci_t0:    SC1 ECI state at T=0 [6,] m, m/s
        sc2_eci_t0:    SC2 ECI state at T=0 [6,] m, m/s
        sc1_eci_tca:   SC1 ECI state at TCA [6,] m, m/s
        sc2_eci_tca:   SC2 ECI state at TCA [6,] m, m/s
    """
    # SC1 at TCA
    sc1_eci_tca = generate_sc1_eci_at_tca(SC1_OE_AT_TCA)

    # SC2 relative to SC1 at TCA in RTN, then convert to absolute ECI
    sc2_rtn_at_tca = generate_sc2_rtn_at_tca(r_mag, v_mag, conjunction_type, seed)
    sc2_eci_tca = np.array(state_rtn_to_eci(sc1_eci_tca, sc2_rtn_at_tca))

    # Build propagators at TCA, then propagate backwards to T=0
    prop1 = make_propagator(EPOCH_TCA, sc1_eci_tca, two_body)
    prop2 = make_propagator(EPOCH_TCA, sc2_eci_tca, two_body)

    epoch_start = EPOCH_TCA - tca_hours * 3600.0
    prop1.propagate_to(epoch_start)
    prop2.propagate_to(epoch_start)

    sc1_eci_t0 = np.array(prop1.current_state()[:6])
    sc2_eci_t0 = np.array(prop2.current_state()[:6])

    return epoch_start, EPOCH_TCA, sc1_eci_t0, sc2_eci_t0, sc1_eci_tca, sc2_eci_tca


# ---------------------------------------------------------------------------
# Sample RTN trajectory over planning horizon
# ---------------------------------------------------------------------------

def sample_rtn_trajectory(epoch_start: Epoch, sc1_eci_t0: np.ndarray,
                           sc2_eci_t0: np.ndarray, n_samples: int = 49,
                           two_body: bool = True):
    """
    Propagate both spacecraft from T=0 to TCA and sample RTN relative state.
    Returns array of shape (n_samples, 7): [t_hours, dR, dT, dN, dVR, dVT, dVN]
    """
    prop1 = make_propagator(epoch_start, sc1_eci_t0, two_body)
    prop2 = make_propagator(epoch_start, sc2_eci_t0, two_body)

    results = []
    for i in range(n_samples):
        t_hours = TCA_HOURS * i / (n_samples - 1)
        t = epoch_start + t_hours * 3600.0

        prop1.propagate_to(t)
        prop2.propagate_to(t)

        s1 = np.array(prop1.current_state()[:6])
        s2 = np.array(prop2.current_state()[:6])

        rtn = np.array(state_eci_to_rtn(s1, s2))  # [dR, dT, dN, dVR, dVT, dVN]
        results.append([t_hours] + list(rtn / 1e3))  # convert m -> km, m/s -> km/s

    return np.array(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    initialize_eop()

    print("=" * 70)
    print("Conjunction Geometry Analysis")
    print("=" * 70)
    print(f"TCA horizon: {TCA_HOURS:.0f} hours")
    print(f"Conjunction type: {CONJUNCTION_TYPE}")
    print(f"Miss distance at TCA: {R_MAG:.0f} m")
    print(f"Relative speed at TCA: {V_MAG:.1f} m/s")

    epoch_start, epoch_tca, sc1_t0, sc2_t0, sc1_tca, sc2_tca = generate_conjunction()

    # Verify TCA geometry
    rtn_tca = np.array(state_eci_to_rtn(sc1_tca, sc2_tca))
    print(f"\nVerification at TCA:")
    print(f"  RTN position: R={rtn_tca[0]:.1f}m  T={rtn_tca[1]:.1f}m  N={rtn_tca[2]:.1f}m")
    print(f"  Miss distance: {np.linalg.norm(rtn_tca[:3]):.1f} m")
    print(f"  Relative speed: {np.linalg.norm(rtn_tca[3:]):.2f} m/s")

    # RTN at T=0
    rtn_t0 = np.array(state_eci_to_rtn(sc1_t0, sc2_t0))
    print(f"\nInitial state (T=0, {TCA_HOURS:.0f}h before TCA):")
    print(f"  RTN position: R={rtn_t0[0]/1e3:.2f}km  T={rtn_t0[1]/1e3:.2f}km  N={rtn_t0[2]/1e3:.2f}km")
    print(f"  Range: {np.linalg.norm(rtn_t0[:3])/1e3:.2f} km")

    # Sample trajectory
    print(f"\nRTN trajectory (every 2 hours):")
    print(f"  {'T(h)':<8} {'dR(km)':<10} {'dT(km)':<10} {'dN(km)':<10} {'range(km)':<12}")
    traj = sample_rtn_trajectory(epoch_start, sc1_t0, sc2_t0, n_samples=49)
    for row in traj[::4]:  # every 2 hours (48 steps total, step 4 = 2h)
        t_h, dR, dT, dN = row[0], row[1], row[2], row[3]
        rng = np.sqrt(dR**2 + dT**2 + dN**2)
        print(f"  {t_h:<8.1f} {dR:<10.3f} {dT:<10.3f} {dN:<10.3f} {rng:<12.3f}")

    # Summary statistics for bin design
    print(f"\nRange statistics over full trajectory:")
    print(f"  dR: [{traj[:,1].min():.2f}, {traj[:,1].max():.2f}] km")
    print(f"  dT: [{traj[:,2].min():.2f}, {traj[:,2].max():.2f}] km")
    print(f"  dN: [{traj[:,3].min():.2f}, {traj[:,3].max():.2f}] km")
    print(f"  range: [{np.sqrt((traj[:,1:4]**2).sum(axis=1)).min():.3f}, "
          f"{np.sqrt((traj[:,1:4]**2).sum(axis=1)).max():.2f}] km")


if __name__ == "__main__":
    main()
