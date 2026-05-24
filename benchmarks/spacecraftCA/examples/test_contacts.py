"""
test_contacts.py

Step 1: Compute ground station contact windows for two spacecraft
over a 24-hour planning horizon before TCA.

Scenario: cooperative shared operator
- 1 ground station at mid-latitude
- 2 spacecraft in similar LEO orbits approaching conjunction
- TCA ~24 hours from scenario start

Prints contact windows for each spacecraft and identifies
which epochs both spacecraft have simultaneous contact
(these become synchronization triggers in RSSDA).
"""

import numpy as np
import brahe
from brahe import (
    Epoch, AngleFormat, R_EARTH,
    location_accesses, ElevationConstraint,
    PointLocation, KeplerianPropagator,
    AccessSearchConfig,
    initialize_eop,
)


# ---------------------------------------------------------------------------
# Scenario parameters
# ---------------------------------------------------------------------------

# Planning start epoch
EPOCH_START = Epoch(2025, 6, 1, 0, 0, 0.0)

# TCA is 24 hours later
TCA_OFFSET_SEC = 24 * 3600.0
EPOCH_TCA = EPOCH_START + TCA_OFFSET_SEC

# Ground station: mid-latitude, roughly Goddard Space Flight Center
GS_LON = -76.8   # degrees
GS_LAT =  39.0   # degrees
GS_ALT =  0.0    # meters
GS_MIN_ELEVATION_DEG = 10.0

# Spacecraft orbital elements [a, e, i, RAAN, omega, M] at EPOCH_START
# Both in circular-ish LEO ~550 km, 55 deg inclination
# SC2 slightly offset in mean anomaly to create a conjunction ~24h later
ALT_KM = 550.0
A = R_EARTH + ALT_KM * 1e3   # semi-major axis (m)

SC1_OE = np.array([A, 0.001, 55.0,  20.0, 0.0,   0.0])   # [a, e, i, RAAN, omega, M] degrees
SC2_OE = np.array([A, 0.001, 55.0,  20.0, 0.0,   0.5])   # slightly offset mean anomaly


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def build_propagator(oe_deg: np.ndarray, epoch: Epoch) -> KeplerianPropagator:
    """Build a Keplerian propagator from orbital elements (degrees)."""
    return KeplerianPropagator.from_keplerian(epoch, oe_deg, AngleFormat.DEGREES, step_size=60.0)


def compute_contact_windows(propagator: KeplerianPropagator, gs: PointLocation,
                             t_start: Epoch, t_end: Epoch, label: str):
    """Compute and print contact windows for one spacecraft."""
    constraint = ElevationConstraint(GS_MIN_ELEVATION_DEG)
    config = AccessSearchConfig(initial_time_step=30.0)

    windows = location_accesses(gs, propagator, t_start, t_end, constraint, config=config)

    print(f"\n{label} contact windows ({len(windows)} passes):")
    print(f"  {'#':<4} {'Open (UTC)':<30} {'Close (UTC)':<30} {'Duration (s)':<14} {'Max El (deg)'}")
    for i, w in enumerate(windows):
        duration = w.duration
        max_el = f"{w.elevation_max:.1f}" if w.elevation_max is not None else '---'
        print(f"  {i:<4} {str(w.window_open):<30} {str(w.window_close):<30} {duration:<14.1f} {max_el}")

    return windows


def find_simultaneous_contacts(windows_sc1, windows_sc2):
    """
    Find epochs where both spacecraft have overlapping contact windows.
    These are candidate synchronization trigger epochs for RSSDA.
    Returns list of (overlap_start, overlap_end, midpoint) tuples.
    """
    simultaneous = []
    for w1 in windows_sc1:
        for w2 in windows_sc2:
            # Use Julian date for comparison (float)
            open1_jd  = w1.window_open.jd()
            close1_jd = w1.window_close.jd()
            open2_jd  = w2.window_open.jd()
            close2_jd = w2.window_close.jd()

            overlap_open_jd  = max(open1_jd, open2_jd)
            overlap_close_jd = min(close1_jd, close2_jd)

            if overlap_close_jd > overlap_open_jd:
                overlap_start = w1.window_open  if open1_jd  >= open2_jd  else w2.window_open
                overlap_end   = w1.window_close if close1_jd <= close2_jd else w2.window_close
                duration_sec  = (overlap_close_jd - overlap_open_jd) * 86400.0
                midpoint      = overlap_start + duration_sec / 2.0
                simultaneous.append((overlap_start, overlap_end, midpoint))
    return simultaneous


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Initialize EOP (required for ECI<->ECEF frame transforms used in access computation)
    initialize_eop()

    print("=" * 70)
    print("Spacecraft CA - Contact Window Analysis")
    print("=" * 70)
    print(f"Planning horizon: {EPOCH_START} -> {EPOCH_TCA}")
    print(f"Duration: 24 hours")
    print(f"Ground station: lon={GS_LON}, lat={GS_LAT} (mid-latitude)")

    # Build propagators
    prop_sc1 = build_propagator(SC1_OE, EPOCH_START)
    prop_sc2 = build_propagator(SC2_OE, EPOCH_START)

    # Ground station
    gs = PointLocation(GS_LON, GS_LAT, GS_ALT)

    # Compute contact windows
    windows_sc1 = compute_contact_windows(prop_sc1, gs, EPOCH_START, EPOCH_TCA, "SC1")
    windows_sc2 = compute_contact_windows(prop_sc2, gs, EPOCH_START, EPOCH_TCA, "SC2")

    # Find simultaneous contacts (synchronization trigger candidates)
    simultaneous = find_simultaneous_contacts(windows_sc1, windows_sc2)

    print(f"\nSimultaneous contacts (both SC visible): {len(simultaneous)}")
    for i, (t_open, t_close, t_mid) in enumerate(simultaneous):
        duration = t_close - t_open  # seconds (float)
        print(f"  [{i}] open={t_open}  close={t_close}  duration={duration:.1f}s  midpoint={t_mid}")

    # Summary for RSSDA planning
    print("\n" + "=" * 70)
    print("RSSDA Planning Summary")
    print("=" * 70)
    print(f"  SC1 total passes:           {len(windows_sc1)}")
    print(f"  SC2 total passes:           {len(windows_sc2)}")
    print(f"  Simultaneous passes:        {len(simultaneous)}")
    print(f"  Suggested horizon H:        {len(windows_sc1)} stages (one per SC1 pass)")
    print(f"  Sync trigger epochs:        {len(simultaneous)} (simultaneous contact windows)")
    print()
    print("  Decision epochs (SC1 contact midpoints):")
    for i, w in enumerate(windows_sc1):
        mid = w.midtime
        t_to_tca = EPOCH_TCA - mid
        print(f"    Stage {i}: {mid}  (T-{t_to_tca/3600:.2f}h to TCA)")


if __name__ == "__main__":
    main()
