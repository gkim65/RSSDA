"""
spacecraft_discretizer.py

State = (miss_distance_bin, dev1_bin, dev2_bin, stage)
  miss_distance: predicted miss at TCA, discretized into refined bins (km)
  dev1/dev2: signed coarse trajectory-deviation bins for SC1/SC2
  stage: planning stage index

Stages: 16 total: 10 regular planning points plus 6 GS contact windows.

Total states: 10 miss bins * 3 dev bins * 3 dev bins * 16 stages + 1 sink
  = 1441 states
"""

import numpy as np
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Miss distance bins (km)
# ---------------------------------------------------------------------------

MISS_EDGES_KM = [
    0.0,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    50.0,
    100.0,
    500.0,
    float('inf'),
]
N_MISS = len(MISS_EDGES_KM) - 1
COLLISION_THRESHOLD_KM = 1.0


# ---------------------------------------------------------------------------
# Signed deviation bins
# ---------------------------------------------------------------------------

# 0 = negative signed deviation, 1 = near nominal, 2 = positive signed deviation.
# In this first coarse model the bins track accumulated signed maneuver tendency
# rather than a high-fidelity 6D orbital deviation.
DEV_VALUES = [-1, 0, 1]
N_DEV = len(DEV_VALUES)
DEV_ZERO = DEV_VALUES.index(0)
DEV_LABELS = ["NEG", "NOM", "POS"]


# ---------------------------------------------------------------------------
# Stage/state sizes
# ---------------------------------------------------------------------------

# N_STAGES is derived from the orbit-dependent stage grid (single source of truth in
# spacecraft_stage_grid). Was hardcoded 16; now follows the computed contact timeline.
from spacecraft_stage_grid import N_STAGES  # noqa: E402
N_STAGE_STATES = N_MISS * N_DEV * N_DEV
N_STATES = N_STAGE_STATES * N_STAGES
N_STATES_TOTAL = N_STATES + 1
SINK_STATE = N_STATES


def miss_to_bin(d_km: float) -> int:
    """Bin a miss distance in km. Returns 0..N_MISS-1."""
    for i in range(N_MISS):
        if d_km < MISS_EDGES_KM[i + 1]:
            return i
    return N_MISS - 1


def bin_center_km(i: int) -> float:
    """Geometric mean of bin edges; 1000 km for the last open bin."""
    edges = MISS_EDGES_KM
    lo = edges[i]
    hi = edges[i + 1]
    if hi == float('inf'):
        return 1000.0
    if lo == 0:
        return 0.5 * hi
    return float(np.sqrt(lo * hi))


def miss_bin_label(i: int) -> str:
    """Readable miss-bin range label."""
    lo = MISS_EDGES_KM[i]
    hi = MISS_EDGES_KM[i + 1]
    lo_str = f"{lo:g}"
    if hi == float('inf'):
        return f"[{lo_str}, inf) km"
    return f"[{lo_str}, {hi:g}) km"


def collision_bin_indices() -> List[int]:
    """Bins wholly below the collision threshold."""
    return [
        i for i in range(N_MISS)
        if MISS_EDGES_KM[i + 1] <= COLLISION_THRESHOLD_KM
    ]


def dev_value(dev_bin: int) -> int:
    """Return signed representative value for a deviation bin."""
    return DEV_VALUES[dev_bin]


def update_dev_bin(dev_bin: int, action: int) -> int:
    """
    Update signed deviation bin after a local maneuver.

    WAIT leaves the deviation unchanged. Positive/negative burns move the
    coarse signed deviation one bin in the corresponding direction, clipped to
    the three-bin support.
    """
    if action == 0:
        return dev_bin
    delta = 1 if action == 1 else -1
    new_val = max(-1, min(1, dev_value(dev_bin) + delta))
    return DEV_VALUES.index(new_val)


def state_index(miss_bin: int, *args: int) -> int:
    """
    Flat state index.

    Canonical form:
      state_index(miss_bin, dev1_bin, dev2_bin, stage)

    Backward-compatible form:
      state_index(miss_bin, stage)
    which assumes both deviations are nominal.
    """
    if len(args) == 1:
        dev1_bin = DEV_ZERO
        dev2_bin = DEV_ZERO
        stage = args[0]
    elif len(args) == 3:
        dev1_bin, dev2_bin, stage = args
    else:
        raise TypeError("state_index expects (miss_bin, stage) or (miss_bin, dev1_bin, dev2_bin, stage)")

    within_stage = miss_bin + N_MISS * (dev1_bin + N_DEV * dev2_bin)
    return stage * N_STAGE_STATES + within_stage


def index_to_state(idx: int) -> Tuple[int, int, int, int]:
    """Inverse of state_index. Returns (miss_bin, dev1_bin, dev2_bin, stage)."""
    stage = idx // N_STAGE_STATES
    rem = idx % N_STAGE_STATES
    miss_bin = rem % N_MISS
    rem //= N_MISS
    dev1_bin = rem % N_DEV
    dev2_bin = rem // N_DEV
    return miss_bin, dev1_bin, dev2_bin, stage


def sync_trigger_states(contact_stages: List[int]) -> List[int]:
    """
    Return sorted state indices that are synchronization triggers.
    A state is a sync trigger if its stage is in contact_stages.
    """
    triggers = []
    for idx in range(N_STATES):
        _, _, _, stage = index_to_state(idx)
        if stage in contact_stages:
            triggers.append(idx)
    return sorted(triggers)


def print_summary():
    print("=" * 60)
    print("Miss/Deviation Discrete State Space")
    print("=" * 60)
    print(f"  N_MISS          = {N_MISS}")
    print(f"  N_DEV           = {N_DEV}  ({DEV_LABELS})")
    print(f"  N_STAGES        = {N_STAGES}  (10 regular points + 6 GS contacts)")
    print(f"  N_STAGE_STATES  = {N_STAGE_STATES}")
    print(f"  N_STATES        = {N_STATES}  (non-sink)")
    print(f"  N_STATES_TOTAL  = {N_STATES_TOTAL}  (+ sink at {SINK_STATE})")
    print()
    print("  Miss bins:")
    for i in range(N_MISS):
        lo = MISS_EDGES_KM[i]
        hi = MISS_EDGES_KM[i + 1]
        hi_str = "inf" if hi == float('inf') else f"{hi}"
        print(f"    bin {i}: [{lo}, {hi_str}) km  center={bin_center_km(i):.2f} km")
    print()
    print("  Deviation bins:")
    for i, label in enumerate(DEV_LABELS):
        print(f"    bin {i}: {label:>3s}  value={DEV_VALUES[i]:+d}")


if __name__ == "__main__":
    print_summary()

    print("\nRound-trip check (index_to_state o state_index):")
    all_ok = True
    for miss_bin in range(N_MISS):
        for dev1_bin in range(N_DEV):
            for dev2_bin in range(N_DEV):
                for stage in range(N_STAGES):
                    idx = state_index(miss_bin, dev1_bin, dev2_bin, stage)
                    mb, d1, d2, st = index_to_state(idx)
                    ok = (mb == miss_bin and d1 == dev1_bin and
                          d2 == dev2_bin and st == stage)
                    if not ok:
                        print(f"  FAIL: ({miss_bin},{dev1_bin},{dev2_bin},{stage}) -> {idx} -> ({mb},{d1},{d2},{st})")
                        all_ok = False
    print(f"  All {N_STATES} round-trips: {'OK' if all_ok else 'FAILED'}")

    print("\nmiss_to_bin spot checks:")
    for d in [
        0.0, 0.49, 0.5, 0.99, 1.0, 1.99, 2.0, 4.9, 5.0,
        9.9, 10.0, 19.9, 20.0, 49.9, 50.0, 99.9, 100.0,
        499.9, 500.0, 1e6,
    ]:
        print(f"  {d:>10.2f} km -> bin {miss_to_bin(d)}")

    print()
    print(f"sync_trigger_states(all) = {len(sync_trigger_states(list(range(N_STAGES))))} states (all non-sink)")
