"""
spacecraft_discretizer.py

State = (miss_distance_bin, stage)
  miss_distance: predicted miss at TCA, discretized into 6 bins (km)
  stage: planning stage index (0 = first contact, 5 = last before TCA)

Total states: 6 bins * 6 stages + 1 sink = 37
  Indices 0..35: (miss_bin, stage) states  -> flat index = stage * N_MISS + miss_bin
  Index 36: sink (absorbing terminal state)
"""

import numpy as np
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Miss distance bins (km)
# ---------------------------------------------------------------------------

MISS_EDGES_KM = [0, 1, 5, 20, 100, 500, float('inf')]  # 6 bins
N_MISS = 6
N_STAGES = 6
N_STATES = N_MISS * N_STAGES        # 36
N_STATES_TOTAL = N_STATES + 1       # 37 (index 36 = sink)
SINK_STATE = N_STATES               # 36

_MISS_EDGES = np.array([0, 1, 5, 20, 100, 500], dtype=float)  # left edges only


def miss_to_bin(d_km: float) -> int:
    """Bin a miss distance in km. Returns 0..5."""
    if d_km < 1.0:
        return 0
    elif d_km < 5.0:
        return 1
    elif d_km < 20.0:
        return 2
    elif d_km < 100.0:
        return 3
    elif d_km < 500.0:
        return 4
    else:
        return 5


def bin_center_km(i: int) -> float:
    """Geometric mean of bin edges; 1000 km for the last (open) bin."""
    edges = MISS_EDGES_KM
    lo = edges[i]
    hi = edges[i + 1]
    if hi == float('inf'):
        return 1000.0
    # geometric mean avoids unrepresentatively large center
    if lo == 0:
        lo = 0.5  # avoid sqrt(0); treat [0,1) center as ~0.5 km
    return float(np.sqrt(lo * hi))


def state_index(miss_bin: int, stage: int) -> int:
    """Flat index: stage * N_MISS + miss_bin."""
    return stage * N_MISS + miss_bin


def index_to_state(idx: int) -> Tuple[int, int]:
    """Inverse of state_index. Returns (miss_bin, stage)."""
    stage = idx // N_MISS
    miss_bin = idx % N_MISS
    return miss_bin, stage


def sync_trigger_states(contact_stages: List[int]) -> List[int]:
    """
    Return sorted list of state indices that are synchronization triggers.
    A state is a sync trigger if its stage is in contact_stages.
    """
    triggers = []
    for idx in range(N_STATES):
        _, stage = index_to_state(idx)
        if stage in contact_stages:
            triggers.append(idx)
    return sorted(triggers)


# ---------------------------------------------------------------------------
# Summary / self-test
# ---------------------------------------------------------------------------

def print_summary():
    print("=" * 55)
    print("Miss-Distance Discrete State Space")
    print("=" * 55)
    print(f"  N_MISS         = {N_MISS}")
    print(f"  N_STAGES       = {N_STAGES}")
    print(f"  N_STATES       = {N_STATES}  (non-sink)")
    print(f"  N_STATES_TOTAL = {N_STATES_TOTAL}  (+ sink at {SINK_STATE})")
    print()
    print("  Miss bins:")
    for i in range(N_MISS):
        lo = MISS_EDGES_KM[i]
        hi = MISS_EDGES_KM[i + 1]
        hi_str = "inf" if hi == float('inf') else f"{hi}"
        print(f"    bin {i}: [{lo}, {hi_str}) km  center={bin_center_km(i):.2f} km")
    print()


if __name__ == "__main__":
    print_summary()

    print("Round-trip check (index_to_state o state_index):")
    all_ok = True
    for miss_bin in range(N_MISS):
        for stage in range(N_STAGES):
            idx = state_index(miss_bin, stage)
            mb, st = index_to_state(idx)
            ok = (mb == miss_bin) and (st == stage)
            if not ok:
                print(f"  FAIL: ({miss_bin},{stage}) -> idx={idx} -> ({mb},{st})")
                all_ok = False
    print(f"  All {N_MISS * N_STAGES} round-trips: {'OK' if all_ok else 'FAILED'}")

    print()
    print("miss_to_bin spot checks:")
    for d in [0.0, 0.5, 1.0, 4.9, 5.0, 19.9, 20.0, 99.9, 100.0, 499.9, 500.0, 1e6]:
        print(f"  {d:>10.1f} km -> bin {miss_to_bin(d)}")

    print()
    print(f"sync_trigger_states([0,1,2,3,4,5]) = {len(sync_trigger_states(list(range(N_STAGES))))} states (all non-sink)")
