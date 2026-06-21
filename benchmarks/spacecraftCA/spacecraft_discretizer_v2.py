"""
spacecraft_discretizer_v2.py

Reduced relative-motion state space (the brahe-validated fix; see
notes/session-log/2026-06-04b.md). Replaces the lossy unsigned miss_bin of the
v1 discretizer, which re-derived the next miss from the bin CENTER each step and
threw away maneuver history (telling the solver a burn+counterburn was safe when
real physics returns to danger).

State = (dT_bin, vdev1, vdev2, stage)
  dT_bin : SIGNED along-track offset at TCA (km), the steered quantity. Negative
           = behind, positive = ahead, central bin straddles 0 (collision zone).
  vdev1  : SC1 net along-track VELOCITY-offset level (-1/0/+1), set by net burns.
  vdev2  : SC2 net along-track VELOCITY-offset level.
           vdev is the drift memory v1 lacked: it DRIVES the dT transition
           (dT += vdev*gain(stage)*dt), so a counterburn (vdev->0) leaves dT's
           banked drift instead of snapping back.
  stage  : planning stage index (unchanged from v1).

The miss distance that drives the terminal reward is NOT dT alone but the
quadrature combination with a per-conjunction sideways standoff `perp`:
    miss = sqrt(dT^2 + perp^2)
An along-track burn cannot move perp (verified dN~=0 across 6 geometries), so
perp is a build-time CONSTANT per conjunction; the conjunction TYPE is the value
of perp (head-on perp~=0 .. cross-track perp~=full miss), swept via init_b.

Total states: 13 dT * 3 vdev * 3 vdev * 16 stages + 1 sink = 1873 states.
"""

import numpy as np
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Signed along-track offset (dT) bins (km)
# ---------------------------------------------------------------------------
# Reuse the v1 fine-near-zero miss edges, mirrored about 0; thin the far tail
# (one 0.5 m/s burn reaches ~128 km, so >100 km bins are rarely occupied).
# Edges refined 2026-06-19b in the REWARD-ACTIVE band so the two-ramp terminal reward
# can be evaluated where it has gradient (coarse bins evaluate the convex displacement
# ramp at a stale bin center -> a policy landing at 6 km and one at 17 km looked
# identical under the old single [5,20) bin). Added 3 (splits the 1-5 km risk ramp) and
# 7,10 (splits 5-20 so the convex return-cost can resolve 5-10 vs 10-20). Far tail
# (>20 km) stays coarse: reward there is flat/steeply-convex but the policy avoids it.
# Positive edges: 0.5,1,2,3,5,7,20,50  (8 pos edges -> N_DT=17 signed bins; +/-inf caps are
# ALWAYS appended automatically below). LEANER than the 19-bin trial (dropped the 10 km edge
# AND tightened the tail 100->50):
#   - 0.5,1,2 : FINE near zero -> keeps the sharp <1km collision / act-no-act boundary.
#   - 3,5,7   : the reward-ACTIVE band where the convex displacement ramp steers the landing.
#   - 20,50   : coarse tail. Dropping 100->50 is SAFE under the convex ramp: an uncountered
#     early burn drifts to ~-128km by TCA (lever -127@T-24h..-17@T-3h), but the convex ramp
#     makes living that far out hugely costly so the OPTIMAL policy never occupies the deep
#     tail -- it lands near 5km. The tail only catches transient over-drift (already "very
#     bad", exact center irrelevant). The dt DYNAMICS stay exact; only the far BINNING is
#     coarse. 2026-06-19b: 19 bins -> full-contact solve ~15x slower; trimming to 17 for
#     speed. Revert to [...,7,10,20,100] (19) if 17 loses the Cen<SDec<Dec differentiation.

# DECISION-BOUNDARY anchors (km), ALWAYS present, fixed: these are where the policy's
# act/no-act classification flips, NOT arbitrary resolution. 1 = collision floor (<1km PoC);
# 4 = "never closer than 4km" lower safe edge; 5,7 = the 4-7km target band the policy aims for.
# Above 7 the edges are AUTO-GENERATED from the burn levers (set_dt_edges_from_levers) so a
# burn+counter composes from a FAITHFUL starting bin (the source-snap that aliased 7 distinct
# burns into one center -31.62 was what made a colliding burn+counter read as "safe -5.92" and
# fly into a 100% collision; see session-log 2026-06-20d). Cap at 50km: past that the convex
# ramp keeps the optimal policy out, so the +/-inf tail bin ("clearly too far") suffices.
_DT_NEAR_ANCHORS_KM = [1.0, 4.0, 5.0, 7.0]
_DT_TAIL_CAP_KM = 50.0

# Default static positive edges (used until set_dt_edges_from_levers reconfigures the grid at
# matrix-build time). MUST match what auto_pos_edges produces for the current head-on case, so
# the init belief (built on this default BEFORE build_T_O reconfigures) and the solved matrices
# agree on N_DT / state indices. If the conjunction changes, build_T_O regenerates these; the
# init-belief callers build init_b from the SAME default, so they stay consistent for the
# default conjunction. (21-bin auto result: anchors 1,4,5,7 + lever edges 13,20,24,29,38,45.)
_DT_POS_EDGES_KM = [1.0, 4.0, 5.0, 7.0, 13.0, 20.0, 24.0, 29.0, 38.0, 45.0]


def auto_pos_edges(levers, near_anchors=None, cap=None, min_gap=2.0):
    """Positive dt bin edges = fixed near anchors + source edges DERIVED from the burn levers.

    Each distinct late-burn |lever| in (max(anchors), cap] gets its own bin (an edge placed at
    the midpoint to the next lever cluster), so the first burn of a burn+counter snaps to a
    faithful starting value and the counter composes to the right miss class. Levers closer than
    min_gap merge into one bin. Levers above cap fold into the +/-inf tail. This makes the grid a
    FUNCTION of the conjunction physics -> it regenerates automatically when the orbit / horizon /
    dV changes (different levers -> different source edges)."""
    if near_anchors is None:
        near_anchors = _DT_NEAR_ANCHORS_KM
    if cap is None:
        cap = _DT_TAIL_CAP_KM
    src = sorted({round(abs(L), 1) for L in levers if near_anchors[-1] < abs(L) < cap})
    edges = list(near_anchors)
    i = 0
    while i < len(src):
        j = i
        while j + 1 < len(src) and src[j + 1] - src[i] < min_gap:
            j += 1                              # merge a tight lever cluster into one bin
        # edge just ABOVE this cluster: midpoint to the next cluster, OR (for the last cluster)
        # just past it -- so the deepest source burn gets its own bin and everything beyond folds
        # straight into the +/-inf tail (NO 50km edge; keeps the bin count lean at 21).
        if j + 1 < len(src):
            edge = round((src[j] + src[j + 1]) / 2.0)
        else:
            edge = round(src[j] + min_gap)      # just above the deepest anchored lever
        if edge > edges[-1] and edge < cap:
            edges.append(float(edge))
        i = j + 1
    return edges


def _recompute_grid():
    """(Re)build the signed edge list + derived sizes from _DT_POS_EDGES_KM. Call after changing
    _DT_POS_EDGES_KM so N_DT / N_STATES / state_index stay consistent."""
    global DT_EDGES_KM, N_DT, DT_ZERO_BIN, N_STAGE_STATES, N_STATES, N_STATES_TOTAL, SINK_STATE
    DT_EDGES_KM = (
        [-float('inf')]
        + [-x for x in reversed(_DT_POS_EDGES_KM)]
        + [x for x in _DT_POS_EDGES_KM]
        + [float('inf')]
    )
    N_DT = len(DT_EDGES_KM) - 1
    DT_ZERO_BIN = N_DT // 2              # central bin covers [-anchors[0], +anchors[0]] straddling 0
    N_STAGE_STATES = N_DT * N_VDEV * N_VDEV
    N_STATES = N_STAGE_STATES * N_STAGES
    N_STATES_TOTAL = N_STATES + 1
    SINK_STATE = N_STATES


def set_dt_edges_from_levers(levers, **kw):
    """Reconfigure the dt grid's positive edges from the burn levers, then recompute sizes.
    Called once at matrix-build time (compute_gain_table gives the levers). Returns the new edges."""
    global _DT_POS_EDGES_KM
    _DT_POS_EDGES_KM = auto_pos_edges(levers, **kw)
    _recompute_grid()
    return _DT_POS_EDGES_KM


# Full signed edge list + derived sizes (built from the default _DT_POS_EDGES_KM at import;
# reconfigured by set_dt_edges_from_levers at build time).
DT_EDGES_KM = (
    [-float('inf')]
    + [-x for x in reversed(_DT_POS_EDGES_KM)]
    + [x for x in _DT_POS_EDGES_KM]
    + [float('inf')]
)
N_DT = len(DT_EDGES_KM) - 1
DT_ZERO_BIN = N_DT // 2

COLLISION_THRESHOLD_KM = 1.0


# ---------------------------------------------------------------------------
# Velocity-offset (vdev) bins, per agent
# ---------------------------------------------------------------------------
# -1/0/+1 net along-track velocity offset. Minimum that distinguishes a burn
# from a counterburn. (vdev[5] would add "drifting hard" / same-direction
# double-burn; revisit only if the policy wants double-burns.)
VDEV_VALUES = [-1, 0, 1]
N_VDEV = len(VDEV_VALUES)
VDEV_ZERO = VDEV_VALUES.index(0)
VDEV_LABELS = ["NEG", "NOM", "POS"]


# ---------------------------------------------------------------------------
# Stage/state sizes
# ---------------------------------------------------------------------------

# N_STAGES is derived from the orbit-dependent stage grid (single source of truth in
# spacecraft_stage_grid). Was hardcoded 16; now follows the computed contact timeline.
from spacecraft_stage_grid import N_STAGES  # noqa: E402
N_STAGE_STATES = N_DT * N_VDEV * N_VDEV
N_STATES = N_STAGE_STATES * N_STAGES
N_STATES_TOTAL = N_STATES + 1
SINK_STATE = N_STATES


# ---------------------------------------------------------------------------
# dT binning / centers
# ---------------------------------------------------------------------------

def dt_to_bin(dt_km: float) -> int:
    """Bin a SIGNED along-track offset (km). Returns 0..N_DT-1."""
    for i in range(N_DT):
        if dt_km < DT_EDGES_KM[i + 1]:
            return i
    return N_DT - 1


def dt_bin_center_km(i: int) -> float:
    """
    Representative signed dT for a bin.
    Central bin -> 0. Finite bins -> signed geometric mean of |edges|.
    Open tail bins -> a representative beyond the last finite edge.
    """
    lo = DT_EDGES_KM[i]
    hi = DT_EDGES_KM[i + 1]
    if lo < 0 and hi > 0:                 # central bin straddling 0
        return 0.0
    if not np.isfinite(lo):               # (-inf, -100]
        return -2.0 * abs(hi)
    if not np.isfinite(hi):               # (+100, +inf)
        return 2.0 * abs(lo)
    sign = 1.0 if hi > 0 else -1.0
    return sign * float(np.sqrt(abs(lo) * abs(hi)))


def dt_bin_label(i: int) -> str:
    lo = DT_EDGES_KM[i]
    hi = DT_EDGES_KM[i + 1]
    lo_s = "-inf" if not np.isfinite(lo) else f"{lo:g}"
    hi_s = "+inf" if not np.isfinite(hi) else f"{hi:g}"
    return f"[{lo_s}, {hi_s}) km"


# ---------------------------------------------------------------------------
# Miss readout: quadrature of dT with the per-conjunction perp standoff
# ---------------------------------------------------------------------------

def miss_km_from_dt(dt_km: float, perp_km: float) -> float:
    """True miss distance = sqrt(dT^2 + perp^2). perp is the (fixed) sideways
    standoff an along-track burn cannot change."""
    return float(np.hypot(dt_km, perp_km))


def is_collision_dt(dt_km: float, perp_km: float) -> bool:
    return miss_km_from_dt(dt_km, perp_km) < COLLISION_THRESHOLD_KM


# ---------------------------------------------------------------------------
# vdev update
# ---------------------------------------------------------------------------

def vdev_value(vdev_bin: int) -> int:
    return VDEV_VALUES[vdev_bin]


def update_vdev_bin(vdev_bin: int, action: int) -> int:
    """
    Update the net velocity-offset level after a local maneuver.
      action 0 = WAIT (unchanged), 1 = +dV (move +1), 2 = -dV (move -1),
    clipped to the [-1, +1] support. (Saturation at +/-1 is the vdev[3]
    approximation for same-direction double-burns.)
    """
    if action == 0:
        return vdev_bin
    delta = 1 if action == 1 else -1
    new_val = max(-1, min(1, vdev_value(vdev_bin) + delta))
    return VDEV_VALUES.index(new_val)


# ---------------------------------------------------------------------------
# State indexing
# ---------------------------------------------------------------------------

def state_index(dt_bin: int, *args: int) -> int:
    """
    Flat state index.

    Canonical:  state_index(dt_bin, vdev1, vdev2, stage)
    Shorthand:  state_index(dt_bin, stage)   (both vdev nominal)
    """
    if len(args) == 1:
        vdev1 = VDEV_ZERO
        vdev2 = VDEV_ZERO
        stage = args[0]
    elif len(args) == 3:
        vdev1, vdev2, stage = args
    else:
        raise TypeError("state_index expects (dt_bin, stage) or (dt_bin, vdev1, vdev2, stage)")

    within_stage = dt_bin + N_DT * (vdev1 + N_VDEV * vdev2)
    return stage * N_STAGE_STATES + within_stage


def index_to_state(idx: int) -> Tuple[int, int, int, int]:
    """Inverse of state_index. Returns (dt_bin, vdev1, vdev2, stage)."""
    stage = idx // N_STAGE_STATES
    rem = idx % N_STAGE_STATES
    dt_bin = rem % N_DT
    rem //= N_DT
    vdev1 = rem % N_VDEV
    vdev2 = rem // N_VDEV
    return dt_bin, vdev1, vdev2, stage


def sync_trigger_states(contact_stages: List[int]) -> List[int]:
    """State indices whose stage is in contact_stages."""
    triggers = []
    for idx in range(N_STATES):
        _, _, _, stage = index_to_state(idx)
        if stage in contact_stages:
            triggers.append(idx)
    return sorted(triggers)


def print_summary():
    print("=" * 60)
    print("Reduced relative-motion state space (v2: signed dT + vdev)")
    print("=" * 60)
    print(f"  N_DT            = {N_DT}  (signed along-track offset bins)")
    print(f"  DT_ZERO_BIN     = {DT_ZERO_BIN}  (central, straddles collision line)")
    print(f"  N_VDEV          = {N_VDEV}  ({VDEV_LABELS}) per agent")
    print(f"  N_STAGES        = {N_STAGES}")
    print(f"  N_STAGE_STATES  = {N_STAGE_STATES}")
    print(f"  N_STATES        = {N_STATES}  (non-sink)")
    print(f"  N_STATES_TOTAL  = {N_STATES_TOTAL}  (+ sink at {SINK_STATE})")
    print()
    print("  dT bins (signed along-track offset):")
    for i in range(N_DT):
        mark = "  <- ZERO/collision" if i == DT_ZERO_BIN else ""
        print(f"    bin {i:>2}: {dt_bin_label(i):<22} center={dt_bin_center_km(i):+8.2f} km{mark}")
    print()
    print("  vdev bins (per agent):")
    for i, label in enumerate(VDEV_LABELS):
        print(f"    bin {i}: {label:>3s}  value={VDEV_VALUES[i]:+d}")


if __name__ == "__main__":
    print_summary()
