"""
spacecraft_stage_grid.py

SINGLE SOURCE OF TRUTH for the decision-stage timeline and the GS-contact stages.

Owns:
  - the scenario epoch + SC1/SC2 reference orbits,
  - the static ground-station network (a documented subset of brahe's KSAT data),
  - compute_contact_times_h(): real orbit-dependent simultaneous-contact times,
  - build_stage_grid(): UNION-WITH-MERGE of the ~2h decision cadence with computed
    contacts (a contact within MERGE_THRESHOLD_H of a grid stage merges onto it; a
    contact farther than that spawns its own stage),
  - the derived module-level grid: N_STAGES, STAGE_T_BEFORE_TCA_SEC, STAGE_EPOCHS,
    CONTACT_STAGES (+ set/get).

Both spacecraft_discretizer_v2 and spacecraft_matrices import the grid from here so the
stage count, stage epochs, and contact stages can never drift out of sync.

Design rationale + literature: notes/LITERATURE_GS_NETWORK.md.
Replaces the old hardcoded N_STAGES=16 / frozen _GS_TIMES_H path.
"""
import os
import numpy as np
from brahe import (
    Epoch, AngleFormat, R_EARTH,
    location_accesses, ElevationConstraint, PointLocation, AccessSearchConfig,
    KeplerianPropagator, NumericalOrbitPropagator, NumericalPropagationConfig,
    ForceModelConfig,
    state_koe_to_eci, state_rtn_to_eci, state_eci_to_koe,
    initialize_eop,
)
import brahe

D = AngleFormat.DEGREES

# ---------------------------------------------------------------------------
# Propagator backend — ONE setting governs ALL propagation (matrices, transition,
# generator, AND this module's contact computation). Single source of truth so the
# whole pipeline is consistent: drag everywhere or two-body everywhere.
#   "numerical" : NumericalOrbitPropagator + two_body()  (default; fast; debug/tests)
#   "keplerian" : closed-form two-body KeplerianPropagator (fastest over long horizons)
#   "drag"      : NumericalOrbitPropagator + leo_default() (J2+drag+SRP+third-body) — use
#                 for EXPERIMENT runs. Slower; needs space weather.
# Override at runtime (no code edit) via env SPACECRAFT_PROPAGATOR, or compare_variants_v2
# --backend. Default = two-body for quick tests; set drag explicitly for experiments.
PROPAGATOR_BACKEND = os.environ.get("SPACECRAFT_PROPAGATOR", "numerical").lower()
# Drag-backend smallsat params [mass(kg), drag_area(m^2), Cd, srp_area(m^2), Cr].
DRAG_PARAMS = np.array([150.0, 1.0, 2.2, 1.0, 1.3])

_SW_INITIALIZED = False


def _ensure_sw():
    global _SW_INITIALIZED
    if not _SW_INITIALIZED:
        from brahe import initialize_sw
        initialize_sw()
        _SW_INITIALIZED = True


def set_propagator_backend(backend: str):
    """Set the global propagator backend for the WHOLE pipeline (matrices/transition/
    generator/contacts all read this). One of: numerical | keplerian | drag."""
    global PROPAGATOR_BACKEND
    backend = backend.lower()
    if backend not in ("numerical", "keplerian", "drag"):
        raise ValueError(f"backend must be numerical|keplerian|drag, got {backend!r}")
    PROPAGATOR_BACKEND = backend
    return PROPAGATOR_BACKEND

# ---------------------------------------------------------------------------
# Scenario reference (kept here so discretizer + matrices agree)
# ---------------------------------------------------------------------------
EPOCH_TCA = Epoch(2025, 6, 2, 0, 0, 0.0)

# SC1 orbital elements at TCA [a, e, i, RAAN, omega, M] (deg). 550km, 55deg LEO.
SC1_OE_AT_TCA = np.array([
    R_EARTH + 550e3, 0.001, 55.0, 20.0, 0.0, 0.0,
])

V_REL_MS = 15.0  # along-track closing speed at TCA (m/s). Canonical here; matrices re-exports.

# ---------------------------------------------------------------------------
# Decision cadence + merge threshold
# ---------------------------------------------------------------------------
# Regular ~2h decision cadence (hours before TCA). Contacts merge into / extend this.
_HOUR_GRID_H = [24, 22, 20, 18, 16, 14, 12, 10, 4, 2]

# A computed contact within this many hours of an existing stage MERGES onto it
# (marks it a contact); farther away it SPAWNS its own stage. CHANGEABLE knob:
#   - 0.25h (default): keeps more distinct contacts (spread6 -> 25 stages / 18 contacts).
#   - 0.5h           : coarser; reproduces the historical frozen 16-stage / 6-contact
#                      grid EXACTLY for the single-station gate (validation reference).
# Override at runtime without editing code via env SPACECRAFT_MERGE_THRESHOLD_H (below).
# See notes/scratch/proto_contact_stages.py for the threshold sweep.
MERGE_THRESHOLD_H = 0.25

GS_MIN_ELEVATION_DEG = 10.0  # standard LEO mask (NASA SoA / arXiv:2410.16282)

# ---------------------------------------------------------------------------
# Static ground-station network (used for ALL conjunctions)
# ---------------------------------------------------------------------------
# "spread6": polar N/S anchors (Svalbard, Troll) + globally distributed mid/low-lat
# (Hawaii, Singapore, Weilheim, Awarua). Real KSAT sites from brahe's embedded data.
# Chosen to give ~15 contacts / 5 non-contact stages on the 24h reference grid while
# staying <=22 stages. Rationale + citations: notes/LITERATURE_GS_NETWORK.md.
GS_NETWORK_NAMES = ["Svalbard", "Troll", "Hawaii", "Singapore", "Weilheim", "Awarua"]

_GS_NETWORK_CACHE = None


def gs_network():
    """Return the static network as a list of brahe PointLocation (cached)."""
    global _GS_NETWORK_CACHE
    if _GS_NETWORK_CACHE is None:
        ks = brahe.groundstations.load("ksat")
        by_name = {st.get_name(): st for st in ks}
        missing = [n for n in GS_NETWORK_NAMES if n not in by_name]
        if missing:
            raise ValueError(f"GS network stations not found in KSAT data: {missing}")
        _GS_NETWORK_CACHE = [by_name[n] for n in GS_NETWORK_NAMES]
    return _GS_NETWORK_CACHE


# ---------------------------------------------------------------------------
# SC2 reference orbit (model placement; matches conjunction_generator.sc2_eci_at_tca)
# ---------------------------------------------------------------------------
def sc2_oe_from_rtn(miss_km=0.0, v_rel_ms=None, sc1_oe=None, epoch_tca=None):
    """Reconstruct SC2 KOE from the model's actual RTN placement at TCA: radial miss
    offset + along-track closing speed. This is how the model / generator builds SC2,
    so contacts computed from it are representative of the real sweeps."""
    v_rel_ms = V_REL_MS if v_rel_ms is None else v_rel_ms
    sc1_oe = SC1_OE_AT_TCA if sc1_oe is None else sc1_oe
    sc1_eci = np.array(state_koe_to_eci(np.asarray(sc1_oe, float), D))
    rtn_rel = np.array([miss_km * 1e3, 0.0, 0.0, 0.0, -v_rel_ms, 0.0])
    sc2_eci = np.array(state_rtn_to_eci(sc1_eci, rtn_rel))
    return np.array(state_eci_to_koe(sc2_eci, D))


# ---------------------------------------------------------------------------
# Contact computation (ported from examples/test_contacts.py, generalized)
# ---------------------------------------------------------------------------
_EOP_READY = False


def _ensure_eop():
    global _EOP_READY
    if not _EOP_READY:
        initialize_eop()
        _EOP_READY = True


def _prop(oe_at_tca, epoch_tca, t_start=None):
    """Build a propagator for the contact search, honoring the global PROPAGATOR_BACKEND so
    contacts use the SAME physics as the matrix/transition/generator propagation. Seeded at
    TCA (the natural epoch the elements are defined at). For numerical/drag we PRIME the
    propagator backward to t_start via propagate_to() so its valid range covers the whole
    search window [t_start, TCA] (the integrator extends its range on demand but
    location_accesses needs it pre-covered). Keplerian is closed-form (any epoch)."""
    if PROPAGATOR_BACKEND == "keplerian":
        return KeplerianPropagator.from_keplerian(epoch_tca, oe_at_tca, D, step_size=60.0)
    eci = np.array(state_koe_to_eci(np.asarray(oe_at_tca, float), D))
    if PROPAGATOR_BACKEND == "drag":
        _ensure_sw()
        prop = NumericalOrbitPropagator(
            epoch_tca, eci, NumericalPropagationConfig.default(),
            ForceModelConfig.leo_default(), DRAG_PARAMS)
    else:
        prop = NumericalOrbitPropagator(
            epoch_tca, eci, NumericalPropagationConfig.default(),
            ForceModelConfig.two_body())
    if t_start is not None:
        prop.propagate_to(t_start)   # extend valid range backward to cover the search window
    return prop


def _windows(prop, stations, t_start, t_end):
    constraint = ElevationConstraint(GS_MIN_ELEVATION_DEG)
    cfg = AccessSearchConfig(initial_time_step=30.0)
    w = []
    for st in stations:
        w += list(location_accesses(st, prop, t_start, t_end, constraint, config=cfg))
    return w


def _pass_centers_h(windows, epoch_tca):
    return sorted(
        (epoch_tca.jd() - 0.5 * (w.window_open.jd() + w.window_close.jd())) * 24.0
        for w in windows
    )


def _later_contact_sync_h(win1, win2, epoch_tca):
    """LATER-CONTACT (staggered) sync rule. A sync completes once BOTH spacecraft have had
    a contact SINCE THE LAST SYNC, timed at the LATER pass of that pair (the operator can
    only reconcile the joint state on the ground after the second/later uplink arrives).

    Algorithm (walk the merged timeline, gated by the rarer spacecraft):
      - merge both SCs' pass times onto one timeline in TIME order (latest-in-time first),
      - track whether each SC has reported since the last sync,
      - when BOTH have, fire a sync at the CURRENT event time (= the later completing pass)
        and RESET both flags (each pass is consumed by at most one sync).

    Example (SC2 near-polar, many passes; SC1 few):
        SC1:      A            B          C
        SC2:  1 2   3 4 5   6 7   8 9
      -> syncs at A, B, C (3). SC2's extra passes do NOT add syncs — you can't reconcile a
      JOINT state until SC1 also reports. So the sync count is gated by the SCARCER SC.
      This is why it correctly under-counts vs a naive nearest-pass pairing for crossing /
      high-Δi conjunctions (near-polar SC2: 15 syncs, not 18). Probe: /tmp/sync_cross.py.

    NOTE (model semantics, future work): a sync here grants BOTH shared-state observation
    AND immediate joint action at that stage. Real ops often need a SEPARATE later uplink
    to command a maneuver (observe-now, act-at-next-contact latency). Not modeled yet —
    see notes/SCENARIO_KNOBS.md "sync semantics".

    Returns sync times as hours-before-TCA. (More realistic than strict simultaneous
    visibility when the orbits differ.)"""
    p1 = _pass_centers_h(win1, epoch_tca)
    p2 = _pass_centers_h(win2, epoch_tca)
    if not p1 or not p2:
        return []
    # hours-before-TCA: SMALLER value = later in wall-clock. Sort events latest-first.
    events = sorted([(h, 0) for h in p1] + [(h, 1) for h in p2], key=lambda e: -e[0])
    syncs = []
    seen = [False, False]
    for h, sc in events:
        seen[sc] = True
        if seen[0] and seen[1]:
            syncs.append(h)          # later of the completing pair = current event time
            seen = [False, False]
    return syncs


def _simultaneous_midpoints_h(win1, win2, epoch_tca):
    """SIMULTANEOUS rule (both SC visible at once): SC1 INT SC2 overlap midpoints.
    Kept for the validation gate (reproduces the frozen _GS_TIMES_H)."""
    out = []
    for a in win1:
        for b in win2:
            o = max(a.window_open.jd(), b.window_open.jd())
            c = min(a.window_close.jd(), b.window_close.jd())
            if c > o:
                out.append((epoch_tca.jd() - 0.5 * (o + c)) * 24.0)
    return out


def compute_contact_times_h(sc1_oe, sc2_oe, epoch_tca, horizon_h, stations=None,
                            sync_rule="later"):
    """Real orbit-dependent contact sync times (hours-before-TCA), NOT yet snapped to
    the stage grid. horizon_h = how far before TCA to search (the earliest stage)."""
    _ensure_eop()
    stations = gs_network() if stations is None else stations
    t_start = epoch_tca - horizon_h * 3600.0
    w1 = _windows(_prop(sc1_oe, epoch_tca, t_start), stations, t_start, epoch_tca)
    w2 = _windows(_prop(sc2_oe, epoch_tca, t_start), stations, t_start, epoch_tca)
    if sync_rule == "simultaneous":
        return _simultaneous_midpoints_h(w1, w2, epoch_tca)
    return _later_contact_sync_h(w1, w2, epoch_tca)


def build_stage_grid(hour_grid_h, contact_times_h, merge_threshold_h):
    """UNION-WITH-MERGE. Start from the regular hour grid; for each contact, merge it
    onto the nearest existing stage if within merge_threshold_h (marking that stage a
    contact), else add it as a NEW stage (also a contact). Returns
    (all_times_h_desc, contact_stage_indices)."""
    stages = [float(h) for h in hour_grid_h]
    marks = [False] * len(stages)
    for h in sorted(contact_times_h, reverse=True):
        d = [abs(s - h) for s in stages]
        j = int(np.argmin(d))
        if d[j] <= merge_threshold_h:
            marks[j] = True
        else:
            stages.append(float(h))
            marks.append(True)
    order = sorted(range(len(stages)), key=lambda i: -stages[i])
    all_times = [stages[i] for i in order]
    contact_idx = sorted(i for i, o in enumerate(order) if marks[o])
    return all_times, contact_idx


def compute_stage_grid(sc1_oe=None, sc2_oe=None, epoch_tca=None, hour_grid_h=None,
                       merge_threshold_h=None, stations=None, sync_rule="later"):
    """Full pipeline: compute real contacts for the (sc1, sc2) orbits over the hour-grid
    horizon, then union-with-merge into stage times + contact indices.
    Returns (all_times_h_desc, contact_stage_indices)."""
    sc1_oe = SC1_OE_AT_TCA if sc1_oe is None else sc1_oe
    sc2_oe = sc2_oe_from_rtn() if sc2_oe is None else sc2_oe
    epoch_tca = EPOCH_TCA if epoch_tca is None else epoch_tca
    hour_grid_h = _HOUR_GRID_H if hour_grid_h is None else hour_grid_h
    merge_threshold_h = MERGE_THRESHOLD_H if merge_threshold_h is None else merge_threshold_h
    horizon_h = max(hour_grid_h)
    times = compute_contact_times_h(sc1_oe, sc2_oe, epoch_tca, horizon_h,
                                    stations=stations, sync_rule=sync_rule)
    return build_stage_grid(hour_grid_h, times, merge_threshold_h)


# ---------------------------------------------------------------------------
# Module-level grid (the default reference scenario). Computed once at import.
# Override knobs via env for timing experiments without editing code:
#   SPACECRAFT_MERGE_THRESHOLD_H   (e.g. 0.25)
# ---------------------------------------------------------------------------
_env_thr = os.environ.get("SPACECRAFT_MERGE_THRESHOLD_H")
if _env_thr:
    MERGE_THRESHOLD_H = float(_env_thr)

_ALL_TIMES_H, _CONTACT_STAGES_INIT = compute_stage_grid()

N_STAGES = len(_ALL_TIMES_H)
STAGE_T_BEFORE_TCA_SEC = [h * 3600.0 for h in _ALL_TIMES_H]
STAGE_EPOCHS = [EPOCH_TCA - dt for dt in STAGE_T_BEFORE_TCA_SEC]

# Contact stages (sync/centralization triggers). SINGLE SOURCE OF TRUTH; mutate IN
# PLACE via set_contact_stages so existing list bindings stay consistent.
CONTACT_STAGES = list(_CONTACT_STAGES_INIT)


def set_contact_stages(stages):
    """Override the GS contact-stage list IN PLACE (subset of range(N_STAGES); []=none)."""
    CONTACT_STAGES[:] = sorted({int(s) for s in stages})
    return CONTACT_STAGES


def get_contact_stages():
    return list(CONTACT_STAGES)


if __name__ == "__main__":
    print(f"N_STAGES         = {N_STAGES}")
    print(f"stage times (h)  = {[round(h, 1) for h in _ALL_TIMES_H]}")
    print(f"CONTACT_STAGES   = {CONTACT_STAGES}  ({len(CONTACT_STAGES)} contacts, "
          f"{N_STAGES - len(CONTACT_STAGES)} non-contact)")
    print(f"merge threshold  = {MERGE_THRESHOLD_H}h   network = {GS_NETWORK_NAMES}")
