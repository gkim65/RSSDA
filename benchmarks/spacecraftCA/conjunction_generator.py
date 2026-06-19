"""
Conjunction generator / sweeper for the spacecraft-CA SDec-POMDP (v2).

Produces a STRUCTURED, DEFENSIBLE set of conjunctions that feed straight into the
existing v2 build path. A conjunction is fully specified, for this reduced model, by:

    (perp_km, dt0_km, v_rel_ms[, sc1_oe])

and these go directly into:
    spacecraft_transition_v2.make_sc2_rel_state_at_tca(perp, dt0, v_rel)
    spacecraft_transition_v2.compute_gain_table_and_perp(perp, dt0, dv)
    compare_variants_v2 :  PERP_KM global + init_b (built from dt0)

Design (see notes/MODEL_DEFINITION.md §5):
  * Conjunction TYPE is the value of `perp`, NOT a new state axis. The state space
    is unchanged; we only choose (perp, dt0, v_rel) per conjunction.
  * We parameterize the sweep by (miss, geometry-angle θ) rather than a raw
    (perp, dt0) grid, because that gives EVEN coverage of the (miss, type) plane —
    the axes the paper actually reports:
          dt0  = miss · cos θ      (along-track, STEERABLE by along-track Δv)
          perp = miss · sin θ      (sideways standoff, UN-steerable)
          θ = 0°  → head-on   (all miss along-track; one burn clears it directly)
          θ = 90° → cross-track (all miss sideways; perp is the irreducible floor —
                     but still cleared by RETIMING: an along-track burn grows δT so
                     miss=√(δT²+perp²) exceeds the threshold. NOT "un-fixable"; the
                     burn just has to out-grow a larger floor. brahe-validated, see
                     MODEL_DEFINITION §5/§6 — quadrature is in fact MOST accurate here.)
          0<θ<90  → oblique
  * v_rel here is the ALONG-TRACK (Ṫ) relative velocity at TCA. Sweeping it sweeps the
    along-track period offset; the FEASIBILITY guard caps the Ṫ-injection at ~75–280 m/s
    (altitude-dependent) — but that ceiling is a property of THIS injection method, NOT a
    limit on real conjunctions. SCOPE (brahe-validated 2026-06-11b, see
    notes/EXPERIMENTAL_SETUP.md): real LEO-LEO crossings at ANY inclination difference
    (up to 120° → ~13 km/s relative velocity) stay LEO and ARE covered — a Δinc=10°
    crossing (1.3 km/s) presents as an oblique (perp,dt0) and the model predicts its
    post-burn miss to ~1–2 km. Such high-Δinc crossings must currently be specified via
    real SC2 Keplerian elements (the Ṫ slot can't express their cross-track velocity);
    adding an Ṅ-velocity hook here is an easy future add. Only GEO/HEO (non-LEO-resident,
    large a / high e) objects are out of scope.

Nothing here changes the model. It only enumerates inputs to it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from brahe import (AngleFormat, R_EARTH, state_eci_to_koe, state_eci_to_rtn,
                   state_koe_to_eci, state_rtn_to_eci)

from spacecraft_matrices import (SC1_OE_AT_TCA, DV_MAGNITUDE, EPOCH_TCA,
                                 propagate_batch_to, sc1_eci_at_tca)
import spacecraft_transition_v2 as TV
import spacecraft_stage_grid as SG


# ---------------------------------------------------------------------------
# Defaults / physical constants for the sweep
# ---------------------------------------------------------------------------

# Default along-track relative velocity at TCA (Ṫ slot). The legacy v2 default is
# 15 m/s; we keep a realistic-but-still-co-orbital default and SWEEP it. The true
# 7–14 km/s LEO crossing speed is NOT an along-track speed — it is represented by
# the cross-track geometry (perp), so it must NOT go in this slot.
DEFAULT_V_REL_MS = 15.0

# Feasibility thresholds for SC2's resulting orbit.
PERIGEE_FLOOR_KM = 150.0      # below this, drag/atmosphere — treat as "through Earth"
APOGEE_CEIL_KM   = 2000.0     # above this we leave LEO; reduced model not validated there
ECC_MAX          = 0.25       # near-circular co-orbital regime; high-e breaks the
                              # "along-track burn acts purely in δT" reduction (§3 caveat)

# Geometry-angle → type label buckets (degrees).
def type_label(angle_deg: float) -> str:
    if angle_deg <= 15.0:
        return "head-on"
    if angle_deg >= 75.0:
        return "cross-track"
    return "oblique"


# ---------------------------------------------------------------------------
# Conjunction record
# ---------------------------------------------------------------------------

@dataclass
class Conjunction:
    miss_km: float
    angle_deg: float          # 0 = head-on (all δT), 90 = cross-track (all perp)
    v_rel_ms: float           # along-track relative velocity at TCA (Ṫ slot)
    perp_km: float            # = miss · sin θ   (UN-steerable standoff)
    dt0_km: float             # = miss · cos θ   (steerable along-track offset)
    sc1_oe: np.ndarray        # SC1 orbital elements used [a,e,i,Ω,ω,M] (a in m, angles deg)
    label: str                # head-on / oblique / cross-track
    feasible: bool = True
    reason: str = ""          # why infeasible (empty if feasible)
    name: Optional[str] = None  # set for named case studies

    # --- orbit-first provenance (None for the geometry-first path A) ---
    # The reduction (sc1_oe, sc2_oe) -> (perp, dt0, v_rel) is LOSSY/many-to-one:
    # only the Ṫ relative-velocity survives into v_rel; Ṙ/Ṅ collapse into the
    # geometry (perp) + the build-time gain. We KEEP sc2_oe so the loss is
    # auditable and `true_miss_km` records the brahe 3-D miss vs the model's
    # quadrature miss for that conjunction. The MODEL still only eats the triple.
    sc2_oe: Optional[np.ndarray] = None   # SC2 derived KOE [a,e,i,Ω,ω,M] (a m, deg)
    true_miss_km: Optional[float] = None  # brahe 3-D closest-approach distance
    at_tca: Optional[bool] = None         # closest approach pinned to EPOCH_TCA?
    quad_miss_km: Optional[float] = None  # model's √(δT²+perp²) for this conj
    vrel_rtn: Optional[np.ndarray] = None # full (Ṙ,Ṫ,Ṅ) rel-velocity at CA (m/s)

    def as_build_args(self) -> dict:
        """Exactly what the v2 build path consumes."""
        return dict(perp_km=self.perp_km, dt0_km=self.dt0_km, v_rel_ms=self.v_rel_ms)

    def __repr__(self):
        tag = f" '{self.name}'" if self.name else ""
        ok = "OK " if self.feasible else "XX "
        return (f"<Conj{tag} {ok}{self.label:<11} miss={self.miss_km:6.2f}km "
                f"θ={self.angle_deg:4.0f}°  perp={self.perp_km:6.2f} dt0={self.dt0_km:7.2f} "
                f"v_rel={self.v_rel_ms:7.1f}m/s" + (f"  ({self.reason})" if self.reason else "") + ">")


# ---------------------------------------------------------------------------
# Feasibility guard
# ---------------------------------------------------------------------------

def sc2_eci_at_tca(perp_km: float, dt0_km: float, v_rel_ms: float,
                   sc1_oe: Optional[np.ndarray] = None) -> np.ndarray:
    """Construct SC2's ECI state at TCA from the conjunction params."""
    if sc1_oe is None:
        sc1 = sc1_eci_at_tca()
    else:
        from brahe import state_koe_to_eci
        sc1 = np.array(state_koe_to_eci(np.asarray(sc1_oe, dtype=float), AngleFormat.DEGREES))
    rel = TV.make_sc2_rel_state_at_tca(perp_km, dt0_km, v_rel_ms)
    return np.array(state_rtn_to_eci(sc1, rel))


def check_feasible(perp_km: float, dt0_km: float, v_rel_ms: float,
                   sc1_oe: Optional[np.ndarray] = None) -> Tuple[bool, str]:
    """
    Reject physically-nonsensical conjunctions by building SC2's actual orbit and
    inspecting its Keplerian elements. Returns (feasible, reason).
    """
    try:
        sc2 = sc2_eci_at_tca(perp_km, dt0_km, v_rel_ms, sc1_oe)
        koe = np.array(state_eci_to_koe(sc2, AngleFormat.DEGREES))  # [a,e,i,Ω,ω,M]
    except Exception as e:                                          # pragma: no cover
        return False, f"propagation/convert error: {e}"

    a, e = float(koe[0]), float(koe[1])
    if not np.isfinite(a) or not np.isfinite(e):
        return False, "non-finite elements"
    if e >= 1.0:
        return False, f"hyperbolic/escape (e={e:.3f})"
    if e > ECC_MAX:
        return False, f"e={e:.3f} > {ECC_MAX} (outside near-circular co-orbital regime)"

    perigee_km = (a * (1.0 - e) - R_EARTH) / 1e3
    apogee_km  = (a * (1.0 + e) - R_EARTH) / 1e3
    if perigee_km < PERIGEE_FLOOR_KM:
        return False, f"perigee {perigee_km:.0f}km < {PERIGEE_FLOOR_KM:.0f}km (into atmosphere)"
    if apogee_km > APOGEE_CEIL_KM:
        return False, f"apogee {apogee_km:.0f}km > {APOGEE_CEIL_KM:.0f}km (leaves LEO)"
    return True, ""


# ---------------------------------------------------------------------------
# Per-conjunction GS contacts (orbit-dependent AVAILABLE set)
# ---------------------------------------------------------------------------

def conjunction_sc2_oe(conj: "Conjunction") -> np.ndarray:
    """SC2 Keplerian elements for a conjunction. Uses the stored orbit-first sc2_oe if
    present; otherwise derives it from the conjunction params (geometry-first path)."""
    if conj.sc2_oe is not None:
        return np.asarray(conj.sc2_oe, dtype=float)
    sc2_eci = sc2_eci_at_tca(conj.perp_km, conj.dt0_km, conj.v_rel_ms, conj.sc1_oe)
    return np.array(state_eci_to_koe(sc2_eci, AngleFormat.DEGREES))


def conjunction_contacts(conj: "Conjunction", hour_grid_h=None,
                         merge_threshold_h=None, stations=None, sync_rule="later"):
    """Compute the AVAILABLE GS-contact stage set for THIS conjunction's orbits, against
    the static network. The available set is orbit-DERIVED (depends on conj.sc1_oe /
    sc2_oe); --contact-stages later picks a SUBSET of it. Returns (all_times_h_desc,
    contact_stage_indices) from the union-with-merge grid (see spacecraft_stage_grid)."""
    return SG.compute_stage_grid(
        sc1_oe=np.asarray(conj.sc1_oe, dtype=float),
        sc2_oe=conjunction_sc2_oe(conj),
        hour_grid_h=hour_grid_h, merge_threshold_h=merge_threshold_h,
        stations=stations, sync_rule=sync_rule,
    )


def check_contact_subset(requested, available):
    """The Scenario-1 ablation (--contact-stages) picks a SUBSET of the orbit-derived
    AVAILABLE contact set (only used for SDec subsetting experiments; Cen syncs at all
    stages, Dec at none). HARD ERROR if any requested stage isn't physically available
    for this conjunction (don't silently sync where there's no contact). Returns the
    requested list sorted if valid."""
    avail = set(int(a) for a in available)
    bad = sorted(int(r) for r in requested if int(r) not in avail)
    if bad:
        raise ValueError(
            f"--contact-stages {bad} not in this conjunction's AVAILABLE contact set "
            f"{sorted(avail)}. The SDec subset must be drawn from available contacts.")
    return sorted(int(r) for r in requested)


def apply_conjunction_contacts(conj: "Conjunction", **kwargs):
    """Compute this conjunction's available contacts AND set them as the live contact
    stages (SG.set_contact_stages, the single source of truth read by the matrix builders
    and SDec sync_states). Call before building/solving for a generated conjunction.
    Returns (all_times_h, contact_stage_indices). NOTE: if N_STAGES changes vs the current
    grid, callers must rebuild matrices (the stage count is orbit-dependent)."""
    all_times, contact_idx = conjunction_contacts(conj, **kwargs)
    SG.set_contact_stages(contact_idx)
    return all_times, contact_idx


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def make_conjunction(miss_km: float, angle_deg: float,
                     v_rel_ms: float = DEFAULT_V_REL_MS,
                     sc1_oe: Optional[np.ndarray] = None,
                     name: Optional[str] = None) -> Conjunction:
    """Build a single conjunction record (with feasibility checked)."""
    th = np.deg2rad(angle_deg)
    perp = miss_km * np.sin(th)
    dt0  = miss_km * np.cos(th)
    oe = np.asarray(sc1_oe if sc1_oe is not None else SC1_OE_AT_TCA, dtype=float)
    ok, reason = check_feasible(perp, dt0, v_rel_ms, oe)
    return Conjunction(miss_km=miss_km, angle_deg=angle_deg, v_rel_ms=v_rel_ms,
                       perp_km=perp, dt0_km=dt0, sc1_oe=oe, label=type_label(angle_deg),
                       feasible=ok, reason=reason, name=name)


def max_feasible_v_rel(miss_km: float, angle_deg: float,
                       sc1_oe: Optional[np.ndarray] = None,
                       lo: float = 5.0, hi: float = 1.0e4,
                       tol: float = 1.0) -> float:
    """
    Bisection for the largest along-track v_rel that is still feasible for this
    geometry — "let feasibility decide the upper bound". Returns lo if even lo is
    infeasible. Assumes feasibility is monotone-decreasing in v_rel (larger v_rel →
    more orbit separation → eventually perigee/ecc violation), which holds because
    v_rel sets the period offset.
    """
    if not make_conjunction(miss_km, angle_deg, lo, sc1_oe).feasible:
        return lo
    if make_conjunction(miss_km, angle_deg, hi, sc1_oe).feasible:
        return hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if make_conjunction(miss_km, angle_deg, mid, sc1_oe).feasible:
            lo = mid
        else:
            hi = mid
    return lo


def sweep(miss_levels: List[float],
          angles_deg: List[float],
          v_rels_ms: Optional[List[float]] = None,
          sc1_oe: Optional[np.ndarray] = None,
          v_rel_steps: int = 4,
          keep_infeasible: bool = False) -> List[Conjunction]:
    """
    Structured cross-product sweep over (miss × angle × v_rel).

    If `v_rels_ms` is None, the v_rel axis is auto-discovered PER (miss, angle):
    log-spaced from 5 m/s up to the bisection-found feasibility ceiling, giving
    `v_rel_steps` points. This is the "let feasibility decide the upper bound" mode.
    Otherwise the explicit v_rels are used (and infeasible ones flagged).

    Returns feasible conjunctions only unless keep_infeasible=True (the rejected
    ones carry a `reason` and are useful for the coverage figure).
    """
    out: List[Conjunction] = []
    for miss in miss_levels:
        for ang in angles_deg:
            if v_rels_ms is None:
                vmax = max_feasible_v_rel(miss, ang, sc1_oe)
                vs = np.unique(np.round(
                    np.geomspace(5.0, max(vmax, 5.0 + 1e-6), v_rel_steps), 2)).tolist()
            else:
                vs = v_rels_ms
            for v in vs:
                c = make_conjunction(miss, ang, v, sc1_oe)
                if c.feasible or keep_infeasible:
                    out.append(c)
    return out


# ===========================================================================
# ORBIT-FIRST PATH (B) — inverted construction & search
# ===========================================================================
#
# Path A (above) sets geometry (miss, θ) with all relative velocity in the Ṫ
# slot — clean for a co-orbital sweep but cannot express inclination crossings.
# Path B INVERTS it: place the conjunction on SC1's orbit, choose the full miss
# VECTOR and the full relative-velocity VECTOR (any direction, all 3 RTN slots),
# derive SC2's REAL orbit via brahe, then VERIFY against brahe and REDUCE to the
# model's (perp, dt0, v_rel). This reaches the km/s inclination crossings that A
# could not, and yields SC2's real KOE.
#
# Two gaps from the prototype (notes/scratch/inverted_construct_prototype.py),
# both fixed here:
#   (a) the miss vector now carries a T-component (along-track / oblique geometry),
#       and the velocity vector populates all of Ṙ/Ṫ/Ṅ — not just Ṫ.
#   (b) closest approach is PINNED to EPOCH_TCA by a scalar epoch root-find, so the
#       constructed miss is the miss AT TCA (the @TCA=False prototype rows are gone).
# ---------------------------------------------------------------------------


def _closest_approach(sc1_tca: np.ndarray, sc2_eci: np.ndarray,
                      sc2_epoch, span_s: float = 600.0):
    """
    Brahe closest-approach search between SC1 (state @EPOCH_TCA) and SC2 (state
    @sc2_epoch), over a window ±span_s around EPOCH_TCA. Returns
    (min_dist_km, t_ca_offset_s, rel_rtn_at_ca) where the offset is seconds from
    EPOCH_TCA and rel_rtn is SC2's RTN state relative to SC1 at the true CA.

    Golden-section refine on top of a coarse grid (no scipy dependency).
    """
    def sep(dt):
        ep = EPOCH_TCA + float(dt)
        s1 = propagate_batch_to([EPOCH_TCA], [sc1_tca], ep)[0]
        s2 = propagate_batch_to([sc2_epoch], [sc2_eci], ep)[0]
        return np.linalg.norm(np.asarray(s1)[:3] - np.asarray(s2)[:3]) / 1e3

    # coarse grid -> bracket the minimum
    grid = np.linspace(-span_s, span_s, 41)
    dvals = np.array([sep(t) for t in grid])
    k = int(np.argmin(dvals))
    lo = grid[max(k - 1, 0)]
    hi = grid[min(k + 1, len(grid) - 1)]

    # golden-section
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = sep(c), sep(d)
    for _ in range(40):
        if abs(b - a) < 1e-3:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = sep(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = sep(d)
    t_ca = 0.5 * (a + b)
    ep = EPOCH_TCA + float(t_ca)
    s1 = propagate_batch_to([EPOCH_TCA], [sc1_tca], ep)[0]
    s2 = propagate_batch_to([sc2_epoch], [sc2_eci], ep)[0]
    rel_rtn = np.array(state_eci_to_rtn(np.asarray(s1), np.asarray(s2)))
    return sep(t_ca), t_ca, rel_rtn


def _reduce_rel_to_params(rel_rtn: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
    """
    Reduce a full RTN relative state [R,T,N, Ṙ,Ṫ,Ṅ] (m, m/s) to the model's
    (perp_km, dt0_km, v_rel_ms). LOSSY: perp = √(R²+N²) (sideways standoff),
    dt0 = T (along-track offset), v_rel = -Ṫ (along-track closing speed). The
    Ṙ/Ṅ velocity components are dropped (they live in the geometry + gain, not
    the state — see MODEL_DEFINITION §5). Returns (perp, dt0, v_rel, vrel_rtn).
    """
    R, T, N = rel_rtn[0], rel_rtn[1], rel_rtn[2]
    perp = float(np.hypot(R, N) / 1e3)
    dt0 = float(T / 1e3)
    v_rel = float(-rel_rtn[4])          # closing along-track speed (m/s)
    vrel_rtn = np.array([rel_rtn[3], rel_rtn[4], rel_rtn[5]], dtype=float)
    return perp, dt0, v_rel, vrel_rtn


def make_conjunction_from_encounter(miss_km: float, beta_deg: float, phi_deg: float,
                                    v_rel_ms: float,
                                    sc1_oe: Optional[np.ndarray] = None,
                                    name: Optional[str] = None,
                                    verify_tol_km: float = 0.5) -> Conjunction:
    """
    INVERTED constructor. Place a conjunction on SC1's orbit and derive SC2's real
    orbit, then verify + reduce.

    Parameters
    ----------
    miss_km   : requested closest-approach distance (km).
    beta_deg  : MISS-vector direction in RTN. The miss vector is rotated in the
                radial-T-N space: beta=0 -> pure radial (perp), beta=90 -> pure
                along-track (dt0), with cross-track folded into perp. Gap (a) fix:
                the T-component is now real, so oblique/along-track geometries work.
    phi_deg   : RELATIVE-VELOCITY direction in RTN: phi=0 -> pure along-track (Ṫ,
                co-orbital head-on), phi=90 -> pure cross-track (Ṅ, an inclination
                crossing). This is the all-3-slots velocity the prototype lacked.
    v_rel_ms  : magnitude of the relative velocity at TCA (m/s).
    sc1_oe    : chief KOE [a,e,i,Ω,ω,M] (defaults to SC1_OE_AT_TCA).

    Returns a Conjunction with the reduced (perp,dt0,v_rel) the model consumes
    PLUS sc2_oe / true_miss_km / at_tca / quad_miss_km / vrel_rtn provenance.
    """
    oe = np.asarray(sc1_oe if sc1_oe is not None else SC1_OE_AT_TCA, dtype=float)
    sc1_tca = np.array(state_koe_to_eci(oe, AngleFormat.DEGREES))

    b = np.deg2rad(beta_deg)
    p = np.deg2rad(phi_deg)
    # velocity: phi rotates along-track(-Ṫ, closing) -> cross-track(Ṅ).
    vrel_rtn = np.array([0.0, -v_rel_ms * np.cos(p), -v_rel_ms * np.sin(p)])

    # GAP (a)+(b) FIX, together. The closest approach occurs where rel_pos ⊥ rel_vel
    # (separation rate = 0). So the miss VECTOR must lie in the plane perpendicular
    # to vrel — otherwise the placed epoch is NOT the CA and the requested miss is
    # not realized at TCA (the prototype's @TCA=False rows). This also means an
    # along-track miss (dt0) and an along-track closing velocity (Ṫ) are NOT
    # independent: a pure along-track separation with along-track velocity just
    # means "not yet at closest approach." We therefore build the miss inside the
    # ⊥-plane, with beta rotating WITHIN it. dt0 (the model's along-track offset at
    # TCA) then falls out as the T-projection of that perpendicular miss.
    v_hat = vrel_rtn / np.linalg.norm(vrel_rtn)
    # two orthonormal axes spanning the plane ⊥ v_hat: u1 has the along-track (T)
    # content (so beta controls how much of the miss is along-track), u2 fills it out.
    t_axis = np.array([0.0, 1.0, 0.0])
    u1 = t_axis - np.dot(t_axis, v_hat) * v_hat
    if np.linalg.norm(u1) < 1e-9:           # vrel is purely along-track (phi=0):
        u1 = np.array([1.0, 0.0, 0.0])      # no along-track miss possible -> use radial
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(v_hat, u1); u2 /= np.linalg.norm(u2)
    miss_rtn = miss_km * 1e3 * (np.cos(b) * u2 + np.sin(b) * u1)
    rel0 = np.concatenate([miss_rtn, vrel_rtn])

    sc2_eci = np.array(state_rtn_to_eci(sc1_tca, rel0))
    koe = np.array(state_eci_to_koe(sc2_eci, AngleFormat.DEGREES))

    # GAP (b) FIX: find the TRUE closest approach and pin-check it to EPOCH_TCA.
    true_miss, t_ca, rel_ca = _closest_approach(sc1_tca, sc2_eci, EPOCH_TCA)
    at_tca = abs(t_ca) < 1.0        # within 1 s of the nominal TCA epoch

    # Reduce the verified relative state at the TRUE closest approach.
    perp, dt0, v_rel_reduced, vrel_full = _reduce_rel_to_params(rel_ca)
    quad_miss = float(np.hypot(perp, dt0))

    # feasibility: is SC2 a LEO-resident near-circular orbit?
    a, e = float(koe[0]), float(koe[1])
    feasible, reason = True, ""
    if not (np.isfinite(a) and np.isfinite(e)):
        feasible, reason = False, "non-finite elements"
    elif e >= 1.0:
        feasible, reason = False, f"hyperbolic/escape (e={e:.3f})"
    elif e > ECC_MAX:
        feasible, reason = False, f"e={e:.3f} > {ECC_MAX}"
    else:
        peri = (a * (1.0 - e) - R_EARTH) / 1e3
        apo = (a * (1.0 + e) - R_EARTH) / 1e3
        if peri < PERIGEE_FLOOR_KM:
            feasible, reason = False, f"perigee {peri:.0f}km < {PERIGEE_FLOOR_KM:.0f}km"
        elif apo > APOGEE_CEIL_KM:
            feasible, reason = False, f"apogee {apo:.0f}km > {APOGEE_CEIL_KM:.0f}km"

    # brahe verification: did we actually build the requested miss, at TCA?
    if feasible:
        if abs(true_miss - miss_km) > max(verify_tol_km, 0.02 * miss_km):
            reason = (f"verify FAIL: true_miss {true_miss:.2f} != requested "
                      f"{miss_km:.2f} km")
        elif not at_tca:
            reason = f"verify WARN: CA off TCA by {t_ca:.1f}s"

    angle = float(np.rad2deg(np.arctan2(perp, abs(dt0)))) if (perp or dt0) else 0.0
    return Conjunction(
        miss_km=miss_km, angle_deg=angle, v_rel_ms=v_rel_reduced,
        perp_km=perp, dt0_km=dt0, sc1_oe=oe, label=type_label(angle),
        feasible=feasible, reason=reason, name=name,
        sc2_oe=koe, true_miss_km=float(true_miss), at_tca=bool(at_tca),
        quad_miss_km=quad_miss, vrel_rtn=vrel_full)


def make_conjunction_from_orbits(sc1_koe: np.ndarray, sc2_koe: np.ndarray,
                                 span_s: float = 600.0,
                                 name: Optional[str] = None) -> Conjunction:
    """
    SEARCH constructor for REAL catalogued / TLE orbit pairs. Given two KOEs
    (both [a,e,i,Ω,ω,M], a in m, angles deg, anomaly = M at EPOCH_TCA), find the
    actual closest approach over ±span_s around EPOCH_TCA, read off the geometry,
    verify, and reduce to (perp, dt0, v_rel).

    NB: unlike the inverted constructor this does NOT guarantee a conjunction — it
    REPORTS whatever closest approach the two real orbits have in the window. A huge
    `true_miss_km` just means these two orbits don't actually conjunct here. Pass
    M phased so the encounter falls in the window (or widen span_s / sweep M).
    """
    sc1_oe = np.asarray(sc1_koe, dtype=float)
    sc2_oe = np.asarray(sc2_koe, dtype=float)
    sc1_tca = np.array(state_koe_to_eci(sc1_oe, AngleFormat.DEGREES))
    sc2_tca = np.array(state_koe_to_eci(sc2_oe, AngleFormat.DEGREES))

    true_miss, t_ca, rel_ca = _closest_approach(sc1_tca, sc2_tca, EPOCH_TCA, span_s)
    at_tca = abs(t_ca) < 1.0
    perp, dt0, v_rel, vrel_full = _reduce_rel_to_params(rel_ca)
    quad_miss = float(np.hypot(perp, dt0))

    # feasibility of SC2's own (already-real) orbit
    a, e = float(sc2_oe[0]), float(sc2_oe[1])
    feasible, reason = True, ""
    peri = (a * (1.0 - e) - R_EARTH) / 1e3
    apo = (a * (1.0 + e) - R_EARTH) / 1e3
    if e >= 1.0:
        feasible, reason = False, f"hyperbolic (e={e:.3f})"
    elif e > ECC_MAX:
        feasible, reason = False, f"e={e:.3f} > {ECC_MAX} (non-co-orbital)"
    elif peri < PERIGEE_FLOOR_KM:
        feasible, reason = False, f"perigee {peri:.0f}km < {PERIGEE_FLOOR_KM:.0f}km"
    elif apo > APOGEE_CEIL_KM:
        feasible, reason = False, f"apogee {apo:.0f}km > {APOGEE_CEIL_KM:.0f}km (non-LEO)"

    angle = float(np.rad2deg(np.arctan2(perp, abs(dt0)))) if (perp or dt0) else 0.0
    return Conjunction(
        miss_km=float(true_miss), angle_deg=angle, v_rel_ms=v_rel,
        perp_km=perp, dt0_km=dt0, sc1_oe=sc1_oe, label=type_label(angle),
        feasible=feasible, reason=reason, name=name,
        sc2_oe=sc2_oe, true_miss_km=float(true_miss), at_tca=bool(at_tca),
        quad_miss_km=quad_miss, vrel_rtn=vrel_full)


# ===========================================================================
# COVERAGE DATASET (orbit-first, brahe-measured) — feeds the scope figures
# ===========================================================================
#
# The coverage figures need the MEASURED grid (perp, dt0, v_rel, true_miss, SC2
# a/e/inc, feasible, reduction error) as a usable dataset, NOT scraped stdout.
# `coverage_dataset` builds it from the real orbit-first constructor over a
# (miss × beta × phi × vrel) grid; `coverage_to_rows` / the CSV cache let a
# plotting script regenerate once and re-read cheaply.
# ---------------------------------------------------------------------------

# Columns emitted to the CSV cache (order is the dtype order below).
COVERAGE_FIELDS = [
    "miss_km", "beta_deg", "phi_deg", "vrel_in_ms",   # requested grid point
    "feasible", "reason",                              # scope verdict
    "true_miss_km", "at_tca", "quad_miss_km", "red_err_km",  # brahe verification
    "perp_km", "dt0_km", "v_rel_ms", "angle_deg", "label",   # reduced model triple
    "sc2_a_km", "sc2_e", "sc2_inc_deg",               # SC2 real orbit (provenance)
    "sc2_peri_km", "sc2_apo_km",                       # derived peri/apo altitude
]

# Default coverage grid — spans the along→oblique→cross spectrum (phi), the
# miss-vector direction (beta), miss magnitude, and the velocity regime (vrel)
# that drives the feasibility boundary. The phi×vrel axes are sampled finely so
# the (phi × vrel) coverage grid (Figure B) resolves the scope boundary sharply;
# beta/miss are collapsed in that figure so they stay coarse.
DEFAULT_COVERAGE_GRID = dict(
    miss_levels=[2.0, 5.0],
    betas_deg=[0.0, 45.0, 90.0],
    # DENSE phi × vrel so the coverage HEATMAP (Figure B) resolves the feasible
    # region including the phi≈45° / high-vrel RIDGE (a 90° plane crossing has its
    # relative velocity split 50/50 along/cross → phi=45; the along-track half
    # cancels SC1's orbital speed so SC2 stays circular). vrel reaches ~12 km/s to
    # cover that ridge (a true perpendicular LEO crossing is ~10.7 km/s).
    phis_deg=list(np.linspace(0.0, 90.0, 19)),          # 5° steps
    vrels_ms=list(np.geomspace(15.0, 12000.0, 16)),     # log, reaches the ridge
)


def coverage_dataset(miss_levels=None, betas_deg=None, phis_deg=None,
                     vrels_ms=None, sc1_oe: Optional[np.ndarray] = None
                     ) -> List[Conjunction]:
    """
    Build the orbit-first coverage grid as a list of brahe-measured Conjunctions.

    Each cell runs make_conjunction_from_encounter (real SC2 KOE + brahe-verified
    miss). Both feasible and infeasible cells are returned (infeasible carry a
    `reason`) — the scope figures need both. Defaults to DEFAULT_COVERAGE_GRID.
    """
    g = DEFAULT_COVERAGE_GRID
    miss_levels = miss_levels if miss_levels is not None else g["miss_levels"]
    betas_deg = betas_deg if betas_deg is not None else g["betas_deg"]
    phis_deg = phis_deg if phis_deg is not None else g["phis_deg"]
    vrels_ms = vrels_ms if vrels_ms is not None else g["vrels_ms"]

    out: List[Conjunction] = []
    for miss in miss_levels:
        for beta in betas_deg:
            for phi in phis_deg:
                for vrel in vrels_ms:
                    out.append(make_conjunction_from_encounter(
                        miss, beta, phi, vrel, sc1_oe=sc1_oe))
    return out


def coverage_to_rows(conjs: List[Conjunction],
                     grid_meta: Optional[List[Tuple[float, float, float, float]]] = None
                     ) -> List[dict]:
    """
    Flatten measured Conjunctions to plain dict rows (COVERAGE_FIELDS order),
    deriving SC2 peri/apo altitude and the reduction error. `grid_meta`, if given,
    supplies the requested (miss, beta, phi, vrel_in) per conjunction; otherwise
    those columns fall back to the conjunction's own (post-reduction) values.
    """
    rows: List[dict] = []
    for i, c in enumerate(conjs):
        koe = c.sc2_oe
        if koe is not None:
            a_km = float(koe[0]) / 1e3
            e = float(koe[1]); inc = float(koe[2])
            peri = (float(koe[0]) * (1.0 - e) - R_EARTH) / 1e3
            apo = (float(koe[0]) * (1.0 + e) - R_EARTH) / 1e3
        else:
            a_km = e = inc = peri = apo = float("nan")
        red_err = (float("nan") if (c.true_miss_km is None or c.quad_miss_km is None)
                   else c.true_miss_km - c.quad_miss_km)
        if grid_meta is not None:
            req_miss, beta, phi, vrel_in = grid_meta[i]
        else:
            req_miss, beta, phi, vrel_in = c.miss_km, np.nan, np.nan, c.v_rel_ms
        rows.append(dict(
            miss_km=req_miss, beta_deg=beta, phi_deg=phi, vrel_in_ms=vrel_in,
            feasible=int(bool(c.feasible)), reason=c.reason,
            true_miss_km=c.true_miss_km, at_tca=int(bool(c.at_tca)),
            quad_miss_km=c.quad_miss_km, red_err_km=red_err,
            perp_km=c.perp_km, dt0_km=c.dt0_km, v_rel_ms=c.v_rel_ms,
            angle_deg=c.angle_deg, label=c.label,
            sc2_a_km=a_km, sc2_e=e, sc2_inc_deg=inc,
            sc2_peri_km=peri, sc2_apo_km=apo))
    return rows


def write_coverage_csv(path: str, miss_levels=None, betas_deg=None,
                       phis_deg=None, vrels_ms=None,
                       sc1_oe: Optional[np.ndarray] = None) -> List[dict]:
    """
    Build the coverage grid and CACHE it to CSV so re-plotting doesn't rebuild
    (each cell does a brahe CA search — not free). Returns the rows. The grid
    points are recorded so beta/phi/vrel_in survive into the CSV.
    """
    import csv
    g = DEFAULT_COVERAGE_GRID
    miss_levels = miss_levels if miss_levels is not None else g["miss_levels"]
    betas_deg = betas_deg if betas_deg is not None else g["betas_deg"]
    phis_deg = phis_deg if phis_deg is not None else g["phis_deg"]
    vrels_ms = vrels_ms if vrels_ms is not None else g["vrels_ms"]

    conjs, meta = [], []
    for miss in miss_levels:
        for beta in betas_deg:
            for phi in phis_deg:
                for vrel in vrels_ms:
                    conjs.append(make_conjunction_from_encounter(
                        miss, beta, phi, vrel, sc1_oe=sc1_oe))
                    meta.append((miss, beta, phi, vrel))
    rows = coverage_to_rows(conjs, meta)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COVERAGE_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}  ({len(rows)} rows)")
    return rows


def read_coverage_csv(path: str) -> List[dict]:
    """Read back the cached coverage CSV, coercing numeric columns."""
    import csv
    num = set(COVERAGE_FIELDS) - {"reason", "label", "feasible", "at_tca"}
    rows: List[dict] = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            for k in num:
                r[k] = float(r[k]) if r[k] not in ("", "nan") else float("nan")
            r["feasible"] = int(r["feasible"]); r["at_tca"] = int(r["at_tca"])
            rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Named case studies for the explanatory figures
# ---------------------------------------------------------------------------

def case_studies(sc1_oe: Optional[np.ndarray] = None) -> dict:
    """A handful of NAMED exemplars (one per geometry type) for figures/text."""
    return {
        "head_on":     make_conjunction(5.0,  0.0, DEFAULT_V_REL_MS, sc1_oe, name="head_on"),
        "oblique":     make_conjunction(5.0, 45.0, DEFAULT_V_REL_MS, sc1_oe, name="oblique"),
        "cross_track": make_conjunction(5.0, 90.0, DEFAULT_V_REL_MS, sc1_oe, name="cross_track"),
    }


# ---------------------------------------------------------------------------
# Coverage figure (appendix)
# ---------------------------------------------------------------------------

def plot_coverage(conjs: List[Conjunction], rejected: List[Conjunction],
                  out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))

    # 1: (δT, perp) plane with iso-miss arcs
    dt = np.array([c.dt0_km for c in conjs]); pp = np.array([c.perp_km for c in conjs])
    ax[0].scatter(dt, pp, s=40, c="tab:blue", label=f"feasible ({len(conjs)})")
    if rejected:
        rdt = np.array([c.dt0_km for c in rejected]); rpp = np.array([c.perp_km for c in rejected])
        ax[0].scatter(rdt, rpp, s=40, marker="x", c="tab:red",
                      label=f"infeasible ({len(rejected)})")
    for m in sorted({round(c.miss_km, 3) for c in conjs}):
        t = np.linspace(0, np.pi / 2, 60)
        ax[0].plot(m * np.cos(t), m * np.sin(t), color="0.7", lw=0.5, zorder=0)
    ax[0].set_xlabel("δT (along-track, km) — steerable")
    ax[0].set_ylabel("perp (sideways, km) — UN-steerable")
    ax[0].set_title("Geometry coverage (δT, perp)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3); ax[0].set_aspect("equal")

    # 2: (miss, angle) — even type coverage
    ax[1].scatter([c.miss_km for c in conjs], [c.angle_deg for c in conjs],
                  s=40, c="tab:blue")
    ax[1].set_xscale("log")
    ax[1].axhline(15, color="0.7", lw=0.5); ax[1].axhline(75, color="0.7", lw=0.5)
    ax[1].set_xlabel("total miss (km)")
    ax[1].set_ylabel("geometry angle θ (deg)")
    ax[1].set_title("Type coverage: 0=head-on … 90=cross-track")
    ax[1].grid(alpha=0.3)

    # 3: v_rel feasibility boundary vs angle
    cols = {"head-on": "tab:green", "oblique": "tab:orange", "cross-track": "tab:purple"}
    for c in conjs:
        ax[2].scatter(c.angle_deg, c.v_rel_ms, s=28, c=cols.get(c.label, "0.5"))
    for c in rejected:
        ax[2].scatter(c.angle_deg, c.v_rel_ms, s=28, marker="x", c="tab:red")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("geometry angle θ (deg)")
    ax[2].set_ylabel("along-track v_rel at TCA (m/s)")
    ax[2].set_title("v_rel coverage (× = feasibility-rejected)")
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_floats(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="Conjunction generator / sweeper (v2)")
    ap.add_argument("--miss", default="1,2,5,10,20",
                    help="comma list of total miss magnitudes (km)")
    ap.add_argument("--angles", default="0,15,30,45,60,75,90",
                    help="comma list of geometry angles (deg); 0=head-on,90=cross-track")
    ap.add_argument("--v-rel", default=None,
                    help="comma list of along-track v_rel (m/s); omit to AUTO-discover "
                         "the feasibility-bounded range per geometry")
    ap.add_argument("--v-rel-steps", type=int, default=4,
                    help="points on the auto-discovered v_rel axis")
    ap.add_argument("--keep-infeasible", action="store_true",
                    help="include rejected conjunctions (for the coverage figure)")
    ap.add_argument("--cases", action="store_true", help="print the named case studies")
    ap.add_argument("--plot", default=None, help="path to write the coverage figure PNG")
    args = ap.parse_args()

    miss = _parse_floats(args.miss)
    angles = _parse_floats(args.angles)
    vrels = _parse_floats(args.v_rel) if args.v_rel else None

    if args.cases:
        print("=== named case studies ===")
        for k, c in case_studies().items():
            print(f"  {k:12s} {c!r}")
        print()

    feas = sweep(miss, angles, vrels, v_rel_steps=args.v_rel_steps, keep_infeasible=False)
    rej = []
    if args.keep_infeasible or args.plot:
        allc = sweep(miss, angles, vrels, v_rel_steps=args.v_rel_steps, keep_infeasible=True)
        rej = [c for c in allc if not c.feasible]

    print(f"=== sweep: {len(miss)}×{len(angles)} geometries, "
          f"{'explicit' if vrels else 'auto'} v_rel → {len(feas)} feasible"
          + (f", {len(rej)} infeasible" if rej else "") + " ===")
    for c in feas:
        print(" ", repr(c))
    if rej:
        print("--- rejected ---")
        for c in rej:
            print(" ", repr(c))

    if args.plot:
        plot_coverage(feas, rej, args.plot)


if __name__ == "__main__":
    main()
