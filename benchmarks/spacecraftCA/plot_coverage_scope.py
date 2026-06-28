"""
Coverage / scope figures for the orbit-first conjunction generator (v2).

Renders THREE figures from REAL orbit-first points (conjunction_generator.
make_conjunction_from_encounter), cached to a CSV so re-plotting is cheap:

  A  leo_coverage_physical.png — discrete orbit insets (along-track / oblique /
     cross-track / OUT-of-scope), each shown as ECI orbit + RTN encounter, plus
     the SC2 (a,e) scope map with the LEO-resident IN box and measured points.
  B  leo_coverage_abstract.png — SC2 (inc,e) scope map + a (phi × vrel) coverage
     grid with per-cell feasibility + brahe reduction error.
  C  leo_coverage_continuous.png — CONTINUOUS span: semi-transparent SC2 orbit
     ellipses swept along phi (along→cross velocity), beta (miss direction), and
     vrel (closing speed), in ECI x-y + x-z, so the along→oblique→crossing family
     is visible as a continuum, not isolated dots.

Run from benchmarks/spacecraftCA/:
    PYTHONPATH=. ../../.venv/bin/python -u plot_coverage_scope.py
    PYTHONPATH=. ../../.venv/bin/python -u plot_coverage_scope.py --rebuild
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from brahe import AngleFormat, R_EARTH, state_koe_to_eci, state_eci_to_rtn

import conjunction_generator as G
from spacecraft_matrices import SC1_OE_AT_TCA, EPOCH_TCA, propagate_batch_to

R_E_KM = R_EARTH / 1e3
# Absolute (anchored at this file) so figures land next to this script regardless of cwd
# (running from the repo root previously wrote to ./notes/figures/ at the root).
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notes", "figures")
CSV_PATH = "notes/scratch/coverage_grid.csv"

# Feasibility box (mirrors the generator guards) in (a-altitude, e) and peri/apo.
PERI_FLOOR = G.PERIGEE_FLOOR_KM
APO_CEIL = G.APOGEE_CEIL_KM
ECC_MAX = G.ECC_MAX


# ---------------------------------------------------------------------------
# Orbit geometry helpers (analytic Keplerian ellipse from KOE, for pictures)
# ---------------------------------------------------------------------------

def koe_ellipse_eci(koe: np.ndarray, n: int = 240) -> np.ndarray:
    """Return (n,3) ECI points tracing the full orbit for KOE [a,e,i,Ω,ω,M]."""
    a, e, inc, raan, argp, _ = [float(x) for x in koe]
    nu = np.linspace(0.0, 360.0, n)
    pts = []
    for v in nu:
        st = np.array(state_koe_to_eci(
            np.array([a, e, inc, raan, argp, v]), AngleFormat.DEGREES))
        pts.append(st[:3])
    return np.array(pts) / 1e3  # km


def rtn_encounter_track(sc1_oe: np.ndarray, sc2_oe: np.ndarray,
                        span_s: float = 60.0, n: int = 121) -> np.ndarray:
    """SC2 position in SC1's RTN frame over ±span_s around TCA (km), (n,3)."""
    sc1 = np.array(state_koe_to_eci(np.asarray(sc1_oe, float), AngleFormat.DEGREES))
    sc2 = np.array(state_koe_to_eci(np.asarray(sc2_oe, float), AngleFormat.DEGREES))
    ts = np.linspace(-span_s, span_s, n)
    out = []
    for dt in ts:
        ep = EPOCH_TCA + float(dt)
        s1 = propagate_batch_to([EPOCH_TCA], [sc1], ep)[0]
        s2 = propagate_batch_to([EPOCH_TCA], [sc2], ep)[0]
        rtn = np.array(state_eci_to_rtn(np.asarray(s1), np.asarray(s2)))
        out.append(rtn[:3])
    return np.array(out) / 1e3


def earth_circle(r=R_E_KM, n=200):
    t = np.linspace(0, 2 * np.pi, n)
    return r * np.cos(t), r * np.sin(t)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_rows(rebuild: bool) -> list:
    if rebuild or not os.path.exists(CSV_PATH):
        return G.write_coverage_csv(CSV_PATH)
    return G.read_coverage_csv(CSV_PATH)


# ---------------------------------------------------------------------------
# FIGURE A — physical insets + (a,e) scope map
# ---------------------------------------------------------------------------

def _inset_pair(fig, gs_cell, conj, title, color):
    """Draw an ECI orbit (x-z, occluded) + an RTN encounter mini-pair."""
    inner = gs_cell.subgridspec(1, 2, wspace=0.35)
    sc1_oe = conj.sc1_oe
    axo = fig.add_subplot(inner[0, 0])
    # SC1 + SC2 ECI orbits — x-z projection (shows inclination plane-tilt), with
    # behind-Earth arcs faint-dashed (depth axis = y), same cue as Figure C.
    o1 = koe_ellipse_eci(sc1_oe); o2 = koe_ellipse_eci(conj.sc2_oe)
    h, v, d = _PROJ["xz"]
    ex, ey = earth_circle()
    axo.fill(ex, ey, color="0.85", zorder=1)
    _draw_occluded(axo, o1, "tab:blue", 1.4, 1.0, (0, (1, 1)), h, v, d, zbase=2)
    _draw_occluded(axo, o2, color, 1.4, 1.0, (0, (4, 3)), h, v, d, zbase=4)
    axo.plot([], [], color="tab:blue", lw=1.4, label="SC1")
    axo.plot([], [], color=color, lw=1.4, label="SC2")
    axo.set_aspect("equal"); axo.set_xticks([]); axo.set_yticks([])
    axo.set_title("ECI x-z (behind Earth = faint)", fontsize=7)
    axo.legend(fontsize=5, loc="upper right", framealpha=0.6)

    axr = fig.add_subplot(inner[0, 1])
    trk = rtn_encounter_track(sc1_oe, conj.sc2_oe)
    axr.plot(trk[:, 1], trk[:, 2], color=color, lw=1.2)   # T (along) vs N (cross)
    axr.scatter([0], [0], c="tab:blue", s=22, zorder=3, label="SC1")
    k = len(trk) // 2
    axr.scatter([trk[k, 1]], [trk[k, 2]], c=color, s=22, zorder=3,
                edgecolor="k", lw=0.4, label="TCA")
    axr.axhline(0, color="0.8", lw=0.4)
    axr.axvline(0, color="0.8", lw=0.4)
    # bound the view to a few × the miss so a fast cross-track pass stays readable
    w = max(conj.miss_km * 2.5, 3.0)
    axr.set_xlim(-w, w); axr.set_ylim(-w, w); axr.set_aspect("equal")
    axr.set_xlabel("T (km)", fontsize=6); axr.set_ylabel("N (km)", fontsize=6)
    axr.tick_params(labelsize=5)
    axr.set_title("RTN encounter", fontsize=7)
    axr.legend(fontsize=5, loc="upper right", framealpha=0.6)

    sub = (f"{title}\nperp={conj.perp_km:.1f} dt0={conj.dt0_km:.1f}km  "
           f"e={conj.sc2_oe[1]:.3f} inc={conj.sc2_oe[2]:.1f}°  "
           f"{'IN scope' if conj.feasible else 'OUT: ' + conj.reason}")
    axo.annotate(sub, xy=(0, 1.42), xycoords="axes fraction", fontsize=7.5,
                 ha="left", va="bottom",
                 color=("tab:green" if conj.feasible else "tab:red"))


def _plane_crossing(angle_deg: float = 85.0, miss_km: float = 5.0):
    """A REAL near-perpendicular plane crossing at a chosen plane-to-plane angle.

    Builds SC2 as a co-altitude CIRCULAR orbit whose plane is exactly `angle_deg`
    from SC1's, passing within ~miss_km at TCA, by rotating SC1's velocity about
    the radial axis (this tilts the orbit plane by exactly that angle while
    keeping speed → circular, in-LEO) and offsetting position cross-track for the
    miss. Goes through the from_orbits SEARCH constructor for the verified miss.

    NB the plane-to-plane angle ≠ the naive inclination difference: SC1 is at
    inc=55°, so an inc=90° SC2 is only a 35° crossing. We solve the true angle
    (between the angular-momentum normals) directly. The encounter Δv-injection
    path CANNOT reach high plane angles (caps ~15° before SC2 leaves LEO), so a
    high-angle crossing requires this own-circular-orbit construction."""
    from brahe import state_eci_to_koe
    sc1 = np.asarray(SC1_OE_AT_TCA, float)
    s1 = np.array(state_koe_to_eci(sc1, AngleFormat.DEGREES))
    r1, v1 = s1[:3], s1[3:6]
    r_hat = r1 / np.linalg.norm(r1)
    th = np.radians(angle_deg)
    # Rodrigues rotation of v1 about r_hat by angle_deg
    v2 = (v1 * np.cos(th) + np.cross(r_hat, v1) * np.sin(th)
          + r_hat * np.dot(r_hat, v1) * (1 - np.cos(th)))
    perp = np.cross(r_hat, v2); perp /= np.linalg.norm(perp)
    r2 = r1 + perp * miss_km * 1e3
    koe2 = np.array(state_eci_to_koe(np.concatenate([r2, v2]), AngleFormat.DEGREES))
    return G.make_conjunction_from_orbits(sc1, koe2)


def plane_to_plane_angle(koe_a, koe_b) -> float:
    """True angle (deg) between two orbit planes = angle between their
    angular-momentum normals. Used to LABEL the crossing honestly."""
    def nrm(koe):
        s = np.array(state_koe_to_eci(np.asarray(koe, float), AngleFormat.DEGREES))
        h = np.cross(s[:3], s[3:6]); return h / np.linalg.norm(h)
    return float(np.degrees(np.arccos(
        np.clip(np.dot(nrm(koe_a), nrm(koe_b)), -1, 1))))


def _altitude_sweep_points(alts_km=(400, 550, 800, 1200, 1600)):
    """Auxiliary sweep over SC1 ALTITUDE so the (a,e) scope map isn't a thin
    strip. At each altitude run the same (miss×beta×phi×vrel) grid and collect
    SC2 (a,e,feasible). Returns (a_alt, e, feasible) arrays."""
    from brahe import R_EARTH
    aa, ee, ff = [], [], []
    for alt in alts_km:
        oe = np.asarray(SC1_OE_AT_TCA, float).copy()
        oe[0] = R_EARTH + alt * 1e3
        for c in G.coverage_dataset(sc1_oe=oe):
            if c.sc2_oe is None:
                continue
            aa.append(float(c.sc2_oe[0]) / 1e3 - R_E_KM)
            ee.append(float(c.sc2_oe[1]))
            ff.append(bool(c.feasible))
    return np.array(aa), np.array(ee), np.array(ff)


def fig_physical(rows, out_path):
    coorb = G.make_conjunction_from_encounter(5.0, 90.0,  0.0, 15.0)    # co-orbital, same plane
    shallow = G.make_conjunction_from_encounter(5.0, 0.0, 90.0, 2000.0)  # ~15° crossing (encounter max)
    steep = _plane_crossing(85.0)                                       # real ~85° crossing (from_orbits)
    outsc = G.make_conjunction_from_encounter(5.0,  0.0,  0.0, 1500.0)  # along-track high-v → OUT

    # honest plane-to-plane angles (NOT the naive inc difference) for the labels
    ang_sh = plane_to_plane_angle(shallow.sc1_oe, shallow.sc2_oe)
    ang_st = plane_to_plane_angle(steep.sc1_oe, steep.sc2_oe)

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 0.85], hspace=0.95, wspace=0.5)
    cases = [(coorb, "A1  co-orbital (same plane)", "tab:orange"),
             (shallow, f"A2  shallow crossing ~{ang_sh:.0f}°\n(encounter-path max)", "tab:purple"),
             (steep, f"A3  steep crossing ~{ang_st:.0f}°\n(real, from_orbits)", "tab:green"),
             (outsc, "A4  OUT of scope\n(high-v along-track)", "tab:red")]
    for j, (c, t, col) in enumerate(cases):
        _inset_pair(fig, gs[0, j], c, t, col)

    # bottom row: (a-altitude, e) scope map with TRUE feasibility CURVES + a
    # multi-altitude sweep so the points actually span the map.
    axm = fig.add_subplot(gs[1, :])
    a_alt = np.array([r["sc2_a_km"] - R_E_KM for r in rows])
    e = np.array([r["sc2_e"] for r in rows])
    feas = np.array([r["feasible"] for r in rows], dtype=bool)
    # auxiliary altitude sweep (lighter markers, fills the map)
    sa, se, sf = _altitude_sweep_points()

    # TRUE feasibility region: perigee a(1-e)≥R+150, apogee a(1+e)≤R+2000, e≤ECC_MAX.
    a_grid = np.linspace(100, APO_CEIL + 100, 400)         # a-altitude km
    a_m = (a_grid + R_E_KM)                                 # km radius
    e_peri = 1.0 - (R_E_KM + PERI_FLOOR) / a_m              # perigee-floor curve
    e_apo = (R_E_KM + APO_CEIL) / a_m - 1.0                 # apogee-ceil curve
    e_upper = np.minimum.reduce([e_peri, e_apo,
                                 np.full_like(a_m, ECC_MAX)])
    e_upper = np.clip(e_upper, 0, None)
    feasible_band = e_upper > 0
    axm.fill_between(a_grid[feasible_band], 0, e_upper[feasible_band],
                     color="tab:green", alpha=0.12, zorder=0, label="feasible (LEO) region")
    axm.plot(a_grid, np.clip(e_peri, 0, 0.6), color="tab:red", lw=1.3,
             label="perigee = 150 km (floor)")
    axm.plot(a_grid, np.clip(e_apo, 0, 0.6), color="tab:brown", lw=1.3,
             label="apogee = 2000 km (ceil)")
    axm.axhline(ECC_MAX, color="tab:green", ls="--", lw=1,
                label=f"e = {ECC_MAX} cap")

    axm.scatter(sa[sf], se[sf], s=16, c="tab:cyan", edgecolor="none",
                alpha=0.5, zorder=2, label="alt-sweep feasible")
    axm.scatter(sa[~sf], se[~sf], s=16, marker="x", c="darkorange",
                alpha=0.5, zorder=2, label="alt-sweep infeasible")
    axm.scatter(a_alt[feas], e[feas], s=55, c="tab:blue", edgecolor="k",
                lw=0.3, label="550 km grid: feasible", zorder=3)
    axm.scatter(a_alt[~feas], e[~feas], s=70, marker="x", c="tab:red",
                label="550 km grid: infeasible", zorder=3)
    axm.set_xlabel("SC2 semi-major-axis altitude  a − R⊕ (km)")
    axm.set_ylabel("SC2 eccentricity  e")
    axm.set_title("SC2 orbit regime — TRUE feasibility curves (perigee/apogee/e) "
                  "vs measured points\n(X's inside e<0.25 are infeasible because their "
                  "PERIGEE dips below 150 km — the box was only a 2-D shadow)")
    axm.legend(fontsize=7, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, -0.18))
    axm.grid(alpha=0.3)
    axm.set_xlim(0, APO_CEIL + 100)
    axm.set_ylim(-0.02, 0.4)

    fig.suptitle("Figure A — Physical conjunction types & SC2 orbit scope "
                 "(real orbit-first points)", fontsize=13, y=0.98)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# FIGURE B — abstract scope map + (phi × vrel) coverage grid
# ---------------------------------------------------------------------------

def fig_abstract(rows, out_path):
    """Figure B = the (phi × v_rel) coverage HEATMAP. At dense resolution the
    feasible region — including the phi≈45° / high-vrel RIDGE that reaches 90°
    plane crossings — shows as a continuous band. Two panels:
      B-left  : feasibility fraction per (phi, vrel) cell (collapsed over miss/beta)
      B-right : model reduction error (km) on the FEASIBLE region (accuracy map)
    (The old inc-vs-e panel was dropped: inclination does NOT drive feasibility.)"""
    phis = sorted({r["phi_deg"] for r in rows})
    vrels = sorted({r["vrel_in_ms"] for r in rows})
    P, V = len(phis), len(vrels)
    pidx = {p: i for i, p in enumerate(phis)}
    vidx = {v: i for i, v in enumerate(vrels)}

    feas_frac = np.full((P, V), np.nan)
    rederr = np.full((P, V), np.nan)
    cnt = np.zeros((P, V)); nok = np.zeros((P, V))
    err_acc = np.full((P, V), np.nan)
    for r in rows:
        i, j = pidx[r["phi_deg"]], vidx[r["vrel_in_ms"]]
        cnt[i, j] += 1
        if r["feasible"]:
            nok[i, j] += 1
            e = abs(r["red_err_km"])
            err_acc[i, j] = e if np.isnan(err_acc[i, j]) else max(err_acc[i, j], e)
    feas_frac = np.where(cnt > 0, nok / np.maximum(cnt, 1), np.nan)
    rederr = err_acc

    fig, ax = plt.subplots(1, 2, figsize=(17, 7))
    extent = [0, V, 0, P]

    # B-left: feasibility fraction
    im0 = ax[0].imshow(feas_frac, origin="lower", aspect="auto", extent=extent,
                       cmap="RdYlGn", vmin=0, vmax=1)
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04,
                 label="feasible fraction (over miss × beta)")
    ax[0].set_title("B1  Feasibility region (phi × v_rel)\n"
                    "green = in LEO scope; the bright band rising to the right "
                    "is the reachable region")

    # B-right: reduction-error accuracy on feasible cells (log scale, tiny values)
    with np.errstate(invalid="ignore"):
        logerr = np.log10(np.clip(rederr, 1e-16, None))
    im1 = ax[1].imshow(logerr, origin="lower", aspect="auto", extent=extent,
                       cmap="viridis")
    cb = fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04,
                      label="log₁₀ max |reduction error| (km)")
    ax[1].set_title("B2  Model accuracy on feasible region\n"
                    "(reduction error vs brahe truth — ~1e-15 km everywhere)")

    # mark the phi=45 ridge line on both
    if 45.0 in pidx:
        for a in ax:
            a.axhline(pidx[45.0] + 0.5, color="k", ls=":", lw=1, alpha=0.6)
            a.text(V * 0.5, pidx[45.0] + 0.8, "phi=45° ridge (reaches 90° crossings)",
                   fontsize=8, ha="center", color="k")

    for a in ax:
        a.set_xticks(np.arange(V) + 0.5)
        a.set_xticklabels([f"{v:.0f}" for v in vrels], rotation=60, fontsize=7)
        a.set_yticks(np.arange(P) + 0.5)
        a.set_yticklabels([f"{p:.0f}°" for p in phis], fontsize=7)
        a.set_xlabel("requested relative speed  v_rel  (m/s)")
        a.set_ylabel("velocity direction  phi   (0=along-track, 90=cross-track)")

    fig.suptitle("Figure B — Dense conjunction coverage heatmap "
                 "(velocity direction × relative speed)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# FIGURE C — continuous span of SC2 orbit ellipses
# ---------------------------------------------------------------------------

# Each 2D projection picks (horizontal, vertical, depth) axes from ECI x,y,z.
# "Behind Earth" = the depth coordinate points AWAY from the viewer AND the
# projected (h,v) position falls inside the Earth disk → that arc is occluded.
# Viewer looks down +depth, so depth < 0 is the far side.
_PROJ = {                          # name: (h_idx, v_idx, depth_idx)
    "xy": (0, 1, 2),               # looking down +z
    "xz": (0, 2, 1),               # looking down +y
    "yz": (1, 2, 0),               # looking down +x
}


def _draw_occluded(ax, pts, color, lw, alpha, hidden_ls, h, v, d, zbase):
    """
    Plot a 3D curve in a 2D projection (axes h,v; depth d), drawing the part
    BEHIND the Earth (far side AND projected inside the disk) as faint-dashed.
    Splitting point-by-point with NaN breaks keeps each segment's continuity.
    """
    P = pts[:, [h, v]]
    behind = (pts[:, d] < 0) & (np.hypot(P[:, 0], P[:, 1]) < R_E_KM)
    # front (visible) and hidden masked copies; NaN where the other regime is
    front = P.copy(); front[behind] = np.nan
    hide = P.copy(); hide[~behind] = np.nan
    ax.plot(hide[:, 0], hide[:, 1], color=color, lw=lw * 0.8,
            alpha=alpha * 0.35, ls=hidden_ls, zorder=zbase)
    ax.plot(front[:, 0], front[:, 1], color=color, lw=lw,
            alpha=alpha, ls="-", zorder=zbase + 2)


def _continuous_panel(axes, sweep_vals, build_fn, cmap_name, label, explain,
                      feasible_only=False):
    """
    Draw a family of semi-transparent SC2 ellipses across 3 ECI projections
    (xy, xz, yz) with behind-Earth arcs faint-dashed, plus a text panel
    explaining what the sweep physically changes.
    axes = (ax_xy, ax_xz, ax_yz, ax_txt).
    """
    ax_xy, ax_xz, ax_yz, ax_txt = axes
    cmap = plt.get_cmap(cmap_name)
    proj2d = [(ax_xy, "xy"), (ax_xz, "xz"), (ax_yz, "yz")]
    o1 = koe_ellipse_eci(SC1_OE_AT_TCA)
    ex, ey = earth_circle()

    # Earth disk + SC1 reference on each 2D panel
    for ax, key in proj2d:
        h, v, d = _PROJ[key]
        ax.fill(ex, ey, color="0.85", zorder=1)
        _draw_occluded(ax, o1, "k", 1.8, 1.0, (0, (1, 1)), h, v, d, zbase=3)
    # SC1 label proxy for the legend (occluded draw sets no label)
    ax_xy.plot([], [], color="k", lw=1.8, label="SC1")

    n = len(sweep_vals)
    drawn = []                                       # orbits actually plotted
    for i, val in enumerate(sweep_vals):
        c = build_fn(val)
        if c.sc2_oe is None:
            continue
        if feasible_only and not c.feasible:
            continue
        o2 = koe_ellipse_eci(c.sc2_oe)
        drawn.append(o2)
        col = cmap(0.12 + 0.85 * i / max(n - 1, 1))
        hidden_ls = (0, (4, 3))                      # dash style for hidden arc
        for ax, key in proj2d:
            h, v, d = _PROJ[key]
            _draw_occluded(ax, o2, col, 1.4, 0.6, hidden_ls, h, v, d, zbase=4)

    # when showing only feasible orbits, clip axes to them (else eccentric
    # outliers blow out the scale and hide the in-scope spread)
    if feasible_only and drawn:
        allpts = np.vstack(drawn + [o1])
        for ax, key in proj2d:
            h, v, _ = _PROJ[key]
            m = 1.08 * np.abs(allpts[:, [h, v]]).max()
            ax.set_xlim(-m, m); ax.set_ylim(-m, m)

    for ax, key in proj2d:
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"ECI {key[0]} (km)", fontsize=7)
        ax.set_ylabel(f"ECI {key[1]} (km)", fontsize=7)
        ax.set_title(key.upper(), fontsize=7)
    ax_xy.set_title(label + "\n(behind Earth = faint-dashed)",
                    fontsize=8.5, loc="left")
    ax_xy.legend(fontsize=6, loc="upper right")

    # explanation panel (replaces the old 3D column)
    ax_txt.axis("off")
    ax_txt.text(0.02, 0.98, explain, transform=ax_txt.transAxes,
                fontsize=7.0, va="top", ha="left", linespacing=1.2,
                bbox=dict(boxstyle="round,pad=0.4", fc="0.96", ec="0.7"))


def _phi_vrel_for_plane_angle(theta_deg, v_orb=7593.0):
    """Inverse-solve the (phi, vrel) that yields a CIRCULAR SC2 whose orbit plane
    is theta_deg from SC1's, via from_encounter. Derivation: SC2 must have speed
    V_orb (circular) at angle theta from SC1's velocity, so the relative velocity
    is rel = SC2_vel − SC1_vel with
        along(T) = V_orb(cosθ−1)  (negative),  cross(N) = V_orb sinθ
      ⇒ vrel = 2·V_orb·sin(θ/2),  phi = atan2(sinθ, 1−cosθ).
    Verified exact (achieved plane angle = θ, e≈0.002) up to ~85°."""
    t = np.radians(theta_deg)
    if theta_deg <= 0:
        return 0.0, 15.0
    vrel = 2.0 * v_orb * np.sin(t / 2.0)
    phi = np.degrees(np.arctan2(np.sin(t), 1.0 - np.cos(t)))
    return phi, max(vrel, 15.0)


def _test_space_grid(angles_deg=None, misses_km=None):
    """STRATIFIED coverage of the OUTPUT axes we actually test: plane-crossing
    angle × miss. For each target we inverse-solve (phi, vrel) so the result is a
    feasible circular SC2 at that exact crossing angle. Returns a list of
    (miss, beta, phi, vrel, theta) tuples ordered by theta → clean gradient.
    (Plain Latin-hypercube was rejected: it samples INPUT space evenly but the
    feasible OUTPUT clusters near SC1's plane, giving an uneven gradient.)"""
    angles_deg = angles_deg if angles_deg is not None else np.linspace(0, 85, 12)
    misses_km = misses_km if misses_km is not None else [2.0, 5.0, 8.0]
    out = []
    for th in angles_deg:
        phi, vrel = _phi_vrel_for_plane_angle(float(th))
        for m in misses_km:
            out.append((m, 45.0, phi, vrel, float(th)))
    out.sort(key=lambda t: t[4])     # gradient by plane angle
    return out


def _plane_angle_of(conj):
    """True plane-to-plane angle (deg) of a conjunction's SC2 vs its SC1."""
    return plane_to_plane_angle(conj.sc1_oe, conj.sc2_oe)


def _test_space_2d(n_angle=9, n_ecc=4, seed=1):
    """Cover the feasible (plane-crossing angle × eccentricity) REGION — which is
    a TRIANGLE: e up to ~0.09 at low angles, shrinking to ~0.04 by 60–90°
    (high-angle crossings need the near-circular high-vrel ridge, leaving no room
    for eccentricity before perigee<150). We can't invert (angle,e)→knobs cleanly
    (eccentricity couples through net speed change), so we SAMPLE (miss,beta,phi,
    vrel), keep feasible, then pick representatives nearest each (angle,e) target.
    Returns list of (conj, angle, e) sorted by angle. Circular orbits span all
    angles; eccentric ones appear only at low angles — the triangle shows itself."""
    rng = np.random.default_rng(seed)
    # scan feasible space (coarse but enough to pick representatives from)
    samples = []
    for miss in (2.0, 5.0, 8.0):
        for beta in np.linspace(0, 90, 4):
            for phi in np.linspace(0, 90, 19):
                for vrel in np.geomspace(15, 12000, 28):
                    c = G.make_conjunction_from_encounter(miss, float(beta),
                                                          float(phi), float(vrel))
                    if c.feasible and c.sc2_oe is not None and np.isfinite(c.sc2_oe[1]):
                        samples.append((c, _plane_angle_of(c), float(c.sc2_oe[1])))
    if not samples:
        return []
    A = np.array([[s[1], s[2]] for s in samples])
    amax = A[:, 0].max()
    chosen, used = [], set()
    for th in np.linspace(0, min(amax, 88), n_angle):
        # at this angle, what eccentricities are actually reachable?
        band = A[np.abs(A[:, 0] - th) < 6.0]
        emax = band[:, 1].max() if len(band) else 0.0
        for e_t in np.linspace(0.0, emax, n_ecc):
            d = (A[:, 0] - th) ** 2 / 100.0 + (A[:, 1] - e_t) ** 2 / 0.0009
            k = int(np.argmin(d))
            if k in used:
                continue
            used.add(k)
            chosen.append(samples[k])
    chosen.sort(key=lambda s: s[1])
    return chosen


def _test_space_panel(axes, chosen, explain):
    """Draw the feasible (plane-angle × eccentricity) test-space family.
    COLOR = plane-crossing angle (turbo), LINEWIDTH = eccentricity (thin=circular,
    thick=eccentric). Behind-Earth arcs faint-dashed. axes=(xy,xz,yz,txt)."""
    ax_xy, ax_xz, ax_yz, ax_txt = axes
    cmap = plt.get_cmap("turbo")
    proj2d = [(ax_xy, "xy"), (ax_xz, "xz"), (ax_yz, "yz")]
    o1 = koe_ellipse_eci(SC1_OE_AT_TCA)
    ex, ey = earth_circle()
    for ax, key in proj2d:
        h, v, d = _PROJ[key]
        ax.fill(ex, ey, color="0.85", zorder=1)
        _draw_occluded(ax, o1, "k", 1.8, 1.0, (0, (1, 1)), h, v, d, zbase=3)
    ax_xy.plot([], [], color="k", lw=1.8, label="SC1")

    amax = max((a for _, a, _ in chosen), default=90.0)
    emax = max((e for _, _, e in chosen), default=0.1)
    drawn = []
    for conj, ang, e in chosen:
        o2 = koe_ellipse_eci(conj.sc2_oe); drawn.append(o2)
        col = cmap(0.05 + 0.9 * ang / max(amax, 1e-6))
        lw = 0.9 + 3.2 * (e / max(emax, 1e-6))          # eccentricity → thickness
        for ax, key in proj2d:
            h, v, d = _PROJ[key]
            _draw_occluded(ax, o2, col, lw, 0.6, (0, (4, 3)), h, v, d, zbase=4)
    if drawn:
        allpts = np.vstack(drawn + [o1])
        for ax, key in proj2d:
            h, v, _ = _PROJ[key]
            m = 1.08 * np.abs(allpts[:, [h, v]]).max()
            ax.set_xlim(-m, m); ax.set_ylim(-m, m)
    for ax, key in proj2d:
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"ECI {key[0]} (km)", fontsize=7)
        ax.set_ylabel(f"ECI {key[1]} (km)", fontsize=7)
        ax.set_title(key.upper(), fontsize=7)
    ax_xy.set_title("C4  FULL TEST SPACE: plane-crossing angle × eccentricity\n"
                    "color = angle (0°→~90°), line THICKNESS = eccentricity "
                    "(thin=circular, thick=eccentric)", fontsize=8.5, loc="left")
    ax_xy.legend(fontsize=6, loc="upper right")
    ax_txt.axis("off")
    ax_txt.text(0.02, 0.98, explain, transform=ax_txt.transAxes,
                fontsize=7.0, va="top", ha="left", linespacing=1.2,
                bbox=dict(boxstyle="round,pad=0.4", fc="0.96", ec="0.7"))


def fig_continuous(out_path):
    fig = plt.figure(figsize=(17, 18))
    # 4 sweep rows × 4 cols (xy, xz, yz, explanation text).
    gs = fig.add_gridspec(4, 4, hspace=0.45, wspace=0.20)

    def row_axes(r):
        return (fig.add_subplot(gs[r, 0]), fig.add_subplot(gs[r, 1]),
                fig.add_subplot(gs[r, 2]), fig.add_subplot(gs[r, 3]))

    # Row 1: phi sweep 0→90 (along→cross velocity). High vrel so the
    # velocity-direction effect is VISIBLE: along-track end leaves scope (dashed),
    # cross-track end is the feasible inclination crossing.
    phis = np.linspace(0, 90, 10)
    _continuous_panel(
        row_axes(0),
        phis, lambda p: G.make_conjunction_from_encounter(5.0, 0.0, p, 1500.0),
        "viridis",
        "C1  phi sweep 0°→90°  (along-track → cross-track velocity; vrel=1500 m/s)\n"
        "     dark→bright = phi 0→90;  dashed = out of LEO scope",
        "WHAT phi CHANGES\n\n"
        "phi = direction of the\nrelative velocity at TCA.\n\n"
        "phi=0 : closing velocity is\nALONG-track → both craft on\n"
        "~the same orbit (co-orbital\nhead-on). At km/s this drives\n"
        "SC2 off LEO → OUT (dashed).\n\n"
        "phi=90 : closing velocity is\nCROSS-track → SC2's orbit\n"
        "PLANE tilts vs SC1. An\ninclination crossing — stays\n"
        "in LEO, IN scope at any speed.\n\n"
        "phi turns a co-orbital encounter\ninto a plane-crossing one.")

    # Row 2: beta sweep 0→90 (miss-vector direction) at fixed phi=90, high vrel.
    betas = np.linspace(0, 90, 10)
    _continuous_panel(
        row_axes(1),
        betas, lambda b: G.make_conjunction_from_encounter(5.0, b, 90.0, 1500.0),
        "plasma",
        "C2  beta sweep 0°→90°  (miss-vector direction; phi=90, vrel=1500 m/s)\n"
        "     dark→bright = beta 0→90",
        "WHAT beta CHANGES\n\n"
        "beta = direction of the MISS\nvector within the plane ⊥ the\n"
        "relative velocity.\n\n"
        "It slides the 5 km closest-\napproach offset between the\n"
        "radial/cross (perp) and the\nalong-track (dt0) components:\n"
        "  perp = miss·cos β\n  dt0  = miss·sin β\n\n"
        "The ORBIT barely changes —\nthe family overlaps almost\n"
        "perfectly — because moving a\n5 km miss around hardly\n"
        "perturbs SC2's orbit. beta\nsets WHERE the miss sits (and\n"
        "thus how steerable it is),\nNOT the orbit type.")

    # Row 3: vrel sweep low→high at phi=90 (cross-track) — inclination deepens
    # until SC2 leaves LEO scope (dashed).
    vrels = np.geomspace(15, 6000, 12)
    _continuous_panel(
        row_axes(2),
        vrels, lambda v: G.make_conjunction_from_encounter(5.0, 0.0, 90.0, v),
        "cool",
        "C3  vrel sweep 15→6000 m/s  (cross-track; inclination crossing deepens)\n"
        "     dark→bright = vrel low→high;  dashed = leaves LEO scope",
        "WHAT vrel CHANGES\n\n"
        "vrel = magnitude of the\nrelative velocity at TCA\n(here all cross-track).\n\n"
        "Larger vrel = bigger plane\nangle between the orbits, so\n"
        "SC2's ellipse fans further\nfrom SC1 (inclination crossing\n'deepens'):\n"
        "  15 m/s  → ~0.1° tilt\n  1.5 km/s → ~10° tilt\n"
        "  6 km/s  → leaves LEO box\n\n"
        "This is how a real km/s LEO\nconjunction is represented:\n"
        "cross-track plane geometry,\nNOT along-track speed. Beyond\n"
        "the box (dashed) SC2 is no\nlonger LEO-resident.")

    # Row 4: the FULL TEST SPACE — the feasible (plane-angle × eccentricity)
    # region, sampled & feasibility-filtered. Color=angle, thickness=ecc.
    chosen = _test_space_2d()
    _test_space_panel(
        row_axes(3), chosen,
        "WHAT THIS SHOWS\n\n"
        "The full 2-D span we'd TEST:\n"
        "plane-crossing ANGLE (color)\n× ECCENTRICITY (line thickness).\n\n"
        "All orbits FEASIBLE (in LEO).\n"
        "thin = circular, thick = eccentric.\n\n"
        "The feasible region is a\nTRIANGLE: e up to ~0.09 at low\n"
        "angles, shrinking to ~0.04 by\n60–90°. You CAN'T have both a\n"
        "steep crossing AND high e —\nthey compete for the same\n"
        "velocity budget (perigee floors\nout first). So thick lines only\n"
        "appear at low angles; high-angle\ncrossings are circular-only.\n\n"
        "(Sampled + feasibility-filtered;\ne can't be cleanly inverse-\n"
        "solved — it couples through\nnet speed change.)")

    fig.suptitle("Figure C — Continuous conjunction-type span "
                 "(semi-transparent SC2 orbit family vs fixed SC1)",
                 fontsize=13, y=0.995)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# FIGURE D — the 5-DOF spherical parameterization, one DOF per row, + the REAL
# representative sweep as the final row. Same orbit-ellipse style as Figure C.
# ---------------------------------------------------------------------------

# Absolute (anchored at this file) so the figure renders from ANY cwd — including the
# cluster — not just when run from inside benchmarks/spacecraftCA/.
SWEEP_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "notes", "conj_sweep_spherical.json")


def _spherical_panel(axes, sweep_vals, build_fn, cmap_name, label, explain,
                     cbar_label=None):
    """One DOF row: a family of SC2 ellipses (3 ECI projections) as `build_fn(val)`
    sweeps a single spherical DOF. Reuses the Figure-C occluded-ellipse style."""
    ax_xy, ax_xz, ax_yz, ax_txt = axes
    cmap = plt.get_cmap(cmap_name)
    proj2d = [(ax_xy, "xy"), (ax_xz, "xz"), (ax_yz, "yz")]
    o1 = koe_ellipse_eci(SC1_OE_AT_TCA)
    ex, ey = earth_circle()
    for ax, key in proj2d:
        h, v, d = _PROJ[key]
        ax.fill(ex, ey, color="0.85", zorder=1)
        _draw_occluded(ax, o1, "k", 1.8, 1.0, (0, (1, 1)), h, v, d, zbase=3)
    ax_xy.plot([], [], color="k", lw=1.8, label="SC1")

    n = len(sweep_vals)
    drawn = []
    for i, val in enumerate(sweep_vals):
        c = build_fn(val)
        if c.sc2_oe is None or not np.isfinite(c.sc2_oe[1]) or c.sc2_oe[1] >= 1.0:
            continue
        o2 = koe_ellipse_eci(c.sc2_oe)
        if not np.isfinite(o2).all():
            continue
        drawn.append(o2)
        col = cmap(0.1 + 0.85 * i / max(n - 1, 1))
        for ax, key in proj2d:
            h, v, d = _PROJ[key]
            _draw_occluded(ax, o2, col, 1.4, 0.6, (0, (4, 3)), h, v, d, zbase=4)
    if drawn:
        allpts = np.vstack(drawn + [o1])
        for ax, key in proj2d:
            h, v, _ = _PROJ[key]
            m = 1.08 * np.abs(allpts[:, [h, v]]).max()
            ax.set_xlim(-m, m); ax.set_ylim(-m, m)
    for ax, key in proj2d:
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"ECI {key[0]} (km)", fontsize=7)
        ax.set_ylabel(f"ECI {key[1]} (km)", fontsize=7)
        ax.set_title(key.upper(), fontsize=7)
    ax_xy.set_title(label + "\n(behind Earth = faint-dashed)",
                    fontsize=8.5, loc="left")
    ax_xy.legend(fontsize=6, loc="upper right")
    ax_txt.axis("off")
    ax_txt.text(0.02, 0.98, explain, transform=ax_txt.transAxes,
                fontsize=7.0, va="top", ha="left", linespacing=1.2,
                bbox=dict(boxstyle="round,pad=0.4", fc="0.96", ec="0.7"))


def _sweep_population_panel(axes, json_path, explain):
    """Final row: draw the ACTUAL representative sweep (the 100-conjunction JSON) as
    SC2 ellipse family, COLORED by Δi (plane-crossing angle). Proves the real spread
    across conjunction types visually. axes=(xy,xz,yz,txt)."""
    import json
    ax_xy, ax_xz, ax_yz, ax_txt = axes
    cmap = plt.get_cmap("turbo")
    proj2d = [(ax_xy, "xy"), (ax_xz, "xz"), (ax_yz, "yz")]
    o1 = koe_ellipse_eci(SC1_OE_AT_TCA)
    ex, ey = earth_circle()
    for ax, key in proj2d:
        h, v, d = _PROJ[key]
        ax.fill(ex, ey, color="0.85", zorder=1)
        _draw_occluded(ax, o1, "k", 1.8, 1.0, (0, (1, 1)), h, v, d, zbase=3)
    ax_xy.plot([], [], color="k", lw=1.8, label="SC1")

    with open(json_path) as f:
        specs = json.load(f)
    norm = plt.Normalize(0, 120)                       # Δi color scale (deg)
    drawn = []
    for s in specs:
        sc1 = np.asarray(s["sc1_oe"], float); sc2 = np.asarray(s["sc2_oe"], float)
        di = abs(float(sc2[2]) - float(sc1[2]))
        o2 = koe_ellipse_eci(sc2); drawn.append(o2)
        for ax, key in proj2d:
            h, v, d = _PROJ[key]
            _draw_occluded(ax, o2, cmap(norm(di)), 0.9, 0.45, (0, (4, 3)),
                           h, v, d, zbase=4)
    if drawn:
        allpts = np.vstack(drawn + [o1])
        for ax, key in proj2d:
            h, v, _ = _PROJ[key]
            m = 1.08 * np.abs(allpts[:, [h, v]]).max()
            ax.set_xlim(-m, m); ax.set_ylim(-m, m)
    for ax, key in proj2d:
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"ECI {key[0]} (km)", fontsize=7)
        ax.set_ylabel(f"ECI {key[1]} (km)", fontsize=7)
        ax.set_title(key.upper(), fontsize=7)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = ax_yz.figure.colorbar(sm, ax=ax_yz, pad=0.02, fraction=0.046)
    cb.set_label("Δi (deg)", fontsize=7); cb.ax.tick_params(labelsize=6)
    ax_xy.set_title(f"D6  THE REPRESENTATIVE SWEEP — {len(specs)} conjunctions we test\n"
                    "color = Δi (plane crossing); each ellipse = one SC2 we conjunct with",
                    fontsize=8.5, loc="left")
    ax_xy.legend(fontsize=6, loc="upper right")
    ax_txt.axis("off")
    ax_txt.text(0.02, 0.98, explain, transform=ax_txt.transAxes,
                fontsize=7.0, va="top", ha="left", linespacing=1.2,
                bbox=dict(boxstyle="round,pad=0.4", fc="0.96", ec="0.7"))


def fig_5dof(out_path, sc1_oe=None):
    """The 5-DOF spherical parameterization: one DOF isolated per row, then the real
    representative sweep. Shows what each input does AND that the sweep spans types."""
    sc1 = np.asarray(sc1_oe if sc1_oe is not None else G.CASE_SC1_OE, dtype=float)
    fig = plt.figure(figsize=(17, 26))
    gs = fig.add_gridspec(6, 4, hspace=0.5, wspace=0.20)

    def row(r):
        return (fig.add_subplot(gs[r, 0]), fig.add_subplot(gs[r, 1]),
                fig.add_subplot(gs[r, 2]), fig.add_subplot(gs[r, 3]))

    # DOF 1 — v_rel magnitude (cross-track direction): plane crossing deepens.
    _spherical_panel(
        row(0), np.geomspace(15, 11000, 12),
        lambda v: G.make_conjunction_from_spherical(5.0, v, 90.0, 90.0, 0.0, sc1_oe=sc1),
        "cool",
        "D1  v_rel  15 → 11000 m/s  (φ_v=90 cross-track; Δi DEEPENS)",
        "DOF 1 — |v_rel|\n\nMagnitude of the relative\nvelocity at TCA.\n\n"
        "With cross-track direction\n(φ_v=90), bigger v_rel = a\nSTEEPER plane crossing —\n"
        "SC2's plane tilts further\nfrom SC1. This is the main\nΔi knob.\n\n"
        "On the near-circular ridge it\nreaches retrograde without\nleaving LEO.")

    # DOF 2 — theta_v (velocity polar angle off radial).
    _spherical_panel(
        row(1), np.linspace(5, 90, 12),
        lambda t: G.make_conjunction_from_spherical(5.0, 4000.0, t, 90.0, 0.0, sc1_oe=sc1),
        "viridis",
        "D2  θ_v  5° → 90°  (velocity polar angle off +R; v_rel=4000)",
        "DOF 2 — θ_v\n\nVelocity angle OFF the radial\n(+R) axis. θ_v=90 puts the\n"
        "velocity in the T–N plane\n(the usual crossing); smaller\nθ_v tilts it toward radial,\n"
        "reshaping how SC2's orbit\nopens out. Together with φ_v\nit sets the full velocity\n"
        "DIRECTION (2 of the 5 DOF).")

    # DOF 3 — phi_v (velocity azimuth in T–N plane): along → cross.
    _spherical_panel(
        row(2), np.linspace(0, 90, 12),
        lambda p: G.make_conjunction_from_spherical(5.0, 3000.0, 90.0, p, 0.0, sc1_oe=sc1),
        "plasma",
        "D3  φ_v  0° → 90°  (along-track → cross-track velocity; v_rel=3000)",
        "DOF 3 — φ_v\n\nVelocity AZIMUTH in the T–N\nplane. φ_v=0 → along-track\n"
        "(co-orbital, no plane tilt);\nφ_v=90 → cross-track (max\nplane crossing). Rotating φ_v\n"
        "turns a co-orbital encounter\ninto a plane crossing — the\n"
        "qualitative TYPE of conjunction.")

    # DOF 4 — miss magnitude (the offset size). Orbit barely moves.
    _spherical_panel(
        row(3), np.linspace(0.5, 5.0, 10),
        lambda m: G.make_conjunction_from_spherical(m, 2000.0, 90.0, 60.0, 0.0, sc1_oe=sc1),
        "autumn",
        "D4  miss  0.5 → 5 km  (closest-approach magnitude; v_rel=2000, φ_v=60)",
        "DOF 4 — |miss|\n\nThe closest-approach distance\nat TCA. A few km on a ~7000\n"
        "km orbit is ~0.1% — so the\nSC2 ellipses OVERLAP almost\n"
        "perfectly here. Miss sets the\nDANGER level (collision vs\n"
        "clear), NOT the orbit type.\nIt's handled by the belief\nladder in the model.")

    # DOF 5 — miss alpha (in-plane miss direction): perp ↔ dt0 split = geometry θ.
    _spherical_panel(
        row(4), np.linspace(0, 180, 12),
        lambda a: G.make_conjunction_from_spherical(5.0, 200.0, 60.0, 30.0, a, sc1_oe=sc1),
        "spring",
        "D5  miss α  0° → 180°  (in-plane miss direction → geometry θ; v_rel=200)",
        "DOF 5 — miss α\n\nDirection of the miss WITHIN\nthe plane ⊥ v_rel. Slides the\n"
        "offset between sideways\n(perp) and along-track (dt0):\nthe GEOMETRY angle θ\n"
        "(0=head-on … 90=cross-track).\nLike DOF 4 the orbit barely\n"
        "moves — α sets WHERE the\nmiss sits / how steerable it\nis, not the orbit.\n\n"
        "→ This is the axis that is\nDISTINCT from Δi (DOF 1-3).")

    # Final row — the actual representative sweep.
    _sweep_population_panel(
        row(5), SWEEP_JSON,
        "THE TEST SET\n\nThe ~100 conjunctions we\nactually run, drawn from this\n"
        "5-DOF space. Each ellipse is\none real SC2 we conjunct with;\n"
        "color = Δi.\n\nThe fan of planes (low Δi near\nSC1's plane → steep / "
        "retrograde\nin red) shows we span the FULL\nconjunction-type range, all\n"
        "near-circular and LEO-resident.\n\nΔi and θ (DOF 5) vary\nINDEPENDENTLY — the sweep\n"
        "grids both, so it covers the\n2-D (Δi × θ) type plane, not\njust a line.")

    fig.suptitle("Figure D — The 5-DOF spherical conjunction parameterization "
                 "(one DOF per row) + the representative test sweep (final row)",
                 fontsize=13, y=0.997)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# FIGURE E — "what each DOF drives": compact output-vs-input curves. ONE small
# panel per DOF, plotting the reduced quantity that DOF actually changes, so the
# effect is a readable LINE instead of overlapping orbit ellipses.
# ---------------------------------------------------------------------------

def _di(c):
    return abs(float(c.sc2_oe[2]) - float(c.sc1_oe[2])) if c.sc2_oe is not None else np.nan


def fig_dof_drivers(out_path, sc1_oe=None):
    sc1 = np.asarray(sc1_oe if sc1_oe is not None else G.CASE_SC1_OE, dtype=float)
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    def build(miss=5.0, v=200.0, tv=90.0, pv=30.0, al=0.0):
        return G.make_conjunction_from_spherical(miss, v, tv, pv, al, sc1_oe=sc1)

    # DOF 1 — v_rel → Δi (plane crossing). Use the near-circular crossing direction.
    vv = np.geomspace(15, 11000, 40)
    di1 = [_di(G.make_crossing(d, sc1_oe=sc1)) for d in np.linspace(0, 115, 40)]
    vcross = [G.crossing_velocity_for_delta_i(d, sc1)[0] for d in np.linspace(0, 115, 40)]
    a0 = ax[0, 0]
    a0.plot(vcross, np.linspace(0, 115, 40), "o-", color="tab:blue", ms=3)
    a0.set_xlabel("|v_rel|  (m/s)"); a0.set_ylabel("Δi — plane crossing (deg)")
    a0.set_title("DOF 1 · |v_rel|  →  Δi\n(bigger speed = steeper crossing)")
    a0.grid(alpha=0.3)

    # DOF 2 — theta_v → SC2 eccentricity (radial velocity content pumps e).
    tvs = np.linspace(5, 90, 40)
    e2 = [build(tv=t, v=3000.0, pv=0.0).sc2_oe[1] for t in tvs]
    a1 = ax[0, 1]
    a1.plot(tvs, e2, "o-", color="tab:green", ms=3)
    a1.set_xlabel("θ_v — velocity polar angle (deg)"); a1.set_ylabel("SC2 eccentricity e")
    a1.set_title("DOF 2 · θ_v  →  eccentricity\n(radial velocity content pumps e)")
    a1.grid(alpha=0.3)

    # DOF 3 — phi_v → Δi (along-track→cross-track turns co-orbital into crossing).
    pvs = np.linspace(0, 90, 40)
    di3 = [_di(build(pv=p, v=3000.0)) for p in pvs]
    a2 = ax[0, 2]
    a2.plot(pvs, di3, "o-", color="tab:purple", ms=3)
    a2.set_xlabel("φ_v — velocity azimuth (deg)"); a2.set_ylabel("Δi — plane crossing (deg)")
    a2.set_title("DOF 3 · φ_v  →  Δi\n(along-track → cross-track velocity)")
    a2.grid(alpha=0.3)

    # DOF 4 — miss magnitude → realized miss (and orbit a barely moves).
    mm = np.linspace(0.3, 5.0, 40)
    miss4 = [build(miss=m).true_miss_km for m in mm]
    da4 = [build(miss=m).sc2_oe[0] / 1e3 - sc1[0] / 1e3 for m in mm]
    a3 = ax[1, 0]
    a3.plot(mm, miss4, "o-", color="tab:red", ms=3, label="realized miss (km)")
    a3.plot(mm, da4, "s--", color="0.6", ms=3, label="Δ semi-major axis (km)")
    a3.set_xlabel("requested |miss|  (km)"); a3.set_ylabel("km")
    a3.set_title("DOF 4 · |miss|  →  miss distance\n(orbit Δa ≈ 0: sets danger, not type)")
    a3.legend(fontsize=8); a3.grid(alpha=0.3)

    # DOF 5 — miss alpha → geometry θ (perp/dt0 split). Use a velocity tilt so the
    # ⊥-plane contains the T axis (theta_v=60) and α can swing head-on↔cross.
    als = np.linspace(0, 180, 60)
    th5 = [build(al=a, tv=60.0, pv=0.0, v=200.0).angle_deg for a in als]
    perp5 = [build(al=a, tv=60.0, pv=0.0, v=200.0).perp_km for a in als]
    dt5 = [build(al=a, tv=60.0, pv=0.0, v=200.0).dt0_km for a in als]
    a4 = ax[1, 1]
    a4.plot(als, th5, "o-", color="tab:orange", ms=3, label="geometry θ (deg)")
    a4.set_xlabel("miss α — in-plane direction (deg)")
    a4.set_ylabel("geometry θ  (0=head-on, 90=cross)")
    a4.set_title("DOF 5 · miss α  →  geometry θ\n(slides perp ↔ along-track miss)")
    a4.grid(alpha=0.3); a4.legend(fontsize=8)

    # Panel 6 — the punchline: Δi (velocity DOF) vs θ (miss DOF) are INDEPENDENT.
    a5 = ax[1, 2]
    for dval, col in [(10, "tab:blue"), (40, "tab:green"), (80, "tab:red")]:
        ths, dis = [], []
        for a in np.linspace(0, 170, 24):
            c = G.make_crossing(float(dval), miss_alpha_deg=float(a), sc1_oe=sc1)
            if c.feasible and not c.reason:
                ths.append(c.angle_deg); dis.append(_di(c))
        a5.scatter(ths, dis, s=22, color=col, label=f"Δi≈{dval}°")
    a5.set_xlabel("geometry θ (miss DOF, deg)")
    a5.set_ylabel("Δi (velocity DOF, deg)")
    a5.set_title("θ and Δi are INDEPENDENT axes\n(fix Δi, sweep α → θ moves; Δi stays)")
    a5.legend(fontsize=8); a5.grid(alpha=0.3)

    fig.suptitle("Figure E — What each of the 5 DOF drives "
                 "(reduced output vs swept input)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120); plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# FIGURE F — RTN encounter close-ups: a ZOOMED ±box around SC1 at TCA where the
# miss vector and relative-velocity arrow are drawn directly, so DOF effects are
# large and obvious. One row per DOF group (velocity DOF vs miss DOF).
# ---------------------------------------------------------------------------

def _rtn_at_tca(conj):
    """SC2's RTN position+velocity relative to SC1 at TCA (km, km/s)."""
    sc1 = np.array(state_koe_to_eci(np.asarray(conj.sc1_oe, float), AngleFormat.DEGREES))
    sc2 = np.array(state_koe_to_eci(np.asarray(conj.sc2_oe, float), AngleFormat.DEGREES))
    rel = np.array(state_eci_to_rtn(sc1, sc2))
    return rel[:3] / 1e3, rel[3:] / 1e3       # pos km, vel km/s


def _rtn_2d(ax, samples, cmap_name, box_km=6.0, arrow_km=2.5, draw_vel=True):
    """2D RTN view in the T–N plane (the model's perp/dt0 picture). samples = list of
    (pos_km, vel_kms). dot = SC2 miss offset, unit-length arrow = v_rel direction."""
    cmap = plt.get_cmap(cmap_name)
    ax.axhline(0, color="0.85", lw=0.6); ax.axvline(0, color="0.85", lw=0.6)
    ax.scatter([0], [0], s=170, marker="*", c="k", zorder=6, label="SC1")
    n = len(samples)
    for i, (pos, vel) in enumerate(samples):
        col = cmap(0.12 + 0.82 * i / max(n - 1, 1))
        ax.scatter(pos[1], pos[2], s=60, color=col, edgecolor="k", lw=0.4, zorder=4,
                   label="SC2 miss" if i == 0 else None)
        if draw_vel and np.linalg.norm(vel) > 0:
            vh = vel / np.linalg.norm(vel) * arrow_km
            ax.annotate("", xy=(pos[1] + vh[1], pos[2] + vh[2]), xytext=(pos[1], pos[2]),
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.3, alpha=0.85),
                        zorder=3)
    ax.set_xlabel("T — along-track (km)", fontsize=9)
    ax.set_ylabel("N — cross-track (km)", fontsize=9)
    ax.set_xlim(-box_km, box_km); ax.set_ylim(-box_km, box_km)
    ax.set_aspect("equal"); ax.grid(alpha=0.25)


def _rtn_3d(ax, samples, cmap_name, box_km=5.5, arrow_km=2.5, draw_vel=True):
    """3D RTN view (R,T,N) — dissolves the 2D-projection artifacts so a miss CIRCLE
    looks circular and a rotating ⊥-plane is visible as a tilted disc."""
    cmap = plt.get_cmap(cmap_name)
    ax.scatter([0], [0], [0], s=160, marker="*", c="k", zorder=6)
    n = len(samples)
    for i, (pos, vel) in enumerate(samples):
        col = cmap(0.12 + 0.82 * i / max(n - 1, 1))
        ax.scatter([pos[0]], [pos[1]], [pos[2]], s=42, color=col,
                   edgecolor="k", lw=0.3, zorder=4)
        if draw_vel and np.linalg.norm(vel) > 0:
            vh = vel / np.linalg.norm(vel) * arrow_km
            ax.plot([pos[0], pos[0] + vh[0]], [pos[1], pos[1] + vh[1]],
                    [pos[2], pos[2] + vh[2]], color=col, lw=1.1, alpha=0.8)
    ax.set_xlabel("R", fontsize=8); ax.set_ylabel("T", fontsize=8)
    ax.set_zlabel("N", fontsize=8)
    ax.set_xlim(-box_km, box_km); ax.set_ylim(-box_km, box_km); ax.set_zlim(-box_km, box_km)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=22, azim=-58)
    ax.tick_params(labelsize=6)


def fig_rtn_encounter(out_path, sc1_oe=None):
    """RTN TCA close-ups — ONE ROW PER DOF, each row = [2D T–N view | 3D R-T-N view |
    text]. The 3D view removes the projection artifacts (a miss circle reads as a
    circle; a tilting ⊥-plane reads as a tilting disc)."""
    sc1 = np.asarray(sc1_oe if sc1_oe is not None else G.CASE_SC1_OE, dtype=float)

    def samples_for(vals, build, smooth_sign=False):
        out = []
        for v in vals:
            c = build(v)
            if c.sc2_oe is None or not np.isfinite(c.sc2_oe[1]) or c.sc2_oe[1] >= 1:
                continue
            out.append(_rtn_at_tca(c))
        # DOF 2 only: the constructor's in-plane basis flips sign as the ⊥-plane
        # rotates through alignment, making the α=0 miss "jump" hemispheres. That's a
        # basis-convention artifact (the miss is the same 5 km circle). Flip later
        # points to track the previous one so the EXPLAINER sweeps smoothly. Does not
        # touch the constructor or any sweep data — display only.
        if smooth_sign and len(out) > 1:
            for i in range(1, len(out)):
                if np.dot(out[i][0], out[i - 1][0]) < 0:
                    out[i] = (-out[i][0], out[i][1])
        return out

    def build(miss=5.0, v=200.0, tv=90.0, pv=30.0, al=0.0):
        return G.make_conjunction_from_spherical(miss, v, tv, pv, al, sc1_oe=sc1)

    # one (vals, build_fn, cmap, draw_vel, smooth, title, blurb) per DOF row
    rows = [
        (np.geomspace(15, 11000, 12),
         lambda v: G.make_crossing(  # v_rel via the crossing direction so Δi is real
             float(np.interp(v, [15, 11000], [0, 115])), miss_alpha_deg=0.0, sc1_oe=sc1),
         "cool", True, False, "DOF 1 · |v_rel|  (plane crossing Δi)",
         "Bigger relative speed (cross-track) = a STEEPER plane crossing. The arrow\n"
         "(v_rel direction) tilts further out of SC1's plane; the orbit fans away."),
        (np.linspace(5, 90, 10),
         lambda t: build(tv=t, v=300.0, pv=0.0, al=0.0),
         "viridis", True, True, "DOF 2 · θ_v  (velocity polar angle off radial)",
         "θ_v tilts the velocity between radial and the T–N plane. The miss sits in\n"
         "the plane ⊥ v_rel, so as θ_v turns, that plane ROTATES — in 3D you see the\n"
         "miss disc tilt smoothly (the 2D 'jump' was just a basis-sign convention)."),
        (np.linspace(0, 90, 10),
         lambda p: build(pv=p, v=300.0, al=0.0, tv=90.0),
         "plasma", True, False, "DOF 3 · φ_v  (velocity azimuth: along → cross)",
         "φ_v swings the velocity arrow from ALONG-track (co-orbital) to CROSS-track\n"
         "(plane crossing). The miss dot (α=0) stays put; only the arrow rotates."),
        (np.linspace(0, 180, 13),
         lambda a: build(al=a, tv=60.0, pv=0.0, v=200.0),
         "spring", False, False, "DOF 5 · miss α  (in-plane miss direction → geometry θ)",
         "α rotates the miss around SC1 WITHIN its ⊥-plane — a true CIRCLE of radius\n"
         "5 km (clear in 3D; the 2D T–N view squashes it to an ellipse because the\n"
         "circle's plane is tilted). This is the geometry angle θ (perp↔along-track)."),
        (np.linspace(0.5, 5.0, 9),
         lambda m: build(miss=m, tv=60.0, pv=0.0, v=200.0, al=30.0),
         "autumn", False, False, "DOF 4 · |miss|  (magnitude — danger level, same type)",
         "The miss dot marches straight outward from SC1 along a fixed direction.\n"
         "Magnitude = how close the pass is (collision vs clear); the TYPE is unchanged."),
    ]

    fig = plt.figure(figsize=(15, 4.2 * len(rows)))
    gs = fig.add_gridspec(len(rows), 3, width_ratios=[1.0, 1.0, 0.9],
                          hspace=0.35, wspace=0.25)
    for r, (vals, bfn, cmap, dv, smooth, title, blurb) in enumerate(rows):
        smp = samples_for(vals, bfn, smooth_sign=smooth)
        ax2 = fig.add_subplot(gs[r, 0])
        _rtn_2d(ax2, smp, cmap, draw_vel=dv)
        ax2.set_title(title + "\n2D — T–N plane (model perp/dt0)", fontsize=9, loc="left")
        if r == 0:
            ax2.legend(fontsize=7, loc="upper left")
        ax3 = fig.add_subplot(gs[r, 1], projection="3d")
        _rtn_3d(ax3, smp, cmap, draw_vel=dv)
        ax3.set_title("3D — R-T-N (true geometry)", fontsize=9)
        axt = fig.add_subplot(gs[r, 2]); axt.axis("off")
        axt.text(0.0, 0.95, blurb, transform=axt.transAxes, fontsize=8.0,
                 va="top", ha="left", linespacing=1.5,
                 bbox=dict(boxstyle="round,pad=0.4", fc="0.96", ec="0.7"))

    fig.suptitle("Figure F — RTN encounter at TCA, one DOF per row  "
                 "(SC1 ★ at origin · dot = SC2 miss offset · arrow = v_rel direction)\n"
                 "left: 2D T–N (the model's view) · middle: 3D R-T-N (true geometry, "
                 "no projection squash)", fontsize=12)
    fig.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Coverage/scope figures (orbit-first)")
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild the coverage CSV from the constructor")
    ap.add_argument("--only", default=None,
                    choices=["a", "b", "c", "d", "e", "f"],
                    help="render only one figure")
    args = ap.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)

    if args.only == "d":
        fig_5dof(os.path.join(FIG_DIR, "coverage_5dof_spread.png")); return
    if args.only == "e":
        fig_dof_drivers(os.path.join(FIG_DIR, "coverage_dof_drivers.png")); return
    if args.only == "f":
        fig_rtn_encounter(os.path.join(FIG_DIR, "coverage_rtn_encounter.png")); return

    rows = load_rows(args.rebuild)

    if args.only in (None, "a"):
        fig_physical(rows, os.path.join(FIG_DIR, "leo_coverage_physical.png"))
    if args.only in (None, "b"):
        fig_abstract(rows, os.path.join(FIG_DIR, "leo_coverage_abstract.png"))
    if args.only in (None, "c"):
        fig_continuous(os.path.join(FIG_DIR, "leo_coverage_continuous.png"))
    if args.only is None:
        fig_5dof(os.path.join(FIG_DIR, "coverage_5dof_spread.png"))
        fig_dof_drivers(os.path.join(FIG_DIR, "coverage_dof_drivers.png"))
        fig_rtn_encounter(os.path.join(FIG_DIR, "coverage_rtn_encounter.png"))


if __name__ == "__main__":
    main()
