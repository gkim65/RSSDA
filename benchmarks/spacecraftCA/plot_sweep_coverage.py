"""
Coverage figure for the representative conjunction sweep (AAS/AIAA 2026).

A "lollipop / web" plot on the (a-altitude, e) plane: each conjunction is a TETHER
linking a chief SC1 (circle) to its conjuncting object SC2 (square). The sweep axes:
    geometry beta      -> the perp/dt0 miss split   (encoded as LINE STYLE)
    crossing  Delta-i  -> via (phi, v_rel)          (encoded as TETHER COLOR)
    SC1 altitude       -> a few chief anchors        (the lollipop stems)
    [physical miss]    -> handled by the belief ladder, fixed here at 5 km

The LEO scope (perigee>=150km, apogee<=2000km, e<=0.25) is drawn as boundary curves;
out-of-scope cells are shown as rejected (red x) so the figure also proves we know the
boundary. See notes/spacecraftca-representative-sweep memory + EXPERIMENTAL_SETUP.md.

Run:
    .venv/bin/python benchmarks/spacecraftCA/plot_sweep_coverage.py
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from brahe import R_EARTH

import conjunction_generator as G

R_E_KM = R_EARTH / 1e3
FIG_DIR = os.path.join(os.path.dirname(__file__), "notes", "figures")

# Scope constants (mirror conjunction_generator).
PERIGEE_FLOOR_KM = G.PERIGEE_FLOOR_KM
APOGEE_CEIL_KM = G.APOGEE_CEIL_KM
ECC_MAX = G.ECC_MAX

# --- sweep grid -------------------------------------------------------------
# (phi, v_rel) lookup giving a target Delta-i at the crossing (beta-independent).
DI_TARGETS = {1: (90, 100), 10: (90, 1300), 23: (80, 3000), 39: (70, 5000), 50: (66, 6800)}
BETAS = [90, 67, 45, 22, 0]          # geometry: 90 head-on -> 0 cross
GEOM_LABEL = {90: "head_on", 67: "obl-", 45: "oblique", 22: "obl+", 0: "cross_track"}
# line style per geometry bucket (head-on solid, oblique dashed, cross dotted)
GEOM_STYLE = {90: "-", 67: "-", 45: "--", 22: ":", 0: ":"}
SC1_ALTS = [400, 550, 800, 1200]
MISS_KM = 5.0


def build_rows(sc1_alts=SC1_ALTS, betas=BETAS, di_targets=DI_TARGETS, miss_km=MISS_KM):
    rows = []
    for alt in sc1_alts:
        sc1 = np.array([(R_E_KM + alt) * 1e3, 0.001, 55.0, 20.0, 0.0, 0.0])
        for dik, (phi, vrel) in di_targets.items():
            for beta in betas:
                c = G.make_conjunction_from_encounter(
                    miss_km=miss_km, beta_deg=beta, phi_deg=phi, v_rel_ms=vrel, sc1_oe=sc1)
                koe = c.sc2_oe
                rows.append(dict(
                    sc1_a=alt, sc1_e=0.001, sc1_i=55.0,
                    sc2_a=float(koe[0]) / 1e3 - R_E_KM, sc2_e=float(koe[1]),
                    sc2_i=float(koe[2]), dik=dik, beta=beta,
                    feasible=bool(c.feasible and not c.reason)))
    return rows


def plot(rows, out_path, ymax=0.35):
    fig, ax = plt.subplots(figsize=(13, 8))

    # scope boundary curves in (a-alt, e)
    aa = np.linspace(200, 2300, 400)
    a_m = (aa + R_E_KM) * 1e3
    e_peri = 1 - (PERIGEE_FLOOR_KM + R_E_KM) * 1e3 / a_m
    e_apo = (APOGEE_CEIL_KM + R_E_KM) * 1e3 / a_m - 1
    ub = np.clip(np.minimum(np.minimum(e_peri, np.where(e_apo > 0, e_apo, 0)), ECC_MAX), 0, ymax)
    ax.fill_between(aa, ub, ymax, color="0.9", zorder=0)
    ax.plot(aa, np.clip(e_peri, 0, ymax), "r-", lw=1.1, alpha=0.6)
    ax.plot(aa, np.clip(np.where(e_apo > 0, e_apo, np.nan), 0, ymax), color="tab:purple", lw=1.1, alpha=0.6)
    ax.axhline(ECC_MAX, color="0.5", ls="--", lw=0.9)

    norm = plt.Normalize(vmin=0, vmax=100)   # Delta-i color scale (deg)
    cmap = cm.turbo
    feas = [r for r in rows if r["feasible"]]
    rej = [r for r in rows if not r["feasible"]]

    # tethers colored by Delta-i, styled by geometry
    for r in feas:
        di = abs(r["sc2_i"] - r["sc1_i"])
        ax.plot([r["sc1_a"], r["sc2_a"]], [r["sc1_e"], r["sc2_e"]],
                color=cmap(norm(di)), lw=0.7, alpha=0.55,
                ls=GEOM_STYLE[r["beta"]], zorder=1)
    for r in feas:
        di = abs(r["sc2_i"] - r["sc1_i"])
        ax.scatter(r["sc2_a"], r["sc2_e"], s=34, marker="s",
                   c=[cmap(norm(di))], edgecolor="k", lw=0.3, zorder=3)
    for r in rej:   # rejected (out of scope) — show we know the boundary
        ax.scatter(r["sc2_a"], r["sc2_e"], s=34, marker="x", c="r", lw=1.3, zorder=2)

    # small SC1 anchors
    for alt in sorted({r["sc1_a"] for r in rows}):
        ax.scatter(alt, 0.001, s=90, marker="o", c="w", edgecolor="k", lw=1.4, zorder=5)
        ax.annotate(f"{alt}km", (alt, 0.001), (alt, -0.015), fontsize=8, ha="center")

    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.01); cb.set_label("Δi  (plane difference, deg)")

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="w", markeredgecolor="k", markersize=9, label="SC1 chief"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="0.6", markeredgecolor="k", markersize=8, label="SC2 object"),
        Line2D([0], [0], marker="x", color="r", lw=0, markersize=8, label="rejected (out of scope)"),
        Line2D([0], [0], color="0.4", ls="-", label="head-on geometry"),
        Line2D([0], [0], color="0.4", ls="--", label="oblique geometry"),
        Line2D([0], [0], color="0.4", ls=":", label="cross-track geometry"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_xlabel("semi-major-axis altitude  a − R⊕ (km)")
    ax.set_ylabel("eccentricity e")
    ax.set_title(
        f"Representative conjunction sweep — {len(feas)} feasible / {len(rows)} cells\n"
        "tether = one conjunction (SC1→SC2); color = Δi crossing; line style = geometry; shaded = out of LEO scope")
    ax.set_xlim(200, 2300); ax.set_ylim(-0.025, ymax); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out_path, dpi=120)
    print(f"wrote {out_path}  ({len(feas)} feasible / {len(rows)} cells)")


def plot_sun(rows, out_path):
    """The 'sun' view: SC1 at the centre, each conjunction a RAY to its SC2 placed by
    (Delta-i as ANGLE, |Delta-a| as RADIUS). Rays radiate in all directions -> a sun/web,
    instead of parallel tethers. Color = geometry (beta); ring distance = orbital separation."""
    fig, ax = plt.subplots(figsize=(10.5, 10.5))
    feas = [r for r in rows if r["feasible"]]

    # one panel per SC1 alt would split the sun; instead overlay all chiefs at the centre
    # and use Delta-a (SC2 minus SC1 altitude) as radius so every chief shares the origin.
    norm = plt.Normalize(vmin=0, vmax=90)     # geometry angle proxy via beta
    cmap = cm.viridis
    rmax = max(abs(r["sc2_a"] - r["sc1_a"]) for r in feas) * 1.05

    # faint radius rings (orbital separation) + angle spokes (Delta-i)
    for rr in np.linspace(rmax / 4, rmax, 4):
        ax.add_patch(plt.Circle((0, 0), rr, fill=False, color="0.85", lw=0.7, zorder=0))
        ax.annotate(f"Δa≈{rr:.0f}km", (0, rr), fontsize=7, color="0.6", ha="center")
    for di in [0, 10, 23, 39, 50]:
        ang = np.deg2rad(90 - di * 1.6)       # spread Delta-i across the upper half
        ax.plot([0, rmax * np.cos(ang)], [0, rmax * np.sin(ang)], color="0.9", lw=0.6, zorder=0)
        ax.annotate(f"Δi {di}°", (rmax * np.cos(ang), rmax * np.sin(ang)), fontsize=7, color="0.6")

    for r in feas:
        di = abs(r["sc2_i"] - r["sc1_i"])
        rad = abs(r["sc2_a"] - r["sc1_a"])
        ang = np.deg2rad(90 - di * 1.6)
        x, y = rad * np.cos(ang), rad * np.sin(ang)
        # geometry angle theta from perp/dt0 proxy via beta (0..90)
        geom = 90 - r["beta"]                  # beta90 head-on -> 0 ; beta0 cross -> 90
        ax.plot([0, x], [0, y], color=cmap(norm(geom)), lw=0.6, alpha=0.5, zorder=1)
        ax.scatter(x, y, s=30, marker="s", c=[cmap(norm(geom))], edgecolor="k", lw=0.3, zorder=3)

    ax.scatter(0, 0, s=260, marker="o", c="gold", edgecolor="k", lw=1.6, zorder=5)
    ax.annotate("SC1\nchief", (0, 0), (0, -rmax * 0.08), ha="center", fontsize=9, fontweight="bold")

    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.01, shrink=0.7)
    cb.set_label("geometry θ (0=head-on … 90=cross-track)")
    ax.set_title(f"Conjunction 'sun': {len(feas)} rays from the chief\n"
                 "angle = Δi crossing · radius = orbital separation Δa · color = geometry")
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(); fig.savefig(out_path, dpi=120)
    print(f"wrote {out_path}  ({len(feas)} feasible)")


# ===========================================================================
# SPHERICAL SWEEP coverage — plots the REAL representative sweep (the 100-conj JSON)
# on its actual sweep axes: Δi (plane crossing) × geometry-θ (perp/dt0 split), with
# SC1 altitude as marker, e/altitude as secondary panels. This is the figure that
# PROVES the down-selected sweep spreads over everything, not just (a,e).
# ===========================================================================

import json   # noqa: E402

SWEEP_JSON = os.path.join(os.path.dirname(__file__), "notes", "conj_sweep_spherical.json")
CASES_JSON = os.path.join(os.path.dirname(__file__), "notes", "conj_cases_spherical.json")


def _load_conjs(path):
    with open(path) as f:
        specs = json.load(f)
    out = []
    for s in specs:
        c = G.make_conjunction_from_orbits(
            np.asarray(s["sc1_oe"], float), np.asarray(s["sc2_oe"], float),
            name=s.get("name"))
        out.append(c)
    return out


def plot_spherical_sweep(out_path, sweep_json=SWEEP_JSON, cases_json=CASES_JSON):
    """Coverage of the spherical representative sweep on its real axes."""
    conjs = _load_conjs(sweep_json)
    cases = _load_conjs(cases_json) if os.path.exists(cases_json) else []

    def di(c):
        return abs(float(c.sc2_oe[2]) - float(c.sc1_oe[2]))

    alts = sorted({round(float(c.sc1_oe[0]) / 1e3 - R_E_KM) for c in conjs})
    amark = {a: m for a, m in zip(alts, ["o", "s", "^", "D", "v"])}

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))

    # Panel 1 — THE sweep plane: Δi × geometry-θ, colored by SC2 eccentricity.
    ev = np.array([float(c.sc2_oe[1]) for c in conjs])
    norm_e = plt.Normalize(0, max(ev.max(), 1e-3))
    for c in conjs:
        a = round(float(c.sc1_oe[0]) / 1e3 - R_E_KM)
        ax[0].scatter(di(c), c.angle_deg, s=46, marker=amark.get(a, "o"),
                      c=[cm.viridis(norm_e(float(c.sc2_oe[1])))],
                      edgecolor="k", lw=0.3, zorder=3)
    for c in cases:   # case studies as red stars
        ax[0].scatter(di(c), c.angle_deg, s=240, marker="*", c="red",
                      edgecolor="k", lw=0.6, zorder=5)
        ax[0].annotate(c.name, (di(c), c.angle_deg), (di(c) + 3, c.angle_deg + 3),
                       fontsize=8, color="darkred")
    sm = cm.ScalarMappable(norm=norm_e, cmap=cm.viridis); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax[0], pad=0.01); cb.set_label("SC2 eccentricity")
    ax[0].set_xlabel("Δi  — plane-crossing angle (deg)")
    ax[0].set_ylabel("geometry θ  (0=head-on … 90=cross-track)")
    ax[0].set_title(f"Representative sweep on its real axes ({len(conjs)} conj)\n"
                    "marker = SC1 altitude · color = SC2 e · ★ = case study")
    ax[0].grid(alpha=0.3)

    # Panel 2 — Δi × SC2 altitude (multi-altitude + retrograde coverage).
    for c in conjs:
        a = round(float(c.sc1_oe[0]) / 1e3 - R_E_KM)
        inc = float(c.sc2_oe[2])
        col = "tab:red" if inc > 90 else "tab:blue"
        ax[1].scatter(di(c), float(c.sc2_oe[0]) / 1e3 - R_E_KM, s=40,
                      marker=amark.get(a, "o"), c=col, edgecolor="k", lw=0.3, alpha=0.8)
    ax[1].set_xlabel("Δi (deg)")
    ax[1].set_ylabel("SC2 altitude  a − R⊕ (km)")
    ax[1].set_title("Altitude × Δi coverage\n(red = retrograde SC2, i>90°)")
    ax[1].grid(alpha=0.3)

    # Panel 3 — feasibility envelope (a, e) with scope curves (proves near-circular).
    aa = np.linspace(200, 2300, 400); a_m = (aa + R_E_KM) * 1e3
    e_peri = 1 - (PERIGEE_FLOOR_KM + R_E_KM) * 1e3 / a_m
    e_apo = (APOGEE_CEIL_KM + R_E_KM) * 1e3 / a_m - 1
    ub = np.clip(np.minimum(np.minimum(e_peri, np.where(e_apo > 0, e_apo, 0)), ECC_MAX), 0, 0.3)
    ax[2].fill_between(aa, ub, 0.3, color="0.9", zorder=0)
    ax[2].axhline(ECC_MAX, color="0.5", ls="--", lw=0.9, label=f"e_max={ECC_MAX}")
    for c in conjs:
        ax[2].scatter(float(c.sc2_oe[0]) / 1e3 - R_E_KM, float(c.sc2_oe[1]),
                      s=34, c="tab:blue", edgecolor="k", lw=0.2, alpha=0.7)
    ax[2].set_xlabel("SC2 altitude  a − R⊕ (km)")
    ax[2].set_ylabel("SC2 eccentricity e")
    ax[2].set_title("All near-circular, in LEO scope\n(shaded = out of scope)")
    ax[2].set_xlim(200, 1700); ax[2].set_ylim(-0.01, 0.3)
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)

    # legend for altitude markers
    from matplotlib.lines import Line2D
    hh = [Line2D([0], [0], marker=amark[a], color="w", markerfacecolor="0.6",
                 markeredgecolor="k", markersize=8, label=f"SC1 {a} km") for a in alts]
    ax[0].legend(handles=hh, fontsize=8, loc="lower right")

    fig.tight_layout(); fig.savefig(out_path, dpi=120)
    print(f"wrote {out_path}  ({len(conjs)} conj, {len(cases)} cases)")


if __name__ == "__main__":
    plot_spherical_sweep(os.path.join(FIG_DIR, "sweep_coverage_spherical.png"))