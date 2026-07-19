"""
plot_reward.py  --  the v2 two-ramp TERMINAL REWARD figure (a PAPER figure).

The v2 terminal reward drawn as two opposing ramps on a signed-log axis, with a
zoomed side panel at the optimum. The curves are pulled LIVE from
`spacecraft_transition_v2` so the figure can never drift from the model (falls back
to matching constants if the model isn't importable).

This is a data-free, instant-running plot script. It was factored out of
`plot_v2_concept.py` (which also holds schematic explainer panels not used in the
paper) so the paper figure has a clean, self-contained source.

Outputs (to benchmarks/spacecraftCA/notes/figures/):
  reward.pdf                     -- the paper figure (annotated, normal weight, transparent)
  concept_4_two_ramp_reward.png  -- same, PNG raster (+ .pdf)
  concept_4_two_ramp_reward.svg  -- thicker lines (s=1.8) for slides/web
  reward_black.pdf / .svg        -- black-slide version (white foreground, transparent bg)
  reward_black_bg.pdf            -- black background BAKED IN (for a non-black page)

Usage:
  .venv/bin/python benchmarks/spacecraftCA/plot_reward.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIGDIR = os.path.join(_HERE, "notes", "figures")

# Pull the REAL v2 reward so the two-ramp panel can never drift from the model.
sys.path.insert(0, _HERE)
try:
    import spacecraft_transition_v2 as TV
except Exception as _e:  # figure still renders with matching fallback constants
    TV = None
    print(f"[warn] could not import spacecraft_transition_v2 ({_e}); "
          f"using fallback reward constants.")


# --------------------------------------------------------------------------
# Typography: use real LaTeX (Computer Modern) if available, else mathtext-cm.
# Keeps every label in the same serif the paper body uses.
# --------------------------------------------------------------------------
def _setup_typography():
    import shutil
    common = {
        "font.family": "serif",
        "mathtext.fontset": "cm",       # Computer Modern for math (fallback path)
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "font.size": 11,
        "savefig.dpi": 300,
    }
    if shutil.which("latex"):
        try:
            plt.rcParams.update({
                "text.usetex": True,
                "font.serif": ["Computer Modern Roman"],
                "text.latex.preamble": r"\usepackage{amsmath}",
                **common,
            })
            # smoke-test: usetex fails lazily at draw time, so force one render now
            fig = plt.figure()
            fig.text(0.5, 0.5, r"$\delta T$")
            fig.canvas.draw()
            plt.close(fig)
            print("[typography] using real LaTeX (Computer Modern)")
            return
        except Exception as e:
            plt.rcParams["text.usetex"] = False
            print(f"[typography] LaTeX render failed ({e}); falling back to mathtext-cm")
    plt.rcParams.update({"text.usetex": False,
                         "font.serif": ["CMU Serif", "cmr10", "STIXGeneral", "DejaVu Serif"],
                         **common})
    print("[typography] using mathtext Computer Modern (no usetex)")


_setup_typography()

# consistent palette
C_PERP = "#9467bd"    # perp / frozen standoff / displacement ramp  (purple)
C_MISS = "#d62728"    # miss / danger / risk ramp                   (red)
C_BURN = "#2ca02c"    # delta-v / optimum / 5 km cleared            (green)
C_PEN = "#e6007a"     # 1 km max-penalty guide                      (magenta)


def _save(fig, name, transparent=False, vector=True, exts=(".pdf", ".svg")):
    """Save the PNG plus, when vector=True, crisp vector copies (`exts`, same base name) that
    stay sharp at any zoom -- PDF for LaTeX/paper, SVG for web/slides. Pass exts to control
    which vector formats are written (e.g. only PDF here, a thicker SVG elsewhere)."""
    os.makedirs(_FIGDIR, exist_ok=True)
    base = os.path.join(_FIGDIR, name)
    saved = [base]
    fig.savefig(base, dpi=150, bbox_inches="tight", transparent=transparent)
    if vector:
        for ext in exts:
            vpath = os.path.splitext(base)[0] + ext
            fig.savefig(vpath, bbox_inches="tight", transparent=transparent)
            saved.append(vpath)
    plt.close(fig)
    print("saved " + ", ".join(saved))


def _reward_fns():
    """Return (risk_fn, disp_fn, params) from the LIVE model, or a matching fallback."""
    if TV is not None:
        return (TV.risk_ramp_reward, TV.displacement_cost,
                dict(RISK_COLLISION_KM=TV.RISK_COLLISION_KM,
                     RISK_CLEARED_KM=TV.RISK_CLEARED_KM,
                     RISK_MAX_PENALTY=TV.RISK_MAX_PENALTY,
                     DISP_TUBE=TV.DISP_TUBE_HALFWIDTH_KM,
                     DISP_K=TV.DISP_QUADRATIC_K))
    p = dict(RISK_COLLISION_KM=1.0, RISK_CLEARED_KM=5.0, RISK_MAX_PENALTY=-10000.0,
             DISP_TUBE=5.0, DISP_K=0.2)

    def risk_fn(m):
        if m <= p["RISK_COLLISION_KM"]:
            return p["RISK_MAX_PENALTY"]
        if m >= p["RISK_CLEARED_KM"]:
            return 0.0
        frac = (m - p["RISK_COLLISION_KM"]) / (p["RISK_CLEARED_KM"] - p["RISK_COLLISION_KM"])
        return p["RISK_MAX_PENALTY"] * 0.5 * (1.0 + np.cos(np.pi * frac))

    def disp_fn(dt):
        excess = max(abs(dt) - p["DISP_TUBE"], 0.0)
        return -p["DISP_K"] * excess * excess

    return risk_fn, disp_fn, p


def _signed_log(r):
    """Compress magnitude while keeping sign: y = sign(r)*log10(1+|r|).
    Maps 0->0, -1->-0.3, -10->-1.04, -100->-2, -10000->-4 so a -10000 risk floor and a
    few-hundred displacement bowl are both legible on ONE axis. Tick labels show REAL values."""
    r = np.asarray(r, dtype=float)
    return np.sign(r) * np.log10(1.0 + np.abs(r))


def _draw_two_ramp(ax, x, risk, disp, total, p, *, legend=True, s=1.0, dark=False):
    """Draw the signed-log two-ramp panel onto ax. Total is drawn BEHIND (thick grey) so the
    red risk + purple displacement ramps read on top of it. `s` scales every line width /
    tick / marker thickness uniformly (s=1 normal; larger => thicker, for slide/SVG use).
    `dark=True` recolors foreground (text/ticks/spines/grey-total) to read on a black slide;
    the transparent background is left to the caller so ONE figure drops onto either theme."""
    from matplotlib.ticker import FixedLocator
    fg = "white" if dark else "black"            # axis text / ticks / spines
    total_c = "0.75" if dark else "0.55"         # thick "total" line: lighter grey on black
    disp_lbl = (r"displacement ramp $\propto (|\delta T|-\mathrm{tube})^2$"
                if p["DISP_K"] is not None else r"displacement ramp $\propto |\delta T|$")
    ax.plot(x, _signed_log(total), color=total_c, lw=5 * s, label="total terminal reward", zorder=1)
    ax.plot(x, _signed_log(risk), color=C_MISS, lw=2.2 * s, ls="--",
            label="risk ramp (near-field collision)", zorder=3)
    ax.plot(x, _signed_log(disp), color=C_PERP, lw=2.2 * s, ls="-.", label=disp_lbl, zorder=3)

    ticks = [0, -1, -10, -100, -1000, -10000]
    ax.yaxis.set_major_locator(FixedLocator([_signed_log(t) for t in ticks]))
    ax.set_yticklabels([str(t) for t in ticks])
    ax.set_ylim(_signed_log(1.05 * p["RISK_MAX_PENALTY"]), 0.2)
    ax.set_xlim(0, x.max())
    ax.axhline(0, color=fg, lw=0.5 * s, alpha=0.6)
    # 1 km max-penalty line (magenta) and 5 km screening-cleared line (green)
    ax.axvline(p["RISK_COLLISION_KM"], color=C_PEN, lw=1.2 * s, ls="--")
    ax.axvline(p["RISK_CLEARED_KM"], color=C_BURN, lw=1 * s, ls=":")
    ax.set_xlabel(r"Miss distance at TCA  (km)", color=fg)
    ax.set_ylabel(r"Terminal reward  (signed-log scale)", color=fg)
    ax.grid(True, alpha=0.2, linewidth=0.8 * s)
    ax.tick_params(width=1.0 * s, length=4 * s, colors=fg)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0 * s)
        spine.set_edgecolor(fg)
    if legend:
        leg = ax.legend(loc="lower right", frameon=True, framealpha=0.0 if dark else 0.92)
        if dark:  # legend text must flip to white too
            for txt in leg.get_texts():
                txt.set_color(fg)
            leg.get_frame().set_edgecolor(fg)


def plot_two_ramp_reward():
    """v2 reward = two opposing ramps on a signed-log axis, with a zoomed side panel at the
    optimum. Writes the paper PDF plus PNG/SVG and the black-slide variants."""
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FixedLocator
    risk_fn, disp_fn, p = _reward_fns()

    x = np.linspace(0, 25, 3000)
    risk = np.array([risk_fn(v) for v in x])
    disp = np.array([disp_fn(v) for v in x])
    total = risk + disp

    # optimum = argmax of total past the collision floor (the 5 km knee)
    win = x >= p["RISK_COLLISION_KM"]
    imax = int(np.argmax(np.where(win, total, -np.inf)))

    # zoom window (REAL reward units): x 4..7 km, reward +0.5 .. -2 -- the ramp handoff
    ZX0, ZX1, ZR_TOP, ZR_BOT = 4.0, 7.0, 0.5, -2.0

    def _build(s, dark=False):
        """Build the full 2-panel figure at thickness scale s and return it.
        dark=True flips foreground to white for black slides (background stays transparent)."""
        fig, (axm, axz) = plt.subplots(
            1, 2, figsize=(12.5, 5.4), gridspec_kw={"width_ratios": [2.1, 1]})

        fg = "white" if dark else "black"
        rect_c = "0.7" if dark else "0.25"   # zoom-window dotted rectangle / zoom-panel spines

        opt_x, opt_y = x[imax], _signed_log(total[imax])

        # ---- main signed-log panel ----
        _draw_two_ramp(axm, x, risk, disp, total, p, s=s, dark=dark)
        # dotted rectangle marking the zoom window (real reward units -> signed-log positions)
        ry0, ry1 = _signed_log(ZR_BOT), _signed_log(ZR_TOP)
        axm.add_patch(Rectangle((ZX0, ry0), ZX1 - ZX0, ry1 - ry0, fill=False,
                                ec=rect_c, ls=(0, (2, 2)), lw=1.5 * s, zorder=7))
        # optimum star + green callout
        axm.plot(opt_x, opt_y, "*", color=C_BURN, ms=26, alpha=0.75,
                 markeredgecolor=C_BURN, markeredgewidth=0.6, zorder=8)
        axm.annotate("Optimum\nReward at 5km", (opt_x, opt_y), xytext=(0, 46),
                     textcoords="offset points", color=C_BURN, fontsize=11,
                     ha="center", va="bottom",
                     arrowprops=dict(arrowstyle="->", color=C_BURN, lw=1.4 * s))
        # 1 km max-penalty label (magenta, rotated along its line)
        axm.text(p["RISK_COLLISION_KM"] - 0.25, _signed_log(-200),
                 "1km zone -- maximum penalty", color=C_PEN, fontsize=9,
                 rotation=90, ha="right", va="center")
        # 5 km screening-cleared label (green, below axis)
        axm.text(p["RISK_CLEARED_KM"] + 0.15, _signed_log(-2500),
                 "5km cleared\nscreening zone", color=C_BURN, fontsize=9,
                 ha="left", va="center")

        # ---- side zoom panel (shorter than the main plot) ----
        _draw_two_ramp(axz, x, risk, disp, total, p, legend=False, s=s, dark=dark)
        axz.set_xlim(ZX0, ZX1)
        axz.set_ylim(_signed_log(ZR_BOT), _signed_log(ZR_TOP))
        itk = [0, -1, -2]
        axz.yaxis.set_major_locator(FixedLocator([_signed_log(t) for t in itk]))
        axz.set_yticklabels([str(t) for t in itk])
        axz.set_ylabel(r"Reward (signed-log)", fontsize=10, color=fg)
        axz.set_title(r"Zoom: tradeoff at optimum", fontsize=10, color=fg)
        axz.plot(opt_x, opt_y, "*", color=C_BURN, ms=26, alpha=0.75,
                 markeredgecolor=C_BURN, markeredgewidth=0.6, zorder=8)
        for sp in axz.spines.values():
            sp.set(linestyle=(0, (2, 2)), edgecolor=rect_c, linewidth=1.5 * s)
        pos = axz.get_position()
        axz.set_position([pos.x0, pos.y0 + 0.20, pos.width, pos.height * 0.55])

        # transparent background: clear the figure + both axes patches (legend frame too)
        fig.patch.set_alpha(0.0)
        for a in (axm, axz):
            a.patch.set_alpha(0.0)
        leg = axm.get_legend()
        if leg is not None and not dark:
            leg.get_frame().set_alpha(0.0)
        return fig

    # Annotated PDF (the paper figure); plus PNG + a THICKER SVG for slide/web use.
    _save(_build(s=1.0), "reward.pdf", transparent=True, vector=False)
    _save(_build(s=1.0), "concept_4_two_ramp_reward.png", transparent=True, exts=(".pdf",))
    _save(_build(s=1.8), "concept_4_two_ramp_reward.svg", transparent=True, vector=False)
    # Black-slide versions: white foreground on a transparent background.
    _save(_build(s=1.0, dark=True), "reward_black.pdf", transparent=True, vector=False)
    _save(_build(s=1.8, dark=True), "reward_black.svg", transparent=True, vector=False)
    # Opaque-black-background PDF (background baked in, for placing on a non-black page).
    figk = _build(s=1.0, dark=True)
    figk.patch.set_alpha(1.0)
    figk.patch.set_facecolor("black")
    for a in figk.axes:
        a.patch.set_alpha(1.0)
        a.set_facecolor("black")
    _bpath = os.path.join(_FIGDIR, "reward_black_bg.pdf")
    figk.savefig(_bpath, bbox_inches="tight", facecolor="black")
    plt.close(figk)
    print("saved " + _bpath)


if __name__ == "__main__":
    plot_two_ramp_reward()
    print("\nDone. reward.{pdf} + slide/black variants in notes/figures/")
