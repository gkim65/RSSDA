#!/usr/bin/env python3
"""Greedy contact peel-down heatmap.

Visualizes the "greedy contact peel-down" experiment produced by peel_contacts.py:
the SDec variant starts with all ground-station (GS) sync contacts and greedily
removes them one at a time, keeping a removal only while the SDec expected return
still matches the Centralized rail within a tolerance.

Figure layout (top -> bottom = the greedy story of contacts being peeled away):
  * Each row is one solved SDec subset, in the order the greedy search tried them
    (window first, then greedy_drop candidates in order, then 'minimal').
  * Columns span ALL stages 0 .. N_STAGES-1 (not just contact stages), so the
    reader sees how sparse the kept contacts are against the full stage grid.
    A cell is shaded iff that stage is a kept contact in that row.
  * A right-hand bar panel encodes expected return (returns are negative; bars run
    from a common baseline and the numeric value is printed at each bar end).
    The Centralized rail is drawn as a vertical dashed line with a +/- tol band,
    and the Dec rail (no syncs) as a second reference line.
  * Collision % is annotated per row so we can show safety did not degrade.

Usage:
  python3 plot_peel_heatmap.py --csv notes/results/peel_peel_ready.csv \
      --tag peel_ready --theme light --tol 0.001

Outputs (per conjunction label): notes/figures/peel_heatmap_<tag>[_<label>].{pdf,svg,png}
"""
import argparse
import csv
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

_HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# Typography: real LaTeX (Computer Modern) if available, else mathtext-cm.
# --------------------------------------------------------------------------
def _setup_typography():
    common = {
        "font.family": "serif",
        "mathtext.fontset": "cm",
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
            fig = plt.figure()
            fig.text(0.5, 0.5, r"$\delta$")
            fig.canvas.draw()
            plt.close(fig)
            print("[typography] using real LaTeX (Computer Modern)")
            return
        except Exception as e:  # pragma: no cover - depends on host latex
            plt.rcParams["text.usetex"] = False
            print(f"[typography] LaTeX render failed ({e}); falling back to mathtext-cm")
    plt.rcParams.update({
        "text.usetex": False,
        "font.serif": ["CMU Serif", "cmr10", "STIXGeneral", "DejaVu Serif"],
        **common,
    })
    print("[typography] using mathtext Computer Modern (no usetex)")


# --------------------------------------------------------------------------
# Theme (must read on either a black or a white slide/paper background).
# We keep the axes/patch backgrounds transparent and only pick foreground ink.
# --------------------------------------------------------------------------
def _theme_colors(theme):
    if theme == "dark":
        return {
            "ink": "#f0f0f0",        # text / axes / spines
            "grid": "#666666",
            "cell": "#4da3ff",       # shaded kept-contact cell
            "cell_edge": "#cfe6ff",
            "bar": "#4da3ff",
            "cen": "#ff6b6b",        # centralized rail
            "dec": "#c8a2ff",        # dec rail
            "band": "#ff6b6b",
            "burn": "#ff3b3b",       # maneuver-stage hatch (reads on dark)
        }
    return {
        "ink": "#1a1a1a",
        "grid": "#bcbcbc",
        "cell": "#1f77b4",
        "cell_edge": "#0d3c61",
        "bar": "#1f77b4",
        "cen": "#d62728",
        "dec": "#7d3fb5",
        "band": "#d62728",
        "burn": "#d10000",           # maneuver-stage hatch (reads on light)
    }


# --------------------------------------------------------------------------
# N_STAGES: prefer the live model; robustly fall back to CSV-derived count so
# the script still runs where brahe (the propagator) is not importable.
# --------------------------------------------------------------------------
def _resolve_n_stages(rows, override=None):
    if override:                     # explicit --n-stages wins (escape hatch when the model
        n = int(override)            # can't be imported and the data lacks n_stages)
        print(f"[stages] N_STAGES={n} (from --n-stages override)")
        return n
    try:
        import spacecraft_stage_grid as G  # noqa: F401
        n = int(G.N_STAGES)
        print(f"[stages] N_STAGES={n} (from spacecraft_stage_grid)")
        return n
    except Exception as e:
        n = _n_stages_from_rows(rows)
        print(f"[stages] N_STAGES={n} (data fallback; model import failed: {e})")
        return n


def _n_stages_from_rows(rows):
    n = 0
    for r in rows:
        try:
            n = max(n, int(float(r.get("n_stages", 0) or 0)))
        except (TypeError, ValueError):
            pass
        for s in _parse_contacts(r.get("contacts", ""), n_stages=None):
            n = max(n, s + 1)
    return max(n, 1)


def _parse_contacts(raw, n_stages):
    """Parse a contacts cell into a sorted list of stage ints.

    "ALL" -> every stage (needs n_stages); "" / blank / "NONE" -> [].
    Otherwise a comma string like "0,1,2".
    """
    s = (raw or "").strip()
    if not s or s.upper() in ("NONE", "-"):
        return []
    if s.upper() == "ALL":
        return list(range(n_stages)) if n_stages else []
    out = []
    for tok in s.replace(";", ",").split(","):
        tok = tok.strip()
        if tok == "":
            continue
        try:
            out.append(int(float(tok)))
        except ValueError:
            continue
    return sorted(set(out))


# --------------------------------------------------------------------------
# Clean display labels (sentence case; map raw underscore names).
# --------------------------------------------------------------------------
_SUBSET_LABELS = {
    "__centralized__": "Centralized (all syncs)",
    "__dec__": "Dec (no syncs)",
    "minimal": "Minimal",
}


def _subset_label(name):
    """Pretty, sentence-case display label. Trailing "_c<k>" or a "greedy_drop<k>"
    index render the dropped stage as a math subscript $C_{k}$."""
    if name in _SUBSET_LABELS:
        return _SUBSET_LABELS[name]
    n = (name or "").strip()
    if n.startswith("greedy_drop"):
        idx = n[len("greedy_drop"):].strip("_")
        return f"Greedy drop $C_{{{idx}}}$" if idx else "Greedy drop"
    if n.startswith("window"):
        # window_nearburn_c9 -> "Window (near-burn, $C_9$)"
        idx = n.split("_c")[-1] if "_c" in n else ""
        return f"Window (near-burn, $C_{{{idx}}}$)" if idx.isdigit() else "Window (near-burn)"
    if not n:
        return "(unnamed)"
    # generic: underscores -> spaces, sentence case
    txt = n.replace("_", " ").strip()
    return txt[:1].upper() + txt[1:]


def _fnum(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
def _read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _read_rows_wandb(entity, project, tag=None):
    """Pull peel rows from a wandb project as CSV-shaped dicts (same keys _read_rows yields), so
    the rest of the pipeline is source-agnostic. The peel run logs ONE run per solved subset:
    config carries the strings (label / variant / contacts / subset_name / is_final), history/
    summary carries the numeric metrics (expected_return / collision_prob_matrix / n_stages ...).

    Runs come back unordered; we sort by creation time so the greedy SEQUENCE (rails -> window ->
    greedy drops -> minimal) is preserved, exactly like CSV row order. tag filters on the run
    `group` (peel sets group=tag) so one project can hold many peel runs."""
    import wandb
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    rows = []
    for run in runs:
        cfg = dict(run.config or {})
        if tag is not None and run.group != tag:
            continue
        # metrics: prefer summary (final value); fall back to config if a string slipped in.
        summ = dict(run.summary or {})
        def pick(k):
            v = summ.get(k, cfg.get(k, ""))
            # wandb summary can hold dicts for system metrics; keep only scalars/strings
            return "" if isinstance(v, dict) else v
        rows.append({
            "label": cfg.get("label", ""),
            "variant": cfg.get("variant", ""),
            "contacts": cfg.get("contacts", ""),
            "subset_name": cfg.get("subset_name", ""),
            "is_final": cfg.get("is_final", ""),
            "expected_return": pick("expected_return"),
            "collision_prob_matrix": pick("collision_prob_matrix"),
            "n_stages": pick("n_stages"),
            "n_contacts": pick("n_contacts"),
            "_created_at": getattr(run, "created_at", "") or "",
        })
    rows.sort(key=lambda r: r["_created_at"])   # greedy sequence order
    return rows


def _conj_key(row):
    return (row.get("label") or "").strip()


def _gs_contacts_from_orbits(conj_file):
    """Authoritative per-conjunction GS-contact grid, recomputed from the orbits.

    Returns {label: (n_stages, set(contact_stages))} by running the SAME computation
    the peel/solve pipeline uses (CG.conjunction_contacts on each conjunction's orbit
    pair). This is the TRUE ground-station opportunity set -- unlike the union of
    peeled contacts, it includes contacts the window pre-filter dropped before peeling.
    Returns {} if brahe/the model can't be imported here (caller then falls back to
    the data-union), so the script still runs on hosts without the propagator."""
    try:
        from brahe import initialize_eop
        initialize_eop()
        import sweep_driver as SD
        import conjunction_generator as CG
    except Exception as e:
        print(f"[gs-contacts] orbit recompute unavailable ({e}); "
              f"falling back to peeled-contact union")
        return {}
    out = {}
    for c in SD.conjunctions_from_file(conj_file):
        all_times, cs = CG.conjunction_contacts(c)
        label = (c.name or c.label or "").strip()
        out[label] = (len(all_times), set(int(s) for s in cs))
        print(f"[gs-contacts] {label!r}: N_STAGES={len(all_times)} "
              f"contacts({len(cs)})={sorted(int(s) for s in cs)}")
    return out


def _load_burn_map(path):
    """Load the burn-stage map written by inspect_burn_stages.py.

    Shape: { label: { subset_key: {sc1_rate:[...], sc2_rate:[...], n_stages:N, ...} } }.
    Returns {} (no overlay) if the path is missing/unreadable so the figure still
    renders without it."""
    if not path or not os.path.exists(path):
        return {}
    try:
        import json
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"[burns] could not read {path}: {e}; skipping the maneuver overlay")
        return {}


def _burn_rates_for_row(burn_group, contacts_str, n_stages):
    """Match a figure row (by its kept-contact set) to a burn-map subset entry and
    return (sc1_rate, sc2_rate) as length-n_stages float lists (0 if no match).

    The inspector keys subsets by their contact tag ("c15-16-17-..."); we normalize
    both the tag and the row's `contacts` cell to a frozenset of stage ints so the
    match is order/format independent."""
    zero = [0.0] * n_stages
    if not burn_group:
        return zero, zero
    want = frozenset(_parse_contacts(contacts_str, n_stages))

    def key_to_set(k):
        # "c15-16-17" -> {15,16,17}; rails/other keys -> empty (won't match sdec rows)
        if not k.startswith("c"):
            return frozenset()
        return frozenset(int(t) for t in k[1:].split("-") if t.strip().isdigit())

    match = None
    for k, v in burn_group.items():
        if key_to_set(k) == want:
            match = v
            break
    if match is None:
        print(f"[burns]   no match for contacts={sorted(want)} "
              f"(burn keys: {sorted(burn_group)})")
        return zero, zero

    return _pad_rates(match, n_stages, f"contacts={sorted(want)}")


def _pad_rates(entry, n_stages, tag):
    """Pad a burn-map entry's sc1_rate/sc2_rate to length n_stages (0-fill)."""
    zero = [0.0] * n_stages

    def pad(rates):
        r = [float(x) for x in (rates or [])]
        return (r + zero)[:n_stages]
    s1, s2 = pad(entry.get("sc1_rate")), pad(entry.get("sc2_rate"))
    nz = [i for i in range(n_stages) if s1[i] > 0 or s2[i] > 0]
    print(f"[burns]   matched {tag} -> burn stages {nz}")
    return s1, s2


def _burn_rates_by_key(burn_group, key, n_stages):
    """Fetch burn rates for a rail cell keyed directly (e.g. '__dec__')."""
    zero = [0.0] * n_stages
    entry = (burn_group or {}).get(key)
    if not entry:
        return zero, zero
    return _pad_rates(entry, n_stages, key)


def _make_figure(sdec_rows, cen_row, dec_row, n_stages, contacts_all, burn_group,
                 tol, colors, theme, title):
    """Build one peel heatmap figure for a single conjunction."""
    n_rows = len(sdec_rows)

    # Bar panel shows DELTA return from the Centralized rail: 0 = matches the
    # rail, negative = degradation as contacts are peeled away. Anchors the story.
    cen_ret = _fnum(cen_row.get("expected_return")) if cen_row else None
    dec_ret = _fnum(dec_row.get("expected_return")) if dec_row else None
    base = cen_ret if cen_ret is not None else 0.0
    sdec_d = [_fnum(r.get("expected_return"), base) - base for r in sdec_rows]
    dec_d = (dec_ret - base) if dec_ret is not None else None
    all_d = [x for x in (sdec_d + [0.0, dec_d]) if x is not None]
    dmin, dmax = min(all_d), max(all_d)
    span = (dmax - dmin) or 1.0
    # x-range: left headroom for the value labels (they sit at each bar tip),
    # right up to 0 (the rail) plus a hair.
    xlo, xhi = dmin - 0.28 * span, min(0.0, dmax) + 0.02 * span

    # Extra rows bracket the peel: TOP = full GS-contact grid ("all syncs", the
    # set the peel started from); BOTTOM = Dec (no syncs), the risk floor.
    has_dec_row = dec_row is not None
    n_grid = n_rows + 1 + (1 if has_dec_row else 0)

    # Figure size: fit an 8.5x11 page. Rows are short so the figure stays compact.
    fig_h = min(5.0, 1.3 + 0.24 * n_grid)
    fig = plt.figure(figsize=(8.0, fig_h))
    fig.patch.set_alpha(0.0)
    # Left = stage heatmap, right = a thin return-bar strip (~1/8 of the width).
    gs = fig.add_gridspec(1, 2, width_ratios=[6.5, 0.9], wspace=0.04,
                          left=0.24, right=0.88, top=0.80, bottom=0.17)
    ax_hm = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1], sharey=ax_hm)

    ink = colors["ink"]
    for ax in (ax_hm, ax_bar):
        ax.patch.set_alpha(0.0)
        for sp in ax.spines.values():
            sp.set_color(ink)
        ax.tick_params(colors=ink, labelcolor=ink)

    # y ordering, top -> bottom: GS-contacts reference row, sdec search rows,
    # then (if present) the Dec no-sync row at the very bottom (y = 0).
    y_top = n_grid - 1                       # reference (all GS contacts) row
    y_of = lambda i: (n_grid - 2 - i)        # sdec search rows below it
    y_dec = 0 if has_dec_row else None       # Dec no-sync row at the floor

    # -------- heatmap panel --------
    # Reference "all GS contacts" row = the TRUE per-conjunction GS-contact
    # opportunity set, recomputed from the orbits (passed in as contacts_all).
    # Fall back to the union of peeled contacts if the orbit recompute wasn't
    # available -- the greedy peel only removes contacts, so that union is the
    # set the search STARTED from (a subset of the true opportunities).
    if contacts_all:
        contact_stages_model = set(int(s) for s in contacts_all)
    else:
        contact_stages_model = set()
        for r in sdec_rows:
            contact_stages_model |= set(_parse_contacts(r.get("contacts", ""), n_stages))

    cell_h = 0.78  # cells nearly fill each unit row -> compact, no fat gaps

    def _hatch_density(rate, glyph):
        # Full-alpha hatch; DENSITY encodes burn frequency. matplotlib sets hatch
        # spacing by how many times the glyph repeats: more glyphs => finer lines.
        # 4 tiers so rare vs frequent burns read at a glance.
        if rate <= 0:
            return None
        n = 1 if rate < 0.25 else 2 if rate < 0.5 else 3 if rate < 0.85 else 5
        return glyph * n

    def _draw_row(y, kept, sc1_rate=None, sc2_rate=None):
        for st in range(n_stages):
            on = st in kept
            face = colors["cell"] if on else "none"
            edge = colors["cell_edge"] if on else colors["grid"]
            ax_hm.add_patch(Rectangle((st - 0.5, y - cell_h / 2), 1.0, cell_h,
                                      facecolor=face, edgecolor=edge,
                                      linewidth=0.6, alpha=0.95 if on else 0.45))
            # Maneuver overlay: SC1 -> "/" family, SC2 -> "\" family. Full alpha;
            # hatch density scales with how often that spacecraft burns there.
            for rates, glyph in ((sc1_rate, "/"), (sc2_rate, "\\")):
                if not rates:
                    continue
                hatch = _hatch_density(rates[st], glyph)
                if hatch is None:
                    continue
                ax_hm.add_patch(Rectangle((st - 0.5, y - cell_h / 2), 1.0, cell_h,
                                          facecolor="none", edgecolor=colors["burn"],
                                          hatch=hatch, linewidth=0.7, alpha=1.0))

    # Reference row: all original GS contacts (no policy => no burns).
    _draw_row(y_top, contact_stages_model)
    for i, r in enumerate(sdec_rows):
        contacts_str = r.get("contacts", "")
        s1, s2 = _burn_rates_for_row(burn_group, contacts_str, n_stages)
        _draw_row(y_of(i), set(_parse_contacts(contacts_str, n_stages)), s1, s2)
    # Dec row (bottom): empty sync grid (never syncs) + its own burn hatches.
    if has_dec_row:
        d1, d2 = _burn_rates_by_key(burn_group, "__dec__", n_stages)
        _draw_row(y_dec, set(), d1, d2)

    ax_hm.set_xlim(-0.5, n_stages - 0.5)
    ax_hm.set_ylim(-0.6, n_grid - 0.4)
    ax_hm.set_xticks(range(n_stages))
    ax_hm.set_xticklabels([str(s) for s in range(n_stages)], fontsize=9)
    yticks = [y_top] + [y_of(i) for i in range(n_rows)]
    ylabels = [f"GS contacts (all)  ($n_c$={len(contact_stages_model)})"] + \
        [f"{_subset_label(r.get('subset_name'))}  ($n_c$={_syncs(r)})" for r in sdec_rows]
    if has_dec_row:
        yticks.append(y_dec)
        ylabels.append("Dec (no syncs)  ($n_c$=0)")
    ax_hm.set_yticks(yticks)
    ax_hm.set_yticklabels(ylabels, fontsize=9)
    ax_hm.set_xlabel("Stage index (T$-$24\\,h $\\rightarrow$ TCA)"
                     if plt.rcParams.get("text.usetex") else
                     "Stage index (T-24h to TCA)")
    ax_hm.set_title("Kept sync contacts", fontsize=11, color=ink, pad=6)

    # -------- bar panel (delta return from Centralized) --------
    # Bars run from 0 (the rail) leftward to each row's negative delta. No bar on
    # the top reference row (it has no return of its own).
    def _bar(y, d, color):
        ax_bar.barh(y, d, left=0.0, height=cell_h,
                    color=color, edgecolor=colors["cell_edge"],
                    linewidth=0.5, alpha=0.9)
        # Value label just past each bar's tip (to its left), inside the panel.
        ax_bar.text(d - 0.02 * span, y, f"{d:+.2f}", clip_on=False,
                    va="center", ha="right", fontsize=8, color=ink)

    for i in range(n_rows):
        _bar(y_of(i), sdec_d[i], colors["bar"])
    if has_dec_row and dec_d is not None:
        _bar(y_dec, dec_d, colors["dec"])   # Dec bar in the Dec color = the risk floor

    # Centralized rail = 0 (solid). Dec rail dotted so sdec bars read against it.
    ax_bar.axvline(0.0, color=colors["cen"], ls="--", lw=1.6,
                   label="Centralized rail ($\\Delta$=0)")
    if dec_d is not None and abs(dec_d) > 1e-9:
        ax_bar.axvline(dec_d, color=colors["dec"], ls=":", lw=1.4,
                       label=f"Dec rail ({dec_d:+.2f})")

    ax_bar.set_xlim(xlo, xhi)
    ax_bar.set_xlabel("$\\Delta$ return from centralized"
                      if plt.rcParams.get("text.usetex") else
                      "Delta return from centralized")
    # The panel is intentionally thin: only two x-ticks (the two rails) so the
    # tick labels never collide.
    from matplotlib.ticker import MaxNLocator
    ax_bar.xaxis.set_major_locator(MaxNLocator(nbins=2, prune="both"))
    ax_bar.tick_params(axis="x", labelsize=8)
    plt.setp(ax_bar.get_yticklabels(), visible=False)
    ax_bar.tick_params(axis="y", length=0)

    # Legend placed ABOVE the panels so it never overlaps the data.
    handles = [Patch(facecolor=colors["cell"], edgecolor=colors["cell_edge"],
                     label="Sync contact")]
    if burn_group:  # only advertise the overlay when burn data was supplied
        # Density = burn frequency; show a mid-density swatch for each spacecraft.
        handles.append(Patch(facecolor="none", edgecolor=colors["burn"], lw=0.7,
                             hatch="///", label="SC1 maneuver"))
        handles.append(Patch(facecolor="none", edgecolor=colors["burn"], lw=0.7,
                             hatch="\\\\\\", label="SC2 maneuver"))
    if cen_ret is not None:
        handles.append(plt.Line2D([0], [0], color=colors["cen"], ls="--",
                                  label="Centralized rail"))
    if dec_ret is not None:
        handles.append(plt.Line2D([0], [0], color=colors["dec"], ls=":",
                                  label="Dec rail"))
    leg = fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 5),
                     bbox_to_anchor=(0.57, 0.99), frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(ink)

    fig.suptitle(title, y=1.015, fontsize=13, color=ink)
    return fig


def _syncs(row):
    s = _fnum(row.get("syncs"), None)
    if s is not None:
        return int(s)
    return len(_parse_contacts(row.get("contacts", ""), None))


def _save(fig, out_dir, base, theme):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in (".pdf", ".svg", ".png"):
        p = os.path.join(out_dir, base + ext)
        fig.savefig(p, bbox_inches="tight", transparent=True,
                    dpi=300 if ext == ".png" else None)
        paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser(description="Greedy contact peel-down heatmap.")
    ap.add_argument("--csv", default=os.path.join("notes", "results", "peel_peel_ready.csv"),
                    help="Local peel CSV (used unless --source wandb).")
    ap.add_argument("--source", choices=["csv", "wandb"], default="csv",
                    help="Where peel rows come from: local CSV (default) or the wandb API.")
    ap.add_argument("--wandb-entity", default="kmeans_gsopt")
    ap.add_argument("--wandb-project", default="spacecraftCAsyncs",
                    help="wandb project to pull from (e.g. spacecraftCAsyncs, or "
                         "spacecraftCAsyncsTol for the tol=0.001 run).")
    ap.add_argument("--wandb-tag", default=None,
                    help="Filter wandb runs to this peel `group`/tag (else all runs in project).")
    ap.add_argument("--out-dir", default=os.path.join("notes", "figures"))
    ap.add_argument("--tag", default="peel")
    ap.add_argument("--theme", choices=["light", "dark"], default="light")
    ap.add_argument("--tol", type=float, default=0.001)
    ap.add_argument("--n-stages", type=int, default=None,
                    help="Force the stage-grid width (else from the model, then CSV/wandb "
                         "n_stages). Use if the model can't be imported and n_stages is absent.")
    ap.add_argument("--conj", default=None,
                    help="Filter to a single conjunction label (else one figure per label).")
    ap.add_argument("--conj-file",
                    default=os.path.join("notes", "conj_cases_spherical.json"),
                    help="Conjunction orbit specs; the top 'GS contacts (all)' row + the stage "
                         "grid are recomputed per conjunction from these orbits (authoritative). "
                         "Falls back to the peeled-contact union if brahe can't be imported.")
    ap.add_argument("--burn-json", default=None,
                    help="Burn-stage map from inspect_burn_stages.py. When given, each row gets a "
                         "hashed-red maneuver overlay (SC1 '///', SC2 '\\\\\\\\'; alpha ~ burn "
                         "frequency). Omit for no overlay.")
    args = ap.parse_args()

    # Resolve paths relative to this script's directory when not absolute,
    # so it runs the same from any cwd.
    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(_HERE, args.csv)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(_HERE, args.out_dir)
    conj_file = args.conj_file if os.path.isabs(args.conj_file) \
        else os.path.join(_HERE, args.conj_file)
    # --burn-json is a file the USER generates (often at the repo root), so resolve
    # it against the CWD first (normal CLI behavior); fall back to the script dir.
    burn_json = None
    if args.burn_json:
        if os.path.isabs(args.burn_json):
            burn_json = args.burn_json
        elif os.path.exists(args.burn_json):
            burn_json = os.path.abspath(args.burn_json)
        else:
            burn_json = os.path.join(_HERE, args.burn_json)

    _setup_typography()
    colors = _theme_colors(args.theme)

    if args.source == "wandb":
        print(f"[source] wandb {args.wandb_entity}/{args.wandb_project}"
              f"{f' tag={args.wandb_tag}' if args.wandb_tag else ''}")
        rows = _read_rows_wandb(args.wandb_entity, args.wandb_project, args.wandb_tag)
        if not rows:
            raise SystemExit(f"No peel runs in {args.wandb_entity}/{args.wandb_project}"
                             f"{f' with tag {args.wandb_tag}' if args.wandb_tag else ''}")
    else:
        rows = _read_rows(csv_path)
        if not rows:
            raise SystemExit(f"No rows in {csv_path}")
    # Per-stage maneuver overlay (optional): { label: { subset_key: {sc1_rate,...} } }.
    burn_map = _load_burn_map(burn_json)
    if burn_json:
        print(f"[burns] overlay from {burn_json}: {len(burn_map)} label(s)"
              if burn_map else f"[burns] no usable data in {burn_json}; overlay off")

    # Authoritative per-conjunction GS-contact grid, recomputed from the orbits.
    gs_by_label = _gs_contacts_from_orbits(conj_file) if os.path.exists(conj_file) else {}
    if not gs_by_label:
        print(f"[gs-contacts] no orbit grid (missing {conj_file} or brahe); "
              f"top row falls back to the peeled-contact union")

    # Group by conjunction label.
    labels = []
    for r in rows:
        k = _conj_key(r)
        if k not in labels:
            labels.append(k)
    if args.conj is not None:
        labels = [l for l in labels if l == args.conj]
        if not labels:
            raise SystemExit(f"--conj {args.conj!r} not found. Available: "
                             f"{sorted({_conj_key(r) for r in rows})}")

    saved_all = []
    for label in labels:
        grp = [r for r in rows if _conj_key(r) == label]
        # Preserve CSV row order for the sdec search rows.
        sdec_rows = [r for r in grp if (r.get("variant") or "").strip() == "sdec"]
        cen_row = next((r for r in grp if (r.get("variant") or "").strip() == "centralized"), None)
        dec_row = next((r for r in grp if (r.get("variant") or "").strip() == "dec"), None)
        if not sdec_rows:
            print(f"[skip] label {label!r}: no sdec rows")
            continue

        # Per-conjunction grid (head_on != oblique != cross_track). Precedence:
        # --n-stages override > recomputed-from-orbits > this group's data.
        gs_n, contacts_all = gs_by_label.get(label, (None, None))
        if args.n_stages:
            n_stages = int(args.n_stages)
        elif gs_n:
            n_stages = gs_n
        else:
            n_stages = _n_stages_from_rows(grp)
        # Guard: never clip a peeled contact index that lives beyond the grid.
        n_stages = max(n_stages, _n_stages_from_rows(grp))
        print(f"[stages] label={label!r} N_STAGES={n_stages}"
              f"{' (from orbits)' if gs_n else ''}")

        pretty = label.replace("_", " ")
        pretty = pretty[:1].upper() + pretty[1:] if pretty else "Conjunction"
        title = f"Greedy contact peel-down: {pretty}" if label else "Greedy contact peel-down"
        burn_group = burn_map.get(label, {})
        fig = _make_figure(sdec_rows, cen_row, dec_row, n_stages, contacts_all,
                           burn_group, args.tol, colors, args.theme, title)

        base = f"peel_heatmap_{args.tag}"
        if len(labels) > 1 and label:
            safe = "".join(c if c.isalnum() else "_" for c in label)
            base = f"{base}_{safe}"
        paths = _save(fig, out_dir, base, args.theme)
        plt.close(fig)
        saved_all.extend(paths)
        print(f"[saved] label={label!r}  rows={len(sdec_rows)}  -> {paths[0]}")

    print(f"\nDone. {len(saved_all)} files written across {len(labels)} conjunction(s).")


if __name__ == "__main__":
    main()