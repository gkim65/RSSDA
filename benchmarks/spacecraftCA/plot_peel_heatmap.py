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
    "window_nearburn_c3": "Window (near-burn)",
    "minimal": "Minimal",
}


def _subset_label(name):
    if name in _SUBSET_LABELS:
        return _SUBSET_LABELS[name]
    n = (name or "").strip()
    if n.startswith("greedy_drop"):
        idx = n[len("greedy_drop"):]
        return f"Greedy drop {idx}" if idx else "Greedy drop"
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


def _make_figure(sdec_rows, cen_row, dec_row, n_stages, tol, colors, theme, title):
    """Build one peel heatmap figure for a single conjunction."""
    n_rows = len(sdec_rows)

    # Returns for scaling the bar panel.
    sdec_ret = [_fnum(r.get("expected_return"), 0.0) for r in sdec_rows]
    cen_ret = _fnum(cen_row.get("expected_return")) if cen_row else None
    dec_ret = _fnum(dec_row.get("expected_return")) if dec_row else None
    all_ret = [x for x in (sdec_ret + [cen_ret, dec_ret]) if x is not None]
    rmin, rmax = min(all_ret), max(all_ret)
    span = (rmax - rmin) or 1.0
    pad = 0.12 * span
    xlo, xhi = rmin - pad, rmax + 0.35 * span  # extra right room for value labels

    # Figure size: fit an 8.5x11 page. Height grows gently with row count.
    fig_h = min(10.5, 2.4 + 0.62 * n_rows)
    fig = plt.figure(figsize=(8.0, fig_h))
    fig.patch.set_alpha(0.0)
    # Left = stage heatmap, right = return bars.
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.06,
                          left=0.24, right=0.90, top=0.86, bottom=0.14)
    ax_hm = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1], sharey=ax_hm)

    ink = colors["ink"]
    for ax in (ax_hm, ax_bar):
        ax.patch.set_alpha(0.0)
        for sp in ax.spines.values():
            sp.set_color(ink)
        ax.tick_params(colors=ink, labelcolor=ink)

    # y ordering: first greedy row at TOP -> reads top-to-bottom.
    y_of = lambda i: (n_rows - 1 - i)

    # -------- heatmap panel --------
    contact_stages_model = None
    try:
        import spacecraft_stage_grid as G  # noqa: F811
        contact_stages_model = set(int(s) for s in G.CONTACT_STAGES)
    except Exception:
        pass

    for i, r in enumerate(sdec_rows):
        y = y_of(i)
        kept = set(_parse_contacts(r.get("contacts", ""), n_stages))
        for st in range(n_stages):
            face = colors["cell"] if st in kept else "none"
            edge = colors["cell_edge"] if st in kept else colors["grid"]
            ax_hm.add_patch(Rectangle((st - 0.5, y - 0.42), 1.0, 0.84,
                                      facecolor=face, edgecolor=edge,
                                      linewidth=0.6, alpha=0.95 if st in kept else 0.45))

    ax_hm.set_xlim(-0.5, n_stages - 0.5)
    ax_hm.set_ylim(-0.6, n_rows - 0.4)
    ax_hm.set_xticks(range(n_stages))
    ax_hm.set_xticklabels([str(s) for s in range(n_stages)], fontsize=9)
    ax_hm.set_yticks([y_of(i) for i in range(n_rows)])
    ax_hm.set_yticklabels(
        [f"{_subset_label(r.get('subset_name'))}  ($n_c$={_syncs(r)})" for r in sdec_rows],
        fontsize=10)
    ax_hm.set_xlabel("Stage index (T$-$24\\,h $\\rightarrow$ TCA)"
                     if plt.rcParams.get("text.usetex") else
                     "Stage index (T-24h to TCA)")
    ax_hm.set_title("Kept sync contacts", fontsize=11, color=ink, pad=8)

    # Mark which stages are ever a GS-contact opportunity (model contact grid),
    # so blank columns that were never candidates are visually distinct.
    if contact_stages_model:
        for st in range(n_stages):
            if st in contact_stages_model:
                ax_hm.plot([st], [n_rows - 0.28], marker="v", ms=5,
                           color=colors["cen"], clip_on=False)

    # -------- bar panel (expected return) --------
    for i, r in enumerate(sdec_rows):
        y = y_of(i)
        ret = sdec_ret[i]
        ax_bar.barh(y, ret - xlo, left=xlo, height=0.62,
                    color=colors["bar"], edgecolor=colors["cell_edge"],
                    linewidth=0.5, alpha=0.9)
        coll = _fnum(r.get("collision_prob_matrix"), None)
        coll_txt = f", coll {coll*100:.2f}\\%" if (coll is not None and plt.rcParams.get("text.usetex")) \
            else (f", coll {coll*100:.2f}%" if coll is not None else "")
        ax_bar.text(ret + 0.015 * span, y, f"{ret:.2f}{coll_txt}",
                    va="center", ha="left", fontsize=9, color=ink)

    # Centralized rail: dashed line + tol band.
    if cen_ret is not None:
        ax_bar.axvspan(cen_ret - tol, cen_ret + tol, color=colors["band"],
                       alpha=0.18, lw=0)
        ax_bar.axvline(cen_ret, color=colors["cen"], ls="--", lw=1.6,
                       label=f"Centralized rail ({cen_ret:.2f})")
    if dec_ret is not None and (cen_ret is None or abs(dec_ret - cen_ret) > 1e-9):
        ax_bar.axvline(dec_ret, color=colors["dec"], ls=":", lw=1.6,
                       label=f"Dec rail ({dec_ret:.2f})")

    ax_bar.set_xlim(xlo, xhi)
    ax_bar.set_xlabel("Expected return")
    ax_bar.set_title("Performance vs. rails", fontsize=11, color=ink, pad=8)
    plt.setp(ax_bar.get_yticklabels(), visible=False)
    ax_bar.tick_params(axis="y", length=0)

    # Legend placed ABOVE the panels so it never overlaps the data.
    handles = [Patch(facecolor=colors["cell"], edgecolor=colors["cell_edge"],
                     label="Kept sync contact")]
    if contact_stages_model:
        handles.append(plt.Line2D([0], [0], marker="v", ls="none",
                                  color=colors["cen"], label="GS-contact stage"))
    if cen_ret is not None:
        handles.append(plt.Line2D([0], [0], color=colors["cen"], ls="--",
                                  label="Centralized rail"))
        handles.append(Patch(facecolor=colors["band"], alpha=0.18,
                             label=f"$\\pm$ tol band ($\\pm${tol:g})"
                             if plt.rcParams.get("text.usetex") else f"+/- tol ({tol:g})"))
    if dec_ret is not None:
        handles.append(plt.Line2D([0], [0], color=colors["dec"], ls=":",
                                  label="Dec rail"))
    leg = fig.legend(handles=handles, loc="upper center", ncol=min(3, len(handles)),
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
    args = ap.parse_args()

    # Resolve paths relative to this script's directory when not absolute,
    # so it runs the same from any cwd.
    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(_HERE, args.csv)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(_HERE, args.out_dir)

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
    n_stages = _resolve_n_stages(rows, override=args.n_stages)

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

        pretty = label.replace("_", " ")
        pretty = pretty[:1].upper() + pretty[1:] if pretty else "Conjunction"
        title = f"Greedy contact peel-down: {pretty}" if label else "Greedy contact peel-down"
        fig = _make_figure(sdec_rows, cen_row, dec_row, n_stages, args.tol,
                           colors, args.theme, title)

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