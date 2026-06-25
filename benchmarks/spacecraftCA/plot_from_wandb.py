"""
plot_from_wandb.py — regenerate EVERY figure from a wandb run's logged tables.

The companion to replot_from_csv.py: that one rebuilds the figures from the canonical CSVs;
this one rebuilds the SAME figures from a wandb run (local run dir OR a remote run id). main.py
logs the FULL row dicts into wandb tables (variant_summary / reward_parts / burn_timing /
action_schedule / brahe_summary / brahe_rollouts), and the plot functions are pure functions of
those rows, so this is a thin adapter:

    wandb run  ->  {table: [row dicts]}  ->  the existing plot fns + save_histograms

Two sources (auto-detected; --run can be either):
  - a LOCAL run dir:  wandb/run-<ts>-<id>/   (reads files/media/table/*.table.json directly; no
    network, works offline). Pass the dir, or just the bare <id> to search ./wandb for it.
  - a REMOTE run:     entity/project/<id>     (pulls the tables via wandb.Api()).

Usage:
  python plot_from_wandb.py --run wandb/run-20260624_213015-3jl25uzp
  python plot_from_wandb.py --run 3jl25uzp                       # search ./wandb for the dir
  python plot_from_wandb.py --run kmeans_gsopt/spacecraftCA/3jl25uzp   # remote via API
  python plot_from_wandb.py --run <id> --fig-dir notes/figures --tag from_wandb
"""
import os
import sys
import glob
import json
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from compare_variants import (plot_summary, plot_burn_timing, plot_action_schedule)
from compare_variants_v2 import plot_reward_parts

# The figure tables this pipeline knows how to render, and the plot fn each drives. The brahe
# tables are handled separately (histograms via save_histograms; band bars are summary scalars).
_POMDP_TABLES = [
    ("variant_summary", plot_summary, "summary"),
    ("reward_parts", plot_reward_parts, "reward_parts"),
    ("burn_timing", plot_burn_timing, "burn_timing"),
    ("action_schedule", plot_action_schedule, "action_schedule"),
]


def _table_json_to_rows(d):
    """{'columns': [...], 'data': [[...], ...]} -> [ {col: val}, ... ]."""
    cols = d.get("columns", [])
    return [dict(zip(cols, row)) for row in d.get("data", [])]


def _load_tables_local(run_dir):
    """Read every *.table.json under a local run dir into {table_name: [row dicts]}. wandb names
    the files '<key>_<step>_<hash>.table.json' — strip the trailing _<step>_<hash>."""
    tdir = os.path.join(run_dir, "files", "media", "table")
    if not os.path.isdir(tdir):
        raise FileNotFoundError(f"no table dir at {tdir} (is this a wandb run dir?)")
    tables = {}
    for path in sorted(glob.glob(os.path.join(tdir, "*.table.json"))):
        base = os.path.basename(path)[: -len(".table.json")]
        # strip '_<step>_<hexhash>' -> the logged key
        parts = base.rsplit("_", 2)
        name = parts[0] if len(parts) == 3 else base
        with open(path) as f:
            tables[name] = _table_json_to_rows(json.load(f))
    return tables


def _load_tables_remote(run_path):
    """Pull the logged tables from a remote run (entity/project/id) via wandb.Api."""
    import wandb
    api = wandb.Api()
    run = api.run(run_path)
    tables = {}
    for key in [t[0] for t in _POMDP_TABLES] + ["brahe_rollouts", "brahe_summary"]:
        try:
            tbl = run.use_artifact  # noqa: silence linters; real fetch below
        except Exception:
            pass
        try:
            t = run.summary.get(key) or run.history(keys=[key]).get(key)
        except Exception:
            t = None
        # wandb stores logged Tables as run files <key>.table.json — fetch + parse that.
        try:
            fobj = run.file(f"media/table/{key}.table.json")
            local = fobj.download(replace=True, root="/tmp/wandb_tables")
            with open(local.name if hasattr(local, "name") else local) as f:
                tables[key] = _table_json_to_rows(json.load(f))
        except Exception:
            if t is not None:
                tables[key] = _table_json_to_rows(t if isinstance(t, dict) else {})
    return {k: v for k, v in tables.items() if v}


def _resolve_run_dir(run):
    """A bare id or a path -> a local run dir if one exists under ./wandb."""
    if os.path.isdir(run) and os.path.basename(run.rstrip("/")).startswith("run-"):
        return run
    if os.path.isdir(os.path.join(run, "files")):
        return run
    # search ./wandb for run-*-<id>
    for base in (os.path.join(_HERE, "..", "..", "wandb"), os.path.join(_HERE, "wandb"),
                 os.path.join(os.getcwd(), "wandb")):
        hits = glob.glob(os.path.join(base, f"run-*-{run}"))
        if hits:
            return hits[0]
    return None


def _plot_histograms(brahe_rollouts, fig_dir, tag):
    """Per-policy START/END miss histograms from the brahe_rollouts table, reusing rollout_v2's
    save_histograms (the SAME figure the rollout CLI emits). Rows carry init/brahe/matrix miss."""
    import rollout_v2 as RV
    outs = []
    by_policy = {}
    for r in brahe_rollouts:
        by_policy.setdefault(r.get("policy", "policy"), []).append(r)
    for policy, rows in by_policy.items():
        # save_histograms reads init_miss_km / brahe_miss_km / matrix_miss_km per result dict.
        results = [{"init_miss_km": float(r["init_miss_km"]),
                    "brahe_miss_km": float(r["brahe_miss_km"]),
                    "matrix_miss_km": float(r["matrix_miss_km"])} for r in rows]
        safe = policy.replace("[", "").replace("]", "").replace("/", "_")
        out = os.path.join(fig_dir, f"hist_{tag}_{safe}.png")
        RV.save_histograms(results, policy, "mc", out)
        outs.append(out)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    help="local wandb run dir, a bare run id (searched under ./wandb), "
                         "or a remote entity/project/id.")
    ap.add_argument("--fig-dir", default=os.path.join(_HERE, "notes", "figures"))
    ap.add_argument("--tag", default=None, help="figure filename tag (default: the run id).")
    args = ap.parse_args()

    run_dir = _resolve_run_dir(args.run)
    if run_dir is not None:
        print(f"  source: LOCAL run dir {run_dir}")
        tables = _load_tables_local(run_dir)
        run_id = os.path.basename(run_dir.rstrip("/")).split("-")[-1]
    else:
        print(f"  source: REMOTE run {args.run} (via wandb.Api)")
        tables = _load_tables_remote(args.run)
        run_id = args.run.split("/")[-1]

    tag = args.tag or run_id
    os.makedirs(args.fig_dir, exist_ok=True)
    any_done = False

    for key, fn, label in _POMDP_TABLES:
        rows = tables.get(key)
        if not rows:
            print(f"  SKIP {label}: table '{key}' missing/empty in the run")
            continue
        # coerce the numeric columns the plot fns int()/float() internally expect
        for r in rows:
            for k in ("init_bin", "stage", "joint_action"):
                if k in r and r[k] is not None:
                    r[k] = int(float(r[k]))
        out = fn(rows, args.fig_dir, tag)
        print(f"  {label:16} -> {out}")
        any_done = True

    brahe = tables.get("brahe_rollouts")
    if brahe:
        for out in _plot_histograms(brahe, args.fig_dir, tag):
            print(f"  {'histogram':16} -> {out}")
        any_done = True
    else:
        print("  SKIP histograms: table 'brahe_rollouts' missing (run had no rollout/baseline)")

    if not any_done:
        print(f"No renderable tables found in run '{args.run}'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
