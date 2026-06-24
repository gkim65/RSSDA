"""
main.py — the ONE config-first entry point (Hydra/wandb convention).

    cfg  ->  build_scenario(cfg)  ->  solve(scenario)

The clean wandb pattern, no env vars, no subprocess, no import-time freeze:
  - Hydra loads conf/config.yaml (+ any +preset / dotpath overrides) into `cfg`.
  - scenario_config.apply_scenario(cfg) resolves it to a frozen Scenario and populates the
    pipeline's module globals + reward BEFORE any model module imports (import-order
    discipline — the keystone). main.py imports nothing heavy until after that.
  - compare_variants_v2 then solves + writes the canonical CSVs (the source of truth) using
    the SAME machinery the CLI uses (reused, not reimplemented).
  - wandb (optional) logs the resolved config dict + the per-variant summary rows on top.

Run:
  python main.py                                  # full-res head-on reference (3 variants)
  python main.py +preset=verify                   # the regression anchor (Cen=SDec=-7.83/0%)
  python main.py +preset=coarse solve.variants=[centralized,sdec]   # quick machinery check
  python main.py belief.init_miss=0.5 reward.man_cost=-3
  python main.py wandb.enabled=true wandb.project=spacecraftCA run.tag=exp1
  python main.py -m belief.init_miss=0.5,3,5      # Hydra multirun sweep
"""
import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_BENCH)
for _p in (_ROOT, _BENCH, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scenario_config import scenario_from_cfg, apply_scenario


def _abs(p):
    """Resolve a run path relative to this file (Hydra changes cwd to its run dir)."""
    return p if os.path.isabs(p) else os.path.join(_HERE, p)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=== resolved config ===")
    print(OmegaConf.to_yaml(cfg))

    # 1. cfg -> Scenario -> populate the model globals (KEYSTONE). MUST precede model imports.
    scenario = apply_scenario(scenario_from_cfg(cfg))

    # 2. import the solver entry point AFTER the scenario is applied (its own bootstrap sees
    #    _APPLIED_SCENARIO and no-ops, so the orbit-array conjunction we just built is kept).
    import compare_variants_v2 as CV

    run = cfg.run
    out_dir, fig_dir = _abs(str(run.out_dir)), _abs(str(run.fig_dir))
    tag = str(run.tag)

    # 3. solve via the SAME machinery the CLI uses. We hand it an argv mirroring the run knobs
    #    (the scenario knobs are already applied; CV's bootstrap is a no-op now). CSVs are the
    #    source of truth; CV.main returns the in-memory rows for wandb.
    argv = [
        "compare_variants_v2.py",
        "--variants", ",".join(str(v) for v in scenario.variants),
        "--init-miss", str(scenario.init_miss),
        "--init-spread", str(scenario.init_spread),
        "--perp", str(scenario.perp),
        "--iter-limit", str(scenario.iter_limit),
        "--tag", tag, "--out-dir", out_dir, "--fig-dir", fig_dir,
    ]
    if scenario.contact_stages is not None:
        argv += ["--contact-stages",
                 ",".join(str(s) for s in scenario.contact_stages)]
    if not bool(run.figures):
        argv += ["--no-figures"]
    sys.argv = argv
    result = CV.main()

    # 4. optional brahe-rollout validation of the solved policy (rollout_v2 MC), reused.
    if bool(run.get("rollout", False)):
        _run_rollout(scenario, run, out_dir, tag)

    # 5. wandb on top of the CSV (CSV stays source of truth). Logs the resolved config + the
    #    per-variant summary rows. FAIL-SOFT: a wandb auth/network failure must NOT lose a run
    #    whose CSVs already succeeded — warn and continue (the CSV is the real artifact).
    if bool(cfg.wandb.get("enabled", False)):
        try:
            _log_wandb(cfg, scenario, result, tag)
        except Exception as e:
            print(f"\n  [wandb] logging FAILED ({type(e).__name__}: {e}). The run + CSVs are "
                  f"intact — only the wandb upload was skipped. Fix auth with `wandb login "
                  f"--relogin` (key from https://wandb.ai/authorize), or use wandb.mode=offline "
                  f"and `wandb sync` later.")

    print(f"\n=== done (tag={tag}). CSVs -> {out_dir} ===")


def _run_rollout(scenario, run, out_dir, tag):
    """Brahe MC validation of the solved policy (rollout_v2), reused not reimplemented. Same
    process — the scenario is already applied, so rollout_v2's bootstrap no-ops."""
    import rollout_v2 as RV  # noqa: F401  (import applies/imports against the live scenario)
    print("\n=== brahe rollout validation (rollout_v2 MC) ===")
    for variant in scenario.variants:
        if variant == "dec":
            print("  (skip dec rollout — RS-MAA* policy API not wired in rollout_v2)")
            continue
        argv = [
            "rollout_v2.py", "--variant", variant, "--mode", "mc",
            "--rollouts", str(int(run.rollouts)),
            "--init-miss", str(scenario.init_miss),
            "--init-spread", str(scenario.init_spread),
            "--perp", str(scenario.perp),
            "--csv", os.path.join(out_dir, f"rollout_{tag}_{variant}.csv"),
        ]
        if scenario.contact_stages is not None and variant == "sdec":
            argv += ["--contact-stages",
                     ",".join(str(s) for s in scenario.contact_stages)]
        sys.argv = argv
        RV.main()


def _log_wandb(cfg, scenario, result, tag):
    """Log the resolved config dict + per-variant summary rows to wandb. The CSV remains the
    source of truth; this is the experiment-tracking layer on top."""
    import wandb
    mode = str(cfg.wandb.get("mode", "online"))
    if not bool(cfg.wandb.get("enabled", False)):
        mode = "disabled"
    run = wandb.init(
        project=str(cfg.wandb.get("project", "spacecraftCA")),
        entity=cfg.wandb.get("entity", None),
        name=cfg.wandb.get("name", None) or tag,
        mode=mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    # provenance the run actually used
    wandb.summary["n_stages"] = result.get("n_stages")
    wandb.summary["n_contacts"] = len(result.get("contact_stages", []))
    # per-variant headline metrics (mirrors sweep_driver's tidy row fields)
    table = wandb.Table(columns=[
        "variant", "expected_return", "collision_prob", "expected_dv_ms", "expected_syncs"])
    for r in result.get("summary", []):
        wandb.log({
            f"{r['matrix_variant']}/expected_return": float(r["expected_return"]),
            f"{r['matrix_variant']}/collision_prob": float(r["collision_prob"]),
            f"{r['matrix_variant']}/dv_ms": float(r["expected_dv_ms"]),
            f"{r['matrix_variant']}/syncs": float(r["expected_syncs"]),
        })
        table.add_data(r["matrix_variant"], float(r["expected_return"]),
                       float(r["collision_prob"]), float(r["expected_dv_ms"]),
                       float(r["expected_syncs"]))
    wandb.log({"variant_summary": table})
    run.finish()


if __name__ == "__main__":
    main()
