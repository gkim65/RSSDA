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
    #
    #    solve.variants=[] => SKIP the POMDP solve entirely (a baseline-only run). The solve and
    #    the B1 baseline are independent "policy sets"; either, both, or neither may run. We do
    #    NOT pass an empty --variants through CV.main (untested path); we just don't call it.
    if scenario.variants:
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
    else:
        print("=== solve.variants is empty -> skipping the POMDP solve (baseline-only run) ===")
        result = {"summary": [], "n_stages": None, "contact_stages": []}

    # 4. optional brahe-rollout validation of the solved policy (rollout_v2 MC), reused. We
    #    CAPTURE the per-variant brahe band stats so wandb can log them on the SAME axis as the
    #    B1 baseline (brahe-MC vs brahe-MC = the direct, honest comparison).
    brahe_rows = []
    if bool(run.get("rollout", False)):
        brahe_rows += _run_rollout(scenario, run, out_dir, tag)

    # 4b. optional B1 operator-heuristic FLOOR. Additive, its OWN policy + gate (NOT a
    #     solve.variants entry). Reuses baseline_b1's B1PolicySource through rollout_v2's brahe
    #     engine for the SAME conjunction/belief/backend and reports the SAME band stats +
    #     maneuvers — so its row sits alongside the POMDP brahe rows on one axis. CSV is the
    #     source of truth.
    if bool(cfg.get("baseline", {}).get("enabled", False)):
        brahe_rows.append(_run_baseline_b1(cfg, scenario, out_dir, tag))

    # 5. wandb on top of the CSV (CSV stays source of truth). Logs the resolved config + the
    #    per-variant summary rows. FAIL-SOFT: a wandb auth/network failure must NOT lose a run
    #    whose CSVs already succeeded — warn and continue (the CSV is the real artifact).
    if bool(cfg.wandb.get("enabled", False)):
        try:
            # feasibility provenance (E): is this conjunction dangerous enough to act on?
            import spacecraft_transition_v2 as TV
            _ib, flagged, eff_miss = TV.build_init_b_danger(
                scenario.init_miss, scenario.init_spread, scenario.perp, sign_mode="both")
            _log_wandb(cfg, scenario, result, tag, brahe_rows=brahe_rows,
                       flagged=bool(flagged), eff_miss=float(eff_miss))
        except Exception as e:
            print(f"\n  [wandb] logging FAILED ({type(e).__name__}: {e}). The run + CSVs are "
                  f"intact — only the wandb upload was skipped. Fix auth with `wandb login "
                  f"--relogin` (key from https://wandb.ai/authorize), or use wandb.mode=offline "
                  f"and `wandb sync` later.")

    print(f"\n=== done (tag={tag}). CSVs -> {out_dir} ===")


def _rows_to_table(wandb, rows):
    """Build a wandb.Table from a list of row dicts, columns = the first row's keys (stable
    order). Logging the FULL rows (not a subset) lets plot_from_wandb feed them straight into the
    existing plot fns. Empty -> a single-column placeholder table (wandb requires >=1 column)."""
    if not rows:
        return wandb.Table(columns=["empty"])
    cols = list(rows[0].keys())
    t = wandb.Table(columns=cols)
    for r in rows:
        t.add_data(*[r.get(c) for c in cols])
    return t


# Per-rollout columns carried into the wandb brahe table (the histogram + dV/burns + fidelity
# figures' source). Mirrors rollout_v2's per-rollout dict so wandb can rebuild every distribution
# plot WITHOUT the CSV. Kept small (<=25 stages x <=200 rollouts).
_BRAHE_ROLLOUT_COLS = ["init_miss_km", "brahe_miss_km", "matrix_miss_km", "total_dv", "n_burns",
                       "true_term_reward", "matrix_term_reward", "dt_err_km"]


def _brahe_band_summary(name, results):
    """Reduce a list of rollout_v2 per-rollout dicts to ONE tidy brahe-MC row. Same engine,
    same 4-7km band definition + collision floor for EVERY policy (POMDP variants and the B1
    baseline) so the rows are directly comparable on one axis. Also carries the maneuvers
    (mean dV + mean burns) the user wants tracked for both, plus the per-rollout rows (under the
    non-table key `_per_rollout`) so wandb can rebuild the histograms + fidelity figures."""
    import numpy as np
    import rollout_v2 as RV
    import spacecraft_discretizer_v2 as D
    miss = np.array([r["brahe_miss_km"] for r in results])
    dv = np.array([r["total_dv"] for r in results])
    nb = np.array([r["n_burns"] for r in results])
    tr = np.array([r["true_term_reward"] for r in results])
    mr = np.array([r["matrix_term_reward"] for r in results])
    de = np.array([r["dt_err_km"] for r in results])
    bs = RV._band_stats(miss)
    return {
        "policy": name,
        "brahe_collision_prob": float(np.mean(miss < D.COLLISION_THRESHOLD_KM)),
        "brahe_inband_pct": float(bs["inband"]),
        "brahe_below_pct": float(bs["below"]),
        "brahe_above_pct": float(bs["above"]),
        "brahe_mean_miss_km": float(bs["mean"]),
        "brahe_min_miss_km": float(bs["min"]),
        "brahe_max_miss_km": float(bs["max"]),
        "mean_dv_ms": float(dv.mean()),
        "mean_burns": float(nb.mean()),
        "leq2_burn_pct": float(np.mean(nb <= 2)),
        "term_reward_gap": float(tr.mean() - mr.mean()),   # binning over/under-credit (validation)
        "dt_absmean_err_km": float(np.abs(de).mean()),     # matrix-vs-brahe fidelity
        "n_rollouts": len(results),
        # non-table payloads consumed by _log_wandb (NOT scalar columns):
        "_per_rollout": [{c: r.get(c) for c in _BRAHE_ROLLOUT_COLS} for r in results],
    }


def _run_rollout(scenario, run, out_dir, tag):
    """Brahe MC validation of the solved policy (rollout_v2), reused not reimplemented. Same
    process — the scenario is already applied, so rollout_v2's bootstrap no-ops. Returns a
    brahe band row per rolled-out variant (for wandb + the same-axis B1 comparison)."""
    import rollout_v2 as RV
    import spacecraft_transition_v2 as TV
    print("\n=== brahe rollout validation (rollout_v2 MC) ===")
    rows = []
    for variant in scenario.variants:
        if variant == "dec":
            print("  (skip dec rollout — RS-MAA* policy API not wired in rollout_v2)")
            continue
        if scenario.contact_stages is not None and variant == "sdec":
            import spacecraft_matrices as M
            M.set_contact_stages(list(scenario.contact_stages))
        init_b, _flagged, _eff = TV.build_init_b_danger(
            scenario.init_miss, scenario.init_spread, scenario.perp, sign_mode="both")
        T, O, R, perp, sdec, full = RV.CV.solve_policy(variant, init_b)
        results = RV.run_mc(T, O, R, perp, sdec, full, init_b, TV.N_OBS_AGENT,
                            int(run.rollouts), seed=0)
        RV.save_csv(results, os.path.join(out_dir, f"rollout_{tag}_{variant}.csv"))
        RV.summarize(results, variant, "mc")
        rows.append(_brahe_band_summary(variant, results))
    return rows


def _run_baseline_b1(cfg, scenario, out_dir, tag):
    """Run the B1 operator-heuristic floor IN-PROCESS (the chosen seam): build its PolicySource
    + T/O/R via baseline_b1's own helpers (REUSE, not reimplement) and fly it through
    rollout_v2's vectorized brahe engine for the SAME conjunction/belief/backend the POMDP used.
    Writes the per-rollout CSV (source of truth), prints the band summary, and returns ONE brahe
    band row in the SAME shape as the POMDP rollout rows (same-axis comparison + maneuvers)."""
    import importlib
    import rollout_v2 as RV
    import spacecraft_transition_v2 as TV
    # baseline_b1 lives in the sibling baselines_spacecraftCA/ package; its top-level scenario
    # bootstrap no-ops here because main.py already applied the scenario (_APPLIED_SCENARIO set).
    # Importing it also puts baselines_spacecraftCA/ on sys.path, so we reuse ITS belief_filter
    # handle (B1.BF) rather than re-importing — same module, same constants.
    B1 = importlib.import_module("baselines_spacecraftCA.baseline_b1")
    BF = B1.BF

    bc = cfg.baseline
    s1 = bc.get("strategy_sc1") or bc.strategy
    s2 = bc.get("strategy_sc2") or bc.strategy
    strat_names = [str(s1), str(s2)]
    variant = str(bc.variant)
    tle_sigma = bc.get("tle_sigma")
    tle_sigma = BF.TLE_SIGMA_BASE_KM if tle_sigma is None else float(tle_sigma)

    print(f"\n=== B1 operator-heuristic floor (in-process) ===")
    print(f"  strategy: SC1={s1} SC2={s2}  policy={bc.policy} selfish_model={bc.selfish_model} "
          f"other_obs={bc.other_obs}  obs-variant={variant}")

    init_b, _flagged, _eff = TV.build_init_b_danger(
        scenario.init_miss, scenario.init_spread, scenario.perp, sign_mode="both")
    # B1 builds its OBS-model T/O/R (no solver) + its continuous-Gaussian operator prior.
    T, O, R, perp = B1.build_model(variant, scenario.perp)
    init_dt_mean, init_dt_std = BF.prior_from_init(
        scenario.init_miss, scenario.init_spread, scenario.perp)

    def _source():
        return B1.B1PolicySource(
            init_dt_mean, init_dt_std, perp, TV.N_OBS_AGENT, strat_names,
            str(bc.policy), str(bc.selfish_model),
            other_obs=str(bc.other_obs), tle_sigma_base=tle_sigma)

    results = RV.run_mc(T, O, R, perp, None, None, init_b, TV.N_OBS_AGENT,
                        int(bc.rollouts), seed=int(bc.seed), policy_source=_source())
    csv_path = os.path.join(out_dir, f"b1_{tag}.csv")
    B1.save_csv(results, csv_path)
    # label mirrors baseline_b1.main's strat_tag (strategy[X strategy] [-selfish_model] [-policy]
    # -other_obs) so the in-process row reads identically to the standalone CLI.
    strat_tag = strat_names[0] if strat_names[0] == strat_names[1] \
        else f"{strat_names[0]}X{strat_names[1]}"
    if "selfish" in strat_names:
        strat_tag += f"-{bc.selfish_model}"
    if "threshold" in strat_names:
        strat_tag += f"-{bc.policy}"
    strat_tag += f"-{bc.other_obs}"
    label = f"B1[{strat_tag}]"
    B1.summarize(results, label, "mc")
    return _brahe_band_summary(label, results)


def _log_wandb(cfg, scenario, result, tag, brahe_rows=None, flagged=None, eff_miss=None):
    """Log the resolved config + ALL figure data to wandb. The CSV remains the source of truth;
    this layer makes every paper figure reproducible PURELY from the wandb run.

    SUMMARY SCALARS — provenance + the sweep AXES (so runs are groupable/sortable in the UI):
      n_stages / n_contacts / init_miss / init_spread / perp / contact_stages / propagator /
      man_cost / disp_k / flagged / eff_miss.

    TABLES (each = one figure's data), kept on SEPARATE axes so nothing is apples-to-oranges:
      - variant_summary  : POMDP MATRIX-EXPECTATION (return/coll/dV/syncs). Solved variants only.
      - reward_parts     : per-variant maneuver/deviation/risk/displace -> reward-decomposition fig.
      - burn_timing      : per-variant per-stage mean_agent_burns -> burn-timing fig.
      - action_schedule  : per-variant per-stage dominant joint_action + prob -> policy fig.
      - brahe_summary    : BRAHE-MC band stats + maneuvers + fidelity for EVERY flown policy
                           (POMDP under run.rollout + B1) on ONE axis -> the direct comparison.
      - brahe_rollouts   : per-rollout brahe/matrix miss + dV + burns + term-reward + dt-err ->
                           the histogram / fuel-distribution / fidelity figures (re-plottable)."""
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
    # provenance + sweep AXES promoted to top-level summary fields (group/sort across a sweep)
    cs = result.get("contact_stages", [])
    wandb.summary.update({
        "n_stages": result.get("n_stages"),
        "n_contacts": len(cs),
        "contact_stages": ",".join(str(s) for s in cs),
        "init_miss": float(scenario.init_miss),
        "init_spread": float(scenario.init_spread),
        "perp": float(scenario.perp),
        "propagator": str(scenario.propagator),
        "man_cost": float(scenario.man_cost),
        "disp_k": ("linear" if scenario.disp_k is None else float(scenario.disp_k)),
    })
    if flagged is not None:
        wandb.summary["flagged"] = bool(flagged)
    if eff_miss is not None:
        wandb.summary["eff_miss"] = float(eff_miss)

    # --- POMDP figure tables: log the FULL row dicts CV.main() returns (NOT a subset). The
    #     existing plot fns (plot_summary / plot_burn_timing / plot_action_schedule /
    #     plot_reward_parts) are pure functions of these exact rows + replot_from_csv reuses them,
    #     so logging the whole row makes the wandb tables drive the SAME figures directly. Each
    #     row carries `variant` (display label, what the plot fns key on) AND `matrix_variant`.
    for key, rows in (("variant_summary", result.get("summary", [])),
                      ("reward_parts", result.get("reward_parts", [])),
                      ("burn_timing", result.get("burn", [])),
                      ("action_schedule", result.get("action", []))):
        wandb.log({key: _rows_to_table(wandb, rows)})
    # also surface the solver headline as scalar metrics (quick cross-run charts in the UI)
    for r in result.get("summary", []):
        wandb.log({
            f"{r['matrix_variant']}/expected_return": float(r["expected_return"]),
            f"{r['matrix_variant']}/collision_prob": float(r["collision_prob"]),
            f"{r['matrix_variant']}/dv_ms": float(r["expected_dv_ms"]),
            f"{r['matrix_variant']}/syncs": float(r["expected_syncs"]),
        })

    # --- brahe-MC tables: same-axis summary + the per-rollout distribution source ---
    if brahe_rows:
        bcols = ["policy", "brahe_collision_prob", "brahe_inband_pct", "brahe_below_pct",
                 "brahe_above_pct", "brahe_mean_miss_km", "brahe_min_miss_km",
                 "brahe_max_miss_km", "mean_dv_ms", "mean_burns", "leq2_burn_pct",
                 "term_reward_gap", "dt_absmean_err_km", "n_rollouts"]
        btable = wandb.Table(columns=bcols)
        rtable = wandb.Table(columns=["policy"] + _BRAHE_ROLLOUT_COLS)
        for br in brahe_rows:
            for k in ("brahe_collision_prob", "brahe_inband_pct", "brahe_mean_miss_km",
                      "mean_dv_ms", "mean_burns", "leq2_burn_pct", "term_reward_gap",
                      "dt_absmean_err_km"):
                wandb.log({f"{br['policy']}/{k}": float(br[k])})
            btable.add_data(*[br[c] for c in bcols])
            for pr in br.get("_per_rollout", []):
                rtable.add_data(br["policy"], *[pr.get(c) for c in _BRAHE_ROLLOUT_COLS])
        wandb.log({"brahe_summary": btable, "brahe_rollouts": rtable})
    run.finish()


if __name__ == "__main__":
    main()
