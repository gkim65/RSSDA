"""
scenario_config.py — the ONE config-first surface for the spacecraft-CA SDec-POMDP.

The whole pipeline used to be configured by IMPORT-TIME module globals fed by environment
variables (SPACECRAFT_PROPAGATOR, SPACECRAFT_HOUR_GRID_H, SPACECRAFT_MERGE_THRESHOLD_H,
SPACECRAFT_MAN_COST, SPACECRAFT_DISP_K, SPACECRAFT_CONJ_GRID) plus a subprocess-per-
conjunction model in sweep_driver. That is gone. The surface is now:

    cfg  ->  build_scenario(cfg)  ->  solve(scenario)

`cfg` is a plain mapping (an OmegaConf/dict from Hydra/wandb, or hand-built). `Scenario`
is the frozen, fully-resolved parameter bundle. `build_scenario(scenario)` is the KEYSTONE:
it populates the existing module globals in spacecraft_stage_grid (propagator backend, hour
grid, merge threshold, the per-conjunction orbit pair, contact-stage subset) and then forces
the orbit-dependent stage grid to recompute — the same in-place-global idiom already used by
set_propagator_backend / set_contact_stages / configure_dt_grid. Downstream modules
(discretizer_v2, transition_v2, matrices) keep reading the globals, so all existing references
stay valid while the config becomes the single source of truth.

CRITICAL — import order: build_scenario() must run BEFORE any model module
(spacecraft_discretizer_v2 / spacecraft_transition_v2 / spacecraft_matrices) is imported,
because some of those capture grid values at import (e.g. `from spacecraft_stage_grid import
N_STAGES` is a value-binding). main.py imports nothing heavy until after build_scenario.
build_scenario therefore imports ONLY spacecraft_stage_grid (which imports no model module).

NO environment variables. NO subprocess. NO import-time freeze that the config can't reach.
This is NOT the rejected file>env>default loader — config is the default and only surface.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Sequence
import numpy as np


# ---------------------------------------------------------------------------
# Scenario — the fully-resolved parameter bundle. Every knob that used to be an
# env var / module constant the sweeps turned lives here. Maps 1:1 to conf/ paths
# (see notes/SCENARIO_KNOBS.md). Pure data; no brahe, no model imports.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Scenario:
    # --- grid / timeline (was spacecraft_stage_grid module constants + env) ---
    propagator: str = "numerical"                 # numerical | keplerian | drag
    hour_grid_h: Optional[List[float]] = None     # None => stage_grid default ~2h cadence
    merge_threshold_h: float = 0.25               # contact within this MERGES onto a stage
    sync_rule: str = "later"                       # later (staggered) | simultaneous

    # --- conjunction geometry (was SPACECRAFT_CONJ_GRID subprocess hook) ------
    # Either give an explicit orbit pair (sc1_oe / sc2_oe, KOE deg) OR leave both None
    # to use the stage_grid default reference orbit (SC1_OE_AT_TCA + rtn-placed SC2).
    sc1_oe: Optional[List[float]] = None
    sc2_oe: Optional[List[float]] = None

    # --- initial belief (compare_variants_v2 --init-* / --perp) ---------------
    init_miss: float = 0.0                         # danger center (km)
    init_spread: float = 1.4                       # uncertainty half-width (km)
    perp: float = 0.0                              # sideways standoff (km)

    # --- contact / sync subset (the SDec ablation lever) ---------------------
    contact_stages: Optional[List[int]] = None     # None => full available set; [] => none

    # --- reward knobs (were SPACECRAFT_MAN_COST / SPACECRAFT_DISP_K) ----------
    man_cost: float = -2.0                         # per agent-burn
    disp_k: Optional[float] = 0.2                  # convex displacement curvature; None => legacy linear

    # --- observation fidelity (the obs-quality experiment lever; SDec-ONLY) ----
    # perfect (default delta; anchor unchanged) | gps (sub-km both) | tle (TLE-vs-TLE worst
    # case) | asymmetric (GPS-self/TLE-other, the headline). obs_sigma = raw km override for a
    # smooth sync-value curve (beats the named level when set). Affects ONLY the SDec sync obs;
    # Centralized/Dec are fixed rails. Sigmas grounded in belief_filter (see transition_v2).
    obs_fidelity: str = "perfect"
    obs_sigma: Optional[float] = None              # raw km; None => use the named level
    obs_coarse: bool = False                       # coarse operational obs alphabet (km-scale
                                                   # syncs -> ~5 signed symbols); OFF => fine bins
                                                   # => anchor byte-identical. User-opt-in for the
                                                   # sigma (TLE) experiments (makes them tractable).

    # --- solve knobs ----------------------------------------------------------
    variants: List[str] = field(default_factory=lambda: ["centralized", "sdec", "dec"])
    iter_limit: int = 10000                        # Dec RS-MAA* budget
    sdec_tail_qmdp: bool = False                   # SDec/Cen RS-SDA*: QMDP tail approx + rec_limit=1
                                                   # (graded-obs speedup). OFF => anchor byte-identical.
    sdec_iter_limit: int = 2000                    # SDec/Cen RS-SDA* TI2 pruning budget (default 2000;
                                                   # lower => harder prune for graded-obs solves).
    sdec_ti1: bool = False                         # SDec/Cen RS-SDA* TI1 interleaving (prefix policy +
                                                   # MPC re-solve at syncs). OFF => one-shot full policy
                                                   # (anchor byte-identical). ON => interleaved evaluator.
    sdec_max_clusters: int = 20                    # SDec/Cen RS-SDA* TI4 belief-cluster cap (anchors per
                                                   # sync). 20 == anchor. LOWER (2/3/4) caps the post-sync
                                                   # belief fan-out for graded (TLE) obs -> tractable, but
                                                   # too low collapses the noise (degenerate ~= perfect).

    def to_dict(self) -> dict:
        """Plain dict (for wandb config logging / round-trip)."""
        return asdict(self)


# ---------------------------------------------------------------------------
# cfg -> Scenario  (pure; no global mutation, no brahe)
# ---------------------------------------------------------------------------
def _as_list(v):
    if v is None:
        return None
    return [float(x) for x in v]


def _as_int_list(v):
    if v is None:
        return None
    return [int(x) for x in v]


def scenario_from_cfg(cfg) -> Scenario:
    """Build a Scenario from a plain mapping (OmegaConf dict / dict). Pure: reads cfg,
    returns a frozen Scenario. Does NOT touch any module global — that's build_scenario's job.

    Accepts a Hydra-style cfg laid out as:
        cfg.grid.{propagator,hour_grid_h,merge_threshold_h,sync_rule}
        cfg.conjunction.{sc1_oe,sc2_oe}
        cfg.belief.{init_miss,init_spread,perp}
        cfg.contacts.stages
        cfg.reward.{man_cost,disp_k}
        cfg.obs.{fidelity,sigma}
        cfg.solve.{variants,iter_limit}
    Missing groups/fields fall back to the Scenario defaults."""
    # tolerate either OmegaConf or a plain dict; .get works on dict, getattr on OmegaConf
    def g(node, key, default=None):
        if node is None:
            return default
        if hasattr(node, "get"):
            try:
                val = node.get(key, default)
            except Exception:
                val = getattr(node, key, default)
        else:
            val = getattr(node, key, default)
        return default if val is None else val

    def sub(name):
        if cfg is None:
            return None
        if hasattr(cfg, "get"):
            try:
                return cfg.get(name, None)
            except Exception:
                return getattr(cfg, name, None)
        return getattr(cfg, name, None)

    grid = sub("grid")
    conj = sub("conjunction")
    belief = sub("belief")
    contacts = sub("contacts")
    reward = sub("reward")
    obs = sub("obs")
    solve = sub("solve")

    d = Scenario()  # defaults
    disp_k_raw = g(reward, "disp_k", d.disp_k)
    if isinstance(disp_k_raw, str) and disp_k_raw.lower() in ("none", "linear", "null"):
        disp_k_val = None
    else:
        disp_k_val = None if disp_k_raw is None else float(disp_k_raw)

    variants = g(solve, "variants", d.variants)
    variants = list(variants) if variants is not None else d.variants

    obs_sigma_raw = g(obs, "sigma", d.obs_sigma)
    if isinstance(obs_sigma_raw, str) and obs_sigma_raw.lower() in ("none", "null", ""):
        obs_sigma_val = None
    else:
        obs_sigma_val = None if obs_sigma_raw is None else float(obs_sigma_raw)

    return Scenario(
        propagator=str(g(grid, "propagator", d.propagator)).lower(),
        hour_grid_h=_as_list(g(grid, "hour_grid_h", d.hour_grid_h)),
        merge_threshold_h=float(g(grid, "merge_threshold_h", d.merge_threshold_h)),
        sync_rule=str(g(grid, "sync_rule", d.sync_rule)),
        sc1_oe=_as_list(g(conj, "sc1_oe", d.sc1_oe)),
        sc2_oe=_as_list(g(conj, "sc2_oe", d.sc2_oe)),
        init_miss=float(g(belief, "init_miss", d.init_miss)),
        init_spread=float(g(belief, "init_spread", d.init_spread)),
        perp=float(g(belief, "perp", d.perp)),
        contact_stages=_as_int_list(g(contacts, "stages", d.contact_stages)),
        man_cost=float(g(reward, "man_cost", d.man_cost)),
        disp_k=disp_k_val,
        obs_fidelity=str(g(obs, "fidelity", d.obs_fidelity)).lower(),
        obs_sigma=obs_sigma_val,
        obs_coarse=bool(g(obs, "coarse", d.obs_coarse)),
        variants=[str(v) for v in variants],
        iter_limit=int(g(solve, "iter_limit", d.iter_limit)),
        sdec_tail_qmdp=bool(g(solve, "sdec_tail_qmdp", d.sdec_tail_qmdp)),
        sdec_iter_limit=int(g(solve, "sdec_iter_limit", d.sdec_iter_limit)),
        sdec_ti1=bool(g(solve, "sdec_ti1", d.sdec_ti1)),
        sdec_max_clusters=int(g(solve, "sdec_max_clusters", d.sdec_max_clusters)),
    )


# ---------------------------------------------------------------------------
# build_scenario — the KEYSTONE. Populate the module globals from the Scenario,
# then force the orbit-dependent stage grid + reward globals to recompute.
# MUST be called before any model module is imported (import-order discipline).
# ---------------------------------------------------------------------------
def build_scenario(scenario) -> Scenario:
    """Apply a Scenario (or a cfg mapping) to the pipeline's module globals and recompute
    the orbit-dependent stage grid. Returns the resolved Scenario.

    Reuses the established in-place-global idiom (set_propagator_backend / build_stage_grid /
    set_contact_stages) so the 100+ existing global references stay valid. This is the single
    handoff point from config to the model: call it ONCE at the top of a run, before importing
    spacecraft_discretizer_v2 / spacecraft_transition_v2 / spacecraft_matrices."""
    if not isinstance(scenario, Scenario):
        scenario = scenario_from_cfg(scenario)

    # Import ONLY the stage grid here (it imports no model module, so populating its globals
    # first cannot trigger a premature value-binding capture downstream).
    import spacecraft_stage_grid as SG

    # 1. propagator backend — governs ALL propagation (grid contacts + matrices).
    SG.set_propagator_backend(scenario.propagator)

    # 2. hour grid + merge threshold (the timeline/cadence knobs).
    SG.set_hour_grid(scenario.hour_grid_h)            # None => keep module default
    SG.set_merge_threshold(scenario.merge_threshold_h)

    # 3. recompute the stage grid for this conjunction's orbit pair (replaces the
    #    SPACECRAFT_CONJ_GRID subprocess hook). None/None => default reference orbit.
    sc1 = np.asarray(scenario.sc1_oe, float) if scenario.sc1_oe is not None else None
    sc2 = np.asarray(scenario.sc2_oe, float) if scenario.sc2_oe is not None else None
    SG.rebuild_grid(sc1_oe=sc1, sc2_oe=sc2, sync_rule=scenario.sync_rule)

    # 4. contact-stage subset (the SDec ablation). None => full available set is kept;
    #    otherwise restrict the live CONTACT_STAGES list in place.
    if scenario.contact_stages is not None:
        SG.set_contact_stages(scenario.contact_stages)

    # NOTE: reward globals (man_cost, disp_k) live in spacecraft_transition_v2, which is NOT
    # imported here (it would import the model). They are applied by build_reward(), which the
    # entry point calls AFTER importing transition_v2 (see entry points / main.py).
    return scenario


def load_yaml_cfg(path):
    """Load a YAML config file into a plain dict (Hydra/wandb convention). Used by the
    --scenario-config CLI handoff (the ONLY parent->child config path; NOT an env var)."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


# Set once a scenario has been applied (by main.py's Hydra path OR a CLI bootstrap) so a
# subsequent entry-point import doesn't re-apply/clobber it. main.py calls apply_scenario()
# directly (it can pass orbit-array conjunctions that don't fit CLI flags); the entry points'
# _cli_bootstrap_scenario then sees this flag and becomes a no-op.
_APPLIED_SCENARIO: Optional[Scenario] = None


def apply_scenario(scenario) -> Scenario:
    """Apply a Scenario fully (grid + reward) and mark it applied. The single programmatic
    handoff main.py uses: build_scenario(grid) then build_reward(reward). Idempotent-safe for
    the entry points, which check _APPLIED_SCENARIO before doing their own CLI bootstrap."""
    global _APPLIED_SCENARIO
    if not isinstance(scenario, Scenario):
        scenario = scenario_from_cfg(scenario)
    build_scenario(scenario)
    build_reward(scenario)
    _APPLIED_SCENARIO = scenario
    return scenario


def _cli_bootstrap_scenario(argv):
    """Pre-scan a CLI argv into a Scenario and apply it (build_scenario + build_reward) BEFORE
    the caller imports the model modules. This is the standalone-CLI path for the v2 entry
    points (compare_variants_v2 / rollout_v2 / baseline_b1): it keeps the existing flags
    (--backend / --init-miss / --init-spread / --perp / --contact-stages / --man-cost /
    --disp-k) working while routing them through the ONE config surface. A --scenario-config
    <path> YAML supplies the base; explicit flags override it. NO env vars, NO subprocess.

    Returns the resolved Scenario. The caller still parses argv fully in its own argparse for
    the run-specific knobs (tag/out-dir/figures/rsmaa-*); this only handles the SCENARIO knobs
    that must be applied pre-import."""
    # If main.py (the Hydra runner) already applied a Scenario programmatically, do NOT re-apply
    # from argv — that would clobber an orbit-array conjunction the CLI flags can't express.
    if _APPLIED_SCENARIO is not None:
        return _APPLIED_SCENARIO
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--scenario-config", default=None)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--init-miss", type=float, default=None)
    ap.add_argument("--init-spread", type=float, default=None)
    ap.add_argument("--perp", type=float, default=None)
    ap.add_argument("--contact-stages", default="__unset__")
    ap.add_argument("--man-cost", type=float, default=None)
    ap.add_argument("--disp-k", default=None)
    ap.add_argument("--obs-fidelity", default=None)
    ap.add_argument("--obs-sigma", default=None)
    ap.add_argument("--obs-coarse", dest="obs_coarse", action="store_true", default=None)
    ap.add_argument("--hour-grid", default=None)
    ap.add_argument("--merge-threshold", type=float, default=None)
    ap.add_argument("--sdec-tail-qmdp", dest="sdec_tail_qmdp", action="store_true", default=None)
    ap.add_argument("--sdec-iter-limit", type=int, default=None)
    ap.add_argument("--sdec-ti1", dest="sdec_ti1", action="store_true", default=None)
    ap.add_argument("--sdec-max-clusters", type=int, default=None)
    ns, _ = ap.parse_known_args(argv[1:])

    cfg = load_yaml_cfg(ns.scenario_config) if ns.scenario_config else {}
    base = scenario_from_cfg(cfg)

    # explicit flags override the config file / defaults
    overrides = {}
    if ns.backend is not None:
        overrides["propagator"] = ns.backend.lower()
    if ns.hour_grid is not None:
        overrides["hour_grid_h"] = [float(x) for x in ns.hour_grid.split(",") if x.strip()]
    if ns.merge_threshold is not None:
        overrides["merge_threshold_h"] = ns.merge_threshold
    if ns.init_miss is not None:
        overrides["init_miss"] = ns.init_miss
    if ns.init_spread is not None:
        overrides["init_spread"] = ns.init_spread
    if ns.perp is not None:
        overrides["perp"] = ns.perp
    if ns.contact_stages != "__unset__":
        overrides["contact_stages"] = (
            [int(s) for s in ns.contact_stages.split(",") if s.strip() != ""]
            if ns.contact_stages else [])
    if ns.man_cost is not None:
        overrides["man_cost"] = ns.man_cost
    if ns.disp_k is not None:
        overrides["disp_k"] = (None if str(ns.disp_k).lower() in ("none", "linear")
                               else float(ns.disp_k))
    if ns.obs_fidelity is not None:
        overrides["obs_fidelity"] = ns.obs_fidelity.lower()
    if ns.obs_sigma is not None:
        overrides["obs_sigma"] = (None if str(ns.obs_sigma).lower() in ("none", "null")
                                  else float(ns.obs_sigma))
    if ns.obs_coarse is not None:
        overrides["obs_coarse"] = True
    if ns.sdec_tail_qmdp is not None:
        overrides["sdec_tail_qmdp"] = True
    if ns.sdec_iter_limit is not None:
        overrides["sdec_iter_limit"] = ns.sdec_iter_limit
    if ns.sdec_ti1 is not None:
        overrides["sdec_ti1"] = True
    if ns.sdec_max_clusters is not None:
        overrides["sdec_max_clusters"] = ns.sdec_max_clusters

    from dataclasses import replace
    scenario = replace(base, **overrides)
    return apply_scenario(scenario)   # grid + reward, marks _APPLIED_SCENARIO


def build_reward(scenario) -> None:
    """Apply the reward + obs-fidelity knobs to spacecraft_transition_v2. Separate from
    build_scenario because transition_v2 is a model module (importing it triggers the grid
    value-bindings), so it must be imported only AFTER build_scenario has populated the grid.
    Entry points call this right after `import spacecraft_transition_v2 as TV`. (Obs fidelity
    rides here -- not in build_scenario -- for the same post-import reason; it is a transition_v2
    setter exactly like set_reward.)"""
    if not isinstance(scenario, Scenario):
        scenario = scenario_from_cfg(scenario)
    import spacecraft_transition_v2 as TV
    TV.set_reward(man_cost=scenario.man_cost, disp_k=scenario.disp_k)
    TV.set_obs(fidelity=scenario.obs_fidelity, sigma_km=scenario.obs_sigma,
               coarse=scenario.obs_coarse)
