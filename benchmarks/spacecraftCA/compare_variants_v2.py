"""
compare_variants_v2.py

Focused v1-vs-v2 comparison using the SAME expected-eval machinery that produced
v1's -22.41 (compare_variants.expected_policy_metrics), but fed v2's reduced-state
matrices. Solves centralized/sdec with RS-SDA* and dec with RS-MAA* (as the notes
require: Dec uses RS-MAA*, not RS-SDA*).

Why a separate focused script (not a full copy of compare_variants): the expected
RETURN is computed straight from the R matrix (mass * R[a,s]) inside
expected_policy_metrics, and v2's R already bakes in the quadrature terminal reward
-> the headline value is correct with zero state re-decode. Only the DIAGNOSTIC
breakdowns (terminal_bin_probs / expected_miss_km, which assume miss_bin == miss) are
v1-specific; we report the value (the matched metric) and flag the diagnostics as
v1-semantic for now.

Usage:
  python compare_variants_v2.py                       # default head-on (init_miss=0)
  python compare_variants_v2.py --init-miss 0.3 --perp 3   # perp-aware dangerous conj
  python compare_variants_v2.py --contact-stages 13,15     # Scenario-1 contact subset
"""
import os, sys, argparse, math, io, contextlib
from array import array
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCHMARKS = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_BENCHMARKS)
sys.path.insert(0, _ROOT); sys.path.insert(0, _BENCHMARKS); sys.path.insert(0, _HERE)

# The scenario MUST be applied BEFORE the model modules import, because the stage grid
# (and the discretizers' N_STAGES / N_STATES) are derived from it and some downstream imports
# capture grid values by value (import-order discipline). _bootstrap_scenario() pre-scans
# argv (and --scenario-config) into a Scenario, calls build_scenario, and returns it; the
# model modules below then import against the already-populated grid globals. NO env vars.
from scenario_config import (
    Scenario, build_scenario, build_reward, scenario_from_cfg, _cli_bootstrap_scenario,
)
_SCENARIO = _cli_bootstrap_scenario(sys.argv)

from brahe import initialize_eop
from RSSDA import SDecPOMDP, SDecPOMDPModel, int_tuple
from baselines.decPOMDP import DecPOMDP as RSMAA

import spacecraft_transition_v2 as TV
import spacecraft_discretizer_v2 as D
import spacecraft_matrices as M
from sdec_spacecraft import build_config
from spacecraft_matrices import DV_MAGNITUDE

# --- RS-MAA* defaults (v1-proven, see compare_variants.solve_dec_rsmaa) -------
# Every one is exposed as a CLI flag in main() but defaults to the value below,
# so the Dec variant solves with no tuning required. Override only if Dec misbehaves.
RSMAA_DEFAULTS = dict(
    cluster_type="lossless", maxit=200, q_depth=3, alpha=0.2,
    heuristic="MDP", rec_type="MDP", maxrec=2, memory=None,
    iter_limit=10000, p_threshold_cluster=0.0, p_threshold_expand=0.0,
    memory_limit_gb=None, memory_check_interval=None, verbose=False,
)


def build_config_fixed():
    # fixed-mode (TI1=False) to match the v1 -22.41 setup. Graded-obs speed knobs ride on the
    # applied Scenario (solve.sdec_tail_qmdp / solve.sdec_iter_limit); both default to the anchor
    # values so an unset config is byte-identical to the -7.83 reference.
    cfg = build_config(exact=False, ti1_enabled=False)
    if getattr(_SCENARIO, "sdec_tail_qmdp", False):
        cfg.tail_heuristic_type = "QMDP"
        cfg.rec_limit = 1
    _il = getattr(_SCENARIO, "sdec_iter_limit", None)
    if _il is not None:
        cfg.iter_limit = int(_il)
    return cfg


def v2_model(T, O, R, init_b, sync_stages):
    return SDecPOMDPModel(
        nagents=2, nstates=D.N_STATES_TOTAL, nactions=TV.N_JOINT_ACTIONS,
        nobs=TV.N_JOINT_OBS,
        transitions=T.flatten().tolist(), obs=O.flatten().tolist(),
        rewards=R.flatten().tolist(), init_beliefs=init_b.tolist(),
        nacts_factor=[TV.N_ACT_AGENT, TV.N_ACT_AGENT],
        nobs_factor=[TV.N_OBS_AGENT, TV.N_OBS_AGENT],
        sync_states=D.sync_trigger_states(sync_stages),
        sync_actions=[], sync_observations=[],
    )


def dense_rows_to_pdict(mat3, tol=0.0):
    """Convert (action, row, col) dense arrays to RS-MAA* sparse row tuples.
    Faithful copy of compare_variants.dense_rows_to_pdict."""
    rows = []
    for act in range(mat3.shape[0]):
        for row in range(mat3.shape[1]):
            vals = mat3[act, row, :]
            idx = np.flatnonzero(vals > tol)
            rows.append((
                array("i", [int(x) for x in idx]),
                array("d", [float(vals[x]) for x in idx]),
            ))
    return rows


def solve_dec_rsmaa_v2(T, O, R, init_b, rsmaa_cfg):
    """Solve the v2 reduced-state model fully-decentralized with baseline RS-MAA*.

    Mirrors compare_variants.solve_dec_rsmaa, but on v2 dims: N_STATES_TOTAL states,
    N_JOINT_ACTIONS actions, and the v2 RICHER local-obs factor N_OBS_AGENT (NOT v1's
    N_DEV). Returns (value, policy, clustering) — RS-MAA*'s native policy structure,
    consumed by expected_rsmaa_return_v2 below.
    """
    T_pdict = dense_rows_to_pdict(T)
    O_pdict = dense_rows_to_pdict(O)
    solver = RSMAA(
        nagents=2,
        nstates=D.N_STATES_TOTAL,
        nactions=TV.N_JOINT_ACTIONS,
        nobs=O.shape[2],
        transitions=T_pdict,
        obs=O_pdict,
        rewards=R.reshape(-1).tolist(),
        init_beliefs=init_b.tolist(),
        nacts_factor=[TV.N_ACT_AGENT, TV.N_ACT_AGENT],
        nobs_factor=[TV.N_OBS_AGENT, TV.N_OBS_AGENT],   # v2 richer obs (dt_obs * vdev)
        maxh=D.N_STAGES,
        cluster_type=rsmaa_cfg["cluster_type"],
        maxit=rsmaa_cfg["maxit"],
        q_depth=rsmaa_cfg["q_depth"],
        alpha=rsmaa_cfg["alpha"],
        iter_limit=rsmaa_cfg["iter_limit"],
        maxrec=rsmaa_cfg["maxrec"],
        memory=rsmaa_cfg["memory"],
        heuristic=rsmaa_cfg["heuristic"],
        rec_type=rsmaa_cfg["rec_type"],
        p_threshold_cluster=rsmaa_cfg["p_threshold_cluster"],
        p_threshold_expand=rsmaa_cfg["p_threshold_expand"],
        policyvalfound=-math.inf,
        output=rsmaa_cfg["verbose"],
        memory_limit_gb=rsmaa_cfg["memory_limit_gb"],
        memory_check_interval=rsmaa_cfg["memory_check_interval"],
    )
    solver.decentralized = True
    solver.onesided = False
    if rsmaa_cfg["verbose"]:
        value, policy, clustering = solver.multi_agent_astar(D.N_STAGES)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            value, policy, clustering = solver.multi_agent_astar(D.N_STAGES)
    return float(value), policy, clustering


def expected_rsmaa_return_v2(T, O, R, full_result, init_b, prune=1e-12):
    """
    Exact belief-walk eval of a fixed RS-MAA* Dec policy on the v2 model. Mirrors
    compare_variants.expected_rsmaa_policy_metrics (policy[stage][agent][cluster] +
    clustering[step][agent][cluster][local_obs]), but emits v2's reward decomposition
    (maneuver/deviation/risk/displace) and v2 QUADRATURE collision (miss_km_from_dt),
    so Dec produces byte-identical CSV/figure rows to Cen/SDec.

    Returns the same 6-tuple shape as expected_return_from_policy:
      (expected_return, coll_prob, stage_any_burns, comp, term_dt, act_mass)
    """
    from collections import defaultdict

    value, policy, clustering = full_result
    obs_agent_size = TV.N_OBS_AGENT

    expected_return = 0.0
    coll_prob = 0.0
    stage_any_burns = np.zeros(D.N_STAGES)
    comp = defaultdict(float)
    term_dt = defaultdict(float)
    act_mass = defaultdict(float)

    # Walk from the init_b support (Dec has no belief index; cluster ptrs start at 0).
    support = [(int(s), init_b[s]) for s in np.flatnonzero(init_b)]
    nodes = defaultdict(float)
    for s, p in support:
        nodes[(s, 0, 0)] += p

    for step in range(D.N_STAGES):
        next_nodes = defaultdict(float)
        for (true_state, c0, c1), mass in nodes.items():
            if mass <= prune or true_state == D.SINK_STATE:
                continue
            dt_bin, v1b, v2b, stage = D.index_to_state(true_state)
            if step >= len(policy):
                a1, a2 = 0, 0
            else:
                try:
                    a1 = int(policy[step][0][c0]); a2 = int(policy[step][1][c1])
                except (IndexError, TypeError):
                    a1, a2 = 0, 0
            a1, a2 = max(a1, 0), max(a2, 0)
            joint_act = a1 + TV.N_ACT_AGENT * a2
            b1, b2 = a1, a2

            expected_return += mass * float(R[joint_act, true_state])
            act_mass[(stage, int(joint_act))] += mass
            comp["maneuver"] += mass * TV.REWARD_MANEUVER * ((b1 != 0) + (b2 != 0))
            comp["deviation"] += mass * TV.REWARD_DEVIATION * (
                (v1b != D.VDEV_ZERO) + (v2b != D.VDEV_ZERO))
            if (b1 != 0 or b2 != 0):
                stage_any_burns[stage] += mass
            if stage == D.N_STAGES - 1:
                dt_c = D.dt_bin_center_km(dt_bin)
                miss = D.miss_km_from_dt(dt_c, PERP_KM)
                comp["risk"] += mass * TV.risk_ramp_reward(miss)
                comp["displace"] += mass * TV.displacement_cost(dt_c)
                term_dt[dt_bin] += mass
                if miss < D.COLLISION_THRESHOLD_KM:
                    coll_prob += mass

            nzT = np.flatnonzero(T[joint_act, true_state, :] > prune)
            for sp in nzT:
                bm = mass * float(T[joint_act, true_state, sp])
                if bm <= prune:
                    continue
                if sp == D.SINK_STATE:
                    next_nodes[(int(sp), c0, c1)] += bm; continue
                nzO = np.flatnonzero(O[joint_act, sp, :] > prune)
                for obs in nzO:
                    om = bm * float(O[joint_act, sp, obs])
                    if om <= prune:
                        continue
                    o1 = int(obs) % obs_agent_size; o2 = int(obs) // obs_agent_size
                    if step < len(clustering):
                        try:
                            nc0 = int(clustering[step][0][c0][o1])
                            nc1 = int(clustering[step][1][c1][o2])
                        except (IndexError, TypeError):
                            nc0, nc1 = 0, 0
                        nc0 = max(nc0, 0); nc1 = max(nc1, 0)
                    else:
                        nc0, nc1 = c0, c1
                    next_nodes[(int(sp), nc0, nc1)] += om
        nodes = {n: p for n, p in next_nodes.items()
                 if p > prune and n[0] != D.SINK_STATE}
    return expected_return, coll_prob, stage_any_burns, comp, term_dt, act_mass


def expected_return_from_policy(T, O, R, sdec, full_result, init_b, obs_agent_size,
                                prune=1e-12):
    """
    Belief-tree walk copied faithfully from compare_variants.expected_policy_metrics,
    reduced to what we need: expected_return (= matched -22.41 metric, straight from R),
    plus a v2-correct collision probability using the QUADRATURE miss of the terminal
    dt bin. State-agnostic RSSDA helpers (get_rssda_action/update_oh_rssda/get_terminal)
    are reused from spacecraft_simulator.
    """
    from spacecraft_simulator import get_rssda_action, update_oh_rssda
    from collections import defaultdict

    value, policy, clustering, cent_vector, cen_dists_map, clustering_cen = full_result
    root_belief = sdec.dist_dict[int_tuple(sdec.init_beliefs)]
    # root state: the highest-prob state in init_b (point) OR walk from each support state.
    # To match expected_policy_metrics (which conditions on ONE true init bin), we sum
    # over the init_b support weighted by its probability.
    support = [(s, init_b[s]) for s in np.flatnonzero(init_b)]

    expected_return = 0.0
    coll_prob = 0.0
    stage_any_burns = np.zeros(D.N_STAGES)
    # reward decomposition (mirrors v1 compare_variants.reward_components, v2 rewards)
    comp = defaultdict(float)              # 'maneuver'/'deviation'/'risk'/'displace'
    term_dt = defaultdict(float)           # terminal dt-bin -> mass
    act_mass = defaultdict(float)          # (stage, joint_action) -> mass (for action schedule)

    for root_state, root_p in support:
        nodes = {(int(root_state), root_belief, 0, 0): root_p}
        for step in range(D.N_STAGES):
            next_nodes = defaultdict(float)
            for (true_state, belief_idx, oh0, oh1), mass in nodes.items():
                if mass <= prune or true_state == D.SINK_STATE:
                    continue
                dt_bin, v1b, v2b, stage = D.index_to_state(true_state)
                joint_act, a1, a2, is_cen = get_rssda_action(
                    policy, cen_dists_map, clustering, clustering_cen,
                    step, belief_idx, [oh0, oh1], TV.N_ACT_AGENT)
                if joint_act < 0:
                    joint_act, a1, a2 = 0, 0, 0
                b1, b2 = joint_act % TV.N_ACT_AGENT, joint_act // TV.N_ACT_AGENT
                expected_return += mass * float(R[joint_act, true_state])
                act_mass[(stage, int(joint_act))] += mass
                # --- reward decomposition (v2 reward terms) ---
                comp["maneuver"] += mass * TV.REWARD_MANEUVER * ((b1 != 0) + (b2 != 0))
                comp["deviation"] += mass * TV.REWARD_DEVIATION * (
                    (v1b != D.VDEV_ZERO) + (v2b != D.VDEV_ZERO))
                if (b1 != 0 or b2 != 0):
                    stage_any_burns[stage] += mass
                if stage == D.N_STAGES - 1:
                    dt_c = D.dt_bin_center_km(dt_bin)
                    miss = D.miss_km_from_dt(dt_c, PERP_KM)   # quadrature miss
                    comp["risk"] += mass * TV.risk_ramp_reward(miss)
                    comp["displace"] += mass * TV.displacement_cost(dt_c)
                    term_dt[dt_bin] += mass
                    if miss < D.COLLISION_THRESHOLD_KM:
                        coll_prob += mass

                c_ptr = -1
                if is_cen and step < len(cen_dists_map):
                    dm = cen_dists_map[step]
                    if belief_idx in dm:
                        c_ptr = dm.index(belief_idx)
                nzT = np.flatnonzero(T[joint_act, true_state, :] > prune)
                try:
                    obs_to_belief = {int(o): int(d) for o, _, d in
                                     sdec.get_terminal(belief_idx, joint_act)}
                except KeyError:
                    obs_to_belief = {}
                for sp in nzT:
                    bm = mass * float(T[joint_act, true_state, sp])
                    if bm <= prune: continue
                    if sp == D.SINK_STATE:
                        next_nodes[(int(sp), belief_idx, oh0, oh1)] += bm; continue
                    nzO = np.flatnonzero(O[joint_act, sp, :] > prune)
                    for obs in nzO:
                        om = bm * float(O[joint_act, sp, obs])
                        if om <= prune: continue
                        nb = obs_to_belief.get(int(obs), belief_idx)
                        o1 = int(obs) % obs_agent_size; o2 = int(obs) // obs_agent_size
                        n0, n1 = update_oh_rssda(policy, cen_dists_map, clustering,
                                                 clustering_cen, step, nb, [oh0, oh1],
                                                 is_cen, c_ptr, o1, o2)
                        next_nodes[(int(sp), nb, n0, n1)] += om
            nodes = {n: p for n, p in next_nodes.items()
                     if p > prune and n[0] != D.SINK_STATE}
    return expected_return, coll_prob, stage_any_burns, comp, term_dt, act_mass


PERP_KM = 0.0


def build_matrices(variant):
    """Build the (T, O, R, perp) for a variant. These depend ONLY on the scenario
    (orbit pair / grid / contacts / reward / perp) — NOT on the initial belief — so a
    sweep over beliefs of the SAME conjunction can build these ONCE and reuse them
    across every belief (the matrix-reuse seam consumed by _conj_worker.py). O DOES
    depend on the active contact-stage subset (read inside build_T_O via
    M.get_contact_stages), so rebuild per (variant, contact-subset), not per belief."""
    rate_at, perp, _ = TV.compute_gain_table_and_perp(PERP_KM, 0.0)
    T, O = TV.build_T_O(rate_at, variant)
    R = TV.build_R(perp)
    return T, O, R, perp


def solve_policy(variant, init_b, matrices=None):
    """Solve an RS-SDA* variant (centralized/sdec) and return the SOLVED objects so a
    consumer (e.g. rollout_v2.py) can reuse the CANONICAL policy + matrices without
    rebuilding them. Same build path as solve_and_eval -> identical to the headline
    numbers. Returns (T, O, R, perp, sdec, full). Not for dec (RS-MAA*, different API).

    matrices: optional pre-built (T, O, R, perp) from build_matrices(variant) — pass it
    to REUSE matrices across beliefs of the same conjunction (skips the ~22s rebuild).
    None => build them here (the standalone path; unchanged behaviour)."""
    T, O, R, perp = matrices if matrices is not None else build_matrices(variant)
    cs = {"centralized": list(range(D.N_STAGES)),
          "sdec": M.get_contact_stages()}[variant]
    model = v2_model(T, O, R, init_b, cs)
    sdec = SDecPOMDP(model=model, config=build_config_fixed())
    full = sdec.multi_agent_astar(D.N_STAGES)
    return T, O, R, perp, sdec, full


def solve_dec_policy(init_b, rsmaa_cfg=None, matrices=None):
    """Solve the Dec variant (RS-MAA*) and return the SOLVED objects so a consumer (rollout_v2's
    DecPolicySource) can fly the canonical Dec policy through brahe — the Dec analogue of
    solve_policy. Returns (T, O, R, perp, dec_full) where dec_full = (value, policy, clustering)
    from solve_dec_rsmaa_v2. matrices: optional pre-built (T,O,R,perp) to REUSE across beliefs."""
    T, O, R, perp = matrices if matrices is not None else build_matrices("dec")
    dec_full = solve_dec_rsmaa_v2(T, O, R, init_b, rsmaa_cfg or RSMAA_DEFAULTS)
    return T, O, R, perp, dec_full


def solve_and_eval(variant, init_b, rsmaa=False, rsmaa_cfg=None, matrices=None):
    """Solve + expected-eval one variant. matrices: optional pre-built (T,O,R,perp) from
    build_matrices(variant) to REUSE across beliefs of the same conjunction (skips the
    ~22s rebuild); None => build here (standalone path, unchanged)."""
    T, O, R, perp = matrices if matrices is not None else build_matrices(variant)
    obs_agent_size = TV.N_OBS_AGENT
    if rsmaa:
        # Dec via RS-MAA* (NOT RS-SDA* — per Mahdi's notes the Dec variant is solved
        # by the purpose-built RS-MAA* baseline). Native policy structure, evaluated
        # by the RS-MAA*-specific belief walk.
        full = solve_dec_rsmaa_v2(T, O, R, init_b, rsmaa_cfg or RSMAA_DEFAULTS)
        er, cp, burns, comp, term_dt, act_mass = expected_rsmaa_return_v2(
            T, O, R, full, init_b)
        return er, cp, burns, comp, term_dt, act_mass, None
    cs = {"centralized": list(range(D.N_STAGES)),
          "sdec": M.get_contact_stages()}[variant]
    model = v2_model(T, O, R, init_b, cs)
    sdec = SDecPOMDP(model=model, config=build_config_fixed())
    full = sdec.multi_agent_astar(D.N_STAGES)
    er, cp, burns, comp, term_dt, act_mass = expected_return_from_policy(
        T, O, R, sdec, full, init_b, obs_agent_size)
    return er, cp, burns, comp, term_dt, act_mass, None


_VARIANT_LABEL = {"centralized": "Centralized", "sdec": "SDec", "dec": "Decentralized"}


def plot_reward_parts(parts_rows, fig_dir, tag):
    """
    Stacked reward-component bars per variant (maneuver / deviation / risk /
    displacement) with the net total annotated. Explains the Cen=SDec degeneracy:
    the same total reached by a different maneuver-vs-deviation split. (No v1
    equivalent; matches v1's color/style otherwise.)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)
    variants = list(dict.fromkeys(r["variant"] for r in parts_rows))
    parts = ["maneuver", "deviation", "risk", "displace"]
    part_colors = {"maneuver": "#3b6ea8", "deviation": "#2f8f5b",
                   "risk": "#c0392b", "displace": "#c46a32"}
    by = {(r["variant"], r["component"]): float(r["value"]) for r in parts_rows}

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(variants))
    neg_base = np.zeros(len(variants))
    for part in parts:
        vals = np.array([by.get((v, part), 0.0) for v in variants])
        ax.bar(x, vals, bottom=neg_base, label=part, color=part_colors[part], width=0.55)
        neg_base += vals
    for i, v in enumerate(variants):
        total = sum(by.get((v, p), 0.0) for p in parts)
        ax.text(x[i], neg_base[i] - 0.15, f"net {total:.2f}",
                ha="center", va="top", fontsize=10, fontweight="bold")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(variants)
    ax.set_ylabel("Expected reward contribution")
    ax.set_title("v2 reward decomposition by variant\n(same net via different maneuver/deviation split)")
    ax.legend(loc="lower right", ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = os.path.join(fig_dir, f"reward_parts_{tag}.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    # Scenario knobs already applied pre-import by _cli_bootstrap_scenario; declared here so
    # argparse accepts them (the config surface; values were consumed before model import).
    ap.add_argument("--scenario-config", default=None,
                    help="YAML config supplying scenario knobs (the ONE config surface; "
                         "explicit flags override it). NO env vars.")
    ap.add_argument("--man-cost", type=float, default=None,
                    help="per agent-burn reward (cfg.reward.man_cost; default -2.0).")
    ap.add_argument("--disp-k", default=None,
                    help="convex displacement curvature (cfg.reward.disp_k; "
                         "'none'/'linear' => legacy linear ramp).")
    ap.add_argument("--obs-fidelity", default=None,
                    help="SDec sync obs fidelity (cfg.obs.fidelity): "
                         "perfect|gps|tle|asymmetric. perfect (default) == anchor.")
    ap.add_argument("--obs-sigma", default=None,
                    help="raw SDec sync obs sigma km (cfg.obs.sigma); overrides the named "
                         "fidelity for a smooth sync-value curve. 'none' => perfect delta.")
    ap.add_argument("--obs-coarse", dest="obs_coarse", action="store_true", default=None,
                    help="coarsen km-scale syncs onto the signed operational alphabet "
                         "(cfg.obs.coarse); makes the TLE solves tractable. OFF => fine bins.")
    ap.add_argument("--sdec-tail-qmdp", dest="sdec_tail_qmdp", action="store_true", default=None,
                    help="SDec/Cen RS-SDA* QMDP tail approx + rec_limit=1 (cfg.solve.sdec_tail_qmdp; "
                         "graded-obs speedup). OFF => anchor byte-identical.")
    ap.add_argument("--sdec-iter-limit", type=int, default=None,
                    help="SDec/Cen RS-SDA* TI2 pruning budget (cfg.solve.sdec_iter_limit; "
                         "default 2000 == anchor).")
    ap.add_argument("--hour-grid", default=None,
                    help="base decision cadence, comma hours-before-TCA desc "
                         "(cfg.grid.hour_grid_h; default ~2h).")
    ap.add_argument("--merge-threshold", type=float, default=None,
                    help="contact-merge threshold h (cfg.grid.merge_threshold_h; default 0.25).")
    ap.add_argument("--init-miss", type=float, default=0.0,
                    help="TOTAL initial miss (km), perp-aware DANGER dial (default 0 = "
                         "collision course). dt center is back-solved so "
                         "sqrt(dt^2+perp^2)=init_miss. <1 km => must maneuver; ~5 km => "
                         "already-clear control. If <=perp the conjunction is "
                         "unflaggable (clamped, reported). Default init_miss=0, "
                         "spread=1.4, perp=0 reproduces the historical {0,0.7,1.4} belief.")
    ap.add_argument("--init-spread", type=float, default=1.4,
                    help="half-width (km) of the |dt| spread about the center "
                         "(UNCERTAINTY dial — what sync resolves). Default 1.4.")
    ap.add_argument("--perp", type=float, default=0.0)
    ap.add_argument("--contact-stages", type=str, default=None,
                    help="comma-separated stage indices for SDec sync contacts "
                         "(e.g. '13,15'). Overrides the default GS windows. Empty "
                         "string => no contacts. Affects O matrix + sync_states + count.")
    ap.add_argument("--tag", type=str, default="refined_drift_main_v2",
                    help="output tag (CSV + figures). Sits beside the v1 baseline.")
    ap.add_argument("--out-dir", type=str,
                    default=os.path.join(_HERE, "notes", "results"))
    ap.add_argument("--fig-dir", type=str,
                    default=os.path.join(_HERE, "notes", "figures"))
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--variants", type=str, default="centralized,sdec,dec",
                    help="comma-separated subset of centralized,sdec,dec")
    ap.add_argument("--backend", default=None, choices=["numerical", "keplerian", "drag"],
                    help="propagator backend for the WHOLE pipeline (contacts + matrices). "
                         "drag = experiments; numerical/keplerian = two-body (fast, debug). "
                         "Default: env SPACECRAFT_PROPAGATOR or 'numerical'.")
    # --- RS-MAA* (Dec) knobs: each defaults to the v1-proven value, so Dec solves
    #     with no tuning. Override only if the Dec policy misbehaves. ---
    ap.add_argument("--rsmaa-cluster-type", default=RSMAA_DEFAULTS["cluster_type"])
    ap.add_argument("--rsmaa-maxit", type=int, default=RSMAA_DEFAULTS["maxit"])
    ap.add_argument("--rsmaa-q-depth", type=int, default=RSMAA_DEFAULTS["q_depth"])
    ap.add_argument("--rsmaa-alpha", type=float, default=RSMAA_DEFAULTS["alpha"])
    ap.add_argument("--rsmaa-heuristic", default=RSMAA_DEFAULTS["heuristic"],
                    choices=["MDP", "POMDP"])
    ap.add_argument("--rsmaa-rec-type", default=RSMAA_DEFAULTS["rec_type"])
    ap.add_argument("--rsmaa-maxrec", type=int, default=RSMAA_DEFAULTS["maxrec"])
    ap.add_argument("--rsmaa-memory", type=int, default=RSMAA_DEFAULTS["memory"])
    ap.add_argument("--iter-limit", type=int, default=RSMAA_DEFAULTS["iter_limit"])
    ap.add_argument("--verbose-rsmaa", action="store_true")
    args = ap.parse_args()
    global PERP_KM
    PERP_KM = args.perp
    initialize_eop()

    # --- contact-timing override (Scenario-1 ablation). Sets the SINGLE global that
    #     the v2 O-matrix builder, the v1 builder, and SDec sync_states all read. ---
    if args.contact_stages is not None:
        stages = [int(s) for s in args.contact_stages.split(",") if s.strip() != ""]
        M.set_contact_stages(stages)
    print(f"  propagator backend: {M._SG.PROPAGATOR_BACKEND}  (N_STAGES={D.N_STAGES})")
    print(f"  SDec contact stages: {M.get_contact_stages()}")

    rsmaa_cfg = dict(RSMAA_DEFAULTS)
    rsmaa_cfg.update(
        cluster_type=args.rsmaa_cluster_type, maxit=args.rsmaa_maxit,
        q_depth=args.rsmaa_q_depth, alpha=args.rsmaa_alpha,
        heuristic=args.rsmaa_heuristic, rec_type=args.rsmaa_rec_type,
        maxrec=args.rsmaa_maxrec, memory=args.rsmaa_memory,
        iter_limit=args.iter_limit, verbose=args.verbose_rsmaa,
    )
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    # v1's CSV/plot pipeline (same schema, same figure style) — reused, not reimplemented.
    from compare_variants import (write_csv, plot_summary, plot_burn_timing,
                                  plot_action_schedule)

    # SOLVE belief: perp-AWARE 2-knob spread (danger x uncertainty) so the agents are
    # genuinely uncertain (sync has something to resolve), then evaluate. The default
    # (init_miss=0, spread=1.4, perp=0) reproduces the historical {0,0.7,1.4} belief.
    init_b, flagged, eff_miss = TV.build_init_b_danger(
        args.init_miss, args.init_spread, args.perp, sign_mode="both")
    flag_str = "FLAGGED (dangerous)" if flagged else \
        "NOT flaggable (miss<=perp; geometry already clears it — exclude from sweeps)"
    print(f"  init belief: init_miss={args.init_miss} spread={args.init_spread} "
          f"perp={args.perp} -> effective miss={eff_miss:.3f} km [{flag_str}]")

    # init_bin axis: the v2 spread is a single belief, so we report it as one bin (0).
    INIT_BIN = 0
    summary_rows, burn_rows, parts_rows, action_rows = [], [], [], []

    print(f"\nv2 matched-metric comparison  (init_miss={args.init_miss} km, perp={args.perp} km)")
    print("=" * 60)
    sync_count = {"centralized": D.N_STAGES, "sdec": len(M.get_contact_stages()), "dec": 0}
    for variant in variants:
        is_dec = (variant == "dec")
        er, cp, burns, comp, term_dt, act_mass, note = solve_and_eval(
            variant, init_b, rsmaa=is_dec, rsmaa_cfg=rsmaa_cfg)
        label = _VARIANT_LABEL[variant]
        bstr = ",".join(f"s{k}:{burns[k]:.2f}" for k in range(D.N_STAGES) if burns[k] > 0.01)
        print(f"  {variant:<12} expected_return={er:>10.4f}  collision_prob={cp:.4f}")
        print(f"               burns by stage: {bstr or 'none'}  (total burn mass {burns.sum():.3f})")
        print(f"               reward parts: " + "  ".join(
            f"{k}={comp[k]:.3f}" for k in ("maneuver", "deviation", "risk", "displace")))
        tstr = ",".join(f"b{b}({D.dt_bin_center_km(b):+.1f}km):{term_dt[b]:.2f}"
                        for b in sorted(term_dt) if term_dt[b] > 0.01)
        print(f"               terminal dt bins: {tstr}")

        # --- rows in v1's pipeline schema ---
        summary_rows.append({
            "variant": label, "matrix_variant": variant, "init_bin": INIT_BIN,
            "expected_return": er, "collision_prob": cp,
            "expected_dv_ms": float(burns.sum()) * float(DV_MAGNITUDE),
            "expected_syncs": sync_count[variant],
        })
        for k in range(D.N_STAGES):
            burn_rows.append({
                "variant": label, "matrix_variant": variant, "init_bin": INIT_BIN,
                "rollout_mode": "expected", "stage": k,
                "mean_agent_burns": float(burns[k]), "burn_stage_rate": float(burns[k]),
            })
        for part in ("maneuver", "deviation", "risk", "displace"):
            parts_rows.append({
                "variant": label, "matrix_variant": variant, "init_bin": INIT_BIN,
                "component": part, "value": float(comp[part]),
            })
        # action schedule: dominant joint action + its prob per stage (mass-normalized)
        for k in range(D.N_STAGES):
            stage_items = {a: m for (st, a), m in act_mass.items() if st == k}
            tot = sum(stage_items.values())
            if tot <= 0:
                ja_dom, prob = 0, 0.0   # no mass reaching this stage -> WAIT/empty
            else:
                ja_dom = max(stage_items, key=stage_items.get)
                prob = stage_items[ja_dom] / tot
            action_rows.append({
                "variant": label, "matrix_variant": variant, "init_bin": INIT_BIN,
                "stage": k, "joint_action": int(ja_dom), "action_prob": float(prob),
            })
    print("=" * 60)

    tag = args.tag
    write_csv(os.path.join(args.out_dir, f"variant_expected_{tag}.csv"), summary_rows)
    write_csv(os.path.join(args.out_dir, f"variant_burn_timing_{tag}.csv"), burn_rows)
    write_csv(os.path.join(args.out_dir, f"variant_reward_parts_{tag}.csv"), parts_rows)
    write_csv(os.path.join(args.out_dir, f"variant_action_by_stage_{tag}.csv"), action_rows)
    print(f"  wrote CSVs -> {args.out_dir} (tag={tag})")
    if not args.no_figures:
        p1 = plot_summary(summary_rows, args.fig_dir, tag)
        p2 = plot_burn_timing(burn_rows, args.fig_dir, tag)
        p3 = plot_reward_parts(parts_rows, args.fig_dir, tag)
        p4 = plot_action_schedule(action_rows, args.fig_dir, tag)
        for p in (p1, p2, p3, p4):
            print(f"  wrote figure -> {p}")
    # Return the in-memory rows so a config runner (main.py) can log them to wandb without
    # re-reading the CSV. The CSV stays the source of truth; this is just the convenience seam.
    return {"summary": summary_rows, "burn": burn_rows,
            "reward_parts": parts_rows, "action": action_rows,
            "n_stages": D.N_STAGES, "contact_stages": M.get_contact_stages()}


if __name__ == "__main__":
    main()
