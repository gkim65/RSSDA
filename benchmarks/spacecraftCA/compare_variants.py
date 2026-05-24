"""
compare_variants.py

Generate comparative results for the spacecraft CA benchmark:
centralized vs semi-decentralized vs fully decentralized policies.

The experiment holds dynamics, rewards, initial belief, and rollout seeds fixed.
Only the information structure changes:
  centralized: sync at every stage
  sdec:        sync only at ground-contact stages
  dec:         no sync

Outputs:
  notes/results/variant_expected_<tag>.csv
  notes/results/variant_action_by_stage_<tag>.csv
  notes/results/variant_rollouts_<tag>.csv
  notes/results/variant_summary_<tag>.csv
  notes/results/variant_burn_timing_<tag>.csv
  notes/figures/action_schedule_<tag>.png
  notes/figures/variant_comparison_<tag>.png
  notes/figures/burn_timing_<tag>.png
"""

import argparse
import contextlib
import csv
import io
import math
import os
import sys
import time
from array import array
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCHMARKS = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_BENCHMARKS)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _BENCHMARKS)
sys.path.insert(0, _HERE)

from brahe import initialize_eop

from RSSDA import SDecPOMDP, SDecPOMDPModel, int_tuple
from baselines.decPOMDP import (
    DecPOMDP as RSMAA,
    MemoryLimitExceeded as RsmaaMemoryLimitExceeded,
)
from spacecraft_discretizer import (
    DEV_ZERO,
    N_DEV,
    N_MISS,
    N_STAGES,
    N_STATES_TOTAL,
    SINK_STATE,
    bin_center_km,
    index_to_state,
    miss_to_bin,
    state_index,
)
from spacecraft_matrices import (
    CONTACT_STAGES,
    N_ACT_AGENT,
    N_JOINT_ACTIONS,
    N_JOINT_OBS,
    N_MISS_OBS,
    N_OBS_AGENT,
    REWARD_DEVIATION,
    REWARD_MANEUVER,
    REWARD_STEP,
    collision_probability_from_bin_probs,
    load_matrices,
    split_joint_action,
    terminal_risk_reward,
)
from sdec_spacecraft import build_config
from spacecraft_simulator import (
    get_rssda_action,
    run_simulation,
    update_oh_rssda,
)


ACTION_LABELS = {
    0: "WAIT",
    1: "SC1+",
    2: "SC1-",
    3: "SC2+",
    4: "Both+",
    5: "SC1-\nSC2+",
    6: "SC2-",
    7: "SC1+\nSC2-",
    8: "Both-",
}

ACTION_COLORS = {
    0: "#d1d5db",
    1: "#4c9be8",
    2: "#2f6fb0",
    3: "#e8714c",
    4: "#9b59b6",
    5: "#27ae60",
    6: "#b84c2f",
    7: "#1f8f6b",
    8: "#7441a0",
}


@dataclass(frozen=True)
class VariantSpec:
    label: str
    matrix_variant: str
    sync_stages: List[int]
    compress_dec_obs: bool = False


@dataclass(frozen=True)
class SolverModeSpec:
    label: str
    ti1_enabled: bool
    interleaved: bool


def parse_bins(text: str) -> List[int]:
    bins = [int(part.strip()) for part in text.split(",") if part.strip()]
    bad = [b for b in bins if b < 0 or b >= N_MISS]
    if bad:
        raise argparse.ArgumentTypeError(f"Invalid miss bins {bad}; expected 0..{N_MISS - 1}")
    return bins


def make_init_b(belief_bins: List[int]) -> np.ndarray:
    b = np.zeros(N_STATES_TOTAL, dtype=np.float64)
    for mb in belief_bins:
        b[state_index(mb, DEV_ZERO, DEV_ZERO, 0)] = 1.0
    b /= b.sum()
    return b


def default_variants() -> List[VariantSpec]:
    return [
        VariantSpec("Centralized", "centralized", list(range(N_STAGES))),
        VariantSpec("SDec", "sdec", list(CONTACT_STAGES)),
        VariantSpec("Decentralized", "dec", [], True),
    ]


def filter_variants(specs: List[VariantSpec], names: List[str]) -> List[VariantSpec]:
    if not names:
        return specs
    lookup = {
        "centralized": "Centralized",
        "cen": "Centralized",
        "sdec": "SDec",
        "semi": "SDec",
        "decentralized": "Decentralized",
        "dec": "Decentralized",
    }
    wanted = {lookup.get(name.strip().lower(), name.strip()) for name in names}
    filtered = [spec for spec in specs if spec.label in wanted]
    missing = wanted - {spec.label for spec in filtered}
    if missing:
        raise SystemExit(f"Unknown variants: {sorted(missing)}")
    return filtered


def parse_solver_modes(text: str) -> List[SolverModeSpec]:
    modes = []
    lookup = {
        "fixed": SolverModeSpec("fixed", ti1_enabled=False, interleaved=False),
        "full": SolverModeSpec("fixed", ti1_enabled=False, interleaved=False),
        "ti1": SolverModeSpec("interleaved", ti1_enabled=True, interleaved=True),
        "interleaved": SolverModeSpec("interleaved", ti1_enabled=True, interleaved=True),
        "replan": SolverModeSpec("interleaved", ti1_enabled=True, interleaved=True),
    }
    for part in text.split(","):
        name = part.strip().lower()
        if not name:
            continue
        if name == "both":
            modes.extend([lookup["fixed"], lookup["interleaved"]])
            continue
        if name not in lookup:
            raise argparse.ArgumentTypeError(
                f"Unknown solver mode '{part}'. Use fixed, interleaved, or both."
            )
        modes.append(lookup[name])
    if not modes:
        raise argparse.ArgumentTypeError("At least one solver mode is required.")
    deduped = []
    seen = set()
    for mode in modes:
        if mode.label not in seen:
            deduped.append(mode)
            seen.add(mode.label)
    return deduped


def compact_decentralized_observations(O_full: np.ndarray) -> np.ndarray:
    """
    Compress Dec observations from miss-expanded local symbols to 3x3.

    In the cleaned benchmark, the fully decentralized variant never receives a
    miss observation. Each agent only sees its own deviation bin. Keeping the
    unreachable miss/null-expanded symbols in the solver creates a large and
    unnecessary observation branching factor.
    """
    O_small = np.zeros((O_full.shape[0], O_full.shape[1], N_DEV ** 2), dtype=O_full.dtype)
    for full_o in range(N_JOINT_OBS):
        o1 = full_o % N_OBS_AGENT
        o2 = full_o // N_OBS_AGENT
        dev1 = o1 // N_MISS_OBS
        dev2 = o2 // N_MISS_OBS
        compact_o = dev1 + N_DEV * dev2
        O_small[:, :, compact_o] += O_full[:, :, full_o]
    row_sums = O_small.sum(axis=2, keepdims=True)
    np.divide(O_small, row_sums, out=O_small, where=row_sums > 0)
    return O_small


def dense_rows_to_pdict(mat3: np.ndarray, tol: float = 0.0) -> List[Tuple[array, array]]:
    """Convert (action, row, col) dense arrays to RS-MAA* sparse row tuples."""
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


def build_model_custom(T, O, R, init_b, sync_stages, obs_agent_size: int):
    from spacecraft_discretizer import sync_trigger_states

    sync_states = sync_trigger_states(sync_stages)
    model = SDecPOMDPModel(
        nagents=2,
        nstates=N_STATES_TOTAL,
        nactions=N_JOINT_ACTIONS,
        nobs=O.shape[2],
        transitions=T.flatten().tolist(),
        obs=O.flatten().tolist(),
        rewards=R.flatten().tolist(),
        init_beliefs=init_b.tolist(),
        nacts_factor=[N_ACT_AGENT, N_ACT_AGENT],
        nobs_factor=[obs_agent_size, obs_agent_size],
        sync_states=sync_states,
        sync_actions=[],
        sync_observations=[],
    )
    return model, sync_states


def solve_variant(spec: VariantSpec, T, O, R, init_b, obs_agent_size: int,
                  iter_limit=None, max_clusters=None, rec_limit=None,
                  ti1_enabled: bool = False, memory_limit_gb=None,
                  memory_check_interval=None):
    model, sync_states = build_model_custom(
        T, O, R, init_b, spec.sync_stages, obs_agent_size
    )
    config = build_config(exact=False)
    config.TI1 = ti1_enabled
    if iter_limit is not None:
        config.iter_limit = iter_limit
    if max_clusters is not None:
        config.max_clusters = max_clusters
    if rec_limit is not None:
        config.rec_limit = rec_limit
    if memory_limit_gb is not None:
        config.memory_limit_gb = memory_limit_gb
    if memory_check_interval is not None:
        config.memory_check_interval = memory_check_interval
    sdec = SDecPOMDP(model=model, config=config)

    t0 = time.perf_counter()
    result = sdec.multi_agent_astar(N_STAGES)
    solve_seconds = time.perf_counter() - t0

    return sdec, result, len(sync_states), solve_seconds


def solve_dec_rsmaa(T, O, R, init_b, args):
    """Solve the fully decentralized spacecraft model with baseline RS-MAA*."""
    T_pdict = dense_rows_to_pdict(T)
    O_pdict = dense_rows_to_pdict(O)
    solver = RSMAA(
        nagents=2,
        nstates=N_STATES_TOTAL,
        nactions=N_JOINT_ACTIONS,
        nobs=O.shape[2],
        transitions=T_pdict,
        obs=O_pdict,
        rewards=R.reshape(-1).tolist(),
        init_beliefs=init_b.tolist(),
        nacts_factor=[N_ACT_AGENT, N_ACT_AGENT],
        nobs_factor=[N_DEV, N_DEV],
        maxh=N_STAGES,
        cluster_type=args.rsmaa_cluster_type,
        maxit=args.rsmaa_maxit,
        q_depth=args.rsmaa_q_depth,
        alpha=args.rsmaa_alpha,
        iter_limit=args.iter_limit,
        maxrec=args.rec_limit if args.rec_limit is not None else 2,
        memory=args.rsmaa_memory,
        heuristic=args.rsmaa_heuristic,
        rec_type=args.rsmaa_rec_type,
        p_threshold_cluster=args.p_threshold_cluster,
        p_threshold_expand=args.p_threshold_expand,
        policyvalfound=-math.inf,
        output=args.verbose_rsmaa,
        memory_limit_gb=args.memory_limit_gb,
        memory_check_interval=args.memory_check_interval,
    )
    solver.decentralized = True
    solver.onesided = False

    t0 = time.perf_counter()
    if args.verbose_rsmaa:
        value, policy, clustering = solver.multi_agent_astar(N_STAGES)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            value, policy, clustering = solver.multi_agent_astar(N_STAGES)
    solve_seconds = time.perf_counter() - t0
    return solver, (float(value), policy, clustering), 0, solve_seconds


def policy_length(full_result) -> int:
    policy = full_result[1] if full_result and len(full_result) > 1 else None
    return len(policy) if policy is not None else 0


def display_variant_label(spec: VariantSpec, mode: SolverModeSpec) -> str:
    if mode.label == "fixed":
        return spec.label
    return f"{spec.label} TI1"


def add_solver_metadata(row: dict, mode: SolverModeSpec, full_result,
                        solve_seconds: float, replan_count: int = 0,
                        plan_count: int = 1,
                        interleaved_solve_seconds: float = 0.0,
                        missing_action_mass: float = 0.0) -> None:
    plen = policy_length(full_result)
    row.update({
        "solver_mode": mode.label,
        "ti1_enabled": bool(mode.ti1_enabled),
        "interleaved_replanning": bool(mode.interleaved),
        "policy_len": plen,
        "policy_complete": bool(plen >= N_STAGES),
        "initial_solve_seconds": float(solve_seconds),
        "interleaved_solve_seconds": float(interleaved_solve_seconds),
        "replan_count": int(replan_count),
        "plan_count": int(plan_count),
        "missing_action_mass": float(missing_action_mass),
    })


def sample_index(probs: np.ndarray, rng: np.random.Generator) -> int:
    total = float(probs.sum())
    if total <= 0.0:
        return 0
    return int(rng.choice(len(probs), p=probs / total))


def reward_components(true_state: int, joint_act: int) -> Tuple[float, float, float, float]:
    """Return step, maneuver, deviation, terminal-risk reward components."""
    if true_state == SINK_STATE:
        return 0.0, 0.0, 0.0, 0.0
    miss_bin, dev1, dev2, stage = index_to_state(true_state)
    burn1, burn2 = split_joint_action(joint_act)
    step_reward = REWARD_STEP
    maneuver_reward = REWARD_MANEUVER * ((burn1 != 0) + (burn2 != 0))
    deviation_reward = REWARD_DEVIATION * ((dev1 != DEV_ZERO) + (dev2 != DEV_ZERO))
    terminal_reward = 0.0
    if stage == N_STAGES - 1:
        terminal_reward = terminal_risk_reward(miss_bin)
    return step_reward, maneuver_reward, deviation_reward, terminal_reward


def rollout_abstract(T, O, R, sdec, full_result, sync_stages: List[int],
                     true_init_bin: int, dv_mag: float,
                     rng: np.random.Generator, obs_agent_size: int) -> dict:
    """
    Fast rollout directly on the discrete POMDP model.

    This is the right mode for large policy-comparison sweeps because it samples
    the same T/O model used by RS-SDA*. Use spacecraft_simulator.py or
    --rollout-mode physical for continuous Brahe validation runs.
    """
    value, policy, clustering, cent_vector, cen_dists_map, clustering_cen = full_result

    true_state = state_index(true_init_bin, DEV_ZERO, DEV_ZERO, 0)
    current_belief_idx = sdec.dist_dict[int_tuple(sdec.init_beliefs)]
    current_oh = [0, 0]

    total_dv = 0.0
    sync_count = 0
    maneuver_stages = []
    miss_bin_traj = []
    last_miss_bin = true_init_bin

    for k in range(N_STAGES):
        if true_state == SINK_STATE:
            break

        miss_bin, _, _, stage = index_to_state(true_state)
        last_miss_bin = miss_bin
        miss_bin_traj.append(miss_bin)
        if stage in sync_stages:
            sync_count += 1

        joint_act, a1, a2, is_cen = get_rssda_action(
            policy, cen_dists_map, clustering, clustering_cen,
            k, current_belief_idx, current_oh, N_ACT_AGENT)
        if joint_act < 0:
            joint_act, a1, a2 = 0, 0, 0

        burn1, burn2 = split_joint_action(joint_act)
        if burn1 != 0 or burn2 != 0:
            maneuver_stages.append((stage, burn1, burn2))
        total_dv += dv_mag * ((burn1 != 0) + (burn2 != 0))

        c_ptr = -1
        if is_cen and k < len(cen_dists_map):
            dists_at_step = cen_dists_map[k]
            if current_belief_idx in dists_at_step:
                c_ptr = dists_at_step.index(current_belief_idx)

        next_state = sample_index(T[joint_act, true_state, :], rng)
        if next_state == SINK_STATE:
            true_state = next_state
            continue

        obs_joint = sample_index(O[joint_act, next_state, :], rng)
        obs1 = obs_joint % obs_agent_size
        obs2 = obs_joint // obs_agent_size

        try:
            sparse_transitions = sdec.get_terminal(current_belief_idx, joint_act)
            obs_to_belief = {o: d for o, _, d in sparse_transitions}
            if obs_joint in obs_to_belief:
                current_belief_idx = obs_to_belief[obs_joint]
        except KeyError:
            pass

        current_oh = update_oh_rssda(
            policy, cen_dists_map, clustering, clustering_cen,
            k, current_belief_idx, current_oh, is_cen, c_ptr, obs1, obs2)
        true_state = next_state

    return {
        "miss_km_at_tca": bin_center_km(last_miss_bin),
        "total_dv_ms": total_dv,
        "sync_count": sync_count,
        "maneuver_stages": maneuver_stages,
        "miss_bin_trajectory": miss_bin_traj,
    }


def run_abstract_simulation(T, O, R, sdec, full_result, sync_stages: List[int],
                            n_rollouts: int, init_miss_bin: int,
                            dv_mag: float, seed: int,
                            obs_agent_size: int) -> List[dict]:
    rng = np.random.default_rng(seed)
    return [
        rollout_abstract(
            T, O, R, sdec, full_result, sync_stages,
            init_miss_bin, dv_mag, rng, obs_agent_size,
        )
        for _ in range(n_rollouts)
    ]


def expected_policy_metrics(T, O, R, sdec, full_result, sync_stages: List[int],
                            true_init_bin: int, dv_mag: float,
                            obs_agent_size: int, prune: float = 1e-12):
    """
    Exact model-based evaluation of a fixed RSSDA policy for one true init bin.

    The policy is still the one solved from the requested initial belief. This
    evaluator conditions the true state on a specific initial bin and propagates
    all transition/observation branches without Monte Carlo noise.
    """
    value, policy, clustering, cent_vector, cen_dists_map, clustering_cen = full_result
    root_belief = sdec.dist_dict[int_tuple(sdec.init_beliefs)]
    root_state = state_index(true_init_bin, DEV_ZERO, DEV_ZERO, 0)
    nodes = {(root_state, root_belief, 0, 0): 1.0}

    expected_return = 0.0
    expected_step_reward = 0.0
    expected_maneuver_reward = 0.0
    expected_deviation_reward = 0.0
    expected_terminal_risk_reward = 0.0
    expected_dv = 0.0
    expected_syncs = 0.0
    expected_agent_burns = 0.0
    terminal_bin_probs = np.zeros(N_MISS, dtype=float)
    stage_agent_burns = np.zeros(N_STAGES, dtype=float)
    stage_any_burns = np.zeros(N_STAGES, dtype=float)
    stage_action_probs = np.zeros((N_STAGES, N_JOINT_ACTIONS), dtype=float)
    missing_action_mass = 0.0

    terminal_cache = {}
    nonzero_T_cache = {}
    nonzero_O_cache = {}

    for step in range(N_STAGES):
        next_nodes = defaultdict(float)

        for (true_state, belief_idx, oh0, oh1), mass in nodes.items():
            if mass <= prune or true_state == SINK_STATE:
                continue

            miss_bin, _, _, stage = index_to_state(true_state)
            joint_act, a1, a2, is_cen = get_rssda_action(
                policy, cen_dists_map, clustering, clustering_cen,
                step, belief_idx, [oh0, oh1], N_ACT_AGENT)
            if step >= len(policy) or joint_act < 0:
                missing_action_mass += mass
            if joint_act < 0:
                joint_act, a1, a2 = 0, 0, 0

            burn1, burn2 = split_joint_action(joint_act)
            n_burns = int(burn1 != 0) + int(burn2 != 0)

            expected_return += mass * float(R[joint_act, true_state])
            step_r, maneuver_r, deviation_r, terminal_r = reward_components(true_state, joint_act)
            expected_step_reward += mass * step_r
            expected_maneuver_reward += mass * maneuver_r
            expected_deviation_reward += mass * deviation_r
            expected_terminal_risk_reward += mass * terminal_r
            expected_dv += mass * dv_mag * n_burns
            expected_agent_burns += mass * n_burns
            stage_agent_burns[stage] += mass * n_burns
            stage_any_burns[stage] += mass * float(n_burns > 0)
            stage_action_probs[stage, joint_act] += mass

            if stage in sync_stages:
                expected_syncs += mass
            if stage == N_STAGES - 1:
                terminal_bin_probs[miss_bin] += mass

            c_ptr = -1
            if is_cen and step < len(cen_dists_map):
                dists_at_step = cen_dists_map[step]
                if belief_idx in dists_at_step:
                    c_ptr = dists_at_step.index(belief_idx)

            key_T = (joint_act, true_state)
            if key_T not in nonzero_T_cache:
                nz = np.flatnonzero(T[joint_act, true_state, :] > prune)
                nonzero_T_cache[key_T] = [(int(sp), float(T[joint_act, true_state, sp])) for sp in nz]

            key_terminal = (belief_idx, joint_act)
            if key_terminal not in terminal_cache:
                try:
                    terminal_cache[key_terminal] = {
                        int(o): int(d)
                        for o, _, d in sdec.get_terminal(belief_idx, joint_act)
                    }
                except KeyError:
                    terminal_cache[key_terminal] = {}
            obs_to_belief = terminal_cache[key_terminal]

            for next_state, p_state in nonzero_T_cache[key_T]:
                branch_mass = mass * p_state
                if branch_mass <= prune:
                    continue
                if next_state == SINK_STATE:
                    next_nodes[(next_state, belief_idx, oh0, oh1)] += branch_mass
                    continue

                key_O = (joint_act, next_state)
                if key_O not in nonzero_O_cache:
                    nz = np.flatnonzero(O[joint_act, next_state, :] > prune)
                    nonzero_O_cache[key_O] = [
                        (int(obs), float(O[joint_act, next_state, obs]))
                        for obs in nz
                    ]

                for obs_joint, p_obs in nonzero_O_cache[key_O]:
                    obs_mass = branch_mass * p_obs
                    if obs_mass <= prune:
                        continue
                    next_belief = obs_to_belief.get(obs_joint, belief_idx)
                    obs1 = obs_joint % obs_agent_size
                    obs2 = obs_joint // obs_agent_size
                    next_oh0, next_oh1 = update_oh_rssda(
                        policy, cen_dists_map, clustering, clustering_cen,
                        step, next_belief, [oh0, oh1], is_cen, c_ptr, obs1, obs2)
                    next_nodes[(next_state, next_belief, next_oh0, next_oh1)] += obs_mass

        nodes = {
            node: p for node, p in next_nodes.items()
            if p > prune and node[0] != SINK_STATE
        }

    terminal_total = float(terminal_bin_probs.sum())
    if terminal_total > 0:
        terminal_bin_probs /= terminal_total
    expected_miss_km = float(sum(
        terminal_bin_probs[mb] * bin_center_km(mb)
        for mb in range(N_MISS)
    ))

    summary_row = {
        "variant": None,
        "matrix_variant": None,
        "init_bin": true_init_bin,
        "eval_mode": "expected",
        "expected_return": float(expected_return),
        "expected_step_reward": float(expected_step_reward),
        "expected_maneuver_reward": float(expected_maneuver_reward),
        "expected_deviation_reward": float(expected_deviation_reward),
        "expected_terminal_risk_reward": float(expected_terminal_risk_reward),
        "collision_prob": collision_probability_from_bin_probs(terminal_bin_probs),
        "expected_miss_km": expected_miss_km,
        "expected_dv_ms": float(expected_dv),
        "expected_syncs": float(expected_syncs),
        "expected_agent_burns": float(expected_agent_burns),
        "missing_action_mass": float(missing_action_mass),
    }
    for mb in range(N_MISS):
        summary_row[f"terminal_bin_{mb}_prob"] = float(terminal_bin_probs[mb])

    burn_rows = []
    action_rows = []
    for stage in range(N_STAGES):
        burn_rows.append({
            "variant": None,
            "matrix_variant": None,
            "init_bin": true_init_bin,
            "eval_mode": "expected",
            "stage": stage,
            "mean_agent_burns": float(stage_agent_burns[stage]),
            "burn_stage_rate": float(stage_any_burns[stage]),
        })
        total_action_mass = float(stage_action_probs[stage, :].sum())
        if total_action_mass > 0:
            probs = stage_action_probs[stage, :] / total_action_mass
        else:
            probs = stage_action_probs[stage, :]
        for action in range(N_JOINT_ACTIONS):
            action_rows.append({
                "variant": None,
                "matrix_variant": None,
                "init_bin": true_init_bin,
                "eval_mode": "expected",
                "stage": stage,
                "joint_action": action,
                "action_label": ACTION_LABELS[action].replace("\n", " "),
                "action_prob": float(probs[action]),
            })

    return summary_row, burn_rows, action_rows


def expected_rsmaa_policy_metrics(T, O, R, full_result,
                                  true_init_bin: int, dv_mag: float,
                                  prune: float = 1e-12):
    """
    Exact model-based evaluation of a fixed RS-MAA* Dec policy.

    RS-MAA* policies are indexed as policy[stage][agent][local_cluster]. The
    clustering object maps each agent's local cluster plus its local observation
    to the next local cluster.
    """
    value, policy, clustering = full_result
    root_state = state_index(true_init_bin, DEV_ZERO, DEV_ZERO, 0)
    nodes = {(root_state, 0, 0): 1.0}

    expected_return = 0.0
    expected_step_reward = 0.0
    expected_maneuver_reward = 0.0
    expected_deviation_reward = 0.0
    expected_terminal_risk_reward = 0.0
    expected_dv = 0.0
    expected_syncs = 0.0
    expected_agent_burns = 0.0
    terminal_bin_probs = np.zeros(N_MISS, dtype=float)
    stage_agent_burns = np.zeros(N_STAGES, dtype=float)
    stage_any_burns = np.zeros(N_STAGES, dtype=float)
    stage_action_probs = np.zeros((N_STAGES, N_JOINT_ACTIONS), dtype=float)
    missing_action_mass = 0.0

    nonzero_T_cache = {}
    nonzero_O_cache = {}

    for step in range(N_STAGES):
        next_nodes = defaultdict(float)

        for (true_state, c0, c1), mass in nodes.items():
            if mass <= prune or true_state == SINK_STATE:
                continue

            miss_bin, _, _, stage = index_to_state(true_state)
            if step >= len(policy):
                missing_action_mass += mass
                a1, a2 = 0, 0
            else:
                try:
                    a1 = int(policy[step][0][c0])
                    a2 = int(policy[step][1][c1])
                except (IndexError, TypeError):
                    missing_action_mass += mass
                    a1, a2 = 0, 0
            if a1 < 0 or a2 < 0:
                missing_action_mass += mass
                a1, a2 = max(a1, 0), max(a2, 0)
            joint_act = a1 + N_ACT_AGENT * a2

            burn1, burn2 = split_joint_action(joint_act)
            n_burns = int(burn1 != 0) + int(burn2 != 0)

            expected_return += mass * float(R[joint_act, true_state])
            step_r, maneuver_r, deviation_r, terminal_r = reward_components(true_state, joint_act)
            expected_step_reward += mass * step_r
            expected_maneuver_reward += mass * maneuver_r
            expected_deviation_reward += mass * deviation_r
            expected_terminal_risk_reward += mass * terminal_r
            expected_dv += mass * dv_mag * n_burns
            expected_agent_burns += mass * n_burns
            stage_agent_burns[stage] += mass * n_burns
            stage_any_burns[stage] += mass * float(n_burns > 0)
            stage_action_probs[stage, joint_act] += mass

            if stage == N_STAGES - 1:
                terminal_bin_probs[miss_bin] += mass

            key_T = (joint_act, true_state)
            if key_T not in nonzero_T_cache:
                nz = np.flatnonzero(T[joint_act, true_state, :] > prune)
                nonzero_T_cache[key_T] = [
                    (int(sp), float(T[joint_act, true_state, sp]))
                    for sp in nz
                ]

            for next_state, p_state in nonzero_T_cache[key_T]:
                branch_mass = mass * p_state
                if branch_mass <= prune:
                    continue
                if next_state == SINK_STATE:
                    next_nodes[(next_state, c0, c1)] += branch_mass
                    continue

                key_O = (joint_act, next_state)
                if key_O not in nonzero_O_cache:
                    nz = np.flatnonzero(O[joint_act, next_state, :] > prune)
                    nonzero_O_cache[key_O] = [
                        (int(obs), float(O[joint_act, next_state, obs]))
                        for obs in nz
                    ]

                for obs_joint, p_obs in nonzero_O_cache[key_O]:
                    obs_mass = branch_mass * p_obs
                    if obs_mass <= prune:
                        continue
                    obs1 = obs_joint % N_DEV
                    obs2 = obs_joint // N_DEV
                    if step < len(clustering):
                        try:
                            nc0 = int(clustering[step][0][c0][obs1])
                            nc1 = int(clustering[step][1][c1][obs2])
                        except (IndexError, TypeError):
                            missing_action_mass += obs_mass
                            nc0, nc1 = 0, 0
                        if nc0 < 0:
                            nc0 = 0
                        if nc1 < 0:
                            nc1 = 0
                    else:
                        nc0, nc1 = c0, c1
                    next_nodes[(next_state, nc0, nc1)] += obs_mass

        nodes = {
            node: p for node, p in next_nodes.items()
            if p > prune and node[0] != SINK_STATE
        }

    terminal_total = float(terminal_bin_probs.sum())
    if terminal_total > 0:
        terminal_bin_probs /= terminal_total
    expected_miss_km = float(sum(
        terminal_bin_probs[mb] * bin_center_km(mb)
        for mb in range(N_MISS)
    ))

    summary_row = {
        "variant": None,
        "matrix_variant": None,
        "init_bin": true_init_bin,
        "eval_mode": "expected_rsmaa",
        "expected_return": float(expected_return),
        "expected_step_reward": float(expected_step_reward),
        "expected_maneuver_reward": float(expected_maneuver_reward),
        "expected_deviation_reward": float(expected_deviation_reward),
        "expected_terminal_risk_reward": float(expected_terminal_risk_reward),
        "collision_prob": collision_probability_from_bin_probs(terminal_bin_probs),
        "expected_miss_km": expected_miss_km,
        "expected_dv_ms": float(expected_dv),
        "expected_syncs": float(expected_syncs),
        "expected_agent_burns": float(expected_agent_burns),
        "missing_action_mass": float(missing_action_mass),
    }
    for mb in range(N_MISS):
        summary_row[f"terminal_bin_{mb}_prob"] = float(terminal_bin_probs[mb])

    burn_rows = []
    action_rows = []
    for stage in range(N_STAGES):
        burn_rows.append({
            "variant": None,
            "matrix_variant": None,
            "init_bin": true_init_bin,
            "eval_mode": "expected_rsmaa",
            "stage": stage,
            "mean_agent_burns": float(stage_agent_burns[stage]),
            "burn_stage_rate": float(stage_any_burns[stage]),
        })
        total_action_mass = float(stage_action_probs[stage, :].sum())
        probs = (
            stage_action_probs[stage, :] / total_action_mass
            if total_action_mass > 0 else stage_action_probs[stage, :]
        )
        for action in range(N_JOINT_ACTIONS):
            action_rows.append({
                "variant": None,
                "matrix_variant": None,
                "init_bin": true_init_bin,
                "eval_mode": "expected_rsmaa",
                "stage": stage,
                "joint_action": action,
                "action_label": ACTION_LABELS[action].replace("\n", " "),
                "action_prob": float(probs[action]),
            })

    return summary_row, burn_rows, action_rows


def expected_interleaved_policy_metrics(T, O, R, sdec, full_result,
                                        sync_stages: List[int],
                                        true_init_bin: int, dv_mag: float,
                                        obs_agent_size: int,
                                        initial_solve_seconds: float,
                                        prune: float = 1e-12):
    """
    Exact model-based evaluation of TI1 interleaved execution.

    A TI1 solve may return a prefix policy. This evaluator executes that prefix
    only until the next shared-information stage, then solves a fresh RS-SDA*
    subproblem from the updated shared belief and remaining horizon.
    """
    root_belief = sdec.dist_dict[int_tuple(sdec.init_beliefs)]
    root_state = state_index(true_init_bin, DEV_ZERO, DEV_ZERO, 0)
    root_key = (0, root_belief)
    plan_cache = {root_key: full_result}
    solve_seconds_by_plan = {root_key: initial_solve_seconds}

    def get_plan(stage: int, belief_idx: int):
        key = (stage, belief_idx)
        if key not in plan_cache:
            horizon = max(0, N_STAGES - stage)
            t0 = time.perf_counter()
            plan_cache[key] = sdec.multi_agent_astar(horizon, init_beliefs=belief_idx)
            solve_seconds_by_plan[key] = time.perf_counter() - t0
        return key, plan_cache[key]

    # Node: (true_state, belief_idx, oh0, oh1, plan_stage, plan_belief, local_step)
    nodes = {(root_state, root_belief, 0, 0, 0, root_belief, 0): 1.0}

    expected_return = 0.0
    expected_step_reward = 0.0
    expected_maneuver_reward = 0.0
    expected_deviation_reward = 0.0
    expected_terminal_risk_reward = 0.0
    expected_dv = 0.0
    expected_syncs = 0.0
    expected_agent_burns = 0.0
    terminal_bin_probs = np.zeros(N_MISS, dtype=float)
    stage_agent_burns = np.zeros(N_STAGES, dtype=float)
    stage_any_burns = np.zeros(N_STAGES, dtype=float)
    stage_action_probs = np.zeros((N_STAGES, N_JOINT_ACTIONS), dtype=float)
    missing_action_mass = 0.0

    terminal_cache = {}
    nonzero_T_cache = {}
    nonzero_O_cache = {}

    for _ in range(N_STAGES):
        next_nodes = defaultdict(float)

        for node, mass in nodes.items():
            if mass <= prune:
                continue
            true_state, belief_idx, oh0, oh1, plan_stage, plan_belief, local_step = node
            if true_state == SINK_STATE:
                continue

            miss_bin, _, _, stage = index_to_state(true_state)
            if stage in sync_stages and (plan_stage != stage or plan_belief != belief_idx):
                plan_stage, plan_belief, local_step = stage, belief_idx, 0
                oh0, oh1 = 0, 0

            _, plan = get_plan(plan_stage, plan_belief)
            _, policy, clustering, _, cen_dists_map, clustering_cen = plan

            joint_act, a1, a2, is_cen = get_rssda_action(
                policy, cen_dists_map, clustering, clustering_cen,
                local_step, belief_idx, [oh0, oh1], N_ACT_AGENT)
            if local_step >= len(policy) or joint_act < 0:
                missing_action_mass += mass
            if joint_act < 0:
                joint_act, a1, a2 = 0, 0, 0

            burn1, burn2 = split_joint_action(joint_act)
            n_burns = int(burn1 != 0) + int(burn2 != 0)

            expected_return += mass * float(R[joint_act, true_state])
            step_r, maneuver_r, deviation_r, terminal_r = reward_components(true_state, joint_act)
            expected_step_reward += mass * step_r
            expected_maneuver_reward += mass * maneuver_r
            expected_deviation_reward += mass * deviation_r
            expected_terminal_risk_reward += mass * terminal_r
            expected_dv += mass * dv_mag * n_burns
            expected_agent_burns += mass * n_burns
            stage_agent_burns[stage] += mass * n_burns
            stage_any_burns[stage] += mass * float(n_burns > 0)
            stage_action_probs[stage, joint_act] += mass

            if stage in sync_stages:
                expected_syncs += mass
            if stage == N_STAGES - 1:
                terminal_bin_probs[miss_bin] += mass

            c_ptr = -1
            if is_cen and local_step < len(cen_dists_map):
                dists_at_step = cen_dists_map[local_step]
                if belief_idx in dists_at_step:
                    c_ptr = dists_at_step.index(belief_idx)

            key_T = (joint_act, true_state)
            if key_T not in nonzero_T_cache:
                nz = np.flatnonzero(T[joint_act, true_state, :] > prune)
                nonzero_T_cache[key_T] = [(int(sp), float(T[joint_act, true_state, sp])) for sp in nz]

            key_terminal = (belief_idx, joint_act)
            if key_terminal not in terminal_cache:
                try:
                    terminal_cache[key_terminal] = {
                        int(o): int(d)
                        for o, _, d in sdec.get_terminal(belief_idx, joint_act)
                    }
                except KeyError:
                    terminal_cache[key_terminal] = {}
            obs_to_belief = terminal_cache[key_terminal]

            for next_state, p_state in nonzero_T_cache[key_T]:
                branch_mass = mass * p_state
                if branch_mass <= prune:
                    continue
                if next_state == SINK_STATE:
                    next_nodes[(next_state, belief_idx, oh0, oh1,
                                plan_stage, plan_belief, local_step + 1)] += branch_mass
                    continue

                key_O = (joint_act, next_state)
                if key_O not in nonzero_O_cache:
                    nz = np.flatnonzero(O[joint_act, next_state, :] > prune)
                    nonzero_O_cache[key_O] = [
                        (int(obs), float(O[joint_act, next_state, obs]))
                        for obs in nz
                    ]

                _, _, _, next_stage = index_to_state(next_state)
                for obs_joint, p_obs in nonzero_O_cache[key_O]:
                    obs_mass = branch_mass * p_obs
                    if obs_mass <= prune:
                        continue
                    next_belief = obs_to_belief.get(obs_joint, belief_idx)

                    if next_stage in sync_stages:
                        next_nodes[(next_state, next_belief, 0, 0,
                                    next_stage, next_belief, 0)] += obs_mass
                        continue

                    obs1 = obs_joint % obs_agent_size
                    obs2 = obs_joint // obs_agent_size
                    next_oh0, next_oh1 = update_oh_rssda(
                        policy, cen_dists_map, clustering, clustering_cen,
                        local_step, next_belief, [oh0, oh1], is_cen, c_ptr, obs1, obs2)
                    next_nodes[(next_state, next_belief, next_oh0, next_oh1,
                                plan_stage, plan_belief, local_step + 1)] += obs_mass

        nodes = {
            node: p for node, p in next_nodes.items()
            if p > prune and node[0] != SINK_STATE
        }

    terminal_total = float(terminal_bin_probs.sum())
    if terminal_total > 0:
        terminal_bin_probs /= terminal_total
    expected_miss_km = float(sum(
        terminal_bin_probs[mb] * bin_center_km(mb)
        for mb in range(N_MISS)
    ))

    summary_row = {
        "variant": None,
        "matrix_variant": None,
        "init_bin": true_init_bin,
        "eval_mode": "expected_interleaved",
        "expected_return": float(expected_return),
        "expected_step_reward": float(expected_step_reward),
        "expected_maneuver_reward": float(expected_maneuver_reward),
        "expected_deviation_reward": float(expected_deviation_reward),
        "expected_terminal_risk_reward": float(expected_terminal_risk_reward),
        "collision_prob": collision_probability_from_bin_probs(terminal_bin_probs),
        "expected_miss_km": expected_miss_km,
        "expected_dv_ms": float(expected_dv),
        "expected_syncs": float(expected_syncs),
        "expected_agent_burns": float(expected_agent_burns),
        "missing_action_mass": float(missing_action_mass),
    }
    for mb in range(N_MISS):
        summary_row[f"terminal_bin_{mb}_prob"] = float(terminal_bin_probs[mb])

    burn_rows = []
    action_rows = []
    for stage in range(N_STAGES):
        burn_rows.append({
            "variant": None,
            "matrix_variant": None,
            "init_bin": true_init_bin,
            "eval_mode": "expected_interleaved",
            "stage": stage,
            "mean_agent_burns": float(stage_agent_burns[stage]),
            "burn_stage_rate": float(stage_any_burns[stage]),
        })
        total_action_mass = float(stage_action_probs[stage, :].sum())
        probs = stage_action_probs[stage, :] / total_action_mass if total_action_mass > 0 else stage_action_probs[stage, :]
        for action in range(N_JOINT_ACTIONS):
            action_rows.append({
                "variant": None,
                "matrix_variant": None,
                "init_bin": true_init_bin,
                "eval_mode": "expected_interleaved",
                "stage": stage,
                "joint_action": action,
                "action_label": ACTION_LABELS[action].replace("\n", " "),
                "action_prob": float(probs[action]),
            })

    extra_solve_seconds = sum(
        t for key, t in solve_seconds_by_plan.items()
        if key != root_key
    )
    return (
        summary_row,
        burn_rows,
        action_rows,
        len(plan_cache) - 1,
        len(plan_cache),
        extra_solve_seconds,
        missing_action_mass,
    )


def maneuver_string(maneuvers) -> str:
    return ";".join(f"{k}:{a1}:{a2}" for k, a1, a2 in maneuvers)


def first_burn_stage(maneuvers):
    return "" if not maneuvers else min(k for k, _, _ in maneuvers)


def summarize_results(results: List[dict], label: str, matrix_variant: str,
                      init_bin: int, n_sync_states: int, policy_value: float,
                      solve_seconds: float, rollout_seconds: float,
                      dv_mag: float, rollout_mode: str) -> Dict[str, float]:
    misses = np.array([r["miss_km_at_tca"] for r in results], dtype=float)
    dvs = np.array([r["total_dv_ms"] for r in results], dtype=float)
    syncs = np.array([r["sync_count"] for r in results], dtype=float)
    terminal_bins = np.array([miss_to_bin(m) for m in misses], dtype=int)
    n = len(results)
    n_agent_burns = [
        sum((a1 != 0) + (a2 != 0) for _, a1, a2 in r["maneuver_stages"])
        for r in results
    ]

    row = {
        "variant": label,
        "matrix_variant": matrix_variant,
        "init_bin": init_bin,
        "n_rollouts": n,
        "rollout_mode": rollout_mode,
        "policy_value": policy_value,
        "solve_seconds": solve_seconds,
        "rollout_seconds": rollout_seconds,
        "sync_states": n_sync_states,
        "collision_rate_pct": 100.0 * float(np.mean(misses < 1.0)),
        "mean_miss_km": float(np.mean(misses)),
        "median_miss_km": float(np.median(misses)),
        "p05_miss_km": float(np.percentile(misses, 5)),
        "min_miss_km": float(np.min(misses)),
        "mean_dv_ms": float(np.mean(dvs)),
        "mean_syncs": float(np.mean(syncs)),
        "mean_agent_burns": float(np.mean(n_agent_burns)),
        "dv_mag": dv_mag,
    }
    for mb in range(N_MISS):
        row[f"terminal_bin_{mb}_count"] = int(np.sum(terminal_bins == mb))
    return row


def rollout_rows(results: List[dict], label: str, matrix_variant: str,
                 init_bin: int, policy_value: float, seed: int,
                 rollout_mode: str) -> List[dict]:
    rows = []
    for i, r in enumerate(results):
        miss = float(r["miss_km_at_tca"])
        maneuvers = r["maneuver_stages"]
        rows.append({
            "variant": label,
            "matrix_variant": matrix_variant,
            "init_bin": init_bin,
            "rollout": i,
            "rollout_mode": rollout_mode,
            "seed": seed,
            "policy_value": policy_value,
            "miss_km_at_tca": miss,
            "terminal_miss_bin": miss_to_bin(miss),
            "total_dv_ms": float(r["total_dv_ms"]),
            "sync_count": int(r["sync_count"]),
            "n_maneuver_stages": len(maneuvers),
            "first_burn_stage": first_burn_stage(maneuvers),
            "maneuvers": maneuver_string(maneuvers),
            "miss_bin_trajectory": ";".join(str(x) for x in r["miss_bin_trajectory"]),
        })
    return rows


def burn_timing_rows(results: List[dict], label: str, matrix_variant: str,
                     init_bin: int, rollout_mode: str) -> List[dict]:
    stage_agent_burns = np.zeros(N_STAGES, dtype=float)
    stage_any_burns = np.zeros(N_STAGES, dtype=float)
    for r in results:
        for k, a1, a2 in r["maneuver_stages"]:
            stage_agent_burns[k] += (a1 != 0) + (a2 != 0)
            stage_any_burns[k] += 1
    n = max(len(results), 1)
    return [
        {
            "variant": label,
            "matrix_variant": matrix_variant,
            "init_bin": init_bin,
            "rollout_mode": rollout_mode,
            "stage": k,
            "mean_agent_burns": stage_agent_burns[k] / n,
            "burn_stage_rate": stage_any_burns[k] / n,
        }
        for k in range(N_STAGES)
    ]


def write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary_rows: List[dict], fig_dir: str, tag: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)
    variants = list(dict.fromkeys(row["variant"] for row in summary_rows))
    init_bins = sorted(set(int(row["init_bin"]) for row in summary_rows))
    colors = {
        "Centralized": "#3b6ea8",
        "SDec": "#2f8f5b",
        "Decentralized": "#c46a32",
        "Centralized TI1": "#7da6d6",
        "SDec TI1": "#62b985",
        "Decentralized TI1": "#d99a6d",
    }
    expected_mode = summary_rows and "expected_return" in summary_rows[0]
    if expected_mode:
        metrics = [
            ("expected_return", "Expected return"),
            ("collision_prob", "Collision probability (%)"),
            ("expected_dv_ms", "Expected total dv (m/s)"),
            ("expected_syncs", "Expected sync events"),
        ]
    else:
        metrics = [
            ("collision_rate_pct", "Collision rate (%)"),
            ("mean_miss_km", "Mean miss at TCA (km)"),
            ("mean_dv_ms", "Mean total dv (m/s)"),
            ("mean_syncs", "Mean sync events"),
        ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = np.arange(len(init_bins))
    width = 0.78 / max(len(variants), 1)
    by_key = {
        (row["variant"], int(row["init_bin"])): row
        for row in summary_rows
    }

    for ax, (metric, title) in zip(axes.flat, metrics):
        for j, variant in enumerate(variants):
            vals = [
                float(by_key[(variant, b)][metric])
                if (variant, b) in by_key else 0.0
                for b in init_bins
            ]
            if expected_mode and metric == "collision_prob":
                vals = [100.0 * v for v in vals]
            ax.bar(
                x + (j - (len(variants) - 1) / 2) * width,
                vals,
                width=width,
                label=variant,
                color=colors.get(variant),
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in init_bins])
        ax.set_xlabel("True initial miss bin")
        ax.grid(axis="y", alpha=0.25)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(variants))
    fig.suptitle("Spacecraft CA Policy Comparison", fontsize=14)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    out = os.path.join(fig_dir, f"variant_comparison_{tag}.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_burn_timing(burn_rows: List[dict], fig_dir: str, tag: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)
    variants = list(dict.fromkeys(row["variant"] for row in burn_rows))
    init_bins = sorted(set(int(row["init_bin"]) for row in burn_rows))
    colors = {
        "Centralized": "#3b6ea8",
        "SDec": "#2f8f5b",
        "Decentralized": "#c46a32",
        "Centralized TI1": "#7da6d6",
        "SDec TI1": "#62b985",
        "Decentralized TI1": "#d99a6d",
    }

    fig, axes = plt.subplots(len(init_bins), 1, figsize=(11, 3.0 * len(init_bins)),
                             sharex=True, squeeze=False)
    by_key = {
        (row["variant"], int(row["init_bin"]), int(row["stage"])): row
        for row in burn_rows
    }
    stages = np.arange(N_STAGES)

    for ax, init_bin in zip(axes.flat, init_bins):
        for variant in variants:
            vals = [
                float(by_key[(variant, init_bin, k)]["mean_agent_burns"])
                if (variant, init_bin, k) in by_key else 0.0
                for k in stages
            ]
            ax.plot(stages, vals, marker="o", label=variant,
                    color=colors.get(variant))
        ax.set_title(f"Initial miss bin {init_bin}")
        ax.set_ylabel("Agent burns / rollout")
        ax.grid(axis="y", alpha=0.25)

    axes.flat[-1].set_xlabel("Decision stage")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(variants))
    fig.suptitle("Burn Timing by Policy", fontsize=14)
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    out = os.path.join(fig_dir, f"burn_timing_{tag}.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_action_schedule(action_rows: List[dict], fig_dir: str, tag: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    import matplotlib.patches as mpatches

    os.makedirs(fig_dir, exist_ok=True)
    variants = list(dict.fromkeys(row["variant"] for row in action_rows))
    init_bins = sorted(set(int(row["init_bin"]) for row in action_rows))
    best = {}
    for init_bin in init_bins:
        for variant in variants:
            for stage in range(N_STAGES):
                candidates = [
                    row for row in action_rows
                    if row["variant"] == variant
                    and int(row["init_bin"]) == init_bin
                    and int(row["stage"]) == stage
                ]
                if not candidates:
                    continue
                row = max(candidates, key=lambda r: float(r["action_prob"]))
                best[(init_bin, variant, stage)] = (
                    int(row["joint_action"]),
                    float(row["action_prob"]),
                )

    fig, axes = plt.subplots(len(init_bins), 1, figsize=(12, 2.3 * len(init_bins)),
                             squeeze=False)
    for ax, init_bin in zip(axes.flat, init_bins):
        image = np.ones((len(variants), N_STAGES, 3), dtype=float)
        labels = {}
        for y, variant in enumerate(variants):
            for stage in range(N_STAGES):
                action, prob = best.get((init_bin, variant, stage), (0, 0.0))
                image[y, stage, :] = to_rgb(ACTION_COLORS[action])
                labels[(y, stage)] = (ACTION_LABELS[action], prob)

        ax.imshow(image, aspect="auto")
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels(variants)
        ax.set_xticks(range(N_STAGES))
        ax.set_xticklabels([str(k) for k in range(N_STAGES)])
        ax.set_title(f"Dominant joint action by stage, true init bin {init_bin}")
        ax.set_xlabel("Decision stage")
        for y in range(len(variants)):
            for stage in range(N_STAGES):
                label, prob = labels[(y, stage)]
                text = label if prob >= 0.995 else f"{label}\n{prob:.2f}"
                ax.text(stage, y, text, ha="center", va="center",
                        fontsize=7, color="black")
        ax.set_xticks(np.arange(-0.5, N_STAGES, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(variants), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)

    legend_items = [
        mpatches.Patch(color=ACTION_COLORS[a], label=ACTION_LABELS[a].replace("\n", "/"))
        for a in range(N_JOINT_ACTIONS)
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=5, fontsize=8)
    fig.suptitle("Policy Action Schedule", fontsize=14)
    fig.tight_layout(rect=[0, 0.12, 1, 0.94])
    out = os.path.join(fig_dir, f"action_schedule_{tag}.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def print_summary_table(rows: List[dict]) -> None:
    print("\n" + "=" * 96)
    print(f"{'Variant':<16} {'Init':>4} {'Return':>10} {'Coll%':>7} "
          f"{'Exp miss':>11} {'Exp dv':>9} {'Syncs':>7} {'Burns':>7}")
    print("-" * 96)
    for r in rows:
        if "expected_return" in r:
            ret = float(r["expected_return"])
            coll = 100.0 * float(r["collision_prob"])
            miss = float(r["expected_miss_km"])
            dv = float(r["expected_dv_ms"])
            syncs = float(r["expected_syncs"])
            burns = float(r["expected_agent_burns"])
        else:
            ret = float(r.get("policy_value", 0.0))
            coll = float(r["collision_rate_pct"])
            miss = float(r["mean_miss_km"])
            dv = float(r["mean_dv_ms"])
            syncs = float(r["mean_syncs"])
            burns = float(r["mean_agent_burns"])
        print(f"{r['variant']:<16} {int(r['init_bin']):>4} "
              f"{ret:>10.2f} {coll:>7.1f} {miss:>11.3f} "
              f"{dv:>9.3f} {syncs:>7.2f} {burns:>7.2f}")
    print("=" * 96)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=int, default=100)
    parser.add_argument("--eval-mode", choices=["expected", "rollout", "both"],
                        default="expected")
    parser.add_argument("--rollout-mode", choices=["abstract", "physical"],
                        default="abstract",
                        help="abstract is fast POMDP-model simulation; physical uses Brahe")
    parser.add_argument("--variants", default="centralized,sdec,dec",
                        help="Comma-separated subset: centralized,sdec,dec")
    parser.add_argument("--solver-modes", type=parse_solver_modes,
                        default=parse_solver_modes("fixed"),
                        help="Comma-separated solver modes: fixed, interleaved, or both")
    parser.add_argument("--belief-bins", type=parse_bins, default=parse_bins("0,1,2"),
                        help="Comma-separated initial belief support bins")
    parser.add_argument("--eval-bins", type=parse_bins, default=parse_bins("0,1,2"),
                        help="Comma-separated true initial bins to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dv", type=float, default=None,
                        help="Require cached matrices to use this dv")
    parser.add_argument("--fixed-init", action="store_true",
                        help="Physical mode only: use bin centers and precomputed trajectory trees")
    parser.add_argument("--iter-limit", type=int, default=10000)
    parser.add_argument("--max-solve-seconds", type=float, default=600.0,
                        help="Abort after a variant solve exceeds this wall-clock cap")
    parser.add_argument("--max-clusters", type=int, default=None)
    parser.add_argument("--rec-limit", type=int, default=None)
    parser.add_argument("--memory-limit-gb", type=float, default=16.0)
    parser.add_argument("--memory-check-interval", type=int, default=100)
    parser.add_argument("--dec-solver", choices=["rsmaa", "rssda"], default="rsmaa",
                        help="Solver for the fully decentralized variant")
    parser.add_argument("--rsmaa-cluster-type", default="lossless")
    parser.add_argument("--rsmaa-maxit", type=int, default=200)
    parser.add_argument("--rsmaa-q-depth", type=int, default=3)
    parser.add_argument("--rsmaa-alpha", type=float, default=0.2)
    parser.add_argument("--rsmaa-memory", type=int, default=None)
    parser.add_argument("--rsmaa-heuristic", default="MDP", choices=["MDP", "POMDP"])
    parser.add_argument("--rsmaa-rec-type", default="MDP",
                        choices=["max_reward", "MDP", "rec_state", "recursive"])
    parser.add_argument("--p-threshold-cluster", type=float, default=0.0)
    parser.add_argument("--p-threshold-expand", type=float, default=0.0)
    parser.add_argument("--verbose-rsmaa", action="store_true")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--out-dir", default=os.path.join(_HERE, "notes", "results"))
    parser.add_argument("--fig-dir", default=os.path.join(_HERE, "notes", "figures"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    initialize_eop()

    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    init_b = make_init_b(args.belief_bins)
    variants = filter_variants(
        default_variants(),
        [part for part in args.variants.split(",") if part.strip()],
    )

    print("Spacecraft CA policy comparison")
    print(f"  belief_bins={args.belief_bins}")
    print(f"  eval_bins={args.eval_bins}")
    print(f"  variants={[v.label for v in variants]}")
    print(f"  solver_modes={[m.label for m in args.solver_modes]}")
    print(f"  rollouts={args.rollouts}, mode={args.rollout_mode}, "
          f"fixed_init={args.fixed_init}, seed={args.seed}")

    all_rollout_rows = []
    all_rollout_summary_rows = []
    all_expected_rows = []
    all_burn_rows = []
    all_action_rows = []
    cached_dvs = []

    for spec in variants:
        T, O, R, _, _, cached_dv = load_matrices(spec.matrix_variant)
        obs_agent_size = N_OBS_AGENT
        if spec.compress_dec_obs:
            if args.rollout_mode == "physical":
                raise SystemExit(
                    "Physical rollouts with compressed decentralized observations "
                    "are not supported. Use --rollout-mode abstract, or run "
                    "--variants centralized,sdec for physical validation."
                )
            O = compact_decentralized_observations(O)
            obs_agent_size = N_DEV
        cached_dvs.append(cached_dv)
        if args.dv is not None and abs(cached_dv - args.dv) > 1e-9:
            raise SystemExit(
                f"{spec.matrix_variant} cache uses dv={cached_dv}, "
                f"but --dv={args.dv}. Rebuild matrices first."
            )
        dv_mag = args.dv if args.dv is not None else cached_dv

        for mode in args.solver_modes:
            if mode.interleaved and args.eval_mode in ("rollout", "both"):
                raise SystemExit(
                    "Interleaved TI1 replanning is currently implemented for "
                    "expected evaluation. Use --eval-mode expected."
                )
            use_rsmaa_dec = (
                spec.label == "Decentralized" and
                args.dec_solver == "rsmaa"
            )
            if use_rsmaa_dec and mode.interleaved:
                print(
                    "\nSkipping Decentralized TI1: fully decentralized "
                    "baseline uses RS-MAA* with no sync/replanning.",
                    flush=True,
                )
                continue
            if use_rsmaa_dec and args.eval_mode in ("rollout", "both"):
                raise SystemExit(
                    "RS-MAA* Dec integration currently supports expected "
                    "evaluation. Use --eval-mode expected."
                )

            variant_label = display_variant_label(spec, mode)
            print(f"\nSolving {variant_label} ({spec.matrix_variant}, "
                  f"TI1={mode.ti1_enabled}, "
                  f"solver={'RS-MAA*' if use_rsmaa_dec else 'RS-SDA*'})...",
                  flush=True)
            if use_rsmaa_dec:
                sdec = None
                try:
                    rsmaa_solver, full_result, n_sync_states, solve_seconds = solve_dec_rsmaa(
                        T, O, R, init_b, args
                    )
                except RsmaaMemoryLimitExceeded as exc:
                    raise SystemExit(f"Decentralized RS-MAA* memory limit exceeded: {exc}") from exc
            else:
                sdec, full_result, n_sync_states, solve_seconds = solve_variant(
                    spec, T, O, R, init_b, obs_agent_size,
                    iter_limit=args.iter_limit,
                    max_clusters=args.max_clusters,
                    rec_limit=args.rec_limit,
                    ti1_enabled=mode.ti1_enabled,
                    memory_limit_gb=args.memory_limit_gb,
                    memory_check_interval=args.memory_check_interval,
                )
            policy_value = float(full_result[0])
            plen = policy_length(full_result)
            print(f"  value={policy_value:.4f}, policy_len={plen}, "
                  f"sync_states={n_sync_states}, solve={solve_seconds:.2f}s", flush=True)
            if solve_seconds > args.max_solve_seconds:
                raise SystemExit(
                    f"{variant_label} solve exceeded --max-solve-seconds="
                    f"{args.max_solve_seconds:.1f}s after returning "
                    f"({solve_seconds:.1f}s). Reduce --iter-limit or run fewer variants."
                )

            for init_bin in args.eval_bins:
                run_seed = args.seed + 1009 * init_bin
                if args.eval_mode in ("expected", "both"):
                    print(f"  Expected eval: init_bin={init_bin}", flush=True)
                    if use_rsmaa_dec:
                        expected_row, burn_rows, action_rows = expected_rsmaa_policy_metrics(
                            T, O, R, full_result,
                            true_init_bin=init_bin,
                            dv_mag=dv_mag,
                        )
                        replan_count = 0
                        plan_count = 1
                        interleaved_solve_seconds = 0.0
                        missing_action_mass = expected_row.get("missing_action_mass", 0.0)
                    elif mode.interleaved:
                        (
                            expected_row,
                            burn_rows,
                            action_rows,
                            replan_count,
                            plan_count,
                            interleaved_solve_seconds,
                            missing_action_mass,
                        ) = expected_interleaved_policy_metrics(
                            T, O, R, sdec, full_result, spec.sync_stages,
                            true_init_bin=init_bin,
                            dv_mag=dv_mag,
                            obs_agent_size=obs_agent_size,
                            initial_solve_seconds=solve_seconds,
                        )
                    else:
                        expected_row, burn_rows, action_rows = expected_policy_metrics(
                            T, O, R, sdec, full_result, spec.sync_stages,
                            true_init_bin=init_bin,
                            dv_mag=dv_mag,
                            obs_agent_size=obs_agent_size,
                        )
                        replan_count = 0
                        plan_count = 1
                        interleaved_solve_seconds = 0.0
                        missing_action_mass = expected_row.get("missing_action_mass", 0.0)

                    total_solve_seconds = solve_seconds + interleaved_solve_seconds
                    if total_solve_seconds > args.max_solve_seconds:
                        raise SystemExit(
                            f"{variant_label} total solve time exceeded "
                            f"--max-solve-seconds={args.max_solve_seconds:.1f}s "
                            f"after replanning ({total_solve_seconds:.1f}s)."
                        )

                    expected_row.update({
                        "variant": variant_label,
                        "base_variant": spec.label,
                        "matrix_variant": spec.matrix_variant,
                        "policy_value": policy_value,
                        "solve_seconds": total_solve_seconds,
                        "sync_states": n_sync_states,
                        "dv_mag": dv_mag,
                    })
                    add_solver_metadata(
                        expected_row, mode, full_result, solve_seconds,
                        replan_count=replan_count,
                        plan_count=plan_count,
                        interleaved_solve_seconds=interleaved_solve_seconds,
                        missing_action_mass=missing_action_mass,
                    )
                    if use_rsmaa_dec:
                        expected_row.update({
                            "solver_mode": "rsmaa",
                            "ti1_enabled": False,
                            "interleaved_replanning": False,
                            "policy_len": plen,
                            "policy_complete": bool(plen >= N_STAGES),
                            "initial_solve_seconds": float(solve_seconds),
                            "interleaved_solve_seconds": 0.0,
                            "replan_count": 0,
                            "plan_count": 1,
                            "missing_action_mass": float(missing_action_mass),
                            "rsmaa_cluster_type": args.rsmaa_cluster_type,
                            "rsmaa_heuristic": args.rsmaa_heuristic,
                            "rsmaa_rec_type": args.rsmaa_rec_type,
                        })
                    for row in burn_rows:
                        row.update({
                            "variant": variant_label,
                            "base_variant": spec.label,
                            "matrix_variant": spec.matrix_variant,
                            "solver_mode": "rsmaa" if use_rsmaa_dec else mode.label,
                            "ti1_enabled": False if use_rsmaa_dec else bool(mode.ti1_enabled),
                        })
                    for row in action_rows:
                        row.update({
                            "variant": variant_label,
                            "base_variant": spec.label,
                            "matrix_variant": spec.matrix_variant,
                            "solver_mode": "rsmaa" if use_rsmaa_dec else mode.label,
                            "ti1_enabled": False if use_rsmaa_dec else bool(mode.ti1_enabled),
                        })
                    all_expected_rows.append(expected_row)
                    all_burn_rows.extend(burn_rows)
                    all_action_rows.extend(action_rows)

                if args.eval_mode in ("rollout", "both"):
                    print(f"  Rollouts: init_bin={init_bin}, n={args.rollouts}, "
                          f"seed={run_seed}", flush=True)
                    t0 = time.perf_counter()
                    if args.rollout_mode == "abstract":
                        results = run_abstract_simulation(
                            T, O, R, sdec, full_result, spec.sync_stages,
                            n_rollouts=args.rollouts,
                            init_miss_bin=init_bin,
                            dv_mag=dv_mag,
                            seed=run_seed,
                            obs_agent_size=obs_agent_size,
                        )
                    else:
                        results = run_simulation(
                            T, O, R, init_b, spec.sync_stages,
                            n_rollouts=args.rollouts,
                            init_miss_bin=init_bin,
                            dv_mag=dv_mag,
                            seed=run_seed,
                            verbose=False,
                            policy_mode="rssda",
                            sdec=sdec,
                            full_result=full_result,
                            fixed_init=args.fixed_init,
                        )
                    rollout_seconds = time.perf_counter() - t0

                    all_rollout_summary_rows.append(summarize_results(
                        results, variant_label, spec.matrix_variant, init_bin,
                        n_sync_states, policy_value, solve_seconds, rollout_seconds,
                        dv_mag, args.rollout_mode,
                    ))
                    all_rollout_rows.extend(rollout_rows(
                        results, variant_label, spec.matrix_variant, init_bin,
                        policy_value, run_seed, args.rollout_mode,
                    ))
                    all_burn_rows.extend(burn_timing_rows(
                        results, variant_label, spec.matrix_variant, init_bin,
                        args.rollout_mode,
                    ))

    if args.dv is None and len({round(x, 12) for x in cached_dvs}) != 1:
        raise SystemExit(f"Matrix caches have inconsistent dv values: {cached_dvs}")

    os.makedirs(args.out_dir, exist_ok=True)
    expected_path = os.path.join(args.out_dir, f"variant_expected_{tag}.csv")
    action_path = os.path.join(args.out_dir, f"variant_action_by_stage_{tag}.csv")
    summary_path = os.path.join(args.out_dir, f"variant_rollout_summary_{tag}.csv")
    rollouts_path = os.path.join(args.out_dir, f"variant_rollouts_{tag}.csv")
    burn_path = os.path.join(args.out_dir, f"variant_burn_timing_{tag}.csv")
    write_csv(expected_path, all_expected_rows)
    write_csv(action_path, all_action_rows)
    write_csv(summary_path, all_rollout_summary_rows)
    write_csv(rollouts_path, all_rollout_rows)
    write_csv(burn_path, all_burn_rows)

    figure_paths = []
    if not args.no_plots:
        summary_for_plot = all_expected_rows if all_expected_rows else all_rollout_summary_rows
        if summary_for_plot:
            figure_paths.append(plot_summary(summary_for_plot, args.fig_dir, tag))
        if all_burn_rows:
            figure_paths.append(plot_burn_timing(all_burn_rows, args.fig_dir, tag))
        if all_action_rows:
            figure_paths.append(plot_action_schedule(all_action_rows, args.fig_dir, tag))

    print_summary_table(all_expected_rows if all_expected_rows else all_rollout_summary_rows)
    print("\nWrote:")
    for path, rows in [
        (expected_path, all_expected_rows),
        (action_path, all_action_rows),
        (summary_path, all_rollout_summary_rows),
        (rollouts_path, all_rollout_rows),
        (burn_path, all_burn_rows),
    ]:
        if rows:
            print(f"  {path}")
    for path in figure_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
