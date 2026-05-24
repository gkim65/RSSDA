"""
spacecraft_simulator.py

Closed-loop simulation of the spacecraft CA SDec-POMDP policy.

Supports two policy modes:
  - "rssda": Full RS-SDA* policy from multi_agent_astar (default)
  - "greedy": One-step lookahead greedy (baseline/fallback)

Each rollout:
  - Initializes true ECI states from a radial-miss conjunction
  - Steps through N_STAGES decision epochs, applying the policy
  - At each stage: simulates noisy observation, updates belief, takes action
  - At TCA: measures true miss distance with Brahe

Usage:
  python spacecraft_simulator.py                          # RS-SDA* policy, bin 4 init
  python spacecraft_simulator.py --policy greedy          # greedy baseline
  python spacecraft_simulator.py --init-miss-bin 0        # start in collision zone
  python spacecraft_simulator.py --rollouts 100 --dv 0.5
  python spacecraft_simulator.py --compare                # all 3 variants side by side
"""

import os
import sys
import argparse
import math
import numpy as np
from typing import List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCHMARKS = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_BENCHMARKS)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _BENCHMARKS)
sys.path.insert(0, _HERE)

from brahe import (
    initialize_eop,
    state_rtn_to_eci, state_eci_to_rtn,
    par_propagate_to,
)

from spacecraft_matrices import (
    load_matrices, DV_MAGNITUDE,
    STAGE_EPOCHS, EPOCH_TCA, V_REL_MS,
    N_JOINT_ACTIONS, N_ACT_AGENT, N_BURN_AGENT, N_OBS_AGENT,
    MISS_NULL_OBS, TLE_SIGMA_KM, GPS_SIGMA_KM,
    split_joint_action, apply_maneuver, propagate, sc1_eci_at_tca,
    make_prop, CONTACT_STAGES, joint_obs_index, local_obs_index,
)
from spacecraft_discretizer import (
    N_STAGES, N_STATES, N_STATES_TOTAL, SINK_STATE,
    N_MISS, N_DEV, DEV_ZERO, MISS_EDGES_KM,
    miss_to_bin, bin_center_km,
    state_index, index_to_state, update_dev_bin,
    sync_trigger_states, miss_bin_label,
)
from RSSDA import int_tuple

ACT_NAMES = {0: 'WAIT', 1: '+dVT', 2: '-dVT'}
MISS_LABELS = [miss_bin_label(i) for i in range(N_MISS)]


# ---------------------------------------------------------------------------
# Trajectory tree precomputation (--fixed-init mode)
# ---------------------------------------------------------------------------

def precompute_traj_tree(
    sdec, full_result,
    true_init_miss_km: float,
    dv_mag: float,
) -> dict:
    """
    Walk the policy tree via BFS, propagate all reachable ECI states and TCA
    miss distances upfront using par_propagate_to. Returns a traj_cache dict:
      cache[action_prefix_tuple]         -> (sc1_eci, sc2_eci) at that stage
      cache[('tca', action_prefix_tuple)] -> true_miss_km at TCA from that node
    All rollouts from the same init condition can then run with zero Brahe calls.
    """
    value, policy, clustering, cent_vector, cen_dists_map, clustering_cen = full_result

    traj_cache = {}

    # --- Root: SC1/SC2 ECI at stage 0 ---
    sc1_tca_eci = sc1_eci_at_tca()
    sc2_tca = np.array(state_rtn_to_eci(
        sc1_tca_eci,
        np.array([true_init_miss_km * 1e3, 0.0, 0.0, 0.0, -V_REL_MS, 0.0])
    ))
    sc1_s0 = propagate(EPOCH_TCA, sc1_tca_eci, STAGE_EPOCHS[0])
    sc2_s0 = propagate(EPOCH_TCA, sc2_tca,     STAGE_EPOCHS[0])
    traj_cache[()] = (sc1_s0, sc2_s0)

    # BFS state: (belief_idx, oh0, oh1, action_prefix)
    root_belief = sdec.dist_dict[int_tuple(sdec.init_beliefs)]
    frontier = [(root_belief, 0, 0, ())]

    for k in range(N_STAGES - 1):
        # Collect all (parent_prefix, action, sc1_post, sc2_post) for this stage
        to_propagate = []   # (child_prefix, sc1_post_eci, sc2_post_eci)
        next_frontier = []

        seen_child_prefixes = set()

        for (b_idx, oh0, oh1, prefix) in frontier:
            joint_act, a1, a2, is_cen = get_rssda_action(
                policy, cen_dists_map, clustering, clustering_cen,
                k, b_idx, [oh0, oh1], N_ACT_AGENT)
            if joint_act < 0:
                joint_act, a1, a2 = 0, 0, 0

            child_prefix = prefix + (joint_act,)
            sc1_cur, sc2_cur = traj_cache[prefix]

            if child_prefix not in seen_child_prefixes:
                seen_child_prefixes.add(child_prefix)
                burn1_t, burn2_t = a1, a2
                sc1_post = apply_maneuver(sc1_cur, burn1_t, dv=dv_mag)
                sc2_post = apply_maneuver(sc2_cur, burn2_t, dv=dv_mag)
                to_propagate.append((child_prefix, sc1_post, sc2_post))

            # Expand observation branches for next frontier
            c_ptr = -1
            if is_cen and k < len(cen_dists_map) and b_idx in cen_dists_map[k]:
                c_ptr = cen_dists_map[k].index(b_idx)
            try:
                sparse = sdec.get_terminal(b_idx, joint_act)
                for o, p, d in sparse:
                    o1, o2 = o % N_OBS_AGENT, o // N_OBS_AGENT
                    try:
                        if is_cen and c_ptr >= 0:
                            no0 = clustering_cen[k][0][c_ptr][o1]
                            no1 = clustering_cen[k][1][c_ptr][o2]
                        else:
                            no0 = clustering[k][0][oh0][o1]
                            no1 = clustering[k][1][oh1][o2]
                        if no0 < 0: no0 = 0
                        if no1 < 0: no1 = 0
                    except (IndexError, TypeError, KeyError):
                        no0, no1 = 0, 0
                    next_frontier.append((d, no0, no1, child_prefix))
            except KeyError:
                next_frontier.append((b_idx, oh0, oh1, child_prefix))

        # Batch propagate all child nodes at this stage to STAGE_EPOCHS[k+1]
        if to_propagate:
            props_sc1 = [make_prop(STAGE_EPOCHS[k], sc1) for _, sc1, _ in to_propagate]
            props_sc2 = [make_prop(STAGE_EPOCHS[k], sc2) for _, _, sc2 in to_propagate]
            par_propagate_to(props_sc1 + props_sc2, STAGE_EPOCHS[k + 1])
            for i, (child_prefix, _, _) in enumerate(to_propagate):
                traj_cache[child_prefix] = (
                    np.array(props_sc1[i].current_state()[:6]),
                    np.array(props_sc2[i].current_state()[:6]),
                )

        frontier = next_frontier

    # Batch propagate all cached nodes to EPOCH_TCA to get true miss at each node
    all_prefixes = [p for p in traj_cache if isinstance(p, tuple) and
                    (len(p) == 0 or isinstance(p[0], int))]
    for k in range(N_STAGES):
        nodes_at_k = [p for p in all_prefixes if len(p) == k]
        if not nodes_at_k:
            continue
        props_sc1 = [make_prop(STAGE_EPOCHS[k], traj_cache[p][0]) for p in nodes_at_k]
        props_sc2 = [make_prop(STAGE_EPOCHS[k], traj_cache[p][1]) for p in nodes_at_k]
        par_propagate_to(props_sc1 + props_sc2, EPOCH_TCA)
        for i, p in enumerate(nodes_at_k):
            rtn = np.array(state_eci_to_rtn(
                np.array(props_sc1[i].current_state()[:6]),
                np.array(props_sc2[i].current_state()[:6]),
            ))
            traj_cache[('tca', p)] = np.linalg.norm(rtn[:3]) / 1e3

    return traj_cache


# ---------------------------------------------------------------------------
# Greedy policy (baseline)
# ---------------------------------------------------------------------------

def build_greedy_policy(T, R) -> np.ndarray:
    """One-step lookahead greedy: argmax_a [R(a,s) + max_a' R(a', T(s,a))]."""
    policy = np.zeros(N_STATES, dtype=int)
    for s in range(N_STATES):
        best_a, best_q = 0, -1e9
        for a in range(N_JOINT_ACTIONS):
            r = R[a, s]
            next_states = np.where(T[a, s, :] > 0)[0]
            r_next = max((R[:, sp].max() for sp in next_states if sp < N_STATES), default=0.0)
            q = r + r_next
            if q > best_q:
                best_q = q
                best_a = a
        policy[s] = best_a
    return policy


# ---------------------------------------------------------------------------
# Observation simulation
# ---------------------------------------------------------------------------

def simulate_obs(true_miss_km: float, sigma_km: float = None,
                 rng: np.random.Generator = None) -> int:
    if sigma_km is None:
        sigma_km = TLE_SIGMA_KM
    if rng is None:
        rng = np.random.default_rng()
    noisy = max(0.0, rng.normal(true_miss_km, sigma_km))
    return miss_to_bin(noisy)


# ---------------------------------------------------------------------------
# Belief update (for greedy mode)
# ---------------------------------------------------------------------------

def belief_update(belief: np.ndarray, action: int, obs_joint: int,
                  T: np.ndarray, O: np.ndarray) -> np.ndarray:
    b_new = O[action, :, obs_joint] * (T[action].T @ belief)
    total = b_new.sum()
    return b_new / total if total > 1e-15 else belief.copy()


# ---------------------------------------------------------------------------
# RS-SDA* policy action extraction
# ---------------------------------------------------------------------------

def get_rssda_action(policy, cen_dists_map, clustering, clustering_cen,
                     step: int, current_belief_idx: int, current_oh: List[int],
                     act_per_agent: int):
    """
    Extract the joint action from the RS-SDA* policy at a given step.
    Returns (joint_act, a1, a2, is_centralized).
    Mirrors the logic in sdec_mars.py.
    """
    is_centralized = False
    c_ptr = -1

    if step < len(cen_dists_map):
        dists_at_step = cen_dists_map[step]
        if current_belief_idx in dists_at_step:
            is_centralized = True
            c_ptr = dists_at_step.index(current_belief_idx)

    try:
        if not is_centralized:
            act1 = policy[step][0][0][current_oh[0]]
            act2 = policy[step][0][1][current_oh[1]]
            joint_act = act1 + act2 * act_per_agent
        else:
            joint_act = policy[step][1][c_ptr][0]
            act1 = joint_act % act_per_agent
            act2 = joint_act // act_per_agent
    except (IndexError, TypeError):
        # Policy doesn't cover this branch — fall back to WAIT
        joint_act, act1, act2 = 0, 0, 0

    return joint_act, act1, act2, is_centralized


def update_oh_rssda(policy, cen_dists_map, clustering, clustering_cen,
                    step: int, current_belief_idx: int, current_oh: List[int],
                    is_centralized: bool, c_ptr: int, o1: int, o2: int) -> List[int]:
    """Update observation history indices after taking action and receiving obs."""
    try:
        if is_centralized:
            if len(clustering_cen) > step and len(clustering_cen[step]) > 0:
                new_oh_0 = clustering_cen[step][0][c_ptr][o1]
                new_oh_1 = clustering_cen[step][1][c_ptr][o2]
                return [new_oh_0, new_oh_1]
            return [0, 0]
        else:
            next_oh0 = clustering[step][0][current_oh[0]][o1]
            next_oh1 = clustering[step][1][current_oh[1]][o2]
            if next_oh0 == -1 or next_oh1 == -1:
                return [0, 0]
            return [next_oh0, next_oh1]
    except (IndexError, TypeError, KeyError):
        return [0, 0]


# ---------------------------------------------------------------------------
# Single rollout — RS-SDA* policy
# ---------------------------------------------------------------------------

def rollout_rssda(
    T, O, R, sdec,
    full_result,          # (value, policy, clustering, cent_vector, cen_dists_map, clustering_cen)
    contact_stages: List[int],
    true_init_miss_km: float,
    dv_mag: float,
    rng: np.random.Generator,
    traj_cache: Optional[dict] = None,
    # Sync is determined entirely by is_cen from RSSDA's cen_dists_map (sync_states mechanism).
) -> dict:
    """
    Closed-loop rollout using the full RS-SDA* policy.

    traj_cache: shared dict keyed on action-history tuples, mapping to (sc1_eci, sc2_eci)
    at each stage. Built lazily across rollouts — if all rollouts share the same
    true_init_miss_km, propagations are computed once and reused.
    """
    value, policy, clustering, cent_vector, cen_dists_map, clustering_cen = full_result

    if traj_cache is None:
        traj_cache = {}

    # Root node: sc1/sc2 ECI at stage 0 for this init condition.
    root_key = ()
    if root_key not in traj_cache:
        sc1_tca_eci = sc1_eci_at_tca()
        sc1_s0 = propagate(EPOCH_TCA, sc1_tca_eci, STAGE_EPOCHS[0])
        sc2_tca = np.array(state_rtn_to_eci(sc1_tca_eci,
                           np.array([true_init_miss_km * 1e3, 0.0, 0.0, 0.0, -V_REL_MS, 0.0])))
        sc2_s0 = propagate(EPOCH_TCA, sc2_tca, STAGE_EPOCHS[0])
        traj_cache[root_key] = (sc1_s0, sc2_s0)

    current_belief_idx = sdec.dist_dict[int_tuple(sdec.init_beliefs)]
    current_oh = [0, 0]
    dev1_bin = DEV_ZERO
    dev2_bin = DEV_ZERO

    total_dv = 0.0
    sync_count = 0
    maneuver_stages = []
    miss_bin_traj = []
    action_history = []  # joint actions taken so far — cache key prefix

    for k in range(N_STAGES):
        # Look up current ECI state from cache (keyed on actions taken to reach stage k)
        cache_key = tuple(action_history)
        sc1_true, sc2_true = traj_cache[cache_key]

        # True miss at TCA: propagate forward to TCA for measurement.
        # Only do this once per unique cache_key (shared across rollouts).
        tca_key = ('tca', cache_key)
        if tca_key not in traj_cache:
            sc1_proj = propagate(STAGE_EPOCHS[k], sc1_true, EPOCH_TCA)
            sc2_proj = propagate(STAGE_EPOCHS[k], sc2_true, EPOCH_TCA)
            rtn_tca  = np.array(state_eci_to_rtn(sc1_proj, sc2_proj))
            traj_cache[tca_key] = np.linalg.norm(rtn_tca[:3]) / 1e3
        true_miss_km = traj_cache[tca_key]
        miss_bin_traj.append(miss_to_bin(true_miss_km))

        # Choose action first (obs depends on sync flag in chosen action)
        joint_act, a1, a2, is_cen = get_rssda_action(
            policy, cen_dists_map, clustering, clustering_cen,
            k, current_belief_idx, current_oh, N_ACT_AGENT)

        c_ptr = -1
        if is_cen and k < len(cen_dists_map):
            dists_at_step = cen_dists_map[k]
            if current_belief_idx in dists_at_step:
                c_ptr = dists_at_step.index(current_belief_idx)

        if joint_act < 0:
            joint_act, a1, a2 = 0, 0, 0

        burn1, burn2 = a1, a2
        at_contact = k in contact_stages
        # A variant has sync at every contact stage it was given — not just the ones
        # the policy tree happened to expand. Dec has no contact_stages so never syncs.
        variant_syncs = len(contact_stages) > 0

        # Legacy same-stage rollout path. Sync reveals the current abstract state;
        # off-sync observations reveal only each agent's own deviation.
        obs1 = local_obs_index(MISS_NULL_OBS, dev1_bin)
        obs2 = local_obs_index(MISS_NULL_OBS, dev2_bin)
        if at_contact:
            if variant_syncs:
                sync_count += 1
                shared = miss_to_bin(true_miss_km)
                obs1 = local_obs_index(shared, dev1_bin)
                obs2 = local_obs_index(shared, dev2_bin)
            else:
                pass

        obs_joint = joint_obs_index(obs1, obs2)

        if burn1 != 0 or burn2 != 0:
            maneuver_stages.append((k, burn1, burn2))
        if burn1 != 0:
            total_dv += dv_mag
        if burn2 != 0:
            total_dv += dv_mag

        # Advance belief: dict lookup instead of linear scan
        try:
            sparse_transitions = sdec.get_terminal(current_belief_idx, joint_act)
            obs_to_belief = {o: d for o, p, d in sparse_transitions}
            if obs_joint in obs_to_belief:
                current_belief_idx = obs_to_belief[obs_joint]
        except KeyError:
            pass

        if k < N_STAGES - 1:
            current_oh = update_oh_rssda(
                policy, cen_dists_map, clustering, clustering_cen,
                k, current_belief_idx, current_oh, is_cen, c_ptr, obs1, obs2)

            # Propagate to next stage and cache, keyed on actions including this one.
            next_key = tuple(action_history + [joint_act])
            if next_key not in traj_cache:
                sc1_post = apply_maneuver(sc1_true, burn1, dv=dv_mag)
                sc2_post = apply_maneuver(sc2_true, burn2, dv=dv_mag)
                sc1_next = propagate(STAGE_EPOCHS[k], sc1_post, STAGE_EPOCHS[k + 1])
                sc2_next = propagate(STAGE_EPOCHS[k], sc2_post, STAGE_EPOCHS[k + 1])
                traj_cache[next_key] = (sc1_next, sc2_next)

        dev1_bin = update_dev_bin(dev1_bin, burn1)
        dev2_bin = update_dev_bin(dev2_bin, burn2)
        action_history.append(joint_act)

    # Final miss at TCA
    final_tca_key = ('tca', tuple(action_history))
    if final_tca_key not in traj_cache:
        sc1_true, sc2_true = traj_cache[tuple(action_history[:-1])]
        sc1_final = propagate(STAGE_EPOCHS[-1], sc1_true, EPOCH_TCA)
        sc2_final = propagate(STAGE_EPOCHS[-1], sc2_true, EPOCH_TCA)
        rtn_final = np.array(state_eci_to_rtn(sc1_final, sc2_final))
        traj_cache[final_tca_key] = np.linalg.norm(rtn_final[:3]) / 1e3
    miss_km_at_tca = traj_cache[final_tca_key]

    return {
        'miss_km_at_tca':          miss_km_at_tca,
        'total_dv_ms':             total_dv,
        'sync_count':              sync_count,
        'maneuver_stages':         maneuver_stages,
        'miss_bin_trajectory':     miss_bin_traj,
    }


# ---------------------------------------------------------------------------
# Single rollout — greedy policy
# ---------------------------------------------------------------------------

def rollout_rssda_stage_observation(
    T, O, R, sdec,
    full_result,
    contact_stages: List[int],
    true_init_miss_km: float,
    dv_mag: float,
    rng: np.random.Generator,
    traj_cache: Optional[dict] = None,
) -> dict:
    """
    Closed-loop rollout with standard POMDP timing.

    At stage k the policy acts from the current belief/history. The action
    transitions the system to stage k+1; then the agents receive either private
    independent observations or a shared perfect observation if k+1 is a sync
    stage. That observation updates the belief/history used at the next stage.
    """
    value, policy, clustering, cent_vector, cen_dists_map, clustering_cen = full_result

    if traj_cache is None:
        traj_cache = {}

    if () not in traj_cache:
        sc1_tca_eci = sc1_eci_at_tca()
        sc1_s0 = propagate(EPOCH_TCA, sc1_tca_eci, STAGE_EPOCHS[0])
        sc2_tca = np.array(state_rtn_to_eci(
            sc1_tca_eci,
            np.array([true_init_miss_km * 1e3, 0.0, 0.0, 0.0, -V_REL_MS, 0.0])
        ))
        sc2_s0 = propagate(EPOCH_TCA, sc2_tca, STAGE_EPOCHS[0])
        traj_cache[()] = (sc1_s0, sc2_s0)

    current_belief_idx = sdec.dist_dict[int_tuple(sdec.init_beliefs)]
    current_oh = [0, 0]
    dev1_bin = DEV_ZERO
    dev2_bin = DEV_ZERO

    total_dv = 0.0
    sync_count = 0
    maneuver_stages = []
    miss_bin_traj = []
    action_history = []

    for k in range(N_STAGES):
        cache_key = tuple(action_history)
        sc1_true, sc2_true = traj_cache[cache_key]

        tca_key = ('tca', cache_key)
        if tca_key not in traj_cache:
            sc1_proj = propagate(STAGE_EPOCHS[k], sc1_true, EPOCH_TCA)
            sc2_proj = propagate(STAGE_EPOCHS[k], sc2_true, EPOCH_TCA)
            rtn_tca = np.array(state_eci_to_rtn(sc1_proj, sc2_proj))
            traj_cache[tca_key] = np.linalg.norm(rtn_tca[:3]) / 1e3
        miss_bin_traj.append(miss_to_bin(traj_cache[tca_key]))

        if k in contact_stages:
            sync_count += 1

        joint_act, a1, a2, is_cen = get_rssda_action(
            policy, cen_dists_map, clustering, clustering_cen,
            k, current_belief_idx, current_oh, N_ACT_AGENT)
        if joint_act < 0:
            joint_act, a1, a2 = 0, 0, 0

        c_ptr = -1
        if is_cen and k < len(cen_dists_map):
            dists_at_step = cen_dists_map[k]
            if current_belief_idx in dists_at_step:
                c_ptr = dists_at_step.index(current_belief_idx)

        burn1, burn2 = a1, a2
        if burn1 != 0 or burn2 != 0:
            maneuver_stages.append((k, burn1, burn2))
        total_dv += dv_mag * ((burn1 != 0) + (burn2 != 0))

        next_key = tuple(action_history + [joint_act])
        if k < N_STAGES - 1:
            if next_key not in traj_cache:
                sc1_post = apply_maneuver(sc1_true, burn1, dv=dv_mag)
                sc2_post = apply_maneuver(sc2_true, burn2, dv=dv_mag)
                sc1_next = propagate(STAGE_EPOCHS[k], sc1_post, STAGE_EPOCHS[k + 1])
                sc2_next = propagate(STAGE_EPOCHS[k], sc2_post, STAGE_EPOCHS[k + 1])
                traj_cache[next_key] = (sc1_next, sc2_next)

            next_tca_key = ('tca', next_key)
            if next_tca_key not in traj_cache:
                sc1_next, sc2_next = traj_cache[next_key]
                sc1_proj = propagate(STAGE_EPOCHS[k + 1], sc1_next, EPOCH_TCA)
                sc2_proj = propagate(STAGE_EPOCHS[k + 1], sc2_next, EPOCH_TCA)
                rtn_tca = np.array(state_eci_to_rtn(sc1_proj, sc2_proj))
                traj_cache[next_tca_key] = np.linalg.norm(rtn_tca[:3]) / 1e3
            next_miss_km = traj_cache[next_tca_key]

            next_dev1 = update_dev_bin(dev1_bin, burn1)
            next_dev2 = update_dev_bin(dev2_bin, burn2)
            if (k + 1) in contact_stages:
                miss_obs = miss_to_bin(next_miss_km)
                obs1 = local_obs_index(miss_obs, next_dev1)
                obs2 = local_obs_index(miss_obs, next_dev2)
            else:
                obs1 = local_obs_index(MISS_NULL_OBS, next_dev1)
                obs2 = local_obs_index(MISS_NULL_OBS, next_dev2)
            obs_joint = joint_obs_index(obs1, obs2)

            try:
                sparse_transitions = sdec.get_terminal(current_belief_idx, joint_act)
                obs_to_belief = {o: d for o, p, d in sparse_transitions}
                if obs_joint in obs_to_belief:
                    current_belief_idx = obs_to_belief[obs_joint]
            except KeyError:
                pass

            current_oh = update_oh_rssda(
                policy, cen_dists_map, clustering, clustering_cen,
                k, current_belief_idx, current_oh, is_cen, c_ptr, obs1, obs2)
            dev1_bin, dev2_bin = next_dev1, next_dev2
        else:
            final_tca_key = ('tca', next_key)
            if final_tca_key not in traj_cache:
                sc1_post = apply_maneuver(sc1_true, burn1, dv=dv_mag)
                sc2_post = apply_maneuver(sc2_true, burn2, dv=dv_mag)
                sc1_final = propagate(STAGE_EPOCHS[-1], sc1_post, EPOCH_TCA)
                sc2_final = propagate(STAGE_EPOCHS[-1], sc2_post, EPOCH_TCA)
                rtn_final = np.array(state_eci_to_rtn(sc1_final, sc2_final))
                traj_cache[final_tca_key] = np.linalg.norm(rtn_final[:3]) / 1e3

        action_history.append(joint_act)

    return {
        'miss_km_at_tca':          traj_cache[('tca', tuple(action_history))],
        'total_dv_ms':             total_dv,
        'sync_count':              sync_count,
        'maneuver_stages':         maneuver_stages,
        'miss_bin_trajectory':     miss_bin_traj,
    }


def rollout_greedy(
    T, O, R, init_b,
    greedy_policy: np.ndarray,
    contact_stages: List[int],
    true_init_miss_km: float,
    dv_mag: float,
    rng: np.random.Generator,
) -> dict:
    sc1_tca_eci = sc1_eci_at_tca()
    sc1_true = propagate(EPOCH_TCA, sc1_tca_eci, STAGE_EPOCHS[0])
    sc2_tca = np.array(state_rtn_to_eci(sc1_tca_eci,
                       np.array([true_init_miss_km * 1e3, 0.0, 0.0, 0.0, -V_REL_MS, 0.0])))
    sc2_true = propagate(EPOCH_TCA, sc2_tca, STAGE_EPOCHS[0])

    belief = init_b.copy()
    total_dv = 0.0
    sync_count = 0
    maneuver_stages = []
    miss_bin_traj = []
    dev1_bin = DEV_ZERO
    dev2_bin = DEV_ZERO

    for k in range(N_STAGES):
        if k > 0:
            sc1_true = propagate(STAGE_EPOCHS[k - 1], sc1_true, STAGE_EPOCHS[k])
            sc2_true = propagate(STAGE_EPOCHS[k - 1], sc2_true, STAGE_EPOCHS[k])

        sc1_proj = propagate(STAGE_EPOCHS[k], sc1_true, EPOCH_TCA)
        sc2_proj = propagate(STAGE_EPOCHS[k], sc2_true, EPOCH_TCA)
        rtn_tca  = np.array(state_eci_to_rtn(sc1_proj, sc2_proj))
        true_miss_km = np.linalg.norm(rtn_tca[:3]) / 1e3
        miss_bin_traj.append(miss_to_bin(true_miss_km))

        most_likely_s = int(np.argmax(belief[:N_STATES]))
        a = greedy_policy[most_likely_s]
        burn1, burn2 = split_joint_action(a)
        if k in contact_stages:
            sync_count += 1

        if burn1 != 0 or burn2 != 0:
            maneuver_stages.append((k, burn1, burn2))
        if burn1 != 0:
            sc1_true = apply_maneuver(sc1_true, burn1, dv=dv_mag)
            total_dv += dv_mag
        if burn2 != 0:
            sc2_true = apply_maneuver(sc2_true, burn2, dv=dv_mag)
            total_dv += dv_mag

        next_dev1 = update_dev_bin(dev1_bin, burn1)
        next_dev2 = update_dev_bin(dev2_bin, burn2)
        b_pred = np.dot(T[a].T, belief)
        if k < N_STAGES - 1:
            sc1_after = propagate(STAGE_EPOCHS[k], sc1_true, EPOCH_TCA)
            sc2_after = propagate(STAGE_EPOCHS[k], sc2_true, EPOCH_TCA)
            rtn_after = np.array(state_eci_to_rtn(sc1_after, sc2_after))
            next_miss_km = np.linalg.norm(rtn_after[:3]) / 1e3
            if (k + 1) in contact_stages:
                miss_obs = miss_to_bin(next_miss_km)
                obs1 = local_obs_index(miss_obs, next_dev1)
                obs2 = local_obs_index(miss_obs, next_dev2)
            else:
                obs1 = local_obs_index(MISS_NULL_OBS, next_dev1)
                obs2 = local_obs_index(MISS_NULL_OBS, next_dev2)
            obs_joint = joint_obs_index(obs1, obs2)
            belief = O[a, :, obs_joint] * b_pred
        else:
            belief = b_pred
        s = belief.sum()
        if s > 1e-30:
            belief /= s
        dev1_bin, dev2_bin = next_dev1, next_dev2

    sc1_final = propagate(STAGE_EPOCHS[-1], sc1_true, EPOCH_TCA)
    sc2_final = propagate(STAGE_EPOCHS[-1], sc2_true, EPOCH_TCA)
    rtn_final = np.array(state_eci_to_rtn(sc1_final, sc2_final))
    miss_km_at_tca = np.linalg.norm(rtn_final[:3]) / 1e3

    return {
        'miss_km_at_tca':          miss_km_at_tca,
        'total_dv_ms':             total_dv,
        'sync_count':              sync_count,
        'maneuver_stages':         maneuver_stages,
        'miss_bin_trajectory':     miss_bin_traj,
    }


# ---------------------------------------------------------------------------
# Multi-rollout runner
# ---------------------------------------------------------------------------

def run_simulation(
    T, O, R, init_b,
    contact_stages: List[int],
    n_rollouts: int = 100,
    init_miss_bin: int = 4,
    dv_mag: float = None,
    seed: int = 42,
    verbose: bool = True,
    policy_mode: str = "rssda",
    sdec=None,
    full_result=None,
    fixed_init: bool = False,
) -> List[dict]:
    """
    fixed_init=True: all rollouts start from bin_center_km(init_miss_bin).
      Upfront trajectory tree is precomputed with par_propagate_to — rollouts
      then run with zero per-step Brahe calls (pure lookup).
    fixed_init=False (default): each rollout samples true_miss_km uniformly
      within the bin. Lazy per-rollout Brahe propagation (original behavior).
    """
    if dv_mag is None:
        dv_mag = DV_MAGNITUDE
    rng = np.random.default_rng(seed)

    if policy_mode == "greedy":
        greedy_pol = build_greedy_policy(T, R)
        true_init_miss_km = bin_center_km(init_miss_bin) if fixed_init else None
    else:
        assert sdec is not None and full_result is not None, \
            "sdec and full_result required for rssda policy mode"

    # Precompute trajectory tree upfront if fixed_init
    traj_cache = None
    if fixed_init and policy_mode == "rssda":
        true_init_miss_km = bin_center_km(init_miss_bin)
        if verbose:
            print(f"  Precomputing trajectory tree (init_miss={true_init_miss_km:.3f} km)...")
        traj_cache = precompute_traj_tree(sdec, full_result, true_init_miss_km, dv_mag)
        if verbose:
            n_traj = sum(1 for k in traj_cache if isinstance(k, tuple) and
                         (len(k) == 0 or isinstance(k[0], int)))
            print(f"  Tree precomputed: {n_traj} nodes, {len(traj_cache)} cache entries.")

    lo = float(MISS_EDGES_KM[init_miss_bin])
    hi = float(MISS_EDGES_KM[init_miss_bin + 1])
    if hi == float('inf'):
        hi = 1500.0

    results = []
    for i in range(n_rollouts):
        if fixed_init:
            true_miss_km = bin_center_km(init_miss_bin)
        else:
            true_miss_km = rng.uniform(lo, hi)

        if policy_mode == "greedy":
            r = rollout_greedy(T, O, R, init_b, greedy_pol, contact_stages,
                               true_miss_km, dv_mag, rng)
        else:
            r = rollout_rssda_stage_observation(
                T, O, R, sdec, full_result, contact_stages,
                true_miss_km, dv_mag, rng,
                traj_cache=traj_cache)
        results.append(r)
        if verbose and (i + 1) % 10 == 0:
            print(f"  rollout {i+1}/{n_rollouts}  "
                  f"miss={r['miss_km_at_tca']:.2f}km  "
                  f"dv={r['total_dv_ms']:.3f}m/s  "
                  f"syncs={r['sync_count']}")
    return results


def summarize(results: List[dict], label: str, init_miss_bin: int, dv_mag: float):
    misses         = [r['miss_km_at_tca']       for r in results]
    dvs            = [r['total_dv_ms']          for r in results]
    syncs          = [r['sync_count']           for r in results]
    n_coll = sum(1 for m in misses if m < 1.0)
    return {
        'label':      label,
        'n':          len(results),
        'init_bin':   init_miss_bin,
        'coll_rate':  100.0 * n_coll / len(results),
        'mean_miss':  float(np.mean(misses)),
        'min_miss':   float(np.min(misses)),
        'mean_dv':    float(np.mean(dvs)),
        'mean_syncs': float(np.mean(syncs)),
    }


def print_summary(results: List[dict], init_miss_bin: int, dv_mag: float, label: str = ""):
    s = summarize(results, label, init_miss_bin, dv_mag)
    misses = [r['miss_km_at_tca'] for r in results]
    print(f"\n{'='*60}")
    if label:
        print(f"[{label}]  ({s['n']} rollouts, init_bin={init_miss_bin} "
              f"[{MISS_LABELS[init_miss_bin]}], dv={dv_mag} m/s)")
    print(f"{'='*60}")
    print(f"  Collisions:  {s['coll_rate']:.1f}%")
    print(f"  Miss at TCA: min={s['min_miss']:.3f}  mean={s['mean_miss']:.3f}  km")
    print(f"  Total dv:    mean={s['mean_dv']:.4f} m/s")
    print(f"  Sync events: mean={s['mean_syncs']:.1f} of {N_STAGES} stages")
    print(f"\n  Miss-bin distribution at TCA:")
    from collections import Counter
    cnt = Counter(miss_to_bin(m) for m in misses)
    for mb in range(N_MISS):
        bar = '#' * cnt.get(mb, 0)
        print(f"    bin {mb} {MISS_LABELS[mb]:10s}: {cnt.get(mb,0):4d}  {bar}")


def print_comparison_table(summaries: List[dict]):
    print(f"\n{'='*80}")
    print(f"{'Scenario':<22} {'Init bin':<10} {'Coll%':>7} {'Mean miss(km)':>14} "
          f"{'Mean dv(m/s)':>13} {'Mean syncs':>11}")
    print(f"{'-'*80}")
    for s in summaries:
        print(f"  {s['label']:<20} {s['init_bin']:<10} {s['coll_rate']:>7.1f} "
              f"{s['mean_miss']:>14.3f} {s['mean_dv']:>13.4f} {s['mean_syncs']:>11.1f}")
    print(f"{'='*80}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts",      type=int,   default=100)
    parser.add_argument("--init-miss-bin", type=int,   default=4,
                        help="Initial true miss-distance bin (0=collision, 4=nominal)")
    parser.add_argument("--dv",            type=float, default=None)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--policy",        choices=["rssda", "greedy"], default="rssda")
    parser.add_argument("--compare",       action="store_true",
                        help="Run centralized, decentralized, and SDec variants")
    parser.add_argument("--fixed-init",    action="store_true",
                        help="All rollouts start from bin_center_km; precompute trajectory "
                             "tree with par_propagate_to for fast rollouts")
    args = parser.parse_args()

    initialize_eop()

    # Load SDec matrices by default; comparison loop loads per-variant
    T, O, R, init_b, contact_stages, cached_dv = load_matrices("sdec")
    dv_mag = args.dv if args.dv is not None else cached_dv

    from sdec_spacecraft import solve, build_model, build_config
    from RSSDA import SDecPOMDP

    def make_init_b_for_bin(bin_idx, spread: bool = False):
        """Initial belief for stage-0.
        spread=False: point mass at bin_idx (original behavior).
        spread=True:  uniform over bins 0-2 (uncertain conjunction scenario).
        """
        b = np.zeros(N_STATES_TOTAL, dtype=np.float64)
        if spread:
            for mb in [0, 1, 2]:
                b[state_index(mb, DEV_ZERO, DEV_ZERO, 0)] = 1.0
        else:
            b[state_index(bin_idx, DEV_ZERO, DEV_ZERO, 0)] = 1.0
        b /= b.sum()
        return b

    def solve_variant(label, cs, init_belief, T=T, O=O, R=R):
        print(f"\n--- Solving: {label} ---")
        model, _ = build_model(T, O, R, init_belief, cs)
        config = build_config(exact=False)
        sdec_obj = SDecPOMDP(model=model, config=config)
        full_res = sdec_obj.multi_agent_astar(N_STAGES)
        print(f"  Policy value: {full_res[0]:.4f}")
        return sdec_obj, full_res

    if args.compare:
        # Centralized shares histories and acts jointly at every stage.
        # SDec shares histories only at ground-contact sync stages.
        # Decentralized never syncs.
        all_stages = list(range(N_STAGES))
        variant_specs = [
            ("Centralized",     "centralized", all_stages),
            ("Decentralized",   "dec",         []),
            ("SDec (contacts)", "sdec",        contact_stages),
        ]
        all_summaries = []
        for init_bin in [0, 1]:
            for label, mat_variant, cs in variant_specs:
                Tv, Ov, Rv, init_b_v, _, _ = load_matrices(mat_variant)
                init_b_test = make_init_b_for_bin(init_bin, spread=True)
                sdec_obj, full_res = solve_variant(label, cs, init_b_test,
                                                   T=Tv, O=Ov, R=Rv)
                print(f"  Running {args.rollouts} rollouts (init_bin={init_bin})...")
                results = run_simulation(
                    Tv, Ov, Rv, init_b_test, cs,
                    n_rollouts=args.rollouts,
                    init_miss_bin=init_bin,
                    dv_mag=dv_mag,
                    seed=args.seed,
                    verbose=False,
                    policy_mode="rssda",
                    sdec=sdec_obj,
                    full_result=full_res,
                    fixed_init=args.fixed_init,
                )
                s = summarize(results, label, init_bin, dv_mag)
                all_summaries.append(s)
                print_summary(results, init_bin, dv_mag, label)

        print_comparison_table(all_summaries)

    else:
        init_b_test = make_init_b_for_bin(args.init_miss_bin)
        if args.policy == "rssda":
            print("Solving with RS-SDA*...")
            sdec_obj, full_res = solve_variant("SDec", contact_stages, init_b_test)
            results = run_simulation(
                T, O, R, init_b_test, contact_stages,
                n_rollouts=args.rollouts,
                init_miss_bin=args.init_miss_bin,
                dv_mag=dv_mag,
                seed=args.seed,
                verbose=True,
                policy_mode="rssda",
                sdec=sdec_obj,
                full_result=full_res,
                fixed_init=args.fixed_init,
            )
        else:
            results = run_simulation(
                T, O, R, init_b_test, contact_stages,
                n_rollouts=args.rollouts,
                init_miss_bin=args.init_miss_bin,
                dv_mag=dv_mag,
                seed=args.seed,
                verbose=True,
                policy_mode="greedy",
                fixed_init=args.fixed_init,
            )
        print_summary(results, args.init_miss_bin, dv_mag,
                      label=args.policy.upper())
