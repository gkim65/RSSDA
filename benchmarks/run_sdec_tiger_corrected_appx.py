"""
Corrected Monte Carlo runs for IJCAI Table 1 SDec-Tiger Appx. RS-SDA*.

The important guardrail in this script is that the rollout environment samples
from the same T/O/R arrays used by the planner. This avoids the old mismatch
where approximate execution used a different listening accuracy than the
SDec-Tiger model used for exact RS-SDA*.

Example:
    python benchmarks/run_sdec_tiger_corrected_appx.py --trials 100
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from pathlib import Path
from statistics import mean, stdev

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))

import sdec_tiger
from sdec_tiger import TigerProblemFactory
from RSSDA import SDecPOMDP, SDecPOMDPModel, RSSDAConfig, int_tuple


ACCURACY_BY_MODE = {
    "decentralized": 0.85,
    "semi": 0.75,
    "centralized": 0.85,
}

SYNC_ACTIONS_BY_MODE = {
    "decentralized": [],
    "semi": [8],
    "centralized": list(range(9)),
}


def parse_horizons(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def sample_categorical(weights, rng: random.Random) -> int:
    total = sum(weights)
    if total <= 0:
        raise ValueError("cannot sample from all-zero weights")
    draw = rng.random() * total
    acc = 0.0
    for idx, weight in enumerate(weights):
        acc += weight
        if draw <= acc:
            return idx
    return len(weights) - 1


def build_arrays(trigger_mode: str, obs_accuracy: float, lam: float):
    sdec_tiger.TRIGGER_MODE = trigger_mode
    sdec_tiger.OBS_ACCURACY = obs_accuracy
    sdec_tiger.LAMBDA = lam
    factory = TigerProblemFactory(config=None)
    T, O, R, init_b, nacts_fac, nobs_fac = factory.generate()
    return factory, T, O, R, init_b, nacts_fac, nobs_fac


def build_model(factory, T, O, R, init_b, nacts_fac, nobs_fac, sync_actions):
    return SDecPOMDPModel(
        nagents=factory.nagents,
        nstates=factory.nstates,
        nactions=factory.nacts,
        nobs=factory.nobs,
        transitions=T,
        obs=O,
        rewards=R,
        init_beliefs=init_b,
        nacts_factor=nacts_fac,
        nobs_factor=nobs_fac,
        sync_states=[],
        sync_actions=sync_actions,
        sync_observations=[],
    )


def appx_config(horizon: int, profile: str) -> RSSDAConfig:
    if profile == "paper":
        # Closest current-API analogue of the IJCAI Tiger approximate runner:
        # interleaved, progress-limited, recursive tail approximation, lossless
        # clustering, and a QMDP tail beyond the solved head.
        return RSSDAConfig(
            maxh=horizon,
            maxit=200,
            IEmin2=3,
            alpha=0.2,
            algorithm="approximate",
            heuristic_type="POMDP",
            tail_heuristic_type="QMDP",
            TI1=True,
            TI2=True,
            TI3=True,
            TI4=False,
            score_limit=100,
            cen_threshold=0.6,
            sm_temperature=0.6,
            iter_limit=100,
            rec_limit=5,
            hybrid_r=0,
            max_clusters=10,
            output=False,
        )
    if profile == "hybrid":
        return RSSDAConfig(
            maxh=horizon,
            maxit=200,
            IEmin2=3,
            alpha=0.2,
            algorithm="approximate",
            heuristic_type="HYBRID",
            tail_heuristic_type="HYBRID",
            TI1=True,
            TI2=True,
            TI3=True,
            TI4=True,
            score_limit=20,
            cen_threshold=0.6,
            sm_temperature=0.6,
            iter_limit=1000,
            rec_limit=2,
            hybrid_r=2,
            max_clusters=10,
            output=False,
        )
    if profile == "current":
        return RSSDAConfig(
            maxh=horizon,
            maxit=200,
            IEmin2=3,
            alpha=0.2,
            algorithm="approximate",
            heuristic_type="POMDP",
            tail_heuristic_type="POMDP",
            TI1=True,
            TI2=True,
            TI3=True,
            TI4=True,
            score_limit=20,
            cen_threshold=0.6,
            sm_temperature=0.6,
            iter_limit=1000,
            rec_limit=2,
            hybrid_r=1,
            max_clusters=10,
            output=False,
        )
    raise ValueError(f"unknown profile: {profile}")


def exact_config(horizon: int) -> RSSDAConfig:
    return RSSDAConfig(
        maxh=horizon,
        maxit=200,
        IEmin2=3,
        alpha=0.2,
        algorithm="exact",
        heuristic_type="POMDP",
        tail_heuristic_type="POMDP",
        TI1=False,
        TI2=False,
        TI3=False,
        TI4=False,
        score_limit=20,
        cen_threshold=0.6,
        sm_temperature=0.6,
        iter_limit=1000,
        rec_limit=2,
        hybrid_r=1,
        max_clusters=10,
        output=False,
    )


def build_solver(horizon, trigger_mode, obs_accuracy, lam, solver_config):
    sync_actions = SYNC_ACTIONS_BY_MODE[trigger_mode]
    factory, T, O, R, init_b, nacts_fac, nobs_fac = build_arrays(
        trigger_mode, obs_accuracy, lam
    )
    model = build_model(factory, T, O, R, init_b, nacts_fac, nobs_fac, sync_actions)
    solver = SDecPOMDP(model=model, config=solver_config)
    init_idx = solver.dist_dict[int_tuple(solver.init_beliefs)]
    arrays = {
        "T": T,
        "O": O,
        "R": R,
        "init_b": init_b,
        "nstates": factory.nstates,
        "nactions": factory.nacts,
        "nobs": factory.nobs,
        "nsq": factory.nsq,
        "nso": factory.nso,
        "act_per_agent": factory.act_per_agent,
        "obs_per_agent": factory.obs_per_agent,
    }
    return solver, init_idx, arrays


def entry_action(entry) -> int:
    if isinstance(entry, (list, tuple)):
        if not entry:
            return -1
        return int(entry[0])
    return int(entry)


def find_cen_cluster(cen_dists_map, step: int, belief_idx: int) -> int:
    if step >= len(cen_dists_map):
        return 0
    for idx, bid in enumerate(cen_dists_map[step]):
        if bid == belief_idx:
            return idx
    return 0


def has_valid_dec_action(policy, step: int, clusters: list[int]) -> bool:
    if step >= len(policy) or not policy[step] or len(policy[step]) < 1:
        return False
    dec = policy[step][0]
    if len(dec) < 2:
        return False
    for agent in range(2):
        c = clusters[agent]
        if c < 0 or c >= len(dec[agent]):
            return False
        if entry_action(dec[agent][c]) < 0:
            return False
    return True


def select_action(policy, cent_vector, cen_dists_map, step, belief_idx, clusters):
    if step >= len(policy) or not policy[step]:
        raise IndexError(f"policy has no stage {step}")

    stage = policy[step]
    use_central = bool(step < len(cent_vector) and cent_vector[step])
    if not use_central and not has_valid_dec_action(policy, step, clusters):
        use_central = len(stage) > 1 and len(stage[1]) > 0

    if use_central:
        cen = stage[1] if len(stage) > 1 else []
        cluster = find_cen_cluster(cen_dists_map, step, belief_idx)
        if cluster >= len(cen):
            cluster = 0
        joint_action = entry_action(cen[cluster])
        if joint_action < 0:
            raise ValueError(f"invalid centralized action at stage {step}")
        return joint_action, True, cluster

    dec = stage[0]
    a1 = entry_action(dec[0][clusters[0]])
    a2 = entry_action(dec[1][clusters[1]])
    if a1 < 0 or a2 < 0:
        raise ValueError(f"invalid decentralized action at stage {step}")
    joint_action = a1 + 3 * a2
    return joint_action, False, -1


def step_environment(arrays, state: int, joint_action: int, rng: random.Random):
    nstates = arrays["nstates"]
    nobs = arrays["nobs"]
    nsq = arrays["nsq"]
    nso = arrays["nso"]
    T = arrays["T"]
    O = arrays["O"]
    R = arrays["R"]

    reward = R[joint_action * nstates + state]
    t0 = joint_action * nsq + state * nstates
    next_state = sample_categorical(T[t0 : t0 + nstates], rng)

    o0 = joint_action * nso + next_state * nobs
    obs = sample_categorical(O[o0 : o0 + nobs], rng)
    return next_state, obs, reward


def update_belief(solver, belief_idx: int, joint_action: int, obs_id: int) -> int:
    for oid, prob, next_belief in solver.get_terminal(belief_idx, joint_action):
        if oid == obs_id and prob > 0.0:
            return next_belief
    # If a sampled observation sits on a numerically tiny branch, fall back to
    # the matching observation even if the stored probability rounded to zero.
    for oid, _prob, next_belief in solver.get_terminal(belief_idx, joint_action):
        if oid == obs_id:
            return next_belief
    raise ValueError(f"no belief transition for action {joint_action}, obs {obs_id}")


def update_clusters(
    clusters,
    clustering,
    clustering_cen,
    step,
    used_central,
    cen_cluster,
    obs_id,
    obs_per_agent,
):
    o1 = obs_id % obs_per_agent
    o2 = obs_id // obs_per_agent
    source = clustering_cen if used_central else clustering
    parent = cen_cluster if used_central else None
    next_clusters = clusters[:]

    if step >= len(source) or len(source[step]) < 2:
        return next_clusters

    for agent, obs in ((0, o1), (1, o2)):
        maps = source[step][agent]
        parent_cluster = parent if used_central else clusters[agent]
        if parent_cluster < 0 or parent_cluster >= len(maps):
            continue
        if obs < 0 or obs >= len(maps[parent_cluster]):
            continue
        mapped = maps[parent_cluster][obs]
        next_clusters[agent] = mapped if mapped >= 0 else 0
    return next_clusters


def rollout_trial(solver, init_idx, arrays, horizon: int, seed: int):
    rng = random.Random(seed)
    state = sample_categorical(arrays["init_b"], rng)
    belief_idx = init_idx
    remaining = horizon
    total_reward = 0.0
    total_plan_time = 0.0
    replans = 0

    while remaining > 0:
        solver.maxh = remaining
        solver.cluster_dict.clear()
        t0 = time.time()
        val, policy, clustering, cent_vector, cen_dists_map, clustering_cen = (
            solver.multi_agent_astar(remaining, init_beliefs=belief_idx)
        )
        total_plan_time += time.time() - t0
        replans += 1

        if not policy:
            raise RuntimeError("planner returned an empty policy")

        if any(cent_vector):
            steps_to_execute = cent_vector.index(True) + 1
        else:
            steps_to_execute = len(policy)
        steps_to_execute = min(steps_to_execute, remaining, len(policy))

        clusters = [0, 0]
        for step in range(steps_to_execute):
            joint_action, used_central, cen_cluster = select_action(
                policy, cent_vector, cen_dists_map, step, belief_idx, clusters
            )
            state, obs_id, reward = step_environment(arrays, state, joint_action, rng)
            total_reward += reward
            belief_idx = update_belief(solver, belief_idx, joint_action, obs_id)
            clusters = update_clusters(
                clusters,
                clustering,
                clustering_cen,
                step,
                used_central,
                cen_cluster,
                obs_id,
                arrays["obs_per_agent"],
            )

        remaining -= steps_to_execute

    return {
        "reward": total_reward,
        "plan_time": total_plan_time,
        "replans": replans,
    }


def solve_exact(horizon, trigger_mode, obs_accuracy, lam):
    solver, init_idx, _arrays = build_solver(
        horizon, trigger_mode, obs_accuracy, lam, exact_config(horizon)
    )
    t0 = time.time()
    value, *_ = solver.multi_agent_astar(horizon, init_beliefs=init_idx)
    return value, time.time() - t0


def summarize(values):
    avg = mean(values)
    sd = stdev(values) if len(values) > 1 else 0.0
    se = sd / math.sqrt(len(values)) if values else 0.0
    return avg, sd, se, min(values), max(values)


def run_horizon(args, horizon: int, obs_accuracy: float):
    exact_value = None
    exact_time = None
    if not args.skip_exact:
        exact_value, exact_time = solve_exact(
            horizon, args.trigger_mode, obs_accuracy, args.lam
        )

    solver, init_idx, arrays = build_solver(
        horizon,
        args.trigger_mode,
        obs_accuracy,
        args.lam,
        appx_config(horizon, args.profile),
    )

    rewards = []
    plan_times = []
    replans = []
    wall0 = time.time()
    for trial in range(args.trials):
        seed = args.seed + horizon * 100000 + trial
        result = rollout_trial(solver, init_idx, arrays, horizon, seed)
        rewards.append(result["reward"])
        plan_times.append(result["plan_time"])
        replans.append(result["replans"])
        if args.progress and ((trial + 1) % args.progress == 0 or trial == 0):
            avg, _sd, se, _lo, _hi = summarize(rewards)
            print(
                f"h={horizon} trial={trial + 1}/{args.trials} "
                f"mean={avg:.4f} se={se:.4f} "
                f"avg_plan_time={mean(plan_times):.3f}s",
                flush=True,
            )

    avg, sd, se, lo, hi = summarize(rewards)
    row = {
        "horizon": horizon,
        "trigger_mode": args.trigger_mode,
        "obs_accuracy": obs_accuracy,
        "lambda": args.lam,
        "profile": args.profile,
        "trials": args.trials,
        "seed0": args.seed,
        "appx_mean": avg,
        "appx_std": sd,
        "appx_se": se,
        "appx_min": lo,
        "appx_max": hi,
        "avg_plan_time_s": mean(plan_times),
        "avg_replans": mean(replans),
        "wall_time_s": time.time() - wall0,
        "exact_value": exact_value,
        "exact_time_s": exact_time,
        "gap_appx_minus_exact": None if exact_value is None else avg - exact_value,
    }
    return row


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", default="10,12,15,20")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument(
        "--trigger-mode",
        choices=["decentralized", "semi", "centralized"],
        default="semi",
    )
    parser.add_argument("--obs-accuracy", type=float, default=None)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument(
        "--profile",
        choices=["paper", "hybrid", "current"],
        default="paper",
    )
    parser.add_argument("--skip-exact", action="store_true")
    parser.add_argument("--progress", type=int, default=10)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    horizons = parse_horizons(args.horizons)
    obs_accuracy = (
        args.obs_accuracy
        if args.obs_accuracy is not None
        else ACCURACY_BY_MODE[args.trigger_mode]
    )

    print(
        "Corrected SDec-Tiger Appx. RS-SDA* runs "
        f"(mode={args.trigger_mode}, acc={obs_accuracy}, lambda={args.lam}, "
        f"profile={args.profile}, trials={args.trials})"
    )

    rows = []
    for horizon in horizons:
        row = run_horizon(args, horizon, obs_accuracy)
        rows.append(row)
        exact_text = (
            "N/A" if row["exact_value"] is None else f"{row['exact_value']:.5f}"
        )
        gap_text = (
            "N/A"
            if row["gap_appx_minus_exact"] is None
            else f"{row['gap_appx_minus_exact']:+.5f}"
        )
        print(
            f"RESULT h={horizon}: appx={row['appx_mean']:.5f} "
            f"+/- {row['appx_se']:.5f} exact={exact_text} gap={gap_text} "
            f"avg_plan_time={row['avg_plan_time_s']:.3f}s"
        )

    out = (
        Path(args.out)
        if args.out
        else _ROOT
        / "results"
        / f"sdec_tiger_corrected_appx_{args.profile}_{args.trigger_mode}.csv"
    )
    write_csv(out, rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
