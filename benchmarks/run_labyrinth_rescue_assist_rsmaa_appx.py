"""Run Approximate RS-MAA* on canonical Labyrinth Rescue/Assist instances.

This runner intentionally uses ``baselines.decPOMDP.DecPOMDP`` directly, not
RS-SDA* with empty synchronization triggers.  It solves a fully decentralized
policy for the canonical rescue-assist Dec-POMDP, then evaluates that policy on
a target-balanced slate: three seeded rollouts per possible target node.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import math
import os
import random
import statistics
import sys
import time
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "baselines") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "baselines"))

from baselines.decPOMDP import DecPOMDP as RSMAA, MemoryLimitExceeded  # noqa: E402
from labyrinth_cache import (  # noqa: E402
    load_cached_rescue_assist_labyrinth,
    precompute_rescue_assist_all,
)


CANONICAL_INSTANCES = [
    ("ExtCross9", "1"),
    ("LopsidedY10", "2"),
    ("Ladder10", "3"),
    ("Maze12", "4"),
    ("HiddenTail11", "5"),
    ("Mesh10", "7"),
]


@dataclass
class SolveResult:
    instance: str
    bid: str
    value: float
    solve_time: float
    policy: list
    clustering: list


@dataclass
class TrialResult:
    instance: str
    bid: str
    target_idx: int
    target_node: int
    trial_idx: int
    seed: int
    reward: float
    terminal_step: int
    terminal_kind: str


def _parse_bids(raw: str) -> List[str]:
    wanted = [chunk.strip() for chunk in str(raw).split(",") if chunk.strip()]
    valid = {bid for _, bid in CANONICAL_INSTANCES}
    names = {name.lower(): bid for name, bid in CANONICAL_INSTANCES}
    bids = []
    for item in wanted:
        bid = names.get(item.lower(), item)
        if bid not in valid:
            raise ValueError(
                f"Unknown rescue-assist bid/name {item!r}; valid bids are "
                f"{', '.join(sorted(valid))}")
        bids.append(bid)
    return bids


def _selected_instances(bids: Iterable[str]):
    selected = set(bids)
    return [(name, bid) for name, bid in CANONICAL_INSTANCES if bid in selected]


def ensure_cache(bid: str, detection_prob: float):
    cache = load_cached_rescue_assist_labyrinth(
        bid,
        detection_prob,
        assist_reward=100.0,
        unassisted_reward=80.0,
        wrong_reward=-200.0,
        step_cost=-1.0,
    )
    if cache is None:
        precompute_rescue_assist_all(
            bid,
            detection_prob,
            assist_reward=100.0,
            unassisted_reward=80.0,
            wrong_reward=-200.0,
            step_cost=-1.0,
        )
        cache = load_cached_rescue_assist_labyrinth(
            bid,
            detection_prob,
            assist_reward=100.0,
            unassisted_reward=80.0,
            wrong_reward=-200.0,
            step_cost=-1.0,
        )
    if cache is None:
        raise RuntimeError(f"Could not load or precompute rescue-assist cache for bid={bid}")
    return cache


def csr_to_pdict_rows(csr_list) -> List[Tuple[array, array]]:
    rows = []
    for matrix in csr_list:
        for row in range(matrix.shape[0]):
            start = matrix.indptr[row]
            end = matrix.indptr[row + 1]
            rows.append((
                array("i", [int(x) for x in matrix.indices[start:end]]),
                array("d", [float(x) for x in matrix.data[start:end]]),
            ))
    return rows


def solve_rsmaa(instance: str, bid: str, cache_data, args: argparse.Namespace) -> SolveResult:
    cfg = cache_data["config"]
    t_pdict = csr_to_pdict_rows(cache_data["T_csr_list"])
    o_pdict = csr_to_pdict_rows(cache_data["O_csr_list"])

    solver = RSMAA(
        nagents=2,
        nstates=cfg["nstates"],
        nactions=cfg["nacts"],
        nobs=cfg["nobs"],
        transitions=t_pdict,
        obs=o_pdict,
        rewards=list(cache_data["R"]),
        init_beliefs=list(cache_data["init_beliefs"]),
        nacts_factor=cfg["nacts_factor"],
        nobs_factor=cfg["nobs_factor"],
        maxh=args.horizon,
        cluster_type=args.cluster_type,
        maxit=args.maxit,
        q_depth=args.q_depth,
        alpha=args.alpha,
        iter_limit=args.iter_limit,
        maxrec=args.maxrec,
        memory=args.memory,
        heuristic=args.heuristic,
        rec_type=args.rec_type,
        p_threshold_cluster=args.p_threshold_cluster,
        p_threshold_expand=args.p_threshold_expand,
        policyvalfound=-math.inf,
        output=args.verbose_solver,
        memory_limit_gb=args.memory_limit_gb,
        memory_check_interval=args.memory_check_interval,
    )
    solver.decentralized = True
    solver.onesided = False

    t0 = time.time()
    if args.verbose_solver:
        value, policy, clustering = solver.multi_agent_astar(args.horizon)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            value, policy, clustering = solver.multi_agent_astar(args.horizon)
    solve_time = time.time() - t0
    if policy is None or clustering is None:
        raise RuntimeError(f"RS-MAA* did not return a policy for {instance} bid={bid}")
    return SolveResult(instance, bid, float(value), float(solve_time), policy, clustering)


def seed_for_target_trial(seed_base: int, target_idx: int, trial_idx: int) -> int:
    return int(seed_base + target_idx * 1000 + trial_idx * 10_000_000)


def sample_sparse(indices, values, rng: random.Random) -> int:
    if len(indices) == 0:
        raise RuntimeError("Cannot sample from empty sparse distribution")
    draw = rng.random()
    acc = 0.0
    for idx, prob in zip(indices, values):
        acc += float(prob)
        if draw <= acc + 1e-12:
            return int(idx)
    return int(indices[-1])


def classify_terminal(reward: float, sink_state: int, next_state: int) -> str:
    if next_state != sink_state:
        return "none"
    if reward >= 99.0:
        return "assisted"
    if reward > 0.0:
        return "unassisted"
    if reward < -150.0:
        return "wrong"
    return "sink"


def simulate_policy(
    solve: SolveResult,
    cache_data,
    target_idx: int,
    target_node: int,
    trial_idx: int,
    seed: int,
    args: argparse.Namespace,
) -> TrialResult:
    cfg = cache_data["config"]
    loader = cache_data["loader"]
    rng = random.Random(seed)
    nstates = cfg["nstates"]
    act_per = cfg["act_per_agent"]
    obs_per = cfg["obs_per_agent"]
    sink = cfg["sink_state"]
    num_nodes = loader["num_nodes"]
    num_targets = loader["num_targets"]
    start = loader["start_node"]
    state = start * (num_nodes * num_targets) + start * num_targets + target_idx
    oh = [0, 0]
    total_reward = 0.0
    terminal_step = 0
    terminal_kind = "none"
    R = cache_data["R_np"]
    T_csr = cache_data["T_csr_list"]
    O_csr = cache_data["O_csr_list"]

    for step in range(args.horizon):
        try:
            a1 = int(solve.policy[step][0][oh[0]])
            a2 = int(solve.policy[step][1][oh[1]])
        except Exception as exc:
            raise RuntimeError(
                f"Policy lookup failed at step={step}, oh={oh}, "
                f"target={target_node}, seed={seed}") from exc
        joint_act = a1 + act_per * a2
        reward = float(R[joint_act, state])
        total_reward += reward

        trow = T_csr[joint_act].getrow(state)
        next_state = sample_sparse(trow.indices, trow.data, rng)
        orow = O_csr[joint_act].getrow(next_state)
        joint_obs = sample_sparse(orow.indices, orow.data, rng)

        kind = classify_terminal(reward, sink, next_state)
        state = next_state
        if kind != "none":
            terminal_step = step + 1
            terminal_kind = kind
            break

        if step < args.horizon - 1:
            o1 = joint_obs % obs_per
            o2 = joint_obs // obs_per
            next_oh = [
                solve.clustering[step][0][oh[0]][o1],
                solve.clustering[step][1][oh[1]][o2],
            ]
            oh = [0 if x == -1 else int(x) for x in next_oh]

    return TrialResult(
        instance=solve.instance,
        bid=solve.bid,
        target_idx=target_idx,
        target_node=target_node,
        trial_idx=trial_idx + 1,
        seed=seed,
        reward=float(total_reward),
        terminal_step=terminal_step,
        terminal_kind=terminal_kind,
    )


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else math.nan


def _stdev(values: Sequence[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def summarize(trials: Sequence[TrialResult], solves: Sequence[SolveResult]):
    solve_by_bid = {s.bid: s for s in solves}
    rows = []
    by_bid = {}
    for result in trials:
        by_bid.setdefault(result.bid, []).append(result)
    for name, bid in CANONICAL_INSTANCES:
        items = by_bid.get(bid, [])
        if not items:
            continue
        rewards = [r.reward for r in items]
        solve = solve_by_bid[bid]
        rows.append({
            "instance": name,
            "bid": bid,
            "n": len(items),
            "n_targets": len({r.target_idx for r in items}),
            "trials_per_target": len(items) / len({r.target_idx for r in items}),
            "value_mean": _mean(rewards),
            "value_std": _stdev(rewards),
            "value_stderr": _stdev(rewards) / math.sqrt(len(rewards)) if rewards else math.nan,
            "solve_value": solve.value,
            "solve_time": solve.solve_time,
        })
    return rows


def write_outputs(
    trials: Sequence[TrialResult],
    solves: Sequence[SolveResult],
    summary_rows: Sequence[dict],
    args: argparse.Namespace,
) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"aaai_table1_labyrinth_rescue_assist_rsmaa_appx_h{args.horizon}_tpt{args.trials_per_target}"
    trials_path = out_dir / f"{stem}_trials.csv"
    solves_path = out_dir / f"{stem}_solves.csv"
    summary_path = out_dir / f"{stem}_summary.csv"
    md_path = out_dir / f"{stem}_table.md"

    with trials_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "instance", "bid", "target_idx", "target_node", "trial_idx",
                "seed", "reward", "terminal_step", "terminal_kind",
            ],
        )
        writer.writeheader()
        for r in trials:
            writer.writerow(r.__dict__)

    with solves_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["instance", "bid", "solve_value", "solve_time"],
        )
        writer.writeheader()
        for s in solves:
            writer.writerow({
                "instance": s.instance,
                "bid": s.bid,
                "solve_value": f"{s.value:.12g}",
                "solve_time": f"{s.solve_time:.12g}",
            })

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "instance", "bid", "horizon", "n", "n_targets",
                "trials_per_target", "value_mean", "value_std",
                "value_stderr", "solve_value", "solve_time",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            row = dict(row)
            row["horizon"] = args.horizon
            writer.writerow(row)

    with md_path.open("w", newline="") as f:
        f.write("| Instance | h | trials | Approx. RS-MAA* expected value | rollout check | solve time |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            f.write(
                f"| {row['instance']} Search & Rescue | {args.horizon} | "
                f"{row['n']} | "
                f"{row['solve_value']:.2f} | "
                f"{row['value_mean']:.2f} +/- {row['value_stderr']:.2f} | "
                f"{row['solve_time']:.2f} |\n"
            )

    print(f"\nWrote trials:  {trials_path}")
    print(f"Wrote solves:  {solves_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote table:   {md_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run AAAI Table 1 Labyrinth Search & Rescue Approx. RS-MAA* sweeps."
    )
    parser.add_argument("--bids", default="1,2,3,4,5,7")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--trials-per-target", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=4242)
    parser.add_argument("--detection-prob", type=float, default=0.90)
    parser.add_argument("--output-dir", default=os.path.join("results", "aaai_table1"))
    parser.add_argument("--verbose-solver", action="store_true")

    parser.add_argument("--cluster-type", default="lossless")
    parser.add_argument("--maxit", type=int, default=200)
    parser.add_argument("--q-depth", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--iter-limit", type=int, default=1000)
    parser.add_argument("--maxrec", type=int, default=2)
    parser.add_argument("--memory", type=int, default=None)
    parser.add_argument("--heuristic", default="MDP", choices=["MDP", "POMDP"])
    parser.add_argument("--rec-type", default="MDP",
                        choices=["max_reward", "MDP", "rec_state", "recursive"])
    parser.add_argument("--p-threshold-cluster", type=float, default=0.0)
    parser.add_argument("--p-threshold-expand", type=float, default=0.0)
    parser.add_argument("--memory-limit-gb", type=float, default=16.0)
    parser.add_argument("--memory-check-interval", type=int, default=100)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.trials_per_target <= 0:
        parser.error("--trials-per-target must be positive")
    bids = _parse_bids(args.bids)
    instances = _selected_instances(bids)

    print("Approx. RS-MAA* rescue-assist sweep")
    print(
        f"h={args.horizon}, target-balanced={args.trials_per_target}xN, "
        f"detection={args.detection_prob}, "
        "rewards=(assist=100, unassisted=80, wrong=-200, step=-1)"
    )
    print(
        "solver=baselines.decPOMDP.DecPOMDP, "
        f"cluster_type={args.cluster_type}, iter_limit={args.iter_limit}, "
        f"maxrec={args.maxrec}, heuristic={args.heuristic}, rec_type={args.rec_type}"
    )

    all_trials: List[TrialResult] = []
    solves: List[SolveResult] = []
    for instance, bid in instances:
        print(f"\n=== {instance} (bid={bid}) ===", flush=True)
        cache_data = ensure_cache(bid, args.detection_prob)
        targets = [int(x) for x in cache_data["loader"]["targets"]]
        print(f"  targets={len(targets)} ({targets})", flush=True)

        try:
            solve = solve_rsmaa(instance, bid, cache_data, args)
        except MemoryLimitExceeded as exc:
            print(f"  RS-MAA* memory limit exceeded: {exc}", flush=True)
            continue
        except (IndexError, RuntimeError) as exc:
            print(f"  RS-MAA* solve failed: {type(exc).__name__}: {exc}", flush=True)
            continue
        solves.append(solve)
        print(
            f"  solved policy: value={solve.value:.4f}, "
            f"solve_time={solve.solve_time:.2f}s",
            flush=True,
        )

        total = len(targets) * args.trials_per_target
        bid_trials = []
        run_idx = 0
        for target_idx, target_node in enumerate(targets):
            for trial_idx in range(args.trials_per_target):
                run_idx += 1
                seed = seed_for_target_trial(args.seed_base, target_idx, trial_idx)
                result = simulate_policy(
                    solve, cache_data, target_idx, target_node, trial_idx, seed, args)
                bid_trials.append(result)
                all_trials.append(result)
                print(
                    f"  rollout {run_idx}/{total}: target={target_node}, "
                    f"trial={trial_idx + 1}/{args.trials_per_target}, "
                    f"seed={seed}, reward={result.reward:.2f}, "
                    f"terminal={result.terminal_kind}@{result.terminal_step}",
                    flush=True,
                )

        rewards = [r.reward for r in bid_trials]
        print(
            f"  rollout summary: value={_mean(rewards):.2f} +/- "
            f"{(_stdev(rewards) / math.sqrt(len(rewards))):.2f}",
            flush=True,
        )

    summary_rows = summarize(all_trials, solves)
    write_outputs(all_trials, solves, summary_rows, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
