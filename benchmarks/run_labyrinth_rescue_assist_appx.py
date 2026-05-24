"""Run Approximate RS-SDA* on canonical Labyrinth Rescue/Assist instances.

This fills the Approx. RS-SDA* column for the AAAI Table 1 Search & Rescue
Labyrinth rows.  Runs are target-balanced by default: each possible target node
is evaluated with three seeded trials, for 3N total trials where N is the
number of possible targets in that instance.  The benchmark configuration is
the canonical rescue-assist variant:

    detection_prob = 0.90
    rewards = (assist=100, unassisted=80, wrong=-200, step=-1)

By default, runs use the historical Labyrinth approximate profile from the
archived runner: TI1/TI2/TI3 enabled, TI4 disabled, HYBRID heuristics,
iter_limit=300, rec_limit=2.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import sdec_labyrinth as lab  # noqa: E402
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
class TrialResult:
    instance: str
    bid: str
    target_idx: int
    target_node: int
    trial_idx: int
    seed: int
    reward: float
    plan_time: float
    wall_time: float


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


def apply_approx_profile(args: argparse.Namespace) -> None:
    lab.ALGORITHM = "approximate"
    lab.TI1 = True
    lab.TI2 = True
    lab.TI3 = True
    lab.TI4 = False
    lab.HEURISTIC_TYPE = args.heuristic
    lab.TAIL_HEURISTIC_TYPE = args.tail_heuristic
    lab.HYBRID_R = args.hybrid_r
    lab.ITER_LIMIT = args.iter_limit
    lab.REC_LIMIT = args.rec_limit
    lab.MAX_CLUSTERS = args.max_clusters
    lab.SCORE_LIMIT = args.score_limit
    lab.CEN_THRESHOLD = args.cen_threshold
    lab.SM_TEMPERATURE = args.sm_temperature
    lab.ADAPTIVE_CHECK = args.adaptive_check


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


def target_nodes_from_cache(cache_data) -> List[int]:
    targets = cache_data.get("loader", {}).get("targets", [])
    if not targets:
        raise ValueError("Rescue-assist cache did not include target nodes")
    return [int(t) for t in targets]


def seed_for_target_trial(seed_base: int, target_idx: int, trial_idx: int) -> int:
    return int(seed_base + target_idx * 1000 + trial_idx * 10_000_000)


def run_one(
    instance: str,
    bid: str,
    target_idx: int,
    target_node: int,
    trial_idx: int,
    seed: int,
    args: argparse.Namespace,
) -> TrialResult:
    def _execute():
        config = lab.LabyrinthConfig(
            bid,
            args.horizon,
            args.maxit,
            args.ie_min2,
            args.alpha,
            replan_at_all_syncs=args.replan_syncs,
            decentralized=False,
            centralized=False,
            noisy=True,
            detection_prob=args.detection_prob,
            seed=seed,
            rescue_assist=True,
        )
        return lab.run_labyrinth(
            config, verbose=args.verbose, fixed_target_idx=target_idx)

    start = time.time()
    if args.verbose:
        reward, plan_time = _execute()
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            reward, plan_time = _execute()
    wall_time = time.time() - start
    if isinstance(reward, str):
        raise RuntimeError(f"Planner returned nonnumeric result {reward!r}")
    return TrialResult(
        instance=instance,
        bid=bid,
        target_idx=target_idx,
        target_node=target_node,
        trial_idx=trial_idx + 1,
        seed=seed,
        reward=float(reward),
        plan_time=float(plan_time),
        wall_time=float(wall_time),
    )


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else math.nan


def _stdev(values: Sequence[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def summarize(results: Sequence[TrialResult]):
    rows = []
    by_bid = {}
    for result in results:
        by_bid.setdefault(result.bid, []).append(result)
    for name, bid in CANONICAL_INSTANCES:
        trials = by_bid.get(bid, [])
        if not trials:
            continue
        rewards = [r.reward for r in trials]
        plan_times = [r.plan_time for r in trials]
        wall_times = [r.wall_time for r in trials]
        target_count = len({r.target_idx for r in trials})
        trials_per_target = len(trials) / target_count if target_count else math.nan
        rows.append({
            "instance": name,
            "bid": bid,
            "horizon": None,
            "n": len(trials),
            "n_targets": target_count,
            "trials_per_target": trials_per_target,
            "value_mean": _mean(rewards),
            "value_std": _stdev(rewards),
            "value_stderr": _stdev(rewards) / math.sqrt(len(rewards)) if rewards else math.nan,
            "plan_time_mean": _mean(plan_times),
            "wall_time_mean": _mean(wall_times),
        })
    return rows


def write_outputs(
    results: Sequence[TrialResult],
    summary_rows: Sequence[dict],
    args: argparse.Namespace,
) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        "aaai_table1_labyrinth_rescue_assist_appx_"
        f"h{args.horizon}_tpt{args.trials_per_target}"
    )
    trials_path = out_dir / f"{stem}_trials.csv"
    summary_path = out_dir / f"{stem}_summary.csv"
    md_path = out_dir / f"{stem}_table.md"

    with trials_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "instance",
                "bid",
                "target_idx",
                "target_node",
                "trial_idx",
                "seed",
                "reward",
                "plan_time",
                "wall_time",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow({
                "instance": r.instance,
                "bid": r.bid,
                "target_idx": r.target_idx,
                "target_node": r.target_node,
                "trial_idx": r.trial_idx,
                "seed": r.seed,
                "reward": f"{r.reward:.12g}",
                "plan_time": f"{r.plan_time:.12g}",
                "wall_time": f"{r.wall_time:.12g}",
            })

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "instance",
                "bid",
                "horizon",
                "n",
                "n_targets",
                "trials_per_target",
                "value_mean",
                "value_std",
                "value_stderr",
                "plan_time_mean",
                "wall_time_mean",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            row = dict(row)
            row["horizon"] = args.horizon
            writer.writerow(row)

    with md_path.open("w", newline="") as f:
        f.write("| Instance | h | trials | Approx. RS-SDA* value | time |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for row in summary_rows:
            f.write(
                f"| {row['instance']} Search & Rescue | {args.horizon} | "
                f"{row['n']} | "
                f"{row['value_mean']:.2f} +/- {row['value_stderr']:.2f} | "
                f"{row['plan_time_mean']:.2f} |\n"
            )

    print(f"\nWrote trials:  {trials_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote table:   {md_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run AAAI Table 1 Labyrinth Search & Rescue Approx. RS-SDA* sweeps."
    )
    parser.add_argument("--bids", default="1,2,3,4,5,7",
                        help="Comma-separated bid list or instance names.")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument(
        "--trials-per-target",
        type=int,
        default=3,
        help=(
            "Seeded trials per possible target node; total trials are "
            "trials_per_target*N."
        ),
    )
    parser.add_argument("--seed-base", type=int, default=4242)
    parser.add_argument("--detection-prob", type=float, default=0.90)
    parser.add_argument("--maxit", type=int, default=200)
    parser.add_argument("--ie-min2", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--replan-syncs", action="store_true")
    parser.add_argument("--output-dir", default=os.path.join("results", "aaai_table1"))
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--iter-limit", type=int, default=300)
    parser.add_argument("--rec-limit", type=int, default=2)
    parser.add_argument("--heuristic", default="HYBRID", choices=["QMDP", "POMDP", "HYBRID"])
    parser.add_argument("--tail-heuristic", default="HYBRID", choices=["QMDP", "POMDP", "HYBRID"])
    parser.add_argument("--hybrid-r", type=int, default=1)
    parser.add_argument("--max-clusters", type=int, default=2)
    parser.add_argument("--score-limit", type=int, default=20)
    parser.add_argument("--cen-threshold", type=float, default=0.6)
    parser.add_argument("--sm-temperature", type=float, default=0.6)
    parser.add_argument("--adaptive-check", type=int, default=100)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.trials_per_target <= 0:
        parser.error("--trials-per-target must be positive")
    bids = _parse_bids(args.bids)
    instances = _selected_instances(bids)
    apply_approx_profile(args)

    print("Approx. RS-SDA* rescue-assist sweep")
    print(
        f"h={args.horizon}, target-balanced={args.trials_per_target}xN, "
        f"detection={args.detection_prob}, "
        "rewards=(assist=100, unassisted=80, wrong=-200, step=-1)"
    )
    print(
        f"profile: algorithm={lab.ALGORITHM}, TI1={lab.TI1}, TI2={lab.TI2}, "
        f"TI3={lab.TI3}, TI4={lab.TI4}, iter_limit={lab.ITER_LIMIT}, "
        f"rec_limit={lab.REC_LIMIT}, heuristic={lab.HEURISTIC_TYPE}"
    )

    all_results: List[TrialResult] = []
    for instance, bid in instances:
        print(f"\n=== {instance} (bid={bid}) ===")
        cache_data = ensure_cache(bid, args.detection_prob)
        target_nodes = target_nodes_from_cache(cache_data)
        total_trials = len(target_nodes) * args.trials_per_target
        print(
            f"  targets={len(target_nodes)} ({target_nodes}); "
            f"trials_per_target={args.trials_per_target}; total_trials={total_trials}"
        )
        bid_results = []
        run_idx = 0
        for target_idx, target_node in enumerate(target_nodes):
            for trial_idx in range(args.trials_per_target):
                run_idx += 1
                seed = seed_for_target_trial(args.seed_base, target_idx, trial_idx)
                print(
                    f"  trial {run_idx}/{total_trials}: target_idx={target_idx} "
                    f"node={target_node}, trial={trial_idx + 1}/{args.trials_per_target}, "
                    f"seed={seed}",
                    flush=True,
                )
                result = run_one(
                    instance, bid, target_idx, target_node, trial_idx, seed, args)
                bid_results.append(result)
                all_results.append(result)
                print(
                    f"    reward={result.reward:.2f}, "
                    f"plan_time={result.plan_time:.2f}s, wall={result.wall_time:.2f}s",
                    flush=True,
                )
        rewards = [r.reward for r in bid_results]
        times = [r.plan_time for r in bid_results]
        print(
            f"  summary: value={_mean(rewards):.2f} +/- "
            f"{(_stdev(rewards) / math.sqrt(len(rewards))):.2f}, "
            f"time={_mean(times):.2f}s"
        )

    summary_rows = summarize(all_results)
    for row in summary_rows:
        row["horizon"] = args.horizon
    write_outputs(all_results, summary_rows, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
