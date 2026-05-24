"""Probe exact fully centralized RS-SDA* on one rescue-assist Labyrinth instance."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from RSSDA import (  # noqa: E402
    MemoryLimitExceeded,
    RSSDAConfig,
    SDecPOMDP,
    SDecPOMDPModel,
    int_tuple,
)
from labyrinth_cache import (  # noqa: E402
    load_cached_rescue_assist_labyrinth,
    precompute_rescue_assist_all,
)


def ensure_cache(args: argparse.Namespace):
    cache = load_cached_rescue_assist_labyrinth(
        args.bid,
        args.detection_prob,
        assist_reward=args.assist_reward,
        unassisted_reward=args.unassisted_reward,
        wrong_reward=args.wrong_reward,
        step_cost=args.step_cost,
    )
    if cache is None:
        precompute_rescue_assist_all(
            args.bid,
            args.detection_prob,
            assist_reward=args.assist_reward,
            unassisted_reward=args.unassisted_reward,
            wrong_reward=args.wrong_reward,
            step_cost=args.step_cost,
        )
        cache = load_cached_rescue_assist_labyrinth(
            args.bid,
            args.detection_prob,
            assist_reward=args.assist_reward,
            unassisted_reward=args.unassisted_reward,
            wrong_reward=args.wrong_reward,
            step_cost=args.step_cost,
        )
    if cache is None:
        raise RuntimeError(f"Could not load rescue-assist cache for bid={args.bid}")
    return cache


def solve(args: argparse.Namespace) -> int:
    cache = ensure_cache(args)
    cfg = cache["config"]
    loader = cache["loader"]
    sync_states = list(range(int(cfg["nstates"]) - 1))

    print("Exact fully centralized RS-SDA* probe")
    print(
        f"bid={args.bid}, horizon={args.horizon}, "
        f"detection_prob={args.detection_prob}"
    )
    print(
        f"nstates={cfg['nstates']}, nactions={cfg['nacts']}, "
        f"nobs={cfg['nobs']}, targets={loader['targets']}"
    )
    print(f"sync_states={len(sync_states)} / {cfg['nstates']} (all non-sink states)")
    print("algorithm=exact, heuristic_type=POMDP, TI1=False, TI2=False, TI3=False, TI4=False")

    model = SDecPOMDPModel(
        nagents=int(cfg["nagents"]),
        nstates=int(cfg["nstates"]),
        nactions=int(cfg["nacts"]),
        nobs=int(cfg["nobs"]),
        transitions=cache.get("T"),
        obs=cache.get("O"),
        rewards=cache.get("R"),
        init_beliefs=cache["init_beliefs"],
        nacts_factor=cfg["nacts_factor"],
        nobs_factor=cfg["nobs_factor"],
        cached_data=cache,
        sync_states=sync_states,
        sync_actions=[],
        sync_observations=[],
    )
    model.valid_actions_per_state = cache.get("valid_actions_per_state")
    model.valid_actions_per_position = cache.get("valid_actions_per_position")

    solver_config = RSSDAConfig(
        maxh=args.horizon,
        maxit=args.maxit,
        IEmin2=args.ie_min2,
        alpha=args.alpha,
        algorithm="exact",
        TI1=False,
        TI2=False,
        TI3=False,
        TI4=False,
        iter_limit=10**9,
        rec_limit=2,
        heuristic_type="POMDP",
        tail_heuristic_type="POMDP",
        hybrid_r=0,
        max_clusters=20,
        memory_limit_gb=args.memory_limit_gb,
        memory_check_interval=args.memory_check_interval,
        output=args.verbose_solver,
    )

    solver = SDecPOMDP(model=model, config=solver_config, qmdp_data=None)
    init_idx = solver.dist_dict[int_tuple(cache["init_beliefs"])]

    start = time.time()
    try:
        value, policy, _clustering, cent_vector, _cen_dists_map, _clustering_cen = (
            solver.multi_agent_astar(args.horizon, init_beliefs=init_idx)
        )
    except MemoryLimitExceeded as exc:
        elapsed = time.time() - start
        print(f"RESULT memory_limit_exceeded elapsed_sec={elapsed:.6f} error={exc}")
        return 2

    elapsed = time.time() - start
    print(f"RESULT value={value:.12g}")
    print(f"RESULT elapsed_sec={elapsed:.6f}")
    print(f"RESULT policy_stages={len(policy) if policy is not None else None}")
    print(f"RESULT cent_vector={cent_vector}")
    print(
        "RESULT centralized_stages="
        f"{sum(1 for x in cent_vector if x)} / {len(cent_vector)}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bid", default="1")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--detection-prob", type=float, default=0.90)
    parser.add_argument("--assist-reward", type=float, default=100.0)
    parser.add_argument("--unassisted-reward", type=float, default=80.0)
    parser.add_argument("--wrong-reward", type=float, default=-200.0)
    parser.add_argument("--step-cost", type=float, default=-1.0)
    parser.add_argument("--maxit", type=int, default=200)
    parser.add_argument("--ie-min2", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--memory-limit-gb", type=float, default=16.0)
    parser.add_argument("--memory-check-interval", type=int, default=100)
    parser.add_argument("--verbose-solver", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.ie_min2 is None:
        args.ie_min2 = max(3, args.horizon - 3)
    return solve(args)


if __name__ == "__main__":
    raise SystemExit(main())
