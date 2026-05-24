"""
Compare the spacecraftCA fully decentralized solve through two code paths:

1. RS-SDA* with an empty nonterminal sync set.
2. The baseline RS-MAA* implementation in baselines/decPOMDP.py.

Both paths use the cleaned Dec observation semantics: no miss observation, only
each agent's own deviation bin. The miss-expanded local observation symbols are
therefore compressed to a 3 x 3 joint observation space before solving.
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
from typing import List, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCHMARKS = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_BENCHMARKS)
_BASELINES = os.path.join(_ROOT, "baselines")
for path in (_ROOT, _BENCHMARKS, _HERE, _BASELINES):
    if path not in sys.path:
        sys.path.insert(0, path)

from RSSDA import (  # noqa: E402
    MemoryLimitExceeded as RssdaMemoryLimitExceeded,
    RSSDAConfig,
    SDecPOMDP,
    SDecPOMDPModel,
)
from baselines.decPOMDP import (  # noqa: E402
    DecPOMDP as RSMAA,
    MemoryLimitExceeded as RsmaaMemoryLimitExceeded,
)
from spacecraft_discretizer import (  # noqa: E402
    DEV_ZERO,
    N_DEV,
    N_MISS,
    N_STAGES,
    N_STATES_TOTAL,
    state_index,
)
from spacecraft_matrices import (  # noqa: E402
    N_ACT_AGENT,
    N_JOINT_ACTIONS,
    N_JOINT_OBS,
    N_MISS_OBS,
    N_OBS_AGENT,
    load_matrices,
)


def parse_bins(text: str) -> List[int]:
    bins = [int(part.strip()) for part in text.split(",") if part.strip()]
    bad = [b for b in bins if b < 0 or b >= N_MISS]
    if bad:
        raise argparse.ArgumentTypeError(
            f"Invalid miss bins {bad}; expected 0..{N_MISS - 1}"
        )
    return bins


def make_init_b(belief_bins: List[int]) -> np.ndarray:
    belief = np.zeros(N_STATES_TOTAL, dtype=np.float64)
    for mb in belief_bins:
        belief[state_index(mb, DEV_ZERO, DEV_ZERO, 0)] = 1.0
    belief /= belief.sum()
    return belief


def compact_decentralized_observations(O_full: np.ndarray) -> np.ndarray:
    """Compress miss-expanded local observations to own-deviation-only 3 x 3."""
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
    """Convert (action, row, col) dense arrays to decPOMDP sparse row tuples."""
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


def solve_rssda(T, O, R, init_b, args) -> dict:
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
        nobs_factor=[N_DEV, N_DEV],
        sync_states=[],
        sync_actions=[],
        sync_observations=[],
    )
    config = RSSDAConfig(
        maxh=args.horizon,
        algorithm="approximate",
        TI1=False,
        TI2=True,
        TI3=True,
        TI4=True,
        iter_limit=args.iter_limit,
        rec_limit=args.rec_limit,
        max_clusters=args.max_clusters,
        heuristic_type=args.rssda_heuristic,
        memory_limit_gb=args.memory_limit_gb,
        memory_check_interval=args.memory_check_interval,
    )
    solver = SDecPOMDP(model=model, config=config)

    t0 = time.perf_counter()
    result = solver.multi_agent_astar(args.horizon)
    elapsed = time.perf_counter() - t0
    policy = result[1] if result and len(result) > 1 else None
    return {
        "solver": "RS-SDA*_empty_sync",
        "status": "ok",
        "value": float(result[0]),
        "solve_seconds": elapsed,
        "policy_len": len(policy) if policy is not None else 0,
    }


def solve_rsmaa(T, O, R, init_b, args) -> dict:
    t0 = time.perf_counter()
    T_pdict = dense_rows_to_pdict(T)
    O_pdict = dense_rows_to_pdict(O)
    convert_seconds = time.perf_counter() - t0

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
        maxh=args.horizon,
        cluster_type=args.rsmaa_cluster_type,
        maxit=args.rsmaa_maxit,
        q_depth=args.rsmaa_q_depth,
        alpha=args.rsmaa_alpha,
        iter_limit=args.iter_limit,
        maxrec=args.rec_limit,
        memory=args.rsmaa_memory,
        heuristic=args.rsmaa_heuristic,
        rec_type=args.rsmaa_rec_type,
        p_threshold_cluster=args.p_threshold_cluster,
        p_threshold_expand=args.p_threshold_expand,
        policyvalfound=-math.inf,
        output=args.verbose_solver,
        memory_limit_gb=args.memory_limit_gb,
        memory_check_interval=args.memory_check_interval,
    )
    solver.decentralized = True
    solver.onesided = False

    t1 = time.perf_counter()
    if args.verbose_solver:
        value, policy, clustering = solver.multi_agent_astar(args.horizon)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            value, policy, clustering = solver.multi_agent_astar(args.horizon)
    solve_seconds = time.perf_counter() - t1
    return {
        "solver": "RS-MAA*",
        "status": "ok",
        "value": float(value),
        "solve_seconds": solve_seconds,
        "convert_seconds": convert_seconds,
        "policy_len": len(policy) if policy is not None else 0,
        "cluster_len": len(clustering) if clustering is not None else 0,
    }


def failure_row(solver: str, exc: Exception, elapsed: float) -> dict:
    return {
        "solver": solver,
        "status": type(exc).__name__,
        "error": str(exc),
        "solve_seconds": elapsed,
    }


def write_rows(rows: List[dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fields = [
        "solver", "status", "value", "solve_seconds", "convert_seconds",
        "policy_len", "cluster_len", "error", "horizon", "belief_bins",
        "iter_limit", "rec_limit", "max_clusters", "memory_limit_gb",
        "rsmaa_cluster_type", "rsmaa_heuristic", "rsmaa_rec_type",
        "rssda_heuristic",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["both", "rssda", "rsmaa"], default="both")
    parser.add_argument("--horizon", type=int, default=N_STAGES)
    parser.add_argument("--belief-bins", type=parse_bins, default=parse_bins("0,1,2"))
    parser.add_argument("--iter-limit", type=int, default=1000)
    parser.add_argument("--rec-limit", type=int, default=2)
    parser.add_argument("--max-clusters", type=int, default=20)
    parser.add_argument("--memory-limit-gb", type=float, default=16.0)
    parser.add_argument("--memory-check-interval", type=int, default=100)
    parser.add_argument("--rssda-heuristic", default="HYBRID")
    parser.add_argument("--rsmaa-cluster-type", default="lossless")
    parser.add_argument("--rsmaa-maxit", type=int, default=200)
    parser.add_argument("--rsmaa-q-depth", type=int, default=3)
    parser.add_argument("--rsmaa-alpha", type=float, default=0.2)
    parser.add_argument("--rsmaa-memory", type=int, default=None)
    parser.add_argument("--rsmaa-heuristic", default="MDP", choices=["MDP", "POMDP"])
    parser.add_argument(
        "--rsmaa-rec-type",
        default="MDP",
        choices=["max_reward", "MDP", "rec_state", "recursive"],
    )
    parser.add_argument("--p-threshold-cluster", type=float, default=0.0)
    parser.add_argument("--p-threshold-expand", type=float, default=0.0)
    parser.add_argument("--verbose-solver", action="store_true")
    parser.add_argument("--tag", default="dec_rsmaa_compare")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(_HERE, "notes", "results"),
    )
    args = parser.parse_args()

    T, O_full, R, _, _, dv = load_matrices("dec")
    O = compact_decentralized_observations(O_full)
    init_b = make_init_b(args.belief_bins)

    print("spacecraftCA Dec timing comparison")
    print(f"  horizon={args.horizon}, belief_bins={args.belief_bins}, dv={dv}")
    print(
        f"  states={N_STATES_TOTAL}, actions={N_JOINT_ACTIONS}, "
        f"local_obs={N_DEV}, joint_obs={O.shape[2]}"
    )
    print(
        f"  iter_limit={args.iter_limit}, rec_limit={args.rec_limit}, "
        f"memory_limit={args.memory_limit_gb}GB"
    )

    rows = []
    if args.which in ("both", "rssda"):
        print("\nSolving via RS-SDA* with empty sync set...", flush=True)
        t0 = time.perf_counter()
        try:
            row = solve_rssda(T, O, R, init_b, args)
        except RssdaMemoryLimitExceeded as exc:
            row = failure_row("RS-SDA*_empty_sync", exc, time.perf_counter() - t0)
        rows.append(row)
        print(f"  {row['status']}: value={row.get('value', '')} "
              f"time={float(row.get('solve_seconds', 0.0)):.2f}s")

    if args.which in ("both", "rsmaa"):
        print("\nSolving via baseline RS-MAA*...", flush=True)
        t0 = time.perf_counter()
        try:
            row = solve_rsmaa(T, O, R, init_b, args)
        except RsmaaMemoryLimitExceeded as exc:
            row = failure_row("RS-MAA*", exc, time.perf_counter() - t0)
        rows.append(row)
        print(f"  {row['status']}: value={row.get('value', '')} "
              f"time={float(row.get('solve_seconds', 0.0)):.2f}s "
              f"convert={float(row.get('convert_seconds', 0.0)):.2f}s")

    for row in rows:
        row.update({
            "horizon": args.horizon,
            "belief_bins": ",".join(str(x) for x in args.belief_bins),
            "iter_limit": args.iter_limit,
            "rec_limit": args.rec_limit,
            "max_clusters": args.max_clusters,
            "memory_limit_gb": args.memory_limit_gb,
            "rsmaa_cluster_type": args.rsmaa_cluster_type,
            "rsmaa_heuristic": args.rsmaa_heuristic,
            "rsmaa_rec_type": args.rsmaa_rec_type,
            "rssda_heuristic": args.rssda_heuristic,
        })

    out_path = os.path.join(args.out_dir, f"{args.tag}.csv")
    write_rows(rows, out_path)
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
