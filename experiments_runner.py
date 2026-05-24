"""Bounded experiment-performance runner for RSSDA benchmarks.

This script is intentionally not a full paper-table reproduction.  It verifies
tractable AAMAS/IJCAI paper anchors numerically, then smoke-tests the larger
approximate and spacecraft workflows with bounded settings so regressions show
up without requiring every 20-minute timeout case to be rerun.
"""

from __future__ import annotations

import contextlib
import csv
import io
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
BENCH = ROOT / "benchmarks"
BASELINES = ROOT / "baselines"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BASELINES))


@dataclass
class Check:
    name: str
    status: str
    seconds: float
    value: float | None = None
    target: float | None = None
    detail: str = ""

    @property
    def gap(self) -> float | None:
        if self.value is None or self.target is None:
            return None
        return self.value - self.target


def finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"non-finite value {text!r}")
    return value


def value_check(name: str, value: float, target: float, seconds: float,
                tol: float = 5e-4, detail: str = "") -> Check:
    status = "PASS" if abs(value - target) <= tol else "FAIL"
    return Check(name, status, seconds, value, target, detail)


def command_check(name: str, args: list[str], timeout: int,
                  pattern: str | None = None, target: float | None = None,
                  tol: float = 5e-4, cwd: Path = ROOT) -> Check:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, *args],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - t0
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        tail = "\n".join(combined.splitlines()[-12:])
        return Check(name, "FAIL", elapsed, detail=f"exit={proc.returncode}; {tail}")

    if pattern is None:
        return Check(name, "PASS", elapsed)

    match = re.search(pattern, combined)
    if not match:
        tail = "\n".join(combined.splitlines()[-12:])
        return Check(name, "FAIL", elapsed, detail=f"pattern not found; {tail}")
    value = finite_float(match.group(1))
    if target is None:
        return Check(name, "PASS", elapsed, value=value)
    return value_check(name, value, target, elapsed, tol)


def print_check(check: Check) -> None:
    value = "" if check.value is None else f" value={check.value:.5f}"
    target = "" if check.target is None else f" target={check.target:.5f}"
    gap = "" if check.gap is None else f" gap={check.gap:+.5f}"
    detail = "" if not check.detail else f" :: {check.detail}"
    print(f"{check.status:4} {check.name:<48} {check.seconds:7.2f}s{value}{target}{gap}{detail}")


def exact_config(horizon: int):
    from RSSDA import RSSDAConfig

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


def solve_exact_model(name: str, horizon: int, model, target: float,
                      tol: float = 5e-4) -> Check:
    from RSSDA import SDecPOMDP, int_tuple

    t0 = time.perf_counter()
    solver = SDecPOMDP(model=model, config=exact_config(horizon))
    init_idx = solver.dist_dict[int_tuple(solver.init_beliefs)]
    value, *_ = solver.multi_agent_astar(horizon, init_beliefs=init_idx)
    return value_check(name, float(value), target, time.perf_counter() - t0, tol)


def make_model(nagents: int, nstates: int, nactions: int, nobs: int,
               transitions, obs, rewards, init_beliefs,
               nacts_factor, nobs_factor, sync_states=None,
               sync_actions=None):
    from RSSDA import SDecPOMDPModel

    return SDecPOMDPModel(
        nagents=nagents,
        nstates=nstates,
        nactions=nactions,
        nobs=nobs,
        transitions=transitions,
        obs=obs,
        rewards=rewards,
        init_beliefs=init_beliefs,
        nacts_factor=nacts_factor,
        nobs_factor=nobs_factor,
        sync_states=sync_states or [],
        sync_actions=sync_actions or [],
        sync_observations=[],
    )


def with_argv(argv: list[str]):
    @contextlib.contextmanager
    def _ctx():
        old = sys.argv[:]
        sys.argv = argv[:]
        try:
            yield
        finally:
            sys.argv = old
    return _ctx()


def run_aamas_exact() -> list[Check]:
    checks: list[Check] = []

    # SDec-Tiger exact Table 2 anchors.
    import verify_aamas_table2

    for mode, target, acc, sync in [
        ("decentralized", 12.21726, 0.85, []),
        ("semi", 27.21518, 0.75, [8]),
        ("centralized", 47.71696, 0.85, list(range(9))),
    ]:
        t0 = time.perf_counter()
        value, _ = verify_aamas_table2.solve(mode, acc, sync, horizon=8)
        checks.append(value_check(
            f"AAMAS Tiger h=8 {mode}", float(value), target,
            time.perf_counter() - t0))

    # A slightly larger Tiger row remains tractable and catches h-dependent
    # regressions without rerunning the long h=10 decentralized case.
    for mode, target, acc, sync in [
        ("decentralized", 15.57244, 0.85, []),
        ("semi", 30.90457, 0.75, [8]),
        ("centralized", 53.47353, 0.85, list(range(9))),
    ]:
        t0 = time.perf_counter()
        value, _ = verify_aamas_table2.solve(mode, acc, sync, horizon=9)
        checks.append(value_check(
            f"AAMAS Tiger h=9 {mode}", float(value), target,
            time.perf_counter() - t0))

    import sdec_fireFight3houses as fire

    with with_argv(["sdec_fireFight3houses.py", "4"]):
        fcfg = fire.FireFightConfig()
    T, O, R, init = fire.FireFightProblemLoader(fcfg).load_data()
    for horizon, targets in [
        (3, (-5.73697, -5.73697, -5.72285)),
        (4, (-6.57883, -6.57883, -6.51859)),
    ]:
        for label, sync_actions, target in [
            ("dec", [], targets[0]),
            ("semi", fire.FireFightConfig.TRIG_SEMI[0], targets[1]),
            ("central", fire.FireFightConfig.TRIG_CENTRALIZED[0], targets[2]),
        ]:
            model = make_model(
                2, fire.FireFightConfig.NSTATES, fcfg.nacts, fcfg.nobs,
                T, O, R, init, fcfg.nacts_factor, fcfg.nobs_factor,
                sync_actions=sync_actions)
            checks.append(solve_exact_model(
                f"AAMAS Fire h={horizon} {label}", horizon, model, target))

    import sdec_box as box

    with with_argv(["sdec_box.py", "4"]):
        bcfg = box.BoxConfig()
    T, O, R, init = box.BoxProblemLoader(bcfg).load_data()
    for horizon, targets in [
        (3, (66.08100, 66.81000, 66.81000)),
        (4, (98.59361, 99.55630, 99.55630)),
    ]:
        for label, sync_states, target in [
            ("dec", [], targets[0]),
            ("semi", box.BoxConfig.TRIG_SEMI, targets[1]),
            ("central", box.BoxConfig.TRIG_CENTRALIZED, targets[2]),
        ]:
            model = make_model(
                2, box.BoxConfig.NSTATES, bcfg.nacts, bcfg.nobs,
                T, O, R, init, bcfg.nacts_factor, bcfg.nobs_factor,
                sync_states=sync_states)
            checks.append(solve_exact_model(
                f"AAMAS Box h={horizon} {label}", horizon, model, target))

    import sdec_mars as mars

    for horizon, targets in [
        (4, (10.18080, 10.18080, 10.87020)),
        (5, (13.26654, 14.29038, 14.38556)),
        (6, (18.62317, 20.06430, 20.06706)),
    ]:
        with with_argv(["sdec_mars.py", str(horizon)]):
            mcfg = mars.MarsConfig()
        with contextlib.redirect_stdout(io.StringIO()):
            T, O, R, init = mars.MarsProblemLoader(mcfg).load_data()
        for label, sync_states, target in [
            ("dec", [], targets[0]),
            ("semi", mars.mars_right_band_triggers(), targets[1]),
            ("central", list(range(mcfg.nstates)), targets[2]),
        ]:
            model = make_model(
                2, mcfg.nstates, mcfg.nacts, mcfg.nobs,
                T, O, R, init, mcfg.nacts_factor, mcfg.nobs_factor,
                sync_states=sync_states)
            checks.append(solve_exact_model(
                f"AAMAS Mars right h={horizon} {label}",
                horizon, model, target))

    import maritimemedevac as med

    T, O, R, init = med.build_problem()
    for horizon, targets in [
        (6, (3.18348, 3.19807, 3.19945)),
        (7, (3.26710, 6.36301, 6.61819)),
    ]:
        for label, sync_states, target in [
            ("dec", med.triggers_none(), targets[0]),
            ("semi", med.triggers_semi(), targets[1]),
            ("central", med.triggers_full(), targets[2]),
        ]:
            model = make_model(
                2, med.nstates, med.nactions, med.nobs,
                T, O, R, init, [med.a_per_agent] * 2,
                [med.o_per_agent] * 2, sync_states=sync_states)
            checks.append(solve_exact_model(
                f"AAMAS Maritime h={horizon} {label}",
                horizon, model, target))

    return checks


def run_ijcai_labyrinth_exact() -> list[Check]:
    exp_pattern = r"Exp\. Value:\s*([-+]?\d+(?:\.\d+)?)"
    dec_pattern = r"Expected Value \(decentralized\):\s*([-+]?\d+(?:\.\d+)?)"
    cases = [
        ("ExtCross9", "1", 6, 71.25, 84.375, 84.375),
        ("LopsidedY10", "2", 5, 40.7778, 74.6667, 74.6667),
        ("Ladder10", "3", 5, 40.7778, 74.5556, 74.6667),
        ("Maze12", "4", 5, 32.4545, 60.0909, 60.1818),
        ("HiddenTail11", "5", 4, 36.8, 57.2, 57.2),
        ("Mesh10", "7", 5, 52.0, 85.6667, 85.6667),
    ]
    checks: list[Check] = []
    for name, bid, horizon, dec, semi, central in cases:
        checks.append(command_check(
            f"IJCAI {name} h={horizon} centralized",
            ["benchmarks/sdec_labyrinth.py", bid, str(horizon), "1", "--centralized"],
            timeout=300, pattern=exp_pattern, target=central))
        checks.append(command_check(
            f"IJCAI {name} h={horizon} semi",
            ["benchmarks/sdec_labyrinth.py", bid, str(horizon), "1"],
            timeout=300, pattern=exp_pattern, target=semi))
        checks.append(command_check(
            f"IJCAI {name} h={horizon} decentralized",
            ["benchmarks/sdec_labyrinth.py", bid, str(horizon), "1", "--decentralized"],
            timeout=300, pattern=dec_pattern, target=dec))
    return checks


def read_single_summary_value(path: Path) -> float:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"empty summary CSV: {path}")
    for key in ("appx_mean", "value_mean", "expected_return"):
        if key in rows[0]:
            return finite_float(rows[0][key])
    raise RuntimeError(f"no known value column in {path}")


def run_ijcai_approx_smoke(tmp: Path) -> list[Check]:
    checks: list[Check] = []

    tiger_out = tmp / "sdec_tiger_appx.csv"
    check = command_check(
        "IJCAI Tiger approximate smoke",
        [
            "benchmarks/run_sdec_tiger_corrected_appx.py",
            "--horizons", "10",
            "--trials", "3",
            "--trigger-mode", "semi",
            "--profile", "paper",
            "--skip-exact",
            "--progress", "0",
            "--out", str(tiger_out),
        ],
        timeout=300)
    if check.status == "PASS":
        try:
            check.value = read_single_summary_value(tiger_out)
        except Exception as exc:  # pragma: no cover - diagnostic path
            check.status = "FAIL"
            check.detail = str(exc)
    checks.append(check)

    appx_dir = tmp / "labyrinth_appx"
    check = command_check(
        "IJCAI rescue-assist Approx RS-SDA* smoke",
        [
            "benchmarks/run_labyrinth_rescue_assist_appx.py",
            "--bids", "1",
            "--horizon", "2",
            "--trials-per-target", "1",
            "--output-dir", str(appx_dir),
        ],
        timeout=300)
    if check.status == "PASS":
        try:
            check.value = read_single_summary_value(
                appx_dir / "aaai_table1_labyrinth_rescue_assist_appx_h2_tpt1_summary.csv")
        except Exception as exc:
            check.status = "FAIL"
            check.detail = str(exc)
    checks.append(check)

    rsmaa_dir = tmp / "labyrinth_rsmaa_appx"
    check = command_check(
        "IJCAI rescue-assist RS-MAA* smoke",
        [
            "benchmarks/run_labyrinth_rescue_assist_rsmaa_appx.py",
            "--bids", "1",
            "--horizon", "2",
            "--trials-per-target", "1",
            "--output-dir", str(rsmaa_dir),
            "--iter-limit", "50",
            "--maxrec", "1",
        ],
        timeout=300)
    if check.status == "PASS":
        try:
            check.value = read_single_summary_value(
                rsmaa_dir / "aaai_table1_labyrinth_rescue_assist_rsmaa_appx_h2_tpt1_summary.csv")
        except Exception as exc:
            check.status = "FAIL"
            check.detail = str(exc)
    checks.append(check)
    return checks


def run_spacecraft(tmp: Path) -> list[Check]:
    checks = [
        command_check(
            "SpacecraftCA matrix verification",
            ["benchmarks/spacecraftCA/spacecraft_matrices.py", "--verify"],
            timeout=300),
    ]

    out_dir = tmp / "spacecraft_results"
    fig_dir = tmp / "spacecraft_figures"
    tag = "regression_round_robin"
    check = command_check(
        "SpacecraftCA expected comparison",
        [
            "benchmarks/spacecraftCA/compare_variants.py",
            "--eval-mode", "expected",
            "--variants", "centralized,sdec,dec",
            "--solver-modes", "fixed",
            "--belief-bins", "0,1,2",
            "--eval-bins", "0,1,2",
            "--dec-solver", "rsmaa",
            "--iter-limit", "1000",
            "--max-solve-seconds", "120",
            "--max-clusters", "20",
            "--rec-limit", "2",
            "--no-plots",
            "--tag", tag,
            "--out-dir", str(out_dir),
            "--fig-dir", str(fig_dir),
        ],
        timeout=700)
    if check.status == "PASS":
        expected_path = out_dir / f"variant_expected_{tag}.csv"
        try:
            with expected_path.open(newline="") as f:
                rows = list(csv.DictReader(f))
            targets = {
                ("Centralized", "0"): -22.2860,
                ("SDec", "0"): -22.4454,
                ("Decentralized", "0"): -165.9583,
            }
            worst_gap = 0.0
            details = []
            for row in rows:
                key = (row["variant"], row["init_bin"])
                if key not in targets:
                    continue
                value = finite_float(row["expected_return"])
                gap = value - targets[key]
                worst_gap = max(worst_gap, abs(gap))
                details.append(f"{key[0]}={value:.3f}")
            if not details:
                raise RuntimeError("no target spacecraft rows found")
            check.value = worst_gap
            check.target = 0.0
            check.detail = ", ".join(details)
            if worst_gap > 0.2:
                check.status = "FAIL"
        except Exception as exc:
            check.status = "FAIL"
            check.detail = str(exc)
    checks.append(check)
    return checks


def section(title: str, checks: Iterable[Check]) -> list[Check]:
    print(f"\n{title}")
    print("-" * len(title))
    result = list(checks)
    for check in result:
        print_check(check)
    return result


def main() -> int:
    print("RSSDA experiment-performance runner")
    print(f"root={ROOT}")

    all_checks: list[Check] = []
    with tempfile.TemporaryDirectory(prefix="rssda_round_robin_") as tmp_raw:
        tmp = Path(tmp_raw)
        all_checks.extend(section("AAMAS exact paper anchors", run_aamas_exact()))
        all_checks.extend(section("IJCAI deterministic Labyrinth anchors", run_ijcai_labyrinth_exact()))
        all_checks.extend(section("IJCAI approximate smoke checks", run_ijcai_approx_smoke(tmp)))
        all_checks.extend(section("SpacecraftCA checks", run_spacecraft(tmp)))

    failures = [check for check in all_checks if check.status != "PASS"]
    print("\nSummary")
    print("-------")
    print(f"checks={len(all_checks)} pass={len(all_checks) - len(failures)} fail={len(failures)}")
    if failures:
        for check in failures:
            print_check(check)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
