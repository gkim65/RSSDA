"""
Verify all three AAMAS Table 2 SDec-Tiger h=8 values reproduce after the
regime-conditional rollback.
"""

import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
sys.path.insert(0, _here)
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "baselines"))

import sdec_tiger
from sdec_tiger import TigerProblemFactory
from RSSDA import SDecPOMDP, SDecPOMDPModel, RSSDAConfig, int_tuple


def solve(trigger_mode, obs_accuracy, sync_actions, horizon=8):
    sdec_tiger.TRIGGER_MODE = trigger_mode
    sdec_tiger.OBS_ACCURACY = obs_accuracy
    sdec_tiger.LAMBDA = 1.0

    factory = TigerProblemFactory(config=None)
    T, O, R, init_b, nacts_fac, nobs_fac = factory.generate()

    model = SDecPOMDPModel(
        nagents=2, nstates=2, nactions=9, nobs=4,
        transitions=T, obs=O, rewards=R, init_beliefs=init_b,
        nacts_factor=nacts_fac, nobs_factor=nobs_fac,
        sync_states=[], sync_actions=sync_actions, sync_observations=[],
    )
    cfg = RSSDAConfig(
        maxh=horizon, maxit=200, IEmin2=3, alpha=0.2,
        algorithm="exact", heuristic_type="POMDP", tail_heuristic_type="POMDP",
        TI1=False, TI2=False, TI3=False, TI4=False,
        score_limit=20, cen_threshold=0.6, sm_temperature=0.6,
        iter_limit=1000, rec_limit=2, hybrid_r=1, max_clusters=10,
    )
    solver = SDecPOMDP(model=model, config=cfg)
    init_idx = solver.dist_dict[int_tuple(solver.init_beliefs)]
    t0 = time.time()
    val, *_ = solver.multi_agent_astar(horizon, init_beliefs=init_idx)
    return val, time.time() - t0


def main():
    horizon = 8
    targets = {
        "decentralized": (12.21726, 0.85, []),
        "semi":          (27.21518, 0.75, [8]),
        "centralized":   (47.71696, 0.85, list(range(9))),
    }

    print(f"AAMAS Table 2 SDec-Tiger reproduction at h={horizon}")
    print(f"{'mode':>14}  {'acc':>5}  {'value':>12}  {'AAMAS':>12}  {'gap':>10}  {'time':>8}")
    print("-" * 70)
    for mode, (target, acc, sync) in targets.items():
        val, t = solve(mode, acc, sync, horizon)
        print(f"{mode:>14}  {acc:>5.2f}  {val:>12.5f}  {target:>12.5f}  {val - target:>+10.5f}  {t:>7.3f}s")


if __name__ == "__main__":
    main()
