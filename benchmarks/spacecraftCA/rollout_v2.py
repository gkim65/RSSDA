"""
rollout_v2.py — BRAHE closed-loop ROLLOUT VALIDATOR for the v2 policy.

WHY: compare_variants_v2 solves AND evaluates the policy on the binned MATRIX model
(belief-tree walk over T/O/R). It never executes the solved policy through real brahe
propagation. This script closes that gap: it takes the CANONICAL solved policy
(imported from compare_variants_v2.solve_policy — no matrices/policy rebuilt here) and
FLIES its decoded actions through brahe, reporting:
  - true TCA miss (km)            = norm(rtn[:3]) at TCA
  - matrix-predicted vs brahe terminal dt error (km)
  - the burn schedule actually executed.

SIGN CONVENTION (reconciled, notes/scratch/sign_check_v2.py):
  signed along-track dt = state_eci_to_rtn(sc1, sc2)[1] / 1e3   (SC2 wrt SC1)
  +dV burn = apply_maneuver(state, action=1)  ->  dt drifts NEGATIVE (mean_rate<0).
  The model's gain table is measured from the SAME apply_maneuver(+1) + same rtn[1]
  projection, so model and brahe agree in SIGN by construction. (The prior attempt's
  "+129 vs -44.7" was a flipped rtn convention in the rollout, NOT a model bug.)

TWO MODES:
  point-belief  : single deterministic trajectory from the most-dangerous support
                  state (--init-spread 0 at a dangerous --init-miss matches the
                  matrix "miss" exactly, since matrix miss is an EXPECTATION).
  monte-carlo   : sample the spread belief's support + transition + observation
                  branches and average (the matrix miss is an expectation over these).

DECODE: mirrors compare_variants_v2.expected_return_from_policy exactly —
  get_rssda_action / update_oh_rssda, max(joint_act,0) WAIT clamp, cluster pointers,
  is_cen centralized vs decentralized obs-history advance.

Usage:
  .venv/bin/python -u benchmarks/spacecraftCA/rollout_v2.py \
      --variant sdec --mode point --init-miss 0.0 --init-spread 0.0 --backend numerical
  .venv/bin/python -u benchmarks/spacecraftCA/rollout_v2.py \
      --variant centralized --mode mc --init-miss 0.5 --init-spread 1.4 \
      --rollouts 200 --backend drag
"""
import os, sys, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_BENCH)
for p in (_ROOT, _BENCH, _HERE):
    sys.path.insert(0, p)

# --backend must be honored BEFORE the model modules import (the stage grid + N_STAGES
# are computed at import time from the backend). Pre-scan argv, same as compare_variants_v2.
for _i, _a in enumerate(sys.argv):
    if _a == "--backend" and _i + 1 < len(sys.argv):
        os.environ["SPACECRAFT_PROPAGATOR"] = sys.argv[_i + 1].lower()
    elif _a.startswith("--backend="):
        os.environ["SPACECRAFT_PROPAGATOR"] = _a.split("=", 1)[1].lower()

from brahe import initialize_eop, state_eci_to_rtn, state_rtn_to_eci
from RSSDA import int_tuple

import spacecraft_discretizer_v2 as D
import spacecraft_transition_v2 as TV
import spacecraft_matrices as M
from spacecraft_matrices import (
    sc1_eci_at_tca, apply_maneuver, propagate_batch_to,
    STAGE_EPOCHS, EPOCH_TCA, DV_MAGNITUDE,
)
from spacecraft_simulator import get_rssda_action, update_oh_rssda

import compare_variants_v2 as CV


# ---------------------------------------------------------------------------
# brahe geometry helpers (SIGN-RECONCILED)
# ---------------------------------------------------------------------------

def signed_dt_and_miss_at_tca(sc1_tca, sc2_eci, sc2_epoch):
    """Fly SC2 from its current epoch to TCA; return (signed_dt_km, miss_km).
    signed dt = rtn[1] (along-track, SC2 wrt SC1); miss = norm(rtn[:3])."""
    sc2_tca = propagate_batch_to([sc2_epoch], [sc2_eci], EPOCH_TCA)[0]
    rtn = np.array(state_eci_to_rtn(sc1_tca, sc2_tca))
    return float(rtn[1]) / 1e3, float(np.linalg.norm(rtn[:3])) / 1e3


def root_sc2_eci(sc1_tca, dt0_km, perp_km):
    """SC2 ECI at the conjunction's TCA, for an along-track offset dt0 and standoff
    perp. Same placement the gain table uses (make_sc2_rel_state_at_tca)."""
    return np.array(state_rtn_to_eci(
        sc1_tca, TV.make_sc2_rel_state_at_tca(perp_km, dt0_km)))


# ---------------------------------------------------------------------------
# Policy decode (mirrors expected_return_from_policy)
# ---------------------------------------------------------------------------

def decode_action(full, step, belief_idx, oh):
    """Decoded (joint_act, a1, a2, is_cen, c_ptr) at this node, WAIT-clamped."""
    value, policy, clustering, cent_vector, cen_dists_map, clustering_cen = full
    joint_act, a1, a2, is_cen = get_rssda_action(
        policy, cen_dists_map, clustering, clustering_cen,
        step, belief_idx, list(oh), TV.N_ACT_AGENT)
    if joint_act < 0:                       # -1 cluster sentinel => WAIT (not a burn)
        joint_act, a1, a2 = 0, 0, 0
    c_ptr = -1
    if is_cen and step < len(cen_dists_map):
        dm = cen_dists_map[step]
        if belief_idx in dm:
            c_ptr = dm.index(belief_idx)
    # split exactly as the canonical walk does: a1 = act%N, a2 = act//N
    a1 = joint_act % TV.N_ACT_AGENT
    a2 = joint_act // TV.N_ACT_AGENT
    return joint_act, a1, a2, is_cen, c_ptr


# ---------------------------------------------------------------------------
# Single closed-loop brahe trajectory
# ---------------------------------------------------------------------------

def rollout_once(T, O, R, perp_km, sdec, full, init_state, init_belief_idx,
                 obs_agent_size, rng=None, stochastic=False, trace=False):
    """Fly ONE trajectory through brahe.

    init_state: the true (dt_bin, vdev1, vdev2, stage=0) the trajectory STARTS in.
                Its dt_bin center sets the brahe SC2 placement.
    stochastic=False: deterministic next state = matrix argmax branch + the model's
                MEAN dt (no noise sampling); obs = the matrix's most-likely obs.
    stochastic=True : sample T (next state) and O (obs) from the matrix rows (Monte
                Carlo). The belief/obs-history advance still uses the SAMPLED obs.

    Returns dict: brahe_miss_km, brahe_dt_km, matrix_dt_km, matrix_miss_km,
                  burns (list of (stage,a1,a2)), total_dv, sync_count.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    sc1_tca = sc1_eci_at_tca()
    dt0_bin, v1b0, v2b0, _ = D.index_to_state(init_state)
    dt0_km = D.dt_bin_center_km(dt0_bin)
    sc2_eci = root_sc2_eci(sc1_tca, dt0_km, perp_km)   # SC2 at TCA-epoch placement
    sc2_epoch = EPOCH_TCA

    # propagate SC2 back to stage 0 so we can fly it forward applying burns
    sc2_eci = propagate_batch_to([EPOCH_TCA], [sc2_eci], STAGE_EPOCHS[0])[0]
    sc2_epoch = STAGE_EPOCHS[0]

    belief_idx = init_belief_idx
    true_state = init_state
    oh = (0, 0)
    burns, total_dv, sync_count = [], 0.0, 0

    _ACT = {0: "WAIT", 1: "+dV", 2: "-dV"}
    if trace:
        # Per-stage readout: action | matrix bin/center/vdev (state AFTER this stage's
        # transition, so a burn's lever appears at the stage it is commanded, in phase
        # with brahe) | brahe true dt@TCA flying from the current SC2 state | err.
        print(f"\n  {'stg':>3} {'action':>9} {'mtx_bin':>7} {'mtx_dtc':>8} "
              f"{'mtx_vdev':>9} {'brahe_dt':>9} {'err':>8}")

    for step in range(D.N_STAGES):
        joint_act, a1, a2, is_cen, c_ptr = decode_action(full, step, belief_idx, oh)

        # --- execute burns in brahe (SC1 fixed chief; SC2 carries both? No: each agent
        #     burns its own craft. SC1 is the chief reference; in this reduced model both
        #     vdev act on the RELATIVE dt, captured by burning SC2 by the NET action.
        #     a1 (SC1 +dV) lowers relative dt the same as a2 (SC2 -dV): net along-track.
        #     We fly the NET relative effect on SC2 = (a2 effect) - (a1 effect) is wrong;
        #     vdev1+vdev2 BOTH add +rate. So apply a1 AND a2 as two along-track impulses
        #     to SC2 (each unit shifts the relative drift by +rate). ---
        for act in (a1, a2):
            if act != 0:
                sc2_eci = apply_maneuver(sc2_eci, act, dv=DV_MAGNITUDE)
                total_dv += DV_MAGNITUDE
        if a1 != 0 or a2 != 0:
            burns.append((step, a1, a2))

        if is_cen:
            sync_count += 1

        # --- matrix transition: pick next true_state (argmax or sampled) ---
        row = T[joint_act, true_state, :]
        if stochastic:
            nz = np.flatnonzero(row > 1e-12)
            p = row[nz] / row[nz].sum()
            next_state = int(nz[rng.choice(len(nz), p=p)])
        else:
            next_state = int(np.argmax(row))

        # --- matrix observation -> advance belief + obs-history (mirror the walk) ---
        if next_state != D.SINK_STATE and step < D.N_STAGES - 1:
            orow = O[joint_act, next_state, :]
            if stochastic:
                nzo = np.flatnonzero(orow > 1e-12)
                po = orow[nzo] / orow[nzo].sum()
                obs = int(nzo[rng.choice(len(nzo), p=po)])
            else:
                obs = int(np.argmax(orow))
            try:
                o2b = {int(o): int(d) for o, _, d in sdec.get_terminal(belief_idx, joint_act)}
            except KeyError:
                o2b = {}
            nb = o2b.get(obs, belief_idx)
            o1 = obs % obs_agent_size; o2 = obs // obs_agent_size
            n0, n1 = update_oh_rssda(full[1], full[4], full[2], full[5],
                                     step, nb, list(oh), is_cen, c_ptr, o1, o2)
            belief_idx, oh = nb, (n0, n1)

        if trace:
            # brahe truth: dt@TCA flying from the CURRENT SC2 state (post-burn, pre-coast).
            b_dt, _ = signed_dt_and_miss_at_tca(sc1_tca, sc2_eci, sc2_epoch)
            ns_show = next_state if next_state != D.SINK_STATE else true_state
            nb_, nv1_, nv2_, _ = D.index_to_state(ns_show)
            mdtc = D.dt_bin_center_km(nb_)
            vdev_s = f"({D.vdev_value(nv1_)},{D.vdev_value(nv2_)})"
            print(f"  {step:>3} {_ACT[a1]+'/'+_ACT[a2]:>9} {nb_:>7} {mdtc:>8.2f} "
                  f"{vdev_s:>9} {b_dt:>9.2f} {b_dt - mdtc:>8.2f}")

        # --- propagate SC2 to next stage epoch ---
        if step < D.N_STAGES - 1:
            sc2_eci = propagate_batch_to([sc2_epoch], [sc2_eci], STAGE_EPOCHS[step + 1])[0]
            sc2_epoch = STAGE_EPOCHS[step + 1]
        true_state = next_state if next_state != D.SINK_STATE else true_state

    # --- terminal: brahe truth vs matrix prediction ---
    brahe_dt_km, brahe_miss_km = signed_dt_and_miss_at_tca(sc1_tca, sc2_eci, sc2_epoch)
    tdt_bin, _, _, _ = D.index_to_state(true_state) if true_state != D.SINK_STATE else (D.N_DT // 2, 0, 0, 0)
    matrix_dt_km = D.dt_bin_center_km(tdt_bin)
    matrix_miss_km = D.miss_km_from_dt(matrix_dt_km, perp_km)

    # TRUE terminal reward (from the ACTUAL brahe miss) vs the MATRIX's binned terminal reward
    # (the bin center the solver optimized against). The gap = how much binning over/under-credits
    # the landing -- biggest at the 4-5km risk cliff (bin [4,5) center -423 vs ~0 at 4.9km).
    true_term_reward = TV.terminal_reward(brahe_dt_km, brahe_miss_km)
    matrix_term_reward = TV.terminal_reward(matrix_dt_km, matrix_miss_km)
    # number of agent-burns executed (each burn entry may carry 1 or 2 agent burns)
    n_burns = sum((a1 != 0) + (a2 != 0) for (_s, a1, a2) in burns)

    return dict(
        brahe_miss_km=brahe_miss_km, brahe_dt_km=brahe_dt_km,
        matrix_dt_km=matrix_dt_km, matrix_miss_km=matrix_miss_km,
        dt_err_km=brahe_dt_km - matrix_dt_km,
        true_term_reward=true_term_reward, matrix_term_reward=matrix_term_reward,
        burns=burns, n_burns=n_burns, total_dv=total_dv, sync_count=sync_count,
    )


# ---------------------------------------------------------------------------
# Drivers: point belief + Monte Carlo
# ---------------------------------------------------------------------------

def most_dangerous_support(init_b):
    """Support state with the smallest |dt| (most dangerous) — the point-belief seed."""
    support = np.flatnonzero(init_b[:D.N_STATES])
    best, best_absdt = None, 1e18
    for s in support:
        dtb, _, _, stg = D.index_to_state(int(s))
        if stg != 0:
            continue
        a = abs(D.dt_bin_center_km(dtb))
        if a < best_absdt:
            best, best_absdt = int(s), a
    return best


def run_point(T, O, R, perp_km, sdec, full, init_b, obs_agent_size, trace=False):
    root_belief = sdec.dist_dict[int_tuple(sdec.init_beliefs)]
    seed_state = most_dangerous_support(init_b)
    r = rollout_once(T, O, R, perp_km, sdec, full, seed_state, root_belief,
                     obs_agent_size, stochastic=False, trace=trace)
    return [r]


def run_mc(T, O, R, perp_km, sdec, full, init_b, obs_agent_size, n, seed=0):
    """VECTORIZED Monte Carlo: fly all n trajectories in LOCKSTEP so the per-stage
    brahe propagation of all n SC2 states is ONE par_propagate_to call (via
    propagate_batch_to) instead of n calls. Same physics/decode as rollout_once;
    each trajectory keeps its own belief/obs-history/RNG branch."""
    rng = np.random.default_rng(seed)
    root_belief = sdec.dist_dict[int_tuple(sdec.init_beliefs)]
    support = np.flatnonzero(init_b[:D.N_STATES])
    probs = init_b[support] / init_b[support].sum()

    sc1_tca = sc1_eci_at_tca()

    # --- per-trajectory state vectors ---
    s0 = support[rng.choice(len(support), size=n, p=probs)].astype(int)
    sc2 = []                       # ECI state per trajectory
    for s in s0:
        dt0_bin = D.index_to_state(int(s))[0]
        sc2.append(root_sc2_eci(sc1_tca, D.dt_bin_center_km(dt0_bin), perp_km))
    # batch-place all at stage 0
    sc2 = propagate_batch_to([EPOCH_TCA] * n, sc2, STAGE_EPOCHS[0])
    epoch = STAGE_EPOCHS[0]

    belief = [root_belief] * n
    true_state = list(s0)
    oh = [(0, 0)] * n
    burns = [[] for _ in range(n)]
    total_dv = np.zeros(n)
    sync_count = 0

    for step in range(D.N_STAGES):
        # decode + execute burns + advance belief for each trajectory (cheap, no brahe)
        for i in range(n):
            joint_act, a1, a2, is_cen, c_ptr = decode_action(full, step, belief[i], oh[i])
            for act in (a1, a2):
                if act != 0:
                    sc2[i] = apply_maneuver(sc2[i], act, dv=DV_MAGNITUDE)
                    total_dv[i] += DV_MAGNITUDE
            if a1 != 0 or a2 != 0:
                burns[i].append((step, a1, a2))
            if is_cen and i == 0:
                sync_count += 1

            row = T[joint_act, true_state[i], :]
            nz = np.flatnonzero(row > 1e-12)
            ns = int(nz[rng.choice(len(nz), p=row[nz] / row[nz].sum())])
            if ns != D.SINK_STATE and step < D.N_STAGES - 1:
                orow = O[joint_act, ns, :]
                nzo = np.flatnonzero(orow > 1e-12)
                obs = int(nzo[rng.choice(len(nzo), p=orow[nzo] / orow[nzo].sum())])
                try:
                    o2b = {int(o): int(d) for o, _, d in
                           sdec.get_terminal(belief[i], joint_act)}
                except KeyError:
                    o2b = {}
                nb = o2b.get(obs, belief[i])
                o1 = obs % obs_agent_size; o2 = obs // obs_agent_size
                n0, n1 = update_oh_rssda(full[1], full[4], full[2], full[5],
                                         step, nb, list(oh[i]), is_cen, c_ptr, o1, o2)
                belief[i], oh[i] = nb, (n0, n1)
            true_state[i] = ns if ns != D.SINK_STATE else true_state[i]

        # ONE batched propagation of all n SC2 states to the next stage epoch
        if step < D.N_STAGES - 1:
            sc2 = propagate_batch_to([epoch] * n, sc2, STAGE_EPOCHS[step + 1])
            epoch = STAGE_EPOCHS[step + 1]

    # ONE batched propagation of all n SC2 states to TCA
    sc2_tca = propagate_batch_to([epoch] * n, sc2, EPOCH_TCA)
    results = []
    for i in range(n):
        rtn = np.array(state_eci_to_rtn(sc1_tca, sc2_tca[i]))
        bdt = float(rtn[1]) / 1e3
        bmiss = float(np.linalg.norm(rtn[:3])) / 1e3
        tdt_bin = (D.index_to_state(true_state[i])[0]
                   if true_state[i] != D.SINK_STATE else D.N_DT // 2)
        mdt = D.dt_bin_center_km(tdt_bin)
        mmiss = D.miss_km_from_dt(mdt, perp_km)
        nb = sum((a1 != 0) + (a2 != 0) for (_s, a1, a2) in burns[i])
        results.append(dict(
            brahe_miss_km=bmiss, brahe_dt_km=bdt,
            matrix_dt_km=mdt, matrix_miss_km=mmiss,
            dt_err_km=bdt - mdt, burns=burns[i], n_burns=nb,
            true_term_reward=TV.terminal_reward(bdt, bmiss),
            matrix_term_reward=TV.terminal_reward(mdt, mmiss),
            total_dv=float(total_dv[i]), sync_count=sync_count,
        ))
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def summarize(results, label, mode):
    miss = np.array([r["brahe_miss_km"] for r in results])
    bdt = np.array([r["brahe_dt_km"] for r in results])
    mdt = np.array([r["matrix_dt_km"] for r in results])
    dv = np.array([r["total_dv"] for r in results])
    coll = np.mean(miss < D.COLLISION_THRESHOLD_KM)
    print(f"\n{'='*66}\n[{label}]  mode={mode}  n={len(results)}\n{'='*66}")
    print(f"  brahe TCA miss : mean={miss.mean():8.3f}  min={miss.min():8.3f} km")
    print(f"  brahe collisions (<{D.COLLISION_THRESHOLD_KM}km): {100*coll:.2f}%")
    print(f"  terminal dt    : brahe mean={bdt.mean():8.2f}  matrix mean={mdt.mean():8.2f} km")
    print(f"  matrix-vs-brahe dt error: mean={(bdt-mdt).mean():8.2f}  "
          f"absmean={np.abs(bdt-mdt).mean():7.2f}  max|err|={np.abs(bdt-mdt).max():7.2f} km")
    print(f"  total dV       : mean={dv.mean():.3f} m/s   syncs={results[0]['sync_count']}")
    # TRUE terminal reward (from actual brahe miss) vs the MATRIX's binned terminal reward the
    # solver optimized -- the gap shows how much binning over/under-credits the landing.
    tr = np.array([r["true_term_reward"] for r in results])
    mr = np.array([r["matrix_term_reward"] for r in results])
    nb = np.array([r["n_burns"] for r in results])
    print(f"  terminal reward: true(brahe)={tr.mean():9.2f}  matrix(binned)={mr.mean():9.2f}  "
          f"gap={tr.mean()-mr.mean():+8.2f}")
    print(f"  burns/rollout  : mean={nb.mean():.2f}  max={nb.max()}  "
          f"(<=2 burns: {100*np.mean(nb <= 2):.0f}% of rollouts)")
    if len(results) == 1:
        bs = ",".join(f"s{k}:({a1},{a2})" for (k, a1, a2) in results[0]["burns"])
        print(f"  burn schedule  : {bs or 'none'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["centralized", "sdec"], default="sdec",
                    help="dec uses RS-MAA* (different policy API) — not validated here yet.")
    ap.add_argument("--mode", choices=["point", "mc"], default="point")
    ap.add_argument("--init-miss", type=float, default=0.0)
    ap.add_argument("--init-spread", type=float, default=0.0,
                    help="0 (default) => point belief matches matrix expectation exactly.")
    ap.add_argument("--perp", type=float, default=0.0)
    ap.add_argument("--contact-stages", type=str, default=None)
    ap.add_argument("--rollouts", type=int, default=200, help="MC mode only.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default=None,
                    choices=["numerical", "keplerian", "drag"])
    ap.add_argument("--trace", action="store_true",
                    help="point mode: print per-stage matrix-bin/vdev vs brahe dt + err.")
    args = ap.parse_args()

    initialize_eop()
    if args.contact_stages is not None:
        stages = [int(s) for s in args.contact_stages.split(",") if s.strip() != ""]
        M.set_contact_stages(stages)
    CV.PERP_KM = args.perp
    print(f"  backend={M._SG.PROPAGATOR_BACKEND}  N_STAGES={D.N_STAGES}  "
          f"contacts={M.get_contact_stages()}")

    init_b, flagged, eff_miss = TV.build_init_b_danger(
        args.init_miss, args.init_spread, args.perp, sign_mode="both")
    print(f"  init: miss={args.init_miss} spread={args.init_spread} perp={args.perp} "
          f"-> eff_miss={eff_miss:.3f} km  flagged={flagged}")

    # CANONICAL policy + matrices (no rebuild here — imported solve path).
    T, O, R, perp, sdec, full = CV.solve_policy(args.variant, init_b)
    print(f"  solved {args.variant}: policy value={full[0]:.4f}  perp_meas={perp:.3f}")

    obs_agent_size = TV.N_OBS_AGENT
    if args.mode == "point":
        results = run_point(T, O, R, perp, sdec, full, init_b, obs_agent_size,
                            trace=args.trace)
    else:
        results = run_mc(T, O, R, perp, sdec, full, init_b, obs_agent_size,
                         args.rollouts, seed=args.seed)
    summarize(results, args.variant, args.mode)


if __name__ == "__main__":
    main()