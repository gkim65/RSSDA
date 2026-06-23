"""
belief_collapse.py — how the dt belief NARROWS over the rollout under different observation
fidelities, and when its MEAN crosses the act/no-act thresholds. Diagnostic for: "how noisy must
the initial belief be (and the other-craft channel) before OBSERVATIONS actually matter before we
commit a maneuver?"

USES THE SAME belief filter as B1 (belief_filter.BeliefFilter) — so what this plots is exactly the
belief B1's operators carry, NOT a separate scheme. It runs the filter forward under pure WAIT (no
burns) so we isolate how OBSERVATIONS alone resolve the prior, and contrasts the observation models:
  perfect : every contact reveals the combined dt exactly -> belief collapses to a POINT at the first
            contact (idealized "perfect sync").
  tle     : own contacts give NO collapse; the OTHER craft is a NOISY ~8h TLE fix (sigma grows with
            age) -> belief stays fuzzy and resolves SLOWLY, never to a point.
  frozen  : the other craft is never observed -> belief only DIFFUSES (process drift).
(PoC dropped — we never compute a real probability-of-collision anywhere, so reporting one was
misleading. We track believed MISS mean + spread, which is what the operators actually threshold.)

The act thresholds are the operators' believed-miss lines (conservative 5 km, aggressive 2 km). We
report, per stage, the belief mean miss + spread and whether the mean has crossed each line (the
commit stage), and we can SWEEP --init-spread to show where the prior is wide enough that a contact
flips the commit decision.

Usage:
  # single run, contrast perfect vs tle vs frozen on one plot:
  .venv/bin/python -u benchmarks/spacecraftCA/baselines_spacecraftCA/belief_collapse.py \
      --init-miss 5.0 --init-spread 6.0 --backend numerical --variant sdec --plot bc_im5
  # spread sweep (commit stage vs prior width) for one obs model:
  .venv/bin/python -u .../belief_collapse.py --init-miss 5.0 --other-obs tle --backend numerical \
      --spread-sweep 0,1,2,3,4,6,8 --plot bc_sweep
"""
import os, sys, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCA = os.path.dirname(_HERE)
_BENCH = os.path.dirname(_SCA)
_ROOT = os.path.dirname(_BENCH)
for p in (_ROOT, _BENCH, _SCA, _HERE):
    sys.path.insert(0, p)

for _i, _a in enumerate(sys.argv):
    if _a == "--backend" and _i + 1 < len(sys.argv):
        os.environ["SPACECRAFT_PROPAGATOR"] = sys.argv[_i + 1].lower()
    elif _a.startswith("--backend="):
        os.environ["SPACECRAFT_PROPAGATOR"] = _a.split("=", 1)[1].lower()

from brahe import initialize_eop
import spacecraft_discretizer_v2 as D
import spacecraft_transition_v2 as TV
import spacecraft_matrices as M
import belief_filter as BF

THRESHOLDS_KM = {"conservative": 5.0, "aggressive": 2.0}
OBS_MODELS = ["perfect", "tle", "frozen"]
OBS_COLORS = {"perfect": "#c0392b", "tle": "#2f8f5b", "frozen": "#888888"}


def true_combined_dt_for(eff_miss_km, perp_km):
    """The TRUE combined dt the (no-burn) conjunction sits at, used as the noise-free measurement
    centre fed to the filter each contact. Under pure WAIT the true dt is fixed at the conjunction's
    along-track offset = sqrt(eff_miss^2 - perp^2) (>=0)."""
    v = eff_miss_km * eff_miss_km - perp_km * perp_km
    return float(np.sqrt(v)) if v > 0 else 0.0


def filter_trace(prior_mean, prior_std, perp_km, other_obs, tle_sigma, eff_miss, rng=None):
    """Run the shared CONTINUOUS BeliefFilter forward under pure WAIT. At each GS contact we feed a
    measurement of the combined dt; the filter decides what to do with it (perfect: exact collapse;
    tle: apply a pending 8h-epoch TLE here, aged + NOISY; frozen: ignore). The reading is sampled
    N(true_dt, tle_sigma) so the belief MEAN actually MOVES at each noisy fix. Returns per-stage rows
    + commit stage (first time the mean crosses each threshold)."""
    if rng is None:
        rng = np.random.default_rng(0)
    contacts = set(M.get_contact_stages())
    f = BF.BeliefFilter(prior_mean, prior_std, perp_km, other_obs=other_obs,
                        tle_sigma_base=tle_sigma, contact_stages=contacts)
    true_dt = true_combined_dt_for(eff_miss, perp_km)
    rows, commit = [], {n: None for n in THRESHOLDS_KM}
    for step in range(D.N_STAGES):
        mean_miss = f.mean_miss_km(); std = f.std_dt_km(); lo, hi = f.support_width_km()
        for name, thr in THRESHOLDS_KM.items():
            if commit[name] is None and mean_miss < thr:
                commit[name] = step
        rows.append(dict(stage=step, mean_miss=mean_miss, std=std, lo=lo, hi=hi,
                         sync=(step in contacts), collapsed=f.collapsed_this_stage))
        if step < D.N_STAGES - 1:
            f.predict(step)
            ns = step + 1
            if ns in contacts:
                # a measurement is available on this pass: exact truth for perfect, a NOISY sample
                # for tle (the filter only actually applies it if a TLE is pending). frozen ignores.
                reading = true_dt if other_obs == "perfect" else float(rng.normal(true_dt, tle_sigma))
                f.observe(ns, reading)
            else:
                f.observe(ns, None)
    return rows, commit


def maneuver_trace(prior_mean, prior_std, perp_km, other_obs, tle_sigma, true_miss,
                   levers, plan="target_band", rng=None):
    """The operator ACTS: a conservative threshold operator burns when its belief mean miss crosses
    the 5 km act line. Two timing PLANS:
      'target_band' : DEFER the single avoidance burn to the latest small-lever stage so it lands
                      near the band centre (the B1 default). One burn.
      'fire_return' : FIRE IMMEDIATELY on crossing (big early lever -> over-clears far out), then
                      a RETURN burn that trims back only as far as needed to land near the band
                      (~5.5 km) — clear the threshold and stop, don't over-displace. Out-and-back.
    The burns shift the belief MEAN by the lever (commit_burn) AND the true dt. Returns rows + list
    of burn stages. Shows observe -> decide -> ACT (-> RETURN) -> belief jumps, per obs model."""
    if rng is None:
        rng = np.random.default_rng(0)
    contacts = set(M.get_contact_stages())
    thr = THRESHOLDS_KM["conservative"]
    target = 0.5 * (4.0 + 7.0)                      # band centre (~5.5 km)
    f = BF.BeliefFilter(prior_mean, prior_std, perp_km, other_obs=other_obs,
                        tle_sigma_base=tle_sigma, contact_stages=contacts)
    true_dt = true_combined_dt_for(true_miss, perp_km)
    avoided, returned, burn_stages = False, False, []

    def fire_stage_for(dt):                          # latest stage whose 1-burn lever ~ lands target
        need = max(target - abs(dt), 0.0)
        best = 0
        for k in range(D.N_STAGES - 1):
            if abs(levers[k]) >= need:
                best = k
        return best

    burn_records = []      # (stage, signed_lever_km) for the accumulated-drift ramp

    def do_burn(step, direction):
        sign = +1.0 if direction == 1 else -1.0
        f.commit_burn(direction, levers[step])
        burn_stages.append(step)
        burn_records.append((step, sign * levers[step]))
        return sign * levers[step]

    rows = []
    for step in range(D.N_STAGES):
        acted = 0
        cur_dt = f.mean_dt_km()
        if plan == "target_band":
            if not avoided and f.mean_miss_km() < thr and step >= fire_stage_for(cur_dt):
                direction = 2 if cur_dt >= 0 else 1
                true_dt += do_burn(step, direction); avoided, acted = True, 1
        else:  # fire_return
            if not avoided and f.mean_miss_km() < thr:
                # FIRE NOW: push |dt| away from 0 with this stage's (large, early) lever
                direction = 2 if cur_dt >= 0 else 1
                true_dt += do_burn(step, direction); avoided, acted = True, 1
            elif avoided and not returned and step < D.N_STAGES - 1:
                # RETURN: would a counter-burn now land the belief CLOSER to the target band?
                ret_dir = 1 if f.mean_dt_km() > 0 else 2      # opposite: pull |dt| back toward 0
                rsign = +1.0 if ret_dir == 1 else -1.0
                after = f.mean_dt_km() + rsign * levers[step]
                # only return if it doesn't drop us back below the band floor (stay >=4km) AND it
                # gets us closer to the band centre than staying put
                if abs(after) >= 4.0 and abs(abs(after) - target) < abs(abs(f.mean_dt_km()) - target):
                    true_dt += do_burn(step, ret_dir); returned, acted = True, 2
        rows.append(dict(stage=step, mean_miss=f.mean_miss_km(),
                         std=f.std_dt_km(), lo=f.support_width_km()[0], hi=f.support_width_km()[1],
                         sync=(step in contacts), collapsed=f.collapsed_this_stage, burn=acted))
        if step < D.N_STAGES - 1:
            f.predict(step)
            ns = step + 1
            if ns in contacts:
                reading = true_dt if other_obs == "perfect" else float(rng.normal(true_dt, tle_sigma))
                f.observe(ns, reading)
            else:
                f.observe(ns, None)
    return rows, burn_stages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-miss", type=float, default=5.0,
                    help="the operator's PRIOR belief mean miss (what the CDM/screening first says).")
    ap.add_argument("--true-miss", type=float, default=None,
                    help="the ACTUAL miss the conjunction sits at (what the TLE measurements reveal). "
                         "Defaults to --init-miss (prior is correct). Set it LOWER than init-miss to "
                         "show a conjunction that looks safe at first but the TLE fixes drag the belief "
                         "DOWN across the act line -> the commit MOMENT is visible mid-flight.")
    ap.add_argument("--init-spread", type=float, default=6.0)
    ap.add_argument("--perp", type=float, default=0.0)
    ap.add_argument("--variant", choices=["centralized", "sdec", "dec"], default="sdec")
    ap.add_argument("--other-obs", choices=["perfect", "tle", "frozen"], default="tle",
                    help="obs model for the SWEEP (single-run plots ALL three for contrast).")
    ap.add_argument("--tle-sigma", type=float, default=BF.TLE_SIGMA_BASE_KM)
    ap.add_argument("--contact-stages", type=str, default=None)
    ap.add_argument("--backend", default=None, choices=["numerical", "keplerian", "drag"])
    ap.add_argument("--spread-sweep", type=str, default=None)
    ap.add_argument("--with-maneuver", action="store_true",
                    help="run an ACTING conservative operator per obs model (it BURNS when its belief "
                         "crosses the 5km line) and plot belief + the burn + the post-burn jump.")
    ap.add_argument("--maneuver-plan", choices=["target_band", "fire_return"], default="fire_return",
                    help="target_band=defer one burn to land in the band; fire_return=fire NOW on "
                         "crossing then trim back toward the band (out-and-back).")
    ap.add_argument("--plot", type=str, default=None)
    args = ap.parse_args()

    initialize_eop()
    if args.contact_stages is not None:
        M.set_contact_stages([int(s) for s in args.contact_stages.split(",") if s.strip()])
    rate_at, perp, _ = TV.compute_gain_table_and_perp(args.perp, 0.0)
    TV.build_T_O(rate_at, args.variant)     # configures the dt grid (only the lever table is used here)
    levers = np.array([float(rate_at[k]) * TV.stage_t2go_h(k) for k in range(D.N_STAGES)])
    print(f"  backend={M._SG.PROPAGATOR_BACKEND}  N_STAGES={D.N_STAGES}  perp={perp:.3f}  "
          f"contacts={sorted(M.get_contact_stages())}  TLE@{BF.tle_refresh_stages()}")

    true_miss = args.true_miss if args.true_miss is not None else args.init_miss

    if args.with_maneuver:
        pm, ps = BF.prior_from_init(args.init_miss, args.init_spread, args.perp)
        traces, burns = {}, {}
        for oo in OBS_MODELS:
            rows, bstages = maneuver_trace(pm, ps, perp, oo, args.tle_sigma, true_miss, levers,
                                           plan=args.maneuver_plan)
            traces[oo], burns[oo] = rows, bstages
            print(f"\n  [{oo}] prior={args.init_miss} true={true_miss} plan={args.maneuver_plan}  "
                  f"BURNS@{bstages}")
            for r in rows:
                mk = (" <-AVOID" if r["burn"] == 1 else " <-RETURN" if r["burn"] == 2
                      else ("  <-obs" if r["collapsed"] else ""))
                print(f"     s{r['stage']:>2} {'SYNC' if r['sync'] else '    '} "
                      f"mean_miss={r['mean_miss']:7.3f} std={r['std']:6.3f}{mk}")
        if args.plot:
            _plot_maneuver(traces, burns, args, true_miss)
        return

    if args.spread_sweep is None:
        # prior = N(from init_miss); the TLE measurements reveal the TRUE miss (true_miss). When
        # true_miss < init_miss the belief is dragged DOWN across the act line mid-flight.
        pm, ps = BF.prior_from_init(args.init_miss, args.init_spread, args.perp)
        traces = {}
        for oo in OBS_MODELS:
            rows, commit = filter_trace(pm, ps, perp, oo, args.tle_sigma, true_miss)
            traces[oo] = rows
            cs = " ".join(f"{n}@{commit[n]}" for n in THRESHOLDS_KM)
            print(f"\n  [{oo}] prior_miss={args.init_miss} true_miss={true_miss}  commit: {cs}")
            for r in rows:
                tag = "SYNC" if r["sync"] else "    "
                col = "  <-collapse" if r["collapsed"] else ""
                print(f"     s{r['stage']:>2} {tag} mean_miss={r['mean_miss']:7.3f} "
                      f"std_dt={r['std']:7.3f} supp|dt|=[{r['lo']:5.1f},{r['hi']:5.1f}]{col}")
        if args.plot:
            _plot_single(traces, args, true_miss)
        return

    spreads = [float(s) for s in args.spread_sweep.split(",") if s.strip() != ""]
    print(f"\n  INIT-SPREAD SWEEP  init_miss={args.init_miss} true_miss={true_miss}  "
          f"other-obs={args.other_obs}")
    print(f"  {'spread':>7} {'miss0':>7} {'std0':>7} | commit(cons/aggr)")
    sweep = []
    for sp in spreads:
        pm, ps = BF.prior_from_init(args.init_miss, sp, args.perp)
        rows, commit = filter_trace(pm, ps, perp, args.other_obs, args.tle_sigma, true_miss)
        print(f"  {sp:>7.2f} {rows[0]['mean_miss']:>7.3f} {rows[0]['std']:>7.3f} | "
              f"{commit['conservative']}/{commit['aggressive']}")
        sweep.append((sp, commit))
    if args.plot:
        _plot_sweep(sweep, args)


def _plot_maneuver(traces, burns, args, true_miss):
    """Per obs model: the PREDICTED miss-at-TCA (the belief mean — the operator's decision variable)
    with its ±1σ UNCERTAINTY BAND (the actual belief distribution), and the burns marked. The mean
    JUMPS at a burn — verified vs brahe: a velocity change instantly re-aims the predicted TCA
    endpoint (not teleportation; the physical separation oscillates with orbital phase and is not a
    clean quantity). IMPORTANT: this is the operator's BELIEF, NOT its achieved outcome — the true
    brahe miss is in the B1 MC table (where these same operators land ~35km / collide)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    # BROKEN y-axis: the AVOID spikes (~50-124km) crush the -5..20km zone where the thresholds, the
    # belief motion, and the ±1σ bands all live. Lower panel zooms -5..20km (detail); upper panel
    # zooms a tight window around the spike peaks; a diagonal break separates them.
    peak = max(max(r["mean_miss"] for r in traces[oo]) for oo in OBS_MODELS)
    LOW_HI = 20.0
    hi_lo, hi_hi = peak - 15.0, peak + 5.0            # tight window around the tallest spike(s)
    fig, (axt, axb) = plt.subplots(2, 1, sharex=True, figsize=(11, 6.4),
                                   gridspec_kw=dict(height_ratios=[1, 2.6], hspace=0.07))
    for oo in OBS_MODELS:
        rows = traces[oo]; stg = [r["stage"] for r in rows]
        mean = np.array([r["mean_miss"] for r in rows])
        std = np.array([r["std"] for r in rows])
        c = OBS_COLORS[oo]
        for ax in (axt, axb):
            ax.plot(stg, mean, "-o", color=c, ms=3,
                    label=(f"{oo}  (burns@{burns[oo] or 'none'})" if ax is axb else None))
            ax.fill_between(stg, mean - std, mean + std, color=c, alpha=0.18)
        for j, bs in enumerate(burns[oo]):
            lbl = "AVOID" if j == 0 else "RETURN"
            tgt = axt if mean[bs] > LOW_HI else axb
            axb.axvline(bs, color=c, ls=":", lw=1.1, alpha=0.5)
            axt.axvline(bs, color=c, ls=":", lw=1.1, alpha=0.5)
            tgt.annotate(lbl, xy=(bs, mean[bs]), xytext=(bs + 0.4, mean[bs]),
                         fontsize=7.5, color=c, fontweight="bold")
    for name, thr in THRESHOLDS_KM.items():
        axb.axhline(thr, ls="--", lw=1, color="#555")
        axb.text(0, thr, f" {name} {thr}km", va="bottom", fontsize=8, color="#555")
    axb.axhspan(0, D.COLLISION_THRESHOLD_KM, color="red", alpha=0.10)
    axb.axhspan(4.0, 7.0, color="green", alpha=0.10)
    axb.set_ylim(-5, LOW_HI); axt.set_ylim(hi_lo, hi_hi)
    axt.spines["bottom"].set_visible(False); axb.spines["top"].set_visible(False)
    axt.tick_params(labelbottom=False, bottom=False)
    dd = 0.01
    kw = dict(transform=axt.transAxes, color="k", clip_on=False, lw=0.9)
    axt.plot((-dd, +dd), (-dd, +dd), **kw); axt.plot((1 - dd, 1 + dd), (-dd, +dd), **kw)
    kw["transform"] = axb.transAxes
    axb.plot((-dd, +dd), (1 - dd, 1 + dd), **kw); axb.plot((1 - dd, 1 + dd), (1 - dd, 1 + dd), **kw)
    axb.set_xlabel("stage"); axb.set_ylabel("predicted miss at TCA (km) — belief mean ±1σ")
    axt.set_ylabel("AVOID\nspike")
    axt.set_title(f"SINGLE-OPERATOR belief (predicted miss at TCA) + ±1σ, observe->decide->ACT "
                  f"(prior {args.init_miss}km, true {true_miss}km, spread {args.init_spread})\n"
                  f"ONE craft vs a FIXED conjunction (not the 2-craft game — see B1 MC table). "
                  f"AVOID spikes shown in the upper (broken) panel; BELIEF not achieved outcome.",
                  fontsize=9)
    axb.legend(loc="upper right", fontsize=8)
    out = args.plot if (os.sep in args.plot or args.plot.endswith(".png")) else \
        os.path.join(_SCA, "notes", "figures", f"belief_maneuver_{args.plot}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight"); print(f"\n  figure saved -> {out}")


def _plot_single(traces, args, eff):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for oo in OBS_MODELS:
        rows = traces[oo]; stg = [r["stage"] for r in rows]
        mean = [r["mean_miss"] for r in rows]
        lo = [r["lo"] for r in rows]; hi = [r["hi"] for r in rows]
        c = OBS_COLORS[oo]
        ax.plot(stg, mean, "-o", color=c, ms=3, label=f"{oo}: belief mean miss")
        ax.fill_between(stg, lo, hi, color=c, alpha=0.12)
    for name, thr in THRESHOLDS_KM.items():
        ax.axhline(thr, ls="--", lw=1, color="#555")
        ax.text(0, thr, f" {name} {thr}km", va="bottom", fontsize=8, color="#555")
    for r in traces["perfect"]:
        if r["sync"]:
            ax.axvline(r["stage"], color="#2c7", alpha=0.08, lw=2)
    # COMMIT markers: the first stage each model's belief mean crosses the conservative act line
    # (= when an operator on that obs model would DECIDE to maneuver). Shows perfect acts early,
    # tle acts late, frozen never.
    cthr = THRESHOLDS_KM["conservative"]
    for oo in OBS_MODELS:
        cs = next((r["stage"] for r in traces[oo] if r["mean_miss"] < cthr), None)
        if cs is not None:
            c = OBS_COLORS[oo]
            ax.axvline(cs, color=c, ls=":", lw=1.6, alpha=0.9)
            ax.annotate(f"{oo} commits\n(s{cs})", xy=(cs, cthr), xytext=(cs + 0.3, cthr + 0.6),
                        fontsize=7.5, color=c, fontweight="bold")
    ax.axhspan(0, D.COLLISION_THRESHOLD_KM, color="red", alpha=0.10)
    ax.set_xlabel("stage"); ax.set_ylabel("believed miss (km)")
    tm = getattr(args, "true_miss", None) or args.init_miss
    ax.set_title(f"Belief collapse + commit by observation model "
                 f"(prior {args.init_miss}km, true {tm}km, spread {args.init_spread}, {args.variant})\n"
                 f"perfect=instant collapse (acts early), tle=slow noisy ~8h (acts late), "
                 f"frozen=no other-obs (never acts); bands=5–95% |dt| support, dotted=commit stage")
    ax.legend(loc="upper right", fontsize=8)
    out = args.plot if (os.sep in args.plot or args.plot.endswith(".png")) else \
        os.path.join(_SCA, "notes", "figures", f"belief_collapse_{args.plot}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=120); print(f"\n  figure saved -> {out}")


def _plot_sweep(sweep, args):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sp = [r[0] for r in sweep]
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, col in [("conservative", "#3b6ea8"), ("aggressive", "#c0392b")]:
        cs = [(r[1][name] if r[1][name] is not None else np.nan) for r in sweep]
        ax.plot(sp, cs, "-o", color=col, label=f"commit: {name}")
    ax.set_xlabel("init spread (km)"); ax.set_ylabel("commit stage (mean crosses act line)")
    ax.set_title(f"When the belief mean crosses the act line vs prior width "
                 f"(init_miss={args.init_miss} other-obs={args.other_obs} {args.variant})")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    out = args.plot if (os.sep in args.plot or args.plot.endswith(".png")) else \
        os.path.join(_SCA, "notes", "figures", f"belief_collapse_sweep_{args.plot}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=120); print(f"  figure saved -> {out}")


if __name__ == "__main__":
    main()
