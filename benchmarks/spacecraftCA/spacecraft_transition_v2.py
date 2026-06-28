"""
spacecraft_transition_v2.py

v2 transition model for the reduced relative-motion state
(dT_bin, vdev1, vdev2, stage). See notes/MODEL_DEFINITION.md.

This module builds:
  - per-stage brahe DRIFT GAIN table  gain[k]  (km of dT per unit vdev per stage step)
  - per-conjunction perp (sideways standoff, km), build-time constant
  - the dt transition: dt_next = dt + (vdev1+vdev2)*gain[k]   [+ stochastic spread]
  - the T matrix over v2 states

Kept separate from spacecraft_matrices.py (v1) so the v1 model stays runnable for
A/B and so CHECKPOINT 2 (T vs brahe) is verifiable in isolation before O/R wiring.
"""

import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import spacecraft_matrices as M
from spacecraft_matrices import (
    sc1_eci_at_tca, apply_maneuver, propagate_batch_to,
    STAGE_EPOCHS, EPOCH_TCA, DV_MAGNITUDE,
    PROCESS_DRIFT_SIGMA_KM_PER_SQRT_H,
    TRANSITION_NOISE_N_SAMPLES, ACT_WAIT,
    state_eci_to_rtn, state_rtn_to_eci,
)
# NOTE: CONTACT_STAGES is deliberately NOT imported by name. It is overridable at
# runtime (set_contact_stages, for the contact-timing ablation); reading it live via
# M.CONTACT_STAGES guarantees this builder uses the same list every other consumer
# does. (Importing the name would freeze a stale copy — the old dual-binding footgun.)
import spacecraft_discretizer_v2 as D

N_STAGES = D.N_STAGES
N_ACT_AGENT = 3
N_JOINT_ACTIONS = N_ACT_AGENT ** 2

# ---------------------------------------------------------------------------
# Observation space (v2) — mirrors v1 structure with dt-bin replacing miss-bin
# and vdev replacing dev. Per agent: (dt_bin_obs OR null, own vdev_bin).
# ---------------------------------------------------------------------------
DT_NULL_OBS  = D.N_DT                 # "no new dt observation" off-sync
N_DT_OBS     = D.N_DT + 1             # dt bins + null
N_OBS_AGENT  = N_DT_OBS * D.N_VDEV    # local obs = (dt_obs_or_null, own vdev_bin)
N_JOINT_OBS  = N_OBS_AGENT ** 2


def recompute_obs_alphabet():
    """Recompute the obs-alphabet sizes from the LIVE D.N_DT.

    DT_NULL_OBS / N_DT_OBS / N_OBS_AGENT / N_JOINT_OBS are value-bindings captured at import from
    D.N_DT. But D.N_DT is NOT fixed: build_T_O calls D.set_dt_edges_from_levers(), which regenerates
    the dt grid for THIS conjunction's burn levers and can CHANGE N_DT (a different orbit/horizon/dV
    => different lever set => different bin count). If the obs sizes stay frozen at the import-time
    N_DT while the dt_bin loop ranges over the new (larger) D.N_DT, joint_obs_index overflows the
    stale alphabet — the obs matrix O is allocated N_JOINT_OBS wide but indexed with the new N_DT
    ("index 4378 out of bounds for axis 0 with size 4356"). Re-derive here; build_T_O calls this right
    after set_dt_edges_from_levers so O is allocated at the correct width. When N_DT is unchanged this
    is a no-op (anchor byte-identical)."""
    global DT_NULL_OBS, N_DT_OBS, N_OBS_AGENT, N_JOINT_OBS
    DT_NULL_OBS = D.N_DT
    N_DT_OBS    = D.N_DT + 1
    N_OBS_AGENT = N_DT_OBS * D.N_VDEV
    N_JOINT_OBS = N_OBS_AGENT ** 2
    return N_OBS_AGENT, N_JOINT_OBS


def assert_obs_alphabet_consistent():
    """Cheap fail-loud gate: the obs alphabet must agree with the LIVE discretizer dt grid AND the
    stage grid. Catches the import-order / stale-binding desync class at solve time with a clear
    message instead of an opaque IndexError minutes into a matrix build (see recompute_obs_alphabet).
    Called at the top of build_T_O after the dt grid is regenerated."""
    import spacecraft_stage_grid as SG
    expect_agent = (D.N_DT + 1) * D.N_VDEV
    if N_OBS_AGENT != expect_agent or N_JOINT_OBS != expect_agent ** 2:
        raise RuntimeError(
            f"obs alphabet desynced from D.N_DT={D.N_DT}: N_OBS_AGENT={N_OBS_AGENT} "
            f"(expect {expect_agent}), N_JOINT_OBS={N_JOINT_OBS} (expect {expect_agent ** 2}). "
            f"Call recompute_obs_alphabet() after any D.set_dt_edges_from_levers().")
    if D.N_STAGES != SG.N_STAGES:
        raise RuntimeError(
            f"N_STAGES desynced: D.N_STAGES={D.N_STAGES} != SG.N_STAGES={SG.N_STAGES}. "
            f"The discretizer was imported before the per-conjunction grid rebuild.")

# Reward constants. (v1 used REWARD_MANEUVER=-10, but at -10/burn vs -1/deviation-stage
# a single burn cost as much as 10 off-nominal stages, biasing the policy toward NOT
# maneuvering. Lowered to -1 (2026-06-10 cont.) so the burn/deviation/risk tradeoff is
# balanced and the policy will actually maneuver when warranted.)
REWARD_MANEUVER  =  -2.0   # per agent-burn (config cfg.reward.man_cost sets it)
REWARD_DEVIATION =  -1.0   # per stage per agent off-nominal (vdev != NOM)
REWARD_STEP      =   0.0
# 2026-06-20d: was -1, but at -1 the policy OVER-MANEUVERS (4-burn dance, 2 m/s) because a far
# single burn (~9km, disp ~-3) ~ties 3 extra burns. -2 collapses it to a clean 2-burn (1 m/s,
# brahe 0% coll, lands 5.5-7.5km); -3 == -2. -2 chosen as the default. Too high (-5+) makes the
# policy lazy (one far burn to ~9km instead of threading 4-7km). cfg.reward.man_cost sweeps it.

# ---------------------------------------------------------------------------
# Terminal reward = TWO OPPOSING RAMPS in miss/dT space (v2 redesign).
# See notes/MODEL_DEFINITION.md §7 and LITERATURE_CA_THRESHOLDS.md.
# Replaces v1's monotone graded-risk table (which rewarded distance to 100 km and
# made the policy over-mitigate to ~45 km). Literature objective: MINIMIZE dV to
# CLEAR a threshold, not maximize distance. Optimal sits in the valley between:
#   (1) RISK ramp  — decays from the collision floor (<1km, PoC>=1e-4) to ~0 by the
#       screening-volume clearance (~5 km, PoC<1e-4 for LEO); no reward past it.
#   (2) DISPLACEMENT ramp — grows with |dT_TCA| beyond a station-keeping tube
#       half-width (return-to-orbit cost-to-go proxy, §5b); 0 while in-slot.
# Both knees are physically citable; only the relative WEIGHT is a free knob,
# calibrated so collision ALWAYS dominates.
# ---------------------------------------------------------------------------

# --- RISK ramp parameters (keyed to operational screening thresholds, km) ---
RISK_COLLISION_KM   = 1.0      # < this: PoC >= 1e-4 act/no-act line -> max penalty
RISK_CLEARED_KM     = 5.0      # >= this: screening volume cleared -> 0 risk
RISK_MAX_PENALTY    = -10000.0 # penalty at miss = 0 (collision); dominates everything

# --- DISPLACEMENT ramp parameters (keyed to station-keeping tube, km) ---
DISP_TUBE_HALFWIDTH_KM = 5.0   # within this |dT|: effectively in-slot, no return owed
DISP_COST_PER_KM       = 0.5   # (legacy linear slope; used iff DISP_QUADRATIC_K is None)
# Convex (quadratic) return-cost past the tube. 2026-06-19b: the linear 0.5/km ramp was
# ~100x too gentle vs the -10000 risk floor -> overshooting past 5 km was nearly free, so
# all 3 variants collapsed to ONE late max-lever burn (land ~+10 km), killing differentiation.
# A convex ramp keeps the 5 km knee (cost 0 at/below 5 km, near-0 just past it) but climbs
# steeply, so landing far past 5 km is costly -> the policy must land NEAR 5 km, which from a
# fixed-lever late burn needs finer control / coordination -> revives the sync value.
# cost = -DISP_QUADRATIC_K * (|dT| - tube)^2 .  None => fall back to the legacy linear ramp.
# Physically defensible: phasing-restoration dV grows super-linearly with displacement.
DISP_QUADRATIC_K       = 0.2   # free curvature knob (SWEEP: 0.2 mild .. 1.0 aggressive).
                               # 0.2 = the value that revived Cen<SDec<Dec (2026-06-19b).
# Config sets the curvature (cfg.reward.disp_k). disp_k=None ("none"/"linear" in config)
# falls back to the legacy linear ramp. set_reward() applies both reward knobs.


def set_reward(man_cost=None, disp_k="__keep__"):
    """Apply the reward knobs from the scenario (config-driven; NO env vars). Called by
    scenario_config.build_reward AFTER this module is imported. man_cost=None keeps the
    current REWARD_MANEUVER; disp_k sentinel "__keep__" keeps DISP_QUADRATIC_K (pass None
    explicitly to select the legacy linear ramp, or a float for the convex curvature)."""
    global REWARD_MANEUVER, DISP_QUADRATIC_K
    if man_cost is not None:
        REWARD_MANEUVER = float(man_cost)
    if disp_k != "__keep__":
        DISP_QUADRATIC_K = None if disp_k is None else float(disp_k)
    return REWARD_MANEUVER, DISP_QUADRATIC_K


# ---------------------------------------------------------------------------
# Observation FIDELITY (the obs-quality experiment lever — SDec ONLY).
# ---------------------------------------------------------------------------
# A sync used to be a PERFECT readout of the shared dt_bin (perfect_shared_obs = a delta).
# That made one well-timed sync ~= many syncs -> Cen ~= SDec, killing obs leverage. The
# fidelity knob replaces the sync delta with a GRADED obs: a NORMALIZED prob distribution
# over observed dt_bins, peaked at the true bin and spread by a Gaussian of width sigma_km
# discretized onto the dt grid. perfect == the sigma->0 limit (the existing delta, so the
# regression anchor stays byte-identical). Applied ONLY to the SDec sync branch in build_T_O;
# Centralized and Dec are untouched fixed rails (Cen sync stays the perfect delta, Dec never
# syncs). See notes/SCENARIO_KNOBS.md §E and notes/LITERATURE_CA_THRESHOLDS.md.
#
# Sigmas are GROUNDED in baselines_spacecraftCA/belief_filter.py (the literature along-track
# obs-noise model, cited in LITERATURE_CA_THRESHOLDS.md): TLE_SIGMA_BASE_KM=2.0 grows to
# TLE_SIGMA_MAX_KM=5.0 over TLE_CADENCE_H=8h via TLE_SIGMA_AGE_RATE_KM_PER_H=0.375. GPS (own
# craft) is sub-km (~0.3 km). We import them from belief_filter (do NOT re-declare).
try:
    from baselines_spacecraftCA.belief_filter import (
        TLE_SIGMA_BASE_KM, TLE_SIGMA_MAX_KM,
    )
except Exception:  # pragma: no cover - direct-dir layout fallback
    from belief_filter import TLE_SIGMA_BASE_KM, TLE_SIGMA_MAX_KM  # type: ignore

# GPS sub-km own-craft OD (LITERATURE_CA_THRESHOLDS.md "Onboard GPS": tens of m .. ~100 m;
# ~0.3 km is a conservative km-bin-safe value). Not in belief_filter (which models the OTHER
# craft via TLE), so it lives here with its citation.
GPS_SIGMA_KM = 0.3
# TLE-on-the-other at a sync: use the AGED worst case (5 km). At an 8h cadence the other-craft
# TLE is ~one cadence stale by the time a sync fires, so TLE_SIGMA_MAX_KM is the honest value
# (TLE_SIGMA_BASE_KM=2.0 would be a fresh fix, optimistic). The raw obs.sigma override lets a
# sweep walk this between 2 and 5 km for the sync-value curve.
TLE_SIGMA_KM = TLE_SIGMA_MAX_KM

# Active obs config (set by set_obs / scenario_config). Per-craft sigma over the OTHER craft's
# dt readout at a sync. (None sigma => the perfect delta, regardless of fidelity name.)
OBS_FIDELITY = "perfect"          # perfect | gps | tle | asymmetric | sigma
OBS_SIGMA1_KM = None              # craft-1's sync-obs sigma (km); None => perfect delta
OBS_SIGMA2_KM = None              # craft-2's sync-obs sigma (km); None => perfect delta


def _fidelity_sigmas(fidelity: str, sigma_km=None):
    """Map a fidelity name (+ optional raw sigma override) to per-craft (sigma1, sigma2) km.
    None sigma means a perfect delta for that craft. asymmetric = GPS-self / TLE-other: each
    craft sees the SHARED dt limited by how well IT observes the OTHER craft (craft 1's readout
    of the joint encounter is bounded by its TLE-of-2, craft 2's by its TLE-of-1) -- but since
    each also has precise GPS on ITSELF, the better of the two readouts is what that craft
    contributes. We model the per-craft sync-obs sigma as the craft's view of the joint dt:
    GPS-self gives the craft a near-perfect self anchor, so its limiting noise on the SHARED dt
    is its TLE-of-the-OTHER. => both craft carry TLE-other sigma under asymmetric; the GPS-self
    advantage over pure TLE-vs-TLE is that NEITHER craft's self-position is the bottleneck (in
    tle both self AND other are TLE). Concretely: tle = (TLE, TLE) on a COMBINED self+other
    error; asymmetric = (TLE-other-only, TLE-other-only) -- a smaller sigma because self is GPS.
    See LITERATURE_CA_THRESHOLDS.md (GPS-self / TLE-other split)."""
    if sigma_km is not None:                      # raw override beats the named level
        s = float(sigma_km)
        return (s if s > 0 else None), (s if s > 0 else None)
    f = (fidelity or "perfect").lower()
    if f == "perfect":
        return None, None
    if f == "gps":
        return GPS_SIGMA_KM, GPS_SIGMA_KM
    if f == "tle":
        # both self AND other are TLE -> combined along-track error ~ sqrt(2)*TLE on the joint dt
        s = float(np.hypot(TLE_SIGMA_KM, TLE_SIGMA_KM))
        return s, s
    if f == "asymmetric":
        # GPS-self (negligible) + TLE-other -> ONLY the other-craft TLE limits each readout
        # (no sqrt(2): self is GPS, not a second TLE error). So asymmetric sigma < tle sigma --
        # the headline "knowing your own state precisely helps" case.
        return TLE_SIGMA_KM, TLE_SIGMA_KM
    raise ValueError(f"unknown obs fidelity {fidelity!r} "
                     f"(perfect|gps|tle|asymmetric, or pass obs.sigma)")


def set_obs(fidelity=None, sigma_km="__keep__", coarse="__keep__"):
    """Apply the obs-fidelity knobs (config-driven; NO env vars). Mirrors set_reward; called by
    scenario_config.build_reward AFTER this module is imported. fidelity selects a grounded
    sigma pair; sigma_km (sentinel "__keep__" keeps current; a float overrides the named level
    for the smooth sync-value curve; None/0 => perfect delta). coarse (sentinel "__keep__" keeps;
    True/False toggles the coarse operational obs alphabet) -- OFF by default so the anchor is
    byte-identical. Stores per-craft sigmas so the asymmetric case can carry different fidelity
    per craft."""
    global OBS_FIDELITY, OBS_SIGMA1_KM, OBS_SIGMA2_KM, OBS_COARSE
    if fidelity is not None:
        OBS_FIDELITY = str(fidelity).lower()
    raw = None if sigma_km == "__keep__" else sigma_km
    OBS_SIGMA1_KM, OBS_SIGMA2_KM = _fidelity_sigmas(OBS_FIDELITY, raw)
    if coarse != "__keep__":
        OBS_COARSE = bool(coarse)
    return OBS_FIDELITY, OBS_SIGMA1_KM, OBS_SIGMA2_KM, OBS_COARSE

# Collision threshold used for the collision-probability metric (unchanged).
# (RISK_COLLISION_KM is the reward floor; the binary collision flag uses D's threshold.)


def risk_ramp_reward(miss_km: float) -> float:
    """
    Risk penalty as a SOFT ramp decaying from RISK_MAX_PENALTY at miss=0 to 0 at
    RISK_CLEARED_KM. Flat (max penalty) below RISK_COLLISION_KM (the PoC>=1e-4 floor),
    then smoothly to 0 by the screening-volume clearance distance. Smooth (cosine)
    interpolation avoids the exact-tie cliff of a hard threshold.
    """
    if miss_km <= RISK_COLLISION_KM:
        return RISK_MAX_PENALTY
    if miss_km >= RISK_CLEARED_KM:
        return 0.0
    # smoothstep from 1 (at collision km) to 0 (at cleared km)
    frac = (miss_km - RISK_COLLISION_KM) / (RISK_CLEARED_KM - RISK_COLLISION_KM)
    smooth = 0.5 * (1.0 + np.cos(np.pi * frac))   # 1 -> 0, zero-slope at both ends
    return RISK_MAX_PENALTY * smooth


def displacement_cost(dt_km: float) -> float:
    """
    Return-to-orbit cost-to-go proxy (negative): 0 within the station-keeping tube
    half-width, then grows linearly with |dT| beyond it. Makes far-field drift COST
    something so the policy clears the threshold and stops (no over-mitigation).
    """
    excess = max(abs(dt_km) - DISP_TUBE_HALFWIDTH_KM, 0.0)
    if DISP_QUADRATIC_K is not None:
        return -DISP_QUADRATIC_K * excess * excess     # convex (super-linear) return cost
    return -DISP_COST_PER_KM * excess                  # legacy linear ramp


def terminal_reward(dt_km: float, miss_km: float) -> float:
    """Combined terminal reward = risk ramp (by quadrature miss) + displacement ramp (by |dT|)."""
    return risk_ramp_reward(miss_km) + displacement_cost(dt_km)


def local_obs_index(dt_obs: int, vdev_bin: int) -> int:
    return dt_obs + N_DT_OBS * vdev_bin


def joint_obs_index(o1: int, o2: int) -> int:
    return o1 + N_OBS_AGENT * o2


def _independent_joint_obs(p1, p2):
    pj = np.zeros(N_JOINT_OBS, dtype=np.float64)
    for o1 in range(N_OBS_AGENT):
        if p1[o1] <= 0: continue
        for o2 in range(N_OBS_AGENT):
            if p2[o2] > 0:
                pj[joint_obs_index(o1, o2)] = p1[o1] * p2[o2]
    return pj


def perfect_shared_obs(dt_bin: int, vdev1: int, vdev2: int) -> np.ndarray:
    """Synchronized: both agents see the shared dt_bin plus each own vdev."""
    pj = np.zeros(N_JOINT_OBS, dtype=np.float64)
    o1 = local_obs_index(dt_bin, vdev1)
    o2 = local_obs_index(dt_bin, vdev2)
    pj[joint_obs_index(o1, o2)] = 1.0
    return pj


def private_vdev_only_obs(vdev1: int, vdev2: int) -> np.ndarray:
    """Off-sync: each agent sees only its own vdev (null dt symbol)."""
    p1 = np.zeros(N_OBS_AGENT); p1[local_obs_index(DT_NULL_OBS, vdev1)] = 1.0
    p2 = np.zeros(N_OBS_AGENT); p2[local_obs_index(DT_NULL_OBS, vdev2)] = 1.0
    return _independent_joint_obs(p1, p2)


# ---------------------------------------------------------------------------
# COARSE observation alphabet (opt-in; OFF by default => anchor byte-identical).
# ---------------------------------------------------------------------------
# A km-scale (TLE) sensor cannot resolve the fine 1-3 km dt state bins -- a sigma=5 km readout
# smears across ~10 of the 15 fine bins, so the belief tree branches ~10 ways per sync and never
# prunes (the solve-cost wall). Worse, a 5 km sensor reporting a 1 km-resolution bin is FALSE
# PRECISION. The honest model: a noisy sync resolves the encounter into a few OPERATIONALLY
# MEANINGFUL buckets, not 15 fine bins. We coarsen the OBSERVATION ONLY (the `o` axis of O); the
# STATE grid (T, R, miss, collision, reward) stays FINE and IDENTICAL across fidelities, so TLE
# and perfect runs are directly comparable and propagation/brahe are unaffected.
#
# The coarse symbols are the SIGNED OPERATIONAL THRESHOLDS already in the reward (collision<1,
# "never below 4", safe-band 7 km): a fine dt bin maps to the group its CENTER falls in, by
# |center| against [1,4,7] with sign. ~5 signed groups (far-neg / near-neg / DANGER / near-pos /
# far-pos). A coarsened sync's Gaussian mass is REASSIGNED to each group's REPRESENTATIVE fine
# bin (the fine bin nearest the group's mass-weighted center), so the sync yields at most ~5
# distinct observed symbols instead of ~10 -> the tree prunes -> tractable. O keeps its fine
# shape (N_OBS_AGENT unchanged) so the rollout decode + belief update are untouched.
#
# Per-craft + sigma-aware: coarsening kicks in for a craft ONLY when its sigma exceeds the local
# bin width (set_obs decides). GPS (sigma 0.3 km < bin width) already collapses to 1 bin, so it
# is left FINE automatically; TLE / asymmetric-other (sigma ~5-7 km) coarsen. User-opt-in
# (OBS_COARSE, cfg.obs.coarse=true); default False => fine bins => anchor unchanged.
OBS_COARSE = False                       # cfg.obs.coarse; True => coarsen km-scale (TLE) syncs
_COARSE_THRESHOLDS_KM = (1.0, 4.0, 7.0)  # signed operational cut points (reused from the reward)


def _coarse_group(dt_center_km: float) -> int:
    """Signed operational bucket for a fine bin center: sign * (#thresholds crossed). Groups:
    -3 far-neg | -2 | -1 near-neg | 0 DANGER(|dt|<1) | +1 near-pos | +2 | +3 far-pos."""
    mag = abs(dt_center_km)
    lvl = sum(1 for t in _COARSE_THRESHOLDS_KM if mag >= t)   # 0..3
    return int(np.sign(dt_center_km)) * lvl


def _coarse_reps():
    """For each signed coarse group, the REPRESENTATIVE fine bin (nearest the group's center of
    the fine bins it contains). Returns {group: rep_fine_bin}. Cached per dt grid (N_DT)."""
    global _COARSE_REPS_CACHE
    key = (D.N_DT, tuple(round(D.dt_bin_center_km(b), 4) for b in range(D.N_DT)))
    if _COARSE_REPS_CACHE.get("key") == key:
        return _COARSE_REPS_CACHE["reps"]
    groups = {}
    for b in range(D.N_DT):
        groups.setdefault(_coarse_group(D.dt_bin_center_km(b)), []).append(b)
    reps = {}
    for g, bins in groups.items():
        cs = [D.dt_bin_center_km(b) for b in bins]
        gc = float(np.mean(cs))
        reps[g] = bins[int(np.argmin([abs(c - gc) for c in cs]))]
    _COARSE_REPS_CACHE = {"key": key, "reps": reps}
    return reps


_COARSE_REPS_CACHE = {}


def _dt_obs_dist(true_dt_bin: int, sigma_km, coarse: bool = None) -> np.ndarray:
    """Per-craft observation distribution over the N_DT dt bins given the TRUE bin and a
    Gaussian readout noise of width sigma_km (along-track km). Returns a NORMALIZED length-N_DT
    prob vector over dt bins (NOT including the null symbol -- a sync always yields a dt).

    The Gaussian density is evaluated at each bin CENTER (dt_bin_center_km) and re-normalized
    over the bins, so the result is a proper prob dist peaked at the true bin and spreading to
    neighbours as sigma grows. sigma None / <=0 => the perfect delta (mass 1 on the true bin),
    so the sigma->0 limit reproduces perfect_shared_obs exactly (anchor-preserving).

    coarse (None => module OBS_COARSE): when True, the Gaussian mass is COLLAPSED onto the signed
    operational coarse-group REPRESENTATIVE bins (see above) -- so a km-scale sensor yields ~5
    distinct symbols, not ~10, collapsing the belief tree. Mass is conserved (still sums to 1)
    and still peaked at the true bin's group, so it remains a proper, comparable obs dist."""
    if coarse is None:
        coarse = OBS_COARSE
    if sigma_km is None or sigma_km <= 0.0:
        d = np.zeros(D.N_DT, dtype=np.float64)
        d[true_dt_bin] = 1.0
        return d
    mu = D.dt_bin_center_km(true_dt_bin)
    centers = np.array([D.dt_bin_center_km(b) for b in range(D.N_DT)], dtype=np.float64)
    w = np.exp(-0.5 * ((centers - mu) / float(sigma_km)) ** 2)
    s = w.sum()
    if s <= 1e-300:          # sigma so small every bin underflows -> fall back to delta
        d = np.zeros(D.N_DT, dtype=np.float64)
        d[true_dt_bin] = 1.0
        return d
    d = w / s
    if not coarse:
        return d
    # COARSEN: sum each fine bin's mass into its signed operational group, then place the group
    # total on the group's representative fine bin. Shrinks the # of distinct obs outcomes.
    reps = _coarse_reps()
    gmass = {}
    for b in range(D.N_DT):
        if d[b] > 0.0:
            gmass[_coarse_group(centers[b])] = gmass.get(_coarse_group(centers[b]), 0.0) + d[b]
    out = np.zeros(D.N_DT, dtype=np.float64)
    for g, m in gmass.items():
        out[reps[g]] += m
    return out


def graded_shared_obs(dt_bin: int, vdev1: int, vdev2: int,
                      sigma1_km=None, sigma2_km=None) -> np.ndarray:
    """SDec sync with GRADED fidelity: each agent observes the SHARED true dt_bin through its
    OWN Gaussian readout noise (sigma1 / sigma2 km) plus its own exact vdev. Returns a joint
    obs distribution over N_JOINT_OBS. When both sigmas are None/<=0 this is exactly the perfect
    delta (== perfect_shared_obs), so obs.fidelity=perfect is byte-identical to the old path.

    Coarsening (OBS_COARSE, user-opt-in) collapses each craft's obs onto the operational alphabet.
    GPS needs no special-casing: its sub-bin sigma already lands ~all mass in ONE coarse group, so
    coarsening it is a near no-op (the fine peak is preserved). So a single boolean suffices."""
    if (sigma1_km is None or sigma1_km <= 0.0) and (sigma2_km is None or sigma2_km <= 0.0):
        return perfect_shared_obs(dt_bin, vdev1, vdev2)
    d1 = _dt_obs_dist(dt_bin, sigma1_km, coarse=OBS_COARSE)
    d2 = _dt_obs_dist(dt_bin, sigma2_km, coarse=OBS_COARSE)
    p1 = np.zeros(N_OBS_AGENT, dtype=np.float64)
    p2 = np.zeros(N_OBS_AGENT, dtype=np.float64)
    for b in range(D.N_DT):
        if d1[b] > 0.0:
            p1[local_obs_index(b, vdev1)] = d1[b]
        if d2[b] > 0.0:
            p2[local_obs_index(b, vdev2)] = d2[b]
    return _independent_joint_obs(p1, p2)


# Maneuver-execution error as a fraction of commanded dV. This is the PROPORTIONAL-
# MAGNITUDE term of the standard Gates execution-error model (used operationally, e.g.
# DART TCMs); a low single-digit % is representative (chem looser, electric tighter).
# 2% is a representative value -> SWEEP for sensitivity. See notes/LITERATURE_CA_
# THRESHOLDS.md ("Maneuver execution error"). (We model only proportional magnitude;
# Gates' pointing/cross-track terms are dropped since along-track burns act in dT.)
# Brahe-measured effect: dT spread = frac * |lever_arm(k)| — large early (~2.5 km at
# T-24h @2%), small late (~0.16 km), since an early burn error has longer to amplify.
# (Replaces v1's 50% multiplicative-on-offset noise, ~25x too large for the dT rep.)
EXEC_DV_ERROR_FRAC = 0.02


def split_joint_action(a: int):
    """(a1, a2) per-agent burns; matches v1/RSSDA product order a = a1 + 3*a2."""
    return a % N_ACT_AGENT, a // N_ACT_AGENT


def _action_vdev_delta(action: int) -> int:
    """+dV (1) -> +1, -dV (2) -> -1, WAIT (0) -> 0."""
    if action == ACT_WAIT:
        return 0
    return +1 if action == 1 else -1


def stage_t2go_h(k: int) -> float:
    return (EPOCH_TCA - STAGE_EPOCHS[k]) / 3600.0


# ---------------------------------------------------------------------------
# Per-stage brahe drift-gain table + perp (build-time constants per conjunction)
# ---------------------------------------------------------------------------

def make_sc2_rel_state_at_tca(perp_km: float, dt0_km: float,
                              v_rel_ms: float = 15.0) -> np.ndarray:
    """
    Build SC2's RTN relative state at TCA for a conjunction with a given
    along-track offset dt0 and sideways standoff perp (placed in radial here;
    cross-track gives the same dT dynamics since the burn acts in T regardless).
      dT = dt0 (along-track),  perp = radial standoff,  closing in along-track.
    """
    return np.array([perp_km * 1e3, dt0_km * 1e3, 0.0, 0.0, -v_rel_ms, 0.0])


def compute_gain_table_and_perp(perp_km: float = 0.0, dt0_km: float = 0.0,
                                dv: float = None):
    """
    Per-stage along-track drift RATE from brahe, and the conjunction perp.

    A unit along-track burn does not add a fixed dT; it sets a constant drift RATE
    (km/h) that then accumulates over the remaining time-to-go. We measure the rate
    so the transition can advance dt incrementally per step:
        rate_at[k] = lever_arm(k) / t2go(k)        # km/h per unit vdev, burn at stage k
        dt_step    = rate * (t2go(k) - t2go(k+1))  # per-step increment
    where lever_arm(k) = dT-at-TCA produced by one unit burn applied at stage k.
    (Applying the full lever arm per step instead — the naive reading of "gain" —
    double-counts and blows dt up; this rate form reproduces brahe, see CHECKPOINT 2.)

    Returns (rate_at: np.ndarray[N_STAGES] km/h per unit vdev, perp_km, dt0_km).
    """
    if dv is None:
        dv = DV_MAGNITUDE
    sc1 = sc1_eci_at_tca()
    sc2_tca = np.array(state_rtn_to_eci(sc1, make_sc2_rel_state_at_tca(perp_km, dt0_km)))

    def rtn_T_at_tca(sc2_at_stage_eci, k):
        tca = propagate_batch_to([STAGE_EPOCHS[k]], [sc2_at_stage_eci], EPOCH_TCA)[0]
        return np.array(state_eci_to_rtn(sc1, tca))[1] / 1e3   # along-track km

    rate_at = np.zeros(N_STAGES, dtype=np.float64)
    nom_rtn = np.array(state_eci_to_rtn(sc1, sc2_tca))
    perp_meas = float(np.hypot(nom_rtn[0], nom_rtn[2]) / 1e3)

    for k in range(N_STAGES):
        sc2_s = propagate_batch_to([EPOCH_TCA], [sc2_tca], STAGE_EPOCHS[k])[0]
        dT_nom = rtn_T_at_tca(sc2_s, k)
        sc2_burn = apply_maneuver(sc2_s, 1, dv=dv)          # +dV unit burn
        dT_burn = rtn_T_at_tca(sc2_burn, k)
        lever = dT_burn - dT_nom                            # total dT by TCA, km
        t2go = stage_t2go_h(k)
        rate_at[k] = lever / t2go if t2go > 0 else 0.0      # km/h per unit vdev
    return rate_at, perp_meas, dt0_km


def configure_dt_grid(perp_km: float = 0.0, dt0_km: float = 0.0, dv: float = None):
    """Auto-configure the discretizer's dt bin edges from THIS conjunction's burn levers, the
    SINGLE source of truth for the grid. Both build_init_b_danger and build_T_O call this so the
    init belief and the matrices always agree on N_DT / state indices (no reliance on the static
    default). Idempotent: same conjunction -> same edges. Returns the positive edges."""
    rate_at, _, _ = compute_gain_table_and_perp(perp_km, dt0_km, dv)
    levers = [float(rate_at[k]) * stage_t2go_h(k) for k in range(N_STAGES)]
    return D.set_dt_edges_from_levers(levers)


# ---------------------------------------------------------------------------
# dt transition (deterministic core + stochastic spread)
# ---------------------------------------------------------------------------

_rng = np.random.default_rng(0)
_DRIFT_STD_NORMAL = _rng.standard_normal(TRANSITION_NOISE_N_SAMPLES)
_EXEC_STD_NORMAL = _rng.standard_normal(TRANSITION_NOISE_N_SAMPLES)


def step_hours(stage: int) -> float:
    """Hours of coast between `stage` and the next stage.

    For the LAST decision stage (N_STAGES-2 -> N_STAGES-1) the next stage IS the TCA
    sink, so the coast must run all the way to TCA (the full remaining time-to-go),
    not merely to STAGE_EPOCHS[N-1]. Otherwise a burn at the last decision stage only
    drifts over the (N-2 -> N-1) leg and the final (N-1 -> TCA) leg is never integrated
    -- which truncated a single late burn's lever to ~1/3 (rollout_v2 caught this:
    brahe -17.2 km vs a model that stopped at -5.8 km). Earlier burns were unaffected
    because they accumulate over many stages; only the terminal leg was dropped."""
    if stage >= N_STAGES - 2:
        return max(stage_t2go_h(stage), 0.0)          # coast to TCA
    return max(stage_t2go_h(stage) - stage_t2go_h(stage + 1), 0.0)


def next_dt_distribution(dt_km: float, delta_vdev_sum: int, rate_k: float,
                         stage: int, n_new_burns: int, lever_k: float) -> np.ndarray:
    """
    Distribution over next dt bins.

    TCA-FRAME JUMP transition. The dt STATE is the signed along-track offset AT TCA if
    you coast from here -- a forward PREDICTION, not the current separation. An impulsive
    along-track burn changes velocity permanently, so the new vdev acts all the way to
    TCA: the eventual TCA offset jumps by the FULL remaining lever the instant the burn
    changes vdev, and coasting (WAIT) leaves the prediction unchanged. Hence dt advances
    by the CHANGE in vdev this step times the per-stage lever:

        delta_vdev_sum = (vdev1'+vdev2') - (vdev1+vdev2)   # change in net vdev this step
        dt_next_mean   = dt + delta_vdev_sum * lever_k      # full-lever jump; 0 on WAIT

    lever_k = rate_at[k]*t2go(k) (passed in) = the dT a unit burn at stage k produces BY
    TCA (per-stage rate, matches brahe to <0.1 km vs ~1.5 km for the mean-rate collapse).

    VALIDATED (notes/scratch/diag_optionA.py, deterministic oracle vs brahe): for a
    burn+counter (+dV@s19, -dV@s21) this composes to TCA -6.91 vs brahe -6.98 (0.07 km)
    AND is correct mid-flight at every stage. The residual lever(19)-lever(21) IS the
    real net drift two impulses 2 stages apart leave -- brahe agrees. (The earlier
    "jump composes wrong" alarm (2026-06-20b) was misattributed: see 2026-06-20c.md.)

    ⚠️ KNOWN CAVEAT (next-session work, 2026-06-20c): build_T_O re-reads dt from the BIN
    CENTER each step (no float carry), and the far-field dt grid is coarse (bin 1 center
    -31.62 spans ~[-50,-20]). A -32 km burn snaps to -31.62 then the counter composes
    from that snapped center -- correct only because the grid happens to round favorably.
    Validate the SOLVED policy with rollout_v2 --trace (err ~bin-width at EVERY stage)
    before trusting composition; if it drifts, refine the far-field bins.

    Stochastic spread (added to dt before re-binning):
      - process drift: PROCESS_DRIFT_SIGMA * sqrt(step_hours). PHYSICAL coast-leg
        perturbation that accumulates every stage regardless of burns (TCA-offset
        uncertainty grows toward TCA even on a pure WAIT) -- stays PER-STEP.
      - execution noise: ADDITIVE, sigma = EXEC_DV_ERROR_FRAC * |lever_k| per NEW burn
        this step (the burn's lever is mis-sized by the dV error); already a full-lever
        per-burn quantity, composes with the jump; independent burns add in quadrature.
    """
    dt_step_h = step_hours(stage)
    drift_sigma = PROCESS_DRIFT_SIGMA_KM_PER_SQRT_H * np.sqrt(dt_step_h)
    exec_sigma = (EXEC_DV_ERROR_FRAC * abs(lever_k) * np.sqrt(n_new_burns)
                  if n_new_burns > 0 else 0.0)
    dt_mean = dt_km + delta_vdev_sum * lever_k
    # Vectorized equivalent of the per-sample loop (verified bit-for-bit identical over
    # 984 (mean, sigma) cases incl. extreme tails, max abs diff 0.0; ~9x faster):
    #   for i: x = dt_mean + drift_sigma*Z_d[i] + exec_sigma*Z_e[i]; counts[dt_to_bin(x)] += 1
    # dt_to_bin(x) returns the first bin i with x < DT_EDGES_KM[i+1], clamped to N_DT-1 --
    # i.e. searchsorted on the upper edges (side='right'), clamped. Same fixed normal sample
    # arrays are reused, so it is the exact same empirical distribution.
    x = dt_mean + drift_sigma * _DRIFT_STD_NORMAL + exec_sigma * _EXEC_STD_NORMAL
    bins = np.searchsorted(D.DT_EDGES_KM[1:], x, side="right")
    np.clip(bins, 0, D.N_DT - 1, out=bins)
    counts = np.bincount(bins, minlength=D.N_DT).astype(np.float64)
    return counts / counts.sum()


# ---------------------------------------------------------------------------
# T matrix
# ---------------------------------------------------------------------------

def variant_sync_next(variant: str, next_stage: int) -> bool:
    """Whether the variant has synchronized info arriving at next_stage."""
    if variant == "centralized":
        return True
    if variant == "sdec":
        return next_stage in M.CONTACT_STAGES   # read live (overridable global)
    return False  # dec


def build_T_O(rate_at: np.ndarray, variant: str, verbose: bool = False):
    """
    Build T[a,s,s'] and O[a,s',o] over (dt_bin, vdev1, vdev2, stage).
    `rate_at` mean -> single representative drift rate (Markovian in vdev count).
    Observations: sync stages reveal shared dt_bin + both vdev; off-sync reveal
    only each agent's own vdev. Terminal stage -> sink.
    """
    # AUTO-CONFIGURE the dt grid from THIS conjunction's burn levers BEFORE allocating T/O
    # (T/O sizes depend on D.N_STATES_TOTAL). Fixed near anchors [1,4,5,7] + lever-faithful
    # source edges (<50km) so a burn+counter snaps its first burn to a faithful starting bin
    # (the coarse-tail aliasing was what made a colliding burn+counter read "safe"; 2026-06-20d).
    # Regenerates per conjunction/horizon/dV since the levers change. Markovian: edges depend
    # only on the (build-time) lever table, not on any trajectory.
    levers = [float(rate_at[k]) * stage_t2go_h(k) for k in range(N_STAGES)]
    D.set_dt_edges_from_levers(levers)

    # set_dt_edges_from_levers may have CHANGED D.N_DT for this conjunction's lever set, so the
    # obs-alphabet sizes (frozen at import from the OLD N_DT) must be re-derived BEFORE O is
    # allocated below — otherwise O is sized for the stale alphabet and joint_obs_index overflows
    # it (the 'index 4378 out of bounds size 4356' desync). No-op when N_DT is unchanged.
    recompute_obs_alphabet()
    assert_obs_alphabet_consistent()

    # Per-stage drift rate (rate_at[k]) instead of a single mean over all stages.
    # The gain table already measures the true rate at each stage; using it makes the
    # transition stage-inhomogeneous in vdev dynamics (physically correct: a burn early
    # drifts at the early rate, late at the late rate). The mean-collapse it replaces
    # cost ~0.1 km on the head-on case study (rollout_v2 trace), benign but free to fix.
    T = np.zeros((N_JOINT_ACTIONS, D.N_STATES_TOTAL, D.N_STATES_TOTAL), dtype=np.float64)
    O = np.zeros((N_JOINT_ACTIONS, D.N_STATES_TOTAL, N_JOINT_OBS), dtype=np.float64)
    for k in range(N_STAGES):
        rate_k = float(rate_at[k])
        lever_k = rate_k * stage_t2go_h(k)
        for dt_bin in range(D.N_DT):
            dt_c = D.dt_bin_center_km(dt_bin)
            for v1 in range(D.N_VDEV):
                for v2 in range(D.N_VDEV):
                    s = D.state_index(dt_bin, v1, v2, k)
                    for a in range(N_JOINT_ACTIONS):
                        a1, a2 = split_joint_action(a)
                        if k == N_STAGES - 1:
                            T[a, s, D.SINK_STATE] = 1.0
                            continue
                        nv1 = D.update_vdev_bin(v1, a1)
                        nv2 = D.update_vdev_bin(v2, a2)
                        # CHANGE in net velocity offset this step (not the total): only the
                        # change adds new drift-to-TCA; the prior vdev's lever is already
                        # baked into the incoming dt (TCA-frame jump). vdev saturates at
                        # +/-1, so a burn into the cap yields delta=0 (no new lever) --
                        # the known vdev+-2 limitation (a same-dir double-burn can't add).
                        delta_vdev_sum = ((D.vdev_value(nv1) + D.vdev_value(nv2))
                                          - (D.vdev_value(v1) + D.vdev_value(v2)))
                        n_new_burns = (a1 != ACT_WAIT) + (a2 != ACT_WAIT)
                        sync_next = variant_sync_next(variant, k + 1)
                        dist = next_dt_distribution(dt_c, delta_vdev_sum, rate_k, k,
                                                    n_new_burns, lever_k)
                        for nb, p in enumerate(dist):
                            if p > 0.0:
                                sp = D.state_index(nb, nv1, nv2, k + 1)
                                T[a, s, sp] += p
                                if sync_next:
                                    # GRADED obs lever is SDec-ONLY (the experiment variable):
                                    # Centralized's per-stage sync stays the PERFECT delta (a
                                    # fixed top rail), and Dec never syncs (off-sync branch).
                                    # So obs.fidelity changes ONLY the SDec sync readout; with
                                    # perfect (default) graded_shared_obs == perfect_shared_obs
                                    # for all variants -> anchor byte-identical.
                                    if variant == "sdec":
                                        pj = graded_shared_obs(nb, nv1, nv2,
                                                               OBS_SIGMA1_KM, OBS_SIGMA2_KM)
                                    else:
                                        pj = perfect_shared_obs(nb, nv1, nv2)
                                else:
                                    pj = private_vdev_only_obs(nv1, nv2)
                                O[a, sp, :] += p * pj
        if verbose:
            print(f"  ... v2 T/O stage {k}/{N_STAGES-1} built")
    for a in range(N_JOINT_ACTIONS):
        T[a, D.SINK_STATE, D.SINK_STATE] = 1.0
        O[a, D.SINK_STATE, :] = 1.0 / N_JOINT_OBS
    # normalize O rows
    for a in range(N_JOINT_ACTIONS):
        for sp in range(D.N_STATES_TOTAL):
            rs = O[a, sp, :].sum()
            O[a, sp, :] = (O[a, sp, :] / rs) if rs > 1e-12 else (1.0 / N_JOINT_OBS)
    return T, O


def build_R(perp_km: float) -> np.ndarray:
    """
    Reward R[a, s] (v2 two-ramp terminal reward):
      maneuver cost per burn, per-stage deviation penalty (vdev != NOM),
      and at the final stage the combined terminal reward = RISK ramp (by quadrature
      miss) + DISPLACEMENT ramp (by |dT_TCA|). See terminal_reward() / §7.
    """
    R = np.zeros((N_JOINT_ACTIONS, D.N_STATES_TOTAL), dtype=np.float64)
    for k in range(N_STAGES):
        for dt_bin in range(D.N_DT):
            dt_c = D.dt_bin_center_km(dt_bin)
            miss = D.miss_km_from_dt(dt_c, perp_km)
            for v1 in range(D.N_VDEV):
                for v2 in range(D.N_VDEV):
                    s = D.state_index(dt_bin, v1, v2, k)
                    dev_pen = REWARD_DEVIATION * ((v1 != D.VDEV_ZERO) + (v2 != D.VDEV_ZERO))
                    term = terminal_reward(dt_c, miss) if k == N_STAGES - 1 else 0.0
                    for a in range(N_JOINT_ACTIONS):
                        a1, a2 = split_joint_action(a)
                        man = (REWARD_MANEUVER if a1 != ACT_WAIT else 0.0) + \
                              (REWARD_MANEUVER if a2 != ACT_WAIT else 0.0)
                        R[a, s] = REWARD_STEP + dev_pen + man + term
    # sink: zero
    return R


def build_init_b(dt0_km: float) -> np.ndarray:
    """Point-mass initial belief: nominal vdev, stage 0, dt bin from dt0."""
    init_b = np.zeros(D.N_STATES_TOTAL, dtype=np.float64)
    init_b[D.state_index(D.dt_to_bin(dt0_km), D.VDEV_ZERO, D.VDEV_ZERO, 0)] = 1.0
    return init_b


def build_init_b_spread(miss_bins_km, sign_mode="both") -> np.ndarray:
    """
    v1-matched SPREAD initial belief over uncertain along-track magnitude, to give
    sync something to resolve (v1 spread over unsigned miss bins; here over |dt|).

    miss_bins_km : iterable of candidate |dt| magnitudes (km), e.g. [50, 200] for
                   "far/uncertain" like v1's two farthest bins.
    sign_mode    : "both" -> split each magnitude over +dt and -dt (don't know which
                   side; faithful to v1's unsigned-miss ambiguity, exercises signed
                   dynamics); "pos" -> +dt only (known side).
    All mass at stage 0, vdev = NOM.
    """
    b = np.zeros(D.N_STATES_TOTAL, dtype=np.float64)
    for m in miss_bins_km:
        signs = [+1.0, -1.0] if sign_mode == "both" else [+1.0]
        for s in signs:
            b[D.state_index(D.dt_to_bin(s * m), D.VDEV_ZERO, D.VDEV_ZERO, 0)] += 1.0
    total = b.sum()
    if total <= 0:
        return build_init_b(0.0)
    return b / total


def build_init_b_danger(init_miss_km, spread_km, perp_km, sign_mode="both"):
    """
    Perp-AWARE initial belief, parameterized by TWO interpretable sweep knobs:

      init_miss_km : the TOTAL miss the conjunction starts at (danger dial). The
                     along-track center is back-solved so sqrt(dt_center^2 + perp^2)
                     == init_miss_km. init_miss < COLLISION_THRESHOLD => must-maneuver;
                     init_miss ~ RISK_CLEARED_KM => already-clear control ("needn't
                     maneuver", sync has nothing to resolve).
      spread_km    : half-width of the |dt| spread about that center (uncertainty dial).
                     This is what sync resolves. spread=0 => point belief.
      perp_km      : the fixed sideways standoff (composes with init_miss; an along-track
                     burn cannot reduce it).

    Geometry floor: if init_miss_km <= perp_km the requested total miss is UNREACHABLE
    (perp alone exceeds it). We clamp the center to dt=0 (closest the geometry allows)
    and return (belief, flagged=False) so sweep code can DROP it — such a conjunction
    sits at miss>=perp>=init_miss and would not be flagged in a CDM. Otherwise
    flagged=True.

    Returns (init_b, flagged, effective_miss_km).
    """
    # Configure the dt grid from THIS conjunction's burn levers FIRST, so the init belief is
    # built on the SAME (auto-generated) grid the matrices will use -- no reliance on the static
    # default matching, and correct for any conjunction (different perp -> different levers ->
    # different source edges). build_T_O reconfigures identically at build time (idempotent).
    configure_dt_grid(perp_km)
    perp_km = float(perp_km)
    init_miss_km = float(init_miss_km)
    # Back-solve the along-track center so sqrt(dt^2+perp^2)=init_miss. If perp alone
    # already meets/exceeds the requested total miss, dt is unreachable -> clamp center
    # to 0 (closest the geometry allows) and the effective miss is the perp floor.
    perp_floor = (perp_km >= init_miss_km)
    if perp_floor:
        dt_center = 0.0
        eff_miss = perp_km
    else:
        dt_center = float(np.sqrt(init_miss_km * init_miss_km - perp_km * perp_km))
        eff_miss = init_miss_km
    # "flagged" = would a CDM flag this conjunction = effective miss still inside the
    # screening-clear floor. init_miss=perp=0 (head-on collision course) -> eff 0 ->
    # flagged. Pure cross-track with perp>=RISK_CLEARED_KM -> eff>=floor -> NOT flagged.
    flagged = eff_miss < RISK_CLEARED_KM

    b = np.zeros(D.N_STATES_TOTAL, dtype=np.float64)
    spread_km = max(0.0, float(spread_km))
    # |dt| samples about the center at half-step granularity:
    #   {center-spread, center-spread/2, center, center+spread/2, center+spread},
    # clamped to >=0. At center=0, spread=1.4 this is {0, 0.7, 1.4} — the historical
    # default belief — so dropping the old --init-dt path changes nothing.
    mags = {dt_center}
    if spread_km > 0:
        for d in (spread_km, spread_km / 2.0):
            mags.add(max(0.0, dt_center - d))
            mags.add(dt_center + d)
    signs = [+1.0, -1.0] if sign_mode == "both" else [+1.0]
    for m in mags:
        for s in signs:
            b[D.state_index(D.dt_to_bin(s * m), D.VDEV_ZERO, D.VDEV_ZERO, 0)] += 1.0
    total = b.sum()
    init_b = b / total if total > 0 else build_init_b(dt_center)
    return init_b, flagged, eff_miss


def build_matrices_v2(variant: str, perp_km: float = 0.0, dt0_km: float = 2.0,
                      dv: float = None, init_b=None, verbose: bool = True):
    """Build T, O, R, init_b for the v2 reduced-state model. Mirrors v1 build_matrices.
    If init_b is given (e.g. a spread belief), it overrides the default point-mass."""
    if verbose:
        print(f"Building v2 matrices [{variant}] perp={perp_km} dt0={dt0_km} ...")
    rate_at, perp_meas, _ = compute_gain_table_and_perp(perp_km, dt0_km, dv)
    T, O = build_T_O(rate_at, variant, verbose=verbose)
    R = build_R(perp_meas)
    if init_b is None:
        init_b = build_init_b(dt0_km)
    if verbose:
        print(f"  v2 matrices built: {D.N_STATES_TOTAL} states, {N_JOINT_ACTIONS} acts, "
              f"{N_JOINT_OBS} obs; perp_meas={perp_meas:.3f} km")
    return T, O, R, init_b


if __name__ == "__main__":
    gain, perp, dt0 = compute_gain_table_and_perp(perp_km=0.0, dt0_km=2.0)
    print("Per-stage brahe gain table (km of dT per +1 vdev unit):")
    for k in range(N_STAGES):
        print(f"  stage {k:>2}  t2go={stage_t2go_h(k):>6.2f}h  gain={gain[k]:>8.3f} km/unit")
    print(f"perp(measured) = {perp:.4f} km")
