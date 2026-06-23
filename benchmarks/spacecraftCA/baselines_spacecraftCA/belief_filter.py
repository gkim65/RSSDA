"""
belief_filter.py — CONTINUOUS Gaussian belief over the along-track dt-at-TCA, shared by the B1
operators and the belief-collapse diagnostic. NO binning: the baselines do not solve the POMDP, so
they do not need the discretizer's dt grid — a real operator tracks a continuous state estimate, and
outcomes are flown through brahe (also continuous). The dt-bin grid is only the POMDP solver's
device; it never touches the baseline belief.

STATE: a 1-D Gaussian N(mean_dt, var_dt) over the SIGNED along-track miss at TCA. dt is a concrete
signed quantity in a rollout (the craft is ahead or behind); the operator measures it with a sign, so
the belief is UNIMODAL around that signed value (no +/-dt hedging — that was only the POMDP solver's
init-belief device).

UPDATE SCHEME (textbook 1-D Kalman):
  predict(stage)      : var += process-drift var (the conjunction's own TCA-offset uncertainty grows
                        toward TCA) + TLE-STALENESS var (the OTHER-craft estimate decays as the last
                        fix ages). Mean unchanged (coasting doesn't move the TCA-frame prediction).
  commit_burn(action) : mean += +/-lever (OWN burn is known; no variance added).
  observe(stage, z)   : a contact delivers a measurement z of the COMBINED dt-at-TCA. The likelihood
                        depends on the other-craft fidelity (`other_obs`):
                          'perfect' : exact -> mean=z, var=0 (full collapse; idealized sync).
                          'tle'     : ONLY at a ~8h TLE epoch, a NOISY fix (sigma = aged TLE sigma)
                                      -> Kalman update (partial collapse to ~sigma).
                          'frozen'  : never observe the other craft (no update; belief only widens).
                        Own GS passes carry no information about the OTHER craft (the only uncertain
                        part of dt — own contribution is known via commit_burn), so they do NOT
                        collapse the belief; only TLE/perfect do.

TLE NOISE: measurement sigma at a fix = tle_sigma_base, growing with fix age toward TLE_SIGMA_MAX
(aged-TLE literature; notes/LITERATURE_CA_THRESHOLDS.md). Between fixes the belief widens at
TLE_SIGMA_AGE_RATE per hour (staleness).
"""
import numpy as np
from math import erf
import spacecraft_discretizer_v2 as D
import spacecraft_transition_v2 as TV
from spacecraft_matrices import PROCESS_DRIFT_SIGMA_KM_PER_SQRT_H

# TLE along-track noise model for the OTHER craft (km).
TLE_SIGMA_BASE_KM = 2.0                    # measurement sigma at a fresh fix (the --tle-sigma knob)
TLE_SIGMA_AGE_RATE_KM_PER_H = 0.375        # staleness growth: 2 km -> ~5 km over 8 h
TLE_SIGMA_MAX_KM = 5.0
TLE_CADENCE_H = 8.0                        # other-craft refresh cadence


def tle_refresh_stages():
    """Stages aligned to each TLE epoch (every TLE_CADENCE_H since screening=stage 0); each 8h mark
    snapped to the NEAREST stage on the orbit-derived grid."""
    t0 = TV.stage_t2go_h(0)
    elapsed = [t0 - TV.stage_t2go_h(k) for k in range(D.N_STAGES)]
    marks, m = [], TLE_CADENCE_H
    while m < t0 - 1e-6:
        k = int(np.argmin([abs(e - m) for e in elapsed]))
        if k not in marks:
            marks.append(k)
        m += TLE_CADENCE_H
    return sorted(marks)


class BeliefFilter:
    """Continuous 1-D Gaussian belief N(mean_dt, var_dt) over signed dt-at-TCA. See module docstring.

    init_dt_mean_km : prior mean signed dt (CDM nominal along-track offset).
    init_dt_std_km  : prior std (km) — the operator's initial uncertainty (~ the screening spread).
    other_obs       : 'perfect' | 'tle' | 'frozen'.
    tle_sigma_base  : base TLE measurement sigma at a fresh fix (km).
    tle_stages      : stages a TLE fix arrives (default tle_refresh_stages()).
    """

    def __init__(self, init_dt_mean_km, init_dt_std_km, perp_km, other_obs="tle",
                 tle_sigma_base=TLE_SIGMA_BASE_KM, tle_stages=None, contact_stages=None):
        self.mean = float(init_dt_mean_km)
        self.var = float(init_dt_std_km) ** 2
        self.perp = perp_km
        self.other_obs = other_obs
        self.tle_sigma_base = tle_sigma_base
        self.tle_stages = set(tle_stages if tle_stages is not None else tle_refresh_stages())
        # a TLE GENERATED at an 8h epoch is only APPLIED at the operator's next GS contact (they pull
        # the data on a pass), aged by the epoch->apply gap. Need the contact schedule to know when.
        from spacecraft_matrices import get_contact_stages as _gcs
        self.contact_stages = set(contact_stages if contact_stages is not None else _gcs())
        self._pending_tle_epoch = None      # stage index of a generated-but-not-yet-applied TLE
        self.collapsed_this_stage = False

    # --- readouts ---
    def mean_dt_km(self):
        return self.mean

    def std_dt_km(self):
        return float(np.sqrt(max(self.var, 0.0)))

    def mean_miss_km(self):
        """E[miss] ~ miss(E[dt]) (the point estimate the operator acts on; Jensen gap negligible here)."""
        return D.miss_km_from_dt(self.mean, self.perp)

    def support_width_km(self, z=1.645):
        """5-95% interval on |dt| from the Gaussian. Returns (|dt|_lo, |dt|_hi)."""
        a, b = self.mean - z * self.std_dt_km(), self.mean + z * self.std_dt_km()
        if a <= 0 <= b:
            return 0.0, max(abs(a), abs(b))
        return min(abs(a), abs(b)), max(abs(a), abs(b))

    def mass_below_miss(self, thr_km):
        """P(miss < thr) = P(|dt| < sqrt(thr^2 - perp^2)) under the Gaussian (0 if perp>=thr)."""
        if self.perp >= thr_km:
            return 0.0
        d = np.sqrt(thr_km * thr_km - self.perp * self.perp)
        s = max(self.std_dt_km(), 1e-9)
        cdf = lambda x: 0.5 * (1.0 + erf((x - self.mean) / (s * np.sqrt(2))))
        return float(cdf(d) - cdf(-d))

    # --- updates ---
    def predict(self, stage):
        """Process-drift var growth + TLE staleness var growth over this stage's coast time."""
        if stage >= D.N_STAGES - 1:
            return
        dt_h = max(TV.stage_t2go_h(stage) - TV.stage_t2go_h(stage + 1), 0.0)
        self.var += (PROCESS_DRIFT_SIGMA_KM_PER_SQRT_H ** 2) * dt_h          # process drift
        if self.other_obs in ("tle", "frozen"):                              # other-craft staleness
            self.var += (TLE_SIGMA_AGE_RATE_KM_PER_H ** 2) * dt_h

    def commit_burn(self, action, lever_km):
        """OWN burn (known): shift the mean by +/-lever (1=+dV->+lever, 2=-dV->-lever); no var added."""
        if action == 0:
            return
        self.mean += (+1.0 if action == 1 else -1.0) * lever_km

    def observe(self, stage, combined_dt_km):
        """Condition on the COMBINED-dt measurement available this stage (combined_dt_km, or None if
        no measurement). Behaviour by other-obs:
          perfect : every GS contact reveals the combined dt exactly -> collapse to a point.
          tle     : a TLE is GENERATED at each 8h epoch but only APPLIED at the next GS contact
                    (the operator pulls it on a pass). At apply time the reading is the other craft
                    forward-predicted to now (centre = combined_dt_km), with sigma AGED by the
                    epoch->apply gap -> noisier the longer the wait. Kalman update (partial collapse).
          frozen  : never applied (belief only widens).
        combined_dt_km should carry the caller's measurement (incl. its noise) when one is available."""
        self.collapsed_this_stage = False
        if self.other_obs == "frozen":
            return
        if self.other_obs == "perfect":
            if combined_dt_km is not None and stage in self.contact_stages:
                self.mean, self.var = float(combined_dt_km), 0.0
                self.collapsed_this_stage = True
            return
        # --- tle: track generation at epochs, apply at the next contact ---
        if stage in self.tle_stages:
            self._pending_tle_epoch = stage                  # a fresh TLE was generated
        # apply a pending TLE at the first contact at/after its epoch
        if (self._pending_tle_epoch is not None and stage in self.contact_stages
                and combined_dt_km is not None):
            age_h = max(TV.stage_t2go_h(self._pending_tle_epoch) - TV.stage_t2go_h(stage), 0.0)
            sigma = min(self.tle_sigma_base + TLE_SIGMA_AGE_RATE_KM_PER_H * age_h, TLE_SIGMA_MAX_KM)
            R = sigma * sigma
            K = self.var / (self.var + R)
            self.mean += K * (float(combined_dt_km) - self.mean)
            self.var = (1.0 - K) * self.var
            self._pending_tle_epoch = None
            self.collapsed_this_stage = True


def prior_from_init(init_miss_km, spread_km, perp_km):
    """(mean_dt, std_dt) prior from the conjunction's init_miss / spread / perp. mean_dt = the
    along-track offset giving the nominal miss (back-solved from perp; sign known from geometry, take
    +). std_dt = the screening spread (km)."""
    v = init_miss_km * init_miss_km - perp_km * perp_km
    mean_dt = float(np.sqrt(v)) if v > 0 else 0.0
    return mean_dt, max(float(spread_km), 1e-6)
