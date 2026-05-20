"""
plot_deviation_vs_burn_time.py

Plot SC1 along-track deviation at TCA as a function of when a single
0.5 m/s prograde burn is executed, sampled at 20 and 30-minute intervals.

Shows how early burns cause much larger trajectory deviations than late burns,
motivating the deviation-from-nominal dimension in the state space.

Usage:
  .venv/bin/python spacecraftCA/examples/plot_deviation_vs_burn_time.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(_HERE))

from brahe import initialize_eop, state_eci_to_rtn
initialize_eop()

from spacecraft_matrices import (
    sc1_eci_at_tca, propagate, apply_maneuver,
    EPOCH_TCA, STAGE_T_BEFORE_TCA_SEC, DV_MAGNITUDE,
)

OUT_PATH = os.path.join(_HERE, '..', 'notes', 'figures', 'deviation_vs_burn_time.png')


def compute_deviations(interval_min):
    sc1_tca = sc1_eci_at_tca()
    interval_sec = interval_min * 60
    burn_times_sec = np.arange(interval_sec, 24.5 * 3600, interval_sec)

    deviations, hours = [], []
    for dt in burn_times_sec:
        burn_epoch = EPOCH_TCA - dt
        sc1_at_burn = propagate(EPOCH_TCA, sc1_tca, burn_epoch)
        sc1_burned = apply_maneuver(sc1_at_burn, 1, dv=DV_MAGNITUDE)
        sc1_burned_tca = propagate(burn_epoch, sc1_burned, EPOCH_TCA)
        rtn = np.array(state_eci_to_rtn(sc1_tca, sc1_burned_tca))
        deviations.append(abs(rtn[1]) / 1e3)
        hours.append(dt / 3600)
    return hours, deviations


if __name__ == "__main__":
    hours_30, dev_30 = compute_deviations(30)
    hours_20, dev_20 = compute_deviations(20)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(hours_30, dev_30, 'o-', ms=4, color='steelblue', label='30-min intervals')
    ax.plot(hours_20, dev_20, '.', ms=2, color='lightblue', alpha=0.6, label='20-min intervals')

    for k, dt in enumerate(STAGE_T_BEFORE_TCA_SEC):
        h = dt / 3600
        ax.axvline(h, color='red', alpha=0.4, linestyle='--', linewidth=1)
        ax.text(h + 0.1, 5 + k * 8, f'S{k}', color='red', fontsize=8)

    ax.set_xlabel('Time of burn before TCA (hours)')
    ax.set_ylabel('Along-track deviation at TCA (km)')
    ax.set_title(
        f'SC1 along-track deviation at TCA from single {DV_MAGNITUDE} m/s prograde burn\n'
        f'(measured relative to no-burn nominal trajectory)'
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
    print(f'Saved to {OUT_PATH}')
    print(f'Min deviation: {min(dev_30):.2f} km (T-{hours_30[int(np.argmin(dev_30))]:.1f}h)')
    print(f'Max deviation: {max(dev_30):.2f} km (T-{hours_30[int(np.argmax(dev_30))]:.1f}h)')
