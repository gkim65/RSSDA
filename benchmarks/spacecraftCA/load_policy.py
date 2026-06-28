#!/usr/bin/env python
"""load_policy.py — reload a policy pickle saved by compare_variants_v2 (SPACECRAFT_SAVE_POLICY=1)
and print what it contains: the solve metadata, the initial belief, and the per-stage action
schedule. This is the offline companion to policy-saving — re-inspect an expensive (noisy) solve
with ZERO re-solve.

USAGE:
    python load_policy.py notes/policies/<tag>__sdec__bin0.pkl
    python load_policy.py notes/policies/<tag>__sdec__bin0.pkl --raw   # also dump the raw policy tuple

The pickle (light mode) holds: full (the solved RS-SDA*/RS-MAA* tuple), variant, tag, init_bin,
init_b, perp, n_stages, contact_stages, obs_fidelity, obs_sigma, propagator. (T/O/R are embedded
only if SPACECRAFT_SAVE_MATRICES=1 was set; otherwise rebuild via compare_variants_v2.build_matrices
under the SAME scenario.)

NOTE: decoding the per-OBS-history action schedule (what the agent does after each observation)
needs the discretizer/transition modules, which only resolve under an APPLIED scenario. So the
detailed action decode is best-effort: if the modules import, you get a readable schedule; if not,
you still get the metadata + solved value + raw structure. For the fully-decoded closed-loop
behaviour through brahe, re-fly with rollout_v2 (it reuses the same solve path)."""
import argparse
import pickle
import sys


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _summarize(d):
    print(f"=== policy pickle: {d.get('tag')} / {d.get('variant')} / bin{d.get('init_bin')} ===")
    print(f"  obs_fidelity : {d.get('obs_fidelity')}")
    print(f"  obs_sigma    : {d.get('obs_sigma')}")
    print(f"  propagator   : {d.get('propagator')}")
    print(f"  n_stages     : {d.get('n_stages')}")
    print(f"  contacts     : {d.get('contact_stages')}")
    print(f"  perp_km      : {d.get('perp')}")
    print(f"  has matrices : {'T' in d}  (rebuild via build_matrices if False)")

    full = d.get("full")
    if isinstance(full, (list, tuple)) and len(full) >= 1:
        try:
            print(f"  solved value : {float(full[0]):.6f}")
        except (TypeError, ValueError):
            print(f"  solved value : {full[0]!r}")
    print(f"  full tuple len: {len(full) if hasattr(full, '__len__') else 'n/a'} "
          f"(RS-SDA* = 6: value,policy,clustering,cent_vec,cen_dists_map,clustering_cen; "
          f"RS-MAA* = 3: value,policy,clustering)")

    init_b = d.get("init_b")
    if init_b is not None:
        nz = [(i, float(init_b[i])) for i in range(len(init_b)) if init_b[i] > 1e-9]
        print(f"\n  initial belief: {len(nz)} nonzero state(s)")
        # try a readable dt-bin decode if the discretizer resolves
        decoded = _try_decode_belief(nz)
        for line in decoded:
            print("    " + line)


def _try_decode_belief(nz):
    """Best-effort: decode each nonzero belief state into (dt_bin, vdev1, vdev2, stage) + km."""
    try:
        import spacecraft_discretizer_v2 as D
    except Exception:
        return [f"state {i}: p={p:.4f}  (decode unavailable — import a scenario first)"
                for i, p in nz]
    out = []
    for i, p in nz:
        try:
            dt_bin, v1, v2, stage = D.index_to_state(i)
            km = D.dt_bin_center_km(dt_bin)
            out.append(f"state {i}: p={p:.4f}  dt_bin={dt_bin} ({km:+.2f} km)  "
                       f"vdev=({v1},{v2})  stage={stage}")
        except Exception:
            out.append(f"state {i}: p={p:.4f}")
    return out


def _print_schedule(d):
    """Print the action schedule along the policy's ROOT observation path (oh=[0,0]) — the
    decisions the agents take if every observation lands on the first symbol. This is the open-loop
    spine; the full obs-conditional behaviour (what changes when a noisy sync lands elsewhere) is
    in the policy's other branches — re-fly with rollout_v2 to walk those under brahe.

    Uses the CANONICAL decoder (spacecraft_simulator.get_rssda_action) so it matches the eval/
    rollout exactly, rather than re-guessing the nested structure."""
    full = d.get("full")
    if not (isinstance(full, (list, tuple)) and len(full) >= 2):
        print("\n  (no policy structure to schedule)")
        return
    policy = full[1]
    if policy is None:
        print("\n  (policy is None — likely an iter-limit-capped partial solve; only the "
              "value is meaningful for this cell)")
        return

    # RS-MAA* (Dec) is a 3-tuple with a different policy shape; only RS-SDA* (6-tuple) decodes here.
    if len(full) < 6:
        print(f"\n  action schedule: RS-MAA* (Dec) policy — depth {len(policy)}; "
              f"decode via rollout_v2.DecPolicySource (different structure).")
        return
    _, _, clustering, _, cen_dists_map, clustering_cen = full

    try:
        from spacecraft_simulator import get_rssda_action
        import spacecraft_transition_v2 as TV
    except Exception as e:
        print(f"\n  action schedule: decode needs an applied scenario "
              f"({type(e).__name__}: {e}). Re-run under main.py's scenario context.")
        return

    names = {0: "WAIT", 1: "+dV", 2: "-dV"}
    nact = TV.N_ACT_AGENT
    # Root path: start at belief_idx 0 with obs-history [0,0]. (The true root belief index is the
    # init belief's dist id; for a readable spine, the [0,0] path is the canonical first branch.)
    print(f"\n  action schedule along the root obs-path (policy depth = {len(policy)} stages):")
    belief_idx, oh = 0, [0, 0]
    for step in range(len(policy)):
        try:
            joint, a1, a2, is_cen = get_rssda_action(
                policy, cen_dists_map, clustering, clustering_cen,
                step, belief_idx, oh, nact)
            tag = "CEN" if is_cen else "dec"
            print(f"    stage {step:2d}: SC1={names.get(a1, a1):<5} "
                  f"SC2={names.get(a2, a2):<5} [{tag}]")
        except Exception as e:
            print(f"    stage {step:2d}: (decode error: {type(e).__name__})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="path to a *.pkl saved by SPACECRAFT_SAVE_POLICY=1")
    ap.add_argument("--raw", action="store_true", help="also print the raw policy tuple")
    ap.add_argument("--no-schedule", action="store_true", help="skip the action schedule")
    args = ap.parse_args()

    d = _load(args.path)
    _summarize(d)
    if not args.no_schedule:
        _print_schedule(d)
    if args.raw:
        print("\n=== raw full tuple ===")
        print(d.get("full"))


if __name__ == "__main__":
    sys.exit(main())
