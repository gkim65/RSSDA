"""
Generate an exact-optimal policy report for SDec-Mars.

Usage:
    python benchmarks/report_sdec_mars_policy.py <horizon> [trigger_mode] [com_mode]

trigger_mode defaults to "semi"; com_mode defaults to "right_band".
Output is written to:
    policy_reports/sdec_mars_h<H>_<mode>_<com>.md  (semi)
    policy_reports/sdec_mars_h<H>_<mode>.md        (dec / cen)

The report walks the planner's policy structure stage-by-stage, decodes
state ids into rover (row, col) pairs, and resolves each cluster's belief
id to a sparse list of (state, prob) entries.
"""

import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
sys.path.insert(0, _here)
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "baselines"))

# Make sure sys.argv has the horizon before importing sdec_mars (its MarsConfig
# reads sys.argv[1] eagerly). We'll restore later.
_real_argv = sys.argv

import sdec_mars

# Map COM_MODE name -> trigger-state list builder
COM_MODES = {
    "right_band": sdec_mars.mars_right_band_triggers,
    "chebyshev1": sdec_mars.mars_chebyshev1_triggers,
}


def state_to_positions(s):
    """Decode joint state id into ((row1, col1), (row2, col2))."""
    p1 = s // 16
    p2 = s % 16
    return (p1 // 4, p1 % 4), (p2 // 4, p2 % 4)


def fmt_state(s):
    (r1, c1), (r2, c2) = state_to_positions(s)
    return f"s={s} [rover1=({r1},{c1}), rover2=({r2},{c2})]"


def fmt_belief(dist, top_k=8, eps=1e-6):
    """Sparse belief printout: list nonzero (state, prob) pairs, sorted by prob descending."""
    items = sorted(
        ((i, p) for i, p in enumerate(dist) if p > eps),
        key=lambda x: -x[1],
    )
    if not items:
        return "(empty)"
    head = items[:top_k]
    tail = items[top_k:]
    parts = [
        f"{fmt_state(s)}: P={p:.4f}"
        for s, p in head
    ]
    if tail:
        residual = sum(p for _, p in tail)
        parts.append(f"+ {len(tail)} more states (Σp={residual:.4f})")
    return "; ".join(parts)


def joint_action_label(a, act_per_agent=6):
    a1, a2 = a % act_per_agent, a // act_per_agent
    return f"a={a} (rover1=a{a1}, rover2=a{a2})"


def per_agent_act(a):
    return f"a{a}"


def solve(horizon, trigger_mode, com_mode_name):
    sdec_mars.TRIGGER_MODE = trigger_mode
    if trigger_mode == "semi":
        sdec_mars.COM_MODE = COM_MODES[com_mode_name]()

    sys.argv = ["", str(horizon)]
    config = sdec_mars.MarsConfig()
    sys.argv = _real_argv

    loader = sdec_mars.MarsProblemLoader(config)
    T, O, R, init_b = loader.load_data()

    from RSSDA import SDecPOMDP, SDecPOMDPModel, RSSDAConfig, int_tuple

    model = SDecPOMDPModel(
        nagents=2, nstates=config.nstates, nactions=config.nacts, nobs=config.nobs,
        transitions=T, obs=O, rewards=R, init_beliefs=init_b,
        nacts_factor=config.nacts_factor, nobs_factor=config.nobs_factor,
        sync_states=config.state_trigger, sync_actions=[], sync_observations=[],
    )
    cfg = RSSDAConfig(
        maxh=horizon, maxit=200, IEmin2=3, alpha=0.2,
        algorithm="exact", heuristic_type="POMDP", tail_heuristic_type="POMDP",
        TI1=False, TI2=False, TI3=False, TI4=False,
        score_limit=20, cen_threshold=0.6, sm_temperature=0.6,
        iter_limit=1000000, rec_limit=1000, hybrid_r=1, max_clusters=10,
    )
    solver = SDecPOMDP(model=model, config=cfg)
    init_idx = solver.dist_dict[int_tuple(solver.init_beliefs)]
    t0 = time.time()
    val, policy, clustering, cent_vector, cen_dists_map, clustering_cen = (
        solver.multi_agent_astar(horizon, init_beliefs=init_idx)
    )
    return solver, config, val, policy, clustering, cent_vector, cen_dists_map, clustering_cen, time.time() - t0


def write_report(horizon, trigger_mode, com_mode_name, out_path):
    (
        solver, config,
        val, policy, clustering, cent_vector, cen_dists_map, clustering_cen,
        elapsed,
    ) = solve(horizon, trigger_mode, com_mode_name)

    lines = []
    push = lines.append

    push(f"# SDec-Mars Exact Optimal Policy")
    push("")
    push(f"- **Horizon (h):** {horizon}")
    push(f"- **Trigger mode:** `{trigger_mode}`" + (f" (COM_MODE = `{com_mode_name}`)" if trigger_mode == "semi" else ""))
    push(f"- **Algorithm:** exact RS-SDA*")
    push(f"- **Heuristic:** POMDP (tight)")
    push(f"- **Plan time:** {elapsed:.3f}s")
    push("")

    push("## Expected value")
    push("")
    push(f"```")
    push(f"V*(b0, h={horizon}) = {val:.5f}")
    push(f"```")
    push("")
    push(f"with initial belief b0 deterministic at state 0 (both rovers at grid (0,0)).")
    push("")

    push("## Domain")
    push("")
    push(f"- **States:** {config.nstates} (joint positions on a 4x4 grid; encoding `s = pos1 * 16 + pos2`, where `pos = row*4 + col`)")
    push(f"- **Per-agent actions:** {config.act_per_agent} (movement + sample/transmit)")
    push(f"- **Joint actions:** {config.nacts} (encoding `a = a1 + 6*a2`)")
    push(f"- **Per-agent observations:** {config.obs_per_agent}")
    push(f"- **Joint observations:** {config.nobs}")
    push(f"- **Initial state:** s=0 [rover1=(0,0), rover2=(0,0)]")
    push("")

    push("## Sync trigger states")
    push("")
    if config.state_trigger:
        push(f"`sync_states` ({len(config.state_trigger)} entries) — joint states where centralization fires:")
        push("")
        push("```")
        chunk = []
        for s in config.state_trigger:
            chunk.append(str(s))
            if len(chunk) == 12:
                push(", ".join(chunk))
                chunk = []
        if chunk:
            push(", ".join(chunk))
        push("```")
        push("")
        push("Decoded examples:")
        push("")
        sample_n = min(5, len(config.state_trigger))
        for s in list(config.state_trigger)[:sample_n]:
            push(f"- {fmt_state(s)}")
    else:
        push("No sync trigger states (fully decentralized).")
    push("")

    push("## Centralization vector")
    push("")
    push("Whether agents are operating with a shared joint belief on the most-likely")
    push("trajectory at each stage. (Off-path branches may still centralize and contribute")
    push("to the expected value through `cen_dists_map`.)")
    push("")
    push(f"| stage | regime |")
    push(f"|------:|--------|")
    for t, c in enumerate(cent_vector):
        push(f"| {t}     | {'CEN' if c else 'DEC'} |")
    push("")

    referenced = {0}
    for stage_dists in cen_dists_map:
        for b in stage_dists:
            if b is not None and b >= 0:
                referenced.add(b)

    push("## Belief lookup table")
    push("")
    push("Belief ids referenced by the centralized policy components, with their")
    push("nonzero state probabilities. (Top-8 states shown per belief; remaining mass")
    push("collapsed into a residual.)")
    push("")
    for b in sorted(referenced):
        if b < len(solver.dists):
            push(f"**dist[{b}]:** {fmt_belief(solver.dists[b])}")
            push("")

    push("## Policy by stage")
    push("")
    push("Each stage shows the action taken by the regime active at that stage")
    push("(see the centralization vector above).")
    push("")

    apa = config.act_per_agent
    for t, stage in enumerate(policy):
        regime = "CEN" if (t < len(cent_vector) and cent_vector[t]) else "DEC"
        push(f"### Stage {t} — {regime}")
        push("")

        dec_actions = stage[0] if stage and len(stage) > 0 else []
        cen_actions = stage[1] if stage and len(stage) > 1 else []

        if regime == "DEC":
            push("Decentralized component (active):")
            push("")
            push("| agent | cluster | action |")
            push("|------:|--------:|--------|")
            for ai, agent_clusters in enumerate(dec_actions):
                for ci, a in enumerate(agent_clusters):
                    if a == -2:
                        label = "(unfilled)"
                    elif a == -1:
                        label = "(no-op)"
                    else:
                        label = f"{per_agent_act(a)}"
                    push(f"| {ai} | {ci} | {label} |")
            push("")

            if cen_actions:
                push("Centralized component (off-path; contributes to expected value):")
                push("")
                push("| cluster | belief id | joint action |")
                push("|--------:|----------:|--------------|")
                cluster_dists = cen_dists_map[t] if t < len(cen_dists_map) else []
                for ci, ca in enumerate(cen_actions):
                    if not ca:
                        continue
                    a = ca[0]
                    bid = cluster_dists[ci] if ci < len(cluster_dists) else -1
                    push(f"| {ci} | {bid} | {joint_action_label(a, apa)} |")
                push("")
        else:
            push("Centralized component (active):")
            push("")
            push("| cluster | belief id | joint action |")
            push("|--------:|----------:|--------------|")
            cluster_dists = cen_dists_map[t] if t < len(cen_dists_map) else []
            for ci, ca in enumerate(cen_actions):
                if not ca:
                    continue
                a = ca[0]
                bid = cluster_dists[ci] if ci < len(cluster_dists) else -1
                push(f"| {ci} | {bid} | {joint_action_label(a, apa)} |")
            push("")

    push("## Cluster transitions (replay table)")
    push("")
    push("To replay an episode, walk the cluster index forward stage-by-stage.")
    push("Both agents start in cluster 0 at stage 0. After observing o at stage t,")
    push("agent i's cluster at stage t+1 is `clustering[t][i][parent_cluster][o]`.")
    push("`—` indicates an unreachable observation (zero probability under the")
    push("parent cluster's belief and action).")
    push("")
    push("For each stage you must also know whether the dec or cen branch is")
    push("active. In state-triggered Mars: the cen branch fires whenever the")
    push("**joint state** lies in `sync_states` (right band). Match the resulting")
    push("joint belief against `cen_dists_map[stage]` to find the cen cluster,")
    push("then use the cen `joint action` table for that stage.")
    push("")

    obs_per_agent = config.obs_per_agent
    obs_header = " | ".join(f"o={oi}" for oi in range(obs_per_agent))
    obs_align = " | ".join(["----:"] * obs_per_agent)

    for t in range(len(clustering)):
        push(f"### Stage {t} → Stage {t+1}: decentralized cluster transitions")
        push("")
        push(f"| agent | parent | {obs_header} |")
        push(f"|------:|-------:| {obs_align} |")
        for ai, agent_clusters in enumerate(clustering[t]):
            for pi, transitions in enumerate(agent_clusters):
                cells = []
                for nc in transitions:
                    cells.append(str(nc) if nc != -1 else "—")
                push(f"| {ai}     | {pi}      | {' | '.join(cells)} |")
        push("")

    if clustering_cen and any(any(any(c != -1 for c in row) for row in agent_clusters) for stage in clustering_cen for agent_clusters in stage):
        push("### Centralized-branch cluster transitions")
        push("")
        push("Same format, but for the centralized branch's cluster index. Use")
        push("`cen_dists_map` to look up the joint belief for each cluster.")
        push("")
        for t in range(len(clustering_cen)):
            stage_data = clustering_cen[t]
            has_data = any(any(c != -1 for c in row) for agent_clusters in stage_data for row in agent_clusters)
            if not has_data:
                continue
            push(f"#### Stage {t} → Stage {t+1}")
            push("")
            push(f"| agent | parent | {obs_header} |")
            push(f"|------:|-------:| {obs_align} |")
            for ai, agent_clusters in enumerate(stage_data):
                for pi, transitions in enumerate(agent_clusters):
                    cells = []
                    for nc in transitions:
                        cells.append(str(nc) if nc != -1 else "—")
                    push(f"| {ai}     | {pi}      | {' | '.join(cells)} |")
            push("")

    push("## Notes on Mars")
    push("")
    push("- **State encoding:** `s = pos1 * 16 + pos2`, with `pos = row*4 + col`.")
    push("  So s=0 is both rovers at (0,0); s=85 is both at (1,1); etc.")
    push("- **Trigger semantics (state-based):** centralization fires when the")
    push("  joint state lies in `sync_states`. For `right_band`, both rovers must")
    push("  be in columns 2 or 3 of the 4x4 grid simultaneously.")
    push("- **All-DEC cent_vector:** does NOT mean centralization is unused;")
    push("  it just means the most-likely trajectory doesn't pass through a")
    push("  sync state. The expected value still incorporates centralization on")
    push("  branches where the rovers do enter the right band.")
    push("- **Action labels (a0–a5)** are kept generic here; the full mapping to")
    push("  movement / sample / transmit lives in `mars.data` and is not")
    push("  decoded by the planner.")
    push("")
    push("---")
    push("")
    push(
        f"*Report generated by [benchmarks/report_sdec_mars_policy.py]"
        f"(../benchmarks/report_sdec_mars_policy.py).*"
    )

    text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def main():
    if len(sys.argv) < 2:
        print("Usage: python report_sdec_mars_policy.py <horizon> [trigger_mode] [com_mode]")
        sys.exit(1)
    horizon = int(sys.argv[1])
    trigger_mode = sys.argv[2] if len(sys.argv) > 2 else "semi"
    com_mode_name = sys.argv[3] if len(sys.argv) > 3 else "right_band"

    if trigger_mode not in ("decentralized", "semi", "centralized"):
        print(f"Unknown trigger_mode: {trigger_mode}.")
        sys.exit(1)
    if trigger_mode == "semi" and com_mode_name not in COM_MODES:
        print(f"Unknown com_mode: {com_mode_name}. Available: {list(COM_MODES.keys())}.")
        sys.exit(1)

    out_dir = os.path.join(_root, "policy_reports")
    os.makedirs(out_dir, exist_ok=True)
    if trigger_mode == "semi":
        out_path = os.path.join(out_dir, f"sdec_mars_h{horizon}_{trigger_mode}_{com_mode_name}.md")
    else:
        out_path = os.path.join(out_dir, f"sdec_mars_h{horizon}_{trigger_mode}.md")

    write_report(horizon, trigger_mode, com_mode_name, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
