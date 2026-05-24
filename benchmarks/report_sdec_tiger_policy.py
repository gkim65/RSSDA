"""
Generate an exact-optimal policy report for SDec-Tiger.

Usage:
    python benchmarks/report_sdec_tiger_policy.py <horizon> [trigger_mode]

trigger_mode defaults to "semi". Output is written to
    policy_reports/sdec_tiger_h<H>_<mode>.md

The report walks the planner's policy structure stage-by-stage, naming
joint actions and joint observations in human-readable form, and resolves
each cluster's belief id to the (P(TL), P(TR)) pair it represents.
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


# Action / observation labels.
PER_AGENT_ACT = ["OL", "OR", "Li"]
PER_AGENT_OBS = ["HL", "HR"]


def joint_act_label(a):
    a1, a2 = a % 3, a // 3
    return f"({PER_AGENT_ACT[a1]}, {PER_AGENT_ACT[a2]})"


def joint_obs_label(o):
    o1, o2 = o % 2, o // 2
    return f"({PER_AGENT_OBS[o1]}, {PER_AGENT_OBS[o2]})"


def per_agent_act(a):
    return PER_AGENT_ACT[a]


def fmt_belief(d):
    return f"[P(TL)={d[0]:.5f}, P(TR)={d[1]:.5f}]"


# AAMAS Mod C reproduction defaults per regime.
ACCURACY_BY_MODE = {
    "decentralized": 0.85,
    "semi":          0.75,
    "centralized":   0.85,
}
SYNC_ACTIONS_BY_MODE = {
    "decentralized": [],
    "semi":          [8],
    "centralized":   list(range(9)),
}


def solve(horizon, trigger_mode):
    sdec_tiger.TRIGGER_MODE = trigger_mode
    sdec_tiger.OBS_ACCURACY = ACCURACY_BY_MODE[trigger_mode]
    sdec_tiger.LAMBDA = 1.0

    factory = TigerProblemFactory(config=None)
    T, O, R, init_b, nacts_fac, nobs_fac = factory.generate()

    model = SDecPOMDPModel(
        nagents=2, nstates=2, nactions=9, nobs=4,
        transitions=T, obs=O, rewards=R, init_beliefs=init_b,
        nacts_factor=nacts_fac, nobs_factor=nobs_fac,
        sync_states=[],
        sync_actions=SYNC_ACTIONS_BY_MODE[trigger_mode],
        sync_observations=[],
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
    val, policy, clustering, cent_vector, cen_dists_map, clustering_cen = (
        solver.multi_agent_astar(horizon, init_beliefs=init_idx)
    )
    elapsed = time.time() - t0
    return solver, val, policy, clustering, cent_vector, cen_dists_map, clustering_cen, elapsed


def write_report(horizon, trigger_mode, out_path):
    (
        solver,
        val,
        policy,
        clustering,
        cent_vector,
        cen_dists_map,
        clustering_cen,
        elapsed,
    ) = solve(horizon, trigger_mode)

    lines = []
    push = lines.append

    push(f"# SDec-Tiger Exact Optimal Policy")
    push("")
    push(f"- **Horizon (h):** {horizon}")
    push(f"- **Trigger mode:** `{trigger_mode}`")
    push(f"- **OBS_ACCURACY:** {sdec_tiger.OBS_ACCURACY}")
    push(f"- **LAMBDA:** {sdec_tiger.LAMBDA}")
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
    push(f"with initial belief b0 = [P(TL)=0.5, P(TR)=0.5].")
    push("")

    push("## Domain")
    push("")
    push("- **States:** TL=0, TR=1 (tiger left / right)")
    push("- **Per-agent actions:** OL=0, OR=1, Li=2")
    push("- **Joint action encoding:** `a = a1 + 3*a2` (a1 = agent 1)")
    push("- **Per-agent observations:** HL=0, HR=1")
    push("- **Joint observation encoding:** `o = o1 + 2*o2` (o1 = agent 1)")
    push("- **Reward:** tiger-listen benchmark reward model; see AAMAS appendix Table 3")
    push("- **Sync trigger:** ")
    if SYNC_ACTIONS_BY_MODE[trigger_mode]:
        sync_acts = SYNC_ACTIONS_BY_MODE[trigger_mode]
        labeled = ", ".join(f"a={a} {joint_act_label(a)}" for a in sync_acts)
        push(f"  joint actions {{{labeled}}} cause centralization at the next stage.")
    else:
        push(f"  none (fully decentralized).")
    push("")

    push("## Centralization vector")
    push("")
    push("Tells whether agents are operating with a shared joint belief at each stage.")
    push("")
    push(f"| stage | regime |")
    push(f"|------:|--------|")
    for t, c in enumerate(cent_vector):
        push(f"| {t}     | {'CEN' if c else 'DEC'} |")
    push("")

    # Collect all referenced belief ids and resolve to actual distributions
    referenced = {0}
    for stage_dists in cen_dists_map:
        for b in stage_dists:
            if b is not None and b >= 0:
                referenced.add(b)

    push("## Belief lookup table")
    push("")
    push("Belief ids referenced by the centralized policy components.")
    push("")
    push(f"| id | belief |")
    push(f"|---:|--------|")
    for b in sorted(referenced):
        if b < len(solver.dists):
            push(f"| {b} | {fmt_belief(solver.dists[b])} |")
    push("")

    push("## Policy by stage")
    push("")
    push("Each stage shows the action taken by the regime active at that stage")
    push("(the active regime is given by the centralization vector above).")
    push("")
    push("- **DEC stage:** each agent picks its action from its decentralized")
    push("  policy `dec[agent][cluster]`. Cluster index follows the agent's")
    push("  local observation history within that stage.")
    push("- **CEN stage:** the joint action is selected from `cen[cluster]`,")
    push("  where the cluster's belief id (column \"belief\") is matched against")
    push("  the actual posterior joint belief at the start of the stage.")
    push("")

    for t, stage in enumerate(policy):
        regime = "CEN" if (t < len(cent_vector) and cent_vector[t]) else "DEC"
        push(f"### Stage {t} — {regime}")
        push("")

        dec_actions = stage[0] if stage and len(stage) > 0 else []
        cen_actions = stage[1] if stage and len(stage) > 1 else []

        if regime == "DEC":
            push("Decentralized component (active at this stage):")
            push("")
            push("| agent | cluster | action |")
            push("|------:|--------:|--------|")
            for ai, agent_clusters in enumerate(dec_actions):
                for ci, a in enumerate(agent_clusters):
                    if a == -2:
                        label = "(unfilled)"
                    elif a == -1:
                        label = "(no-op placeholder)"
                    else:
                        label = f"a={a} ({per_agent_act(a)})"
                    push(f"| {ai} | {ci} | {label} |")
            push("")
            if cen_actions:
                push("Centralized component (inactive — shown for reference only):")
                push("")
                push("| cluster | belief | joint action |")
                push("|--------:|--------|--------------|")
                cluster_dists = cen_dists_map[t] if t < len(cen_dists_map) else []
                for ci, ca in enumerate(cen_actions):
                    if not ca:
                        continue
                    a = ca[0]
                    bid = cluster_dists[ci] if ci < len(cluster_dists) else -1
                    belief = (
                        fmt_belief(solver.dists[bid])
                        if 0 <= bid < len(solver.dists)
                        else f"id={bid}"
                    )
                    push(f"| {ci} | {belief} | a={a} {joint_act_label(a)} |")
                push("")
        else:  # CEN
            push("Centralized component (active at this stage):")
            push("")
            push("| cluster | belief | joint action |")
            push("|--------:|--------|--------------|")
            cluster_dists = cen_dists_map[t] if t < len(cen_dists_map) else []
            for ci, ca in enumerate(cen_actions):
                if not ca:
                    continue
                a = ca[0]
                bid = cluster_dists[ci] if ci < len(cluster_dists) else -1
                belief = (
                    fmt_belief(solver.dists[bid])
                    if 0 <= bid < len(solver.dists)
                    else f"id={bid}"
                )
                push(f"| {ci} | {belief} | a={a} {joint_act_label(a)} |")
            push("")

    push("## Cluster transitions (replay table)")
    push("")
    push("To replay an episode, walk the cluster index forward stage-by-stage.")
    push("Both agents start in cluster 0 at stage 0. After observing o at stage t,")
    push("agent i's cluster at stage t+1 is `clustering[t][i][parent_cluster][o]`.")
    push("`—` indicates an unreachable observation under the parent cluster's")
    push("belief and action.")
    push("")
    push("For action-triggered Tiger semi mode: the cen branch fires whenever")
    push("the joint action at stage t was Li,Li (a=8). Match the resulting joint")
    push("belief against `cen_dists_map[t+1]` to find the cen cluster.")
    push("")

    obs_per_agent = 2  # HL, HR
    obs_header = " | ".join(f"o={oi} ({PER_AGENT_OBS[oi]})" for oi in range(obs_per_agent))
    obs_align = " | ".join(["-----:"] * obs_per_agent)

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

    push("## Reading the policy")
    push("")
    push("In semi-decentralized SDec-Tiger, the joint-listen action is the")
    push("only sync trigger. The agents start decentralized, jointly choose")
    push("Listen (action 8) at decentralized stages, and then — when the")
    push("trigger fires — centralize at the next stage to share their")
    push("histories. At a centralized stage, the policy looks up the joint")
    push("belief in the cluster table and emits the corresponding joint")
    push("action; that action either opens a door (which resets the problem)")
    push("or is another joint Listen (which keeps the agents centralized for")
    push("the next stage and tightens the belief).")
    push("")
    push("Belief ids you'll typically see at a CEN stage right after one")
    push("joint Listen from a uniform prior:")
    push("")
    push("- *Posterior after (HL, HL):* belief ≈ [0.9, 0.1] — open right.")
    push("- *Posterior after (HR, HR):* belief ≈ [0.1, 0.9] — open left.")
    push("- *Posterior after disagreeing observations:* belief ≈ [0.5, 0.5]")
    push("  — Listen again.")
    push("")
    push("These three branches are exactly the three clusters listed for")
    push("the first CEN stage above.")
    push("")

    push("---")
    push("")
    push(
        f"*Report generated by [benchmarks/report_sdec_tiger_policy.py]"
        f"(../benchmarks/report_sdec_tiger_policy.py).*"
    )

    text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def main():
    if len(sys.argv) < 2:
        print("Usage: python report_sdec_tiger_policy.py <horizon> [trigger_mode]")
        sys.exit(1)
    horizon = int(sys.argv[1])
    trigger_mode = sys.argv[2] if len(sys.argv) > 2 else "semi"
    if trigger_mode not in ACCURACY_BY_MODE:
        print(f"Unknown trigger_mode: {trigger_mode}. Use one of {list(ACCURACY_BY_MODE.keys())}.")
        sys.exit(1)

    out_dir = os.path.join(_root, "policy_reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"sdec_tiger_h{horizon}_{trigger_mode}.md")

    write_report(horizon, trigger_mode, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
