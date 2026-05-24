"""
Generate stochastic Rescue/Assist Labyrinth benchmarks.

This variant keeps the drilling/noisy benchmark's compact state encoding:

    s = u1 * (N*T) + u2 * T + t_idx

There are no found flags.  Each agent observes its true position plus a noisy
binary sensor.  The terminal commitment action is RESCUE; a valid ASSIST action
by another agent at the target or a neighboring node earns the higher assisted
reward, while a correct unassisted rescue earns the unassisted reward.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(SCRIPT_DIR, "labyrinth_benchmarks")
OUT_DIR = os.path.join(BENCH_DIR, "rescue_assist")

DEFAULT_ASSIST_REWARD = 100.0
DEFAULT_UNASSISTED_REWARD = 80.0
DEFAULT_WRONG_REWARD = -200.0
DEFAULT_STEP_COST = -1.0


@dataclass(frozen=True)
class BaseTopology:
    bid: str
    num_nodes: int
    num_targets: int
    base_actions: int
    target_nodes: List[int]
    start_node: int
    destinations: Dict[Tuple[int, int], int]
    adjacency: List[set]


def encode_standard(num_nodes: int, num_targets: int, u1: int, u2: int,
                    t_idx: int, found1: int = 0, found2: int = 0) -> int:
    return u1 * (num_nodes * num_targets * 4) + u2 * (num_targets * 4) + t_idx * 4 + found1 * 2 + found2


def decode_standard(num_nodes: int, num_targets: int, s: int) -> Tuple[int, int, int, int, int]:
    found2 = s % 2
    tmp = s // 2
    found1 = tmp % 2
    tmp //= 2
    t_idx = tmp % num_targets
    tmp //= num_targets
    u2 = tmp % num_nodes
    u1 = tmp // num_nodes
    return u1, u2, t_idx, found1, found2


def encode_rescue(num_nodes: int, num_targets: int, u1: int, u2: int, t_idx: int) -> int:
    return u1 * (num_nodes * num_targets) + u2 * num_targets + t_idx


def decode_rescue(num_nodes: int, num_targets: int, s: int) -> Tuple[int, int, int]:
    t_idx = s % num_targets
    tmp = s // num_targets
    u2 = tmp % num_nodes
    u1 = tmp // num_nodes
    return u1, u2, t_idx


def _parse_metadata(path: str) -> Tuple[int, int]:
    base_states = None
    base_actions = None
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "states:":
                base_states = int(parts[1])
            elif parts[0] == "actions:":
                base_actions = int(parts[1])
            if base_states is not None and base_actions is not None:
                break
    if base_states is None or base_actions is None:
        raise ValueError(f"Could not parse states/actions metadata from {path}")
    return base_states, base_actions


def _infer_num_nodes(base_states: int) -> int:
    for n in range(2, 200):
        if 4 * n * n * (n - 1) == base_states:
            return n
    raise ValueError(f"Could not infer node count from deterministic state count {base_states}")


def _parse_transition_map(path: str) -> Dict[Tuple[int, int, int], int]:
    transitions: Dict[Tuple[int, int, int], int] = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0] != "T":
                continue
            a1 = int(parts[1])
            a2 = int(parts[2])
            s = int(parts[3])
            sp = int(parts[4])
            prob = float(parts[5])
            if prob > 0.0:
                transitions[(a1, a2, s)] = sp
    return transitions


def load_base_topology(bid: str, start_node: int = 0) -> BaseTopology:
    path = os.path.join(BENCH_DIR, f"labyrinth_{bid}.data")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Base Labyrinth file not found: {path}")

    base_states, base_actions = _parse_metadata(path)
    num_nodes = _infer_num_nodes(base_states)
    num_targets = num_nodes - 1
    target_nodes = [node for node in range(num_nodes) if node != start_node]
    transitions = _parse_transition_map(path)

    destinations: Dict[Tuple[int, int], int] = {}
    adjacency = [set() for _ in range(num_nodes)]

    for node in range(num_nodes):
        destinations[(node, 0)] = node

        ref_a1 = encode_standard(num_nodes, num_targets, node, start_node, 0, 0, 0)
        for action in range(1, base_actions):
            sp = transitions.get((action, 0, ref_a1))
            if sp is None:
                dest = node
            else:
                dest, _, _, _, _ = decode_standard(num_nodes, num_targets, sp)
            destinations[(node, action)] = dest
            if dest != node:
                adjacency[node].add(dest)
                adjacency[dest].add(node)

        ref_a2 = encode_standard(num_nodes, num_targets, start_node, node, 0, 0, 0)
        for action in range(1, base_actions):
            sp = transitions.get((0, action, ref_a2))
            if sp is not None:
                _, dest2, _, _, _ = decode_standard(num_nodes, num_targets, sp)
                if destinations.get((node, action), dest2) == node and dest2 != node:
                    destinations[(node, action)] = dest2
                if dest2 != node:
                    adjacency[node].add(dest2)
                    adjacency[dest2].add(node)

    return BaseTopology(
        bid=str(bid),
        num_nodes=num_nodes,
        num_targets=num_targets,
        base_actions=base_actions,
        target_nodes=target_nodes,
        start_node=start_node,
        destinations=destinations,
        adjacency=adjacency,
    )


def sensor_outcomes(position: int, target_node: int, detection_prob: float) -> List[Tuple[int, float]]:
    beep_prob = detection_prob if position == target_node else 1.0 - detection_prob
    return [
        (position * 2 + 0, 1.0 - beep_prob),
        (position * 2 + 1, beep_prob),
    ]


def _move(topo: BaseTopology, node: int, action: int) -> int:
    if action <= 0 or action >= topo.base_actions:
        return node
    return topo.destinations.get((node, action), node)


def _assist_holds(topo: BaseTopology, node: int, target_node: int) -> bool:
    return node == target_node or target_node in topo.adjacency[node]


def _reward_token(value: float) -> str:
    value = float(value)
    if value.is_integer():
        text = str(int(value))
    else:
        text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def rescue_reward_suffix(assist_reward: float = DEFAULT_ASSIST_REWARD,
                         unassisted_reward: float = DEFAULT_UNASSISTED_REWARD,
                         wrong_reward: float = DEFAULT_WRONG_REWARD,
                         step_cost: float = DEFAULT_STEP_COST) -> str:
    """Return an explicit reward-configuration suffix.

    Rescue/assist rewards define the benchmark instance, so default runs keep
    the reward tuple in the filename instead of using an ambiguous no-suffix
    path.
    """
    return (
        f"_ra{_reward_token(assist_reward)}"
        f"_ru{_reward_token(unassisted_reward)}"
        f"_rw{_reward_token(wrong_reward)}"
        f"_rs{_reward_token(step_cost)}"
    )


def rescue_assist_reward(topo: BaseTopology, u1: int, u2: int, target_node: int,
                         a1: int, a2: int, rescue_action: int, assist_action: int,
                         assist_reward: float = DEFAULT_ASSIST_REWARD,
                         unassisted_reward: float = DEFAULT_UNASSISTED_REWARD,
                         wrong_reward: float = DEFAULT_WRONG_REWARD,
                         step_cost: float = DEFAULT_STEP_COST) -> float:
    rescue1 = a1 == rescue_action
    rescue2 = a2 == rescue_action
    if not (rescue1 or rescue2):
        return float(step_cost)

    wrong_rescue = (rescue1 and u1 != target_node) or (rescue2 and u2 != target_node)
    if wrong_rescue:
        return float(wrong_reward)

    assisted = (
        (a1 == assist_action and _assist_holds(topo, u1, target_node)) or
        (a2 == assist_action and _assist_holds(topo, u2, target_node))
    )
    return float(assist_reward) if assisted else float(unassisted_reward)


def generate_rescue_assist_labyrinth(bid: str, detection_prob: float = 0.85,
                                     start_node: int = 0, force: bool = False,
                                     assist_reward: float = DEFAULT_ASSIST_REWARD,
                                     unassisted_reward: float = DEFAULT_UNASSISTED_REWARD,
                                     wrong_reward: float = DEFAULT_WRONG_REWARD,
                                     step_cost: float = DEFAULT_STEP_COST) -> Tuple[str, BaseTopology]:
    topo = load_base_topology(str(bid), start_node=start_node)
    prob_int = int(detection_prob * 100)
    reward_suffix = rescue_reward_suffix(
        assist_reward, unassisted_reward, wrong_reward, step_cost)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(
        OUT_DIR, f"labyrinth_{bid}_rescue_assist_{prob_int}{reward_suffix}.data")
    if os.path.exists(out_path) and not force:
        return out_path, topo

    num_nodes = topo.num_nodes
    num_targets = topo.num_targets
    nstates = num_nodes * num_nodes * num_targets + 1
    sink = nstates - 1
    obs_per_agent = 2 * num_nodes
    act_per_agent = topo.base_actions + 2
    rescue_action = topo.base_actions
    assist_action = topo.base_actions + 1

    with open(out_path, "w", newline="\n") as f:
        f.write(f"# Rescue/Assist Labyrinth {bid}\n")
        f.write(f"# Detection probability: {detection_prob:g}\n")
        f.write(f"# Nodes: {num_nodes}, Targets: {num_targets}, States: {nstates}\n")
        f.write(f"# Actions per agent: {act_per_agent} (WAIT + {topo.base_actions - 1} MOVEs + RESCUE + ASSIST)\n")
        f.write(f"# Observations per agent: {obs_per_agent} (position * 2 + sensor)\n")
        f.write("# State encoding: s = u1*(N*T) + u2*T + t_idx (NO found flags)\n")
        f.write("# Obs encoding: o = pos*2 + sensor (sensor: 0=Silence, 1=Beep)\n")
        f.write(
            "# RESCUE action: terminal; "
            f"{assist_reward:g} with valid ASSIST, "
            f"{unassisted_reward:g} unassisted, "
            f"{wrong_reward:g} if any RESCUE is wrong\n")
        f.write("# ASSIST condition: assisting agent is at the target or one movement edge away\n")
        f.write("agents: 2\n")
        f.write(f"actions: {act_per_agent}\n")
        f.write(f"observations: {obs_per_agent}\n")
        f.write("discount: 0.95\n")
        f.write("values: reward\n")
        f.write(f"states: {nstates}\n")
        f.write("start:\n")
        f.write("uniform\n\n")

        for s in range(sink):
            u1, u2, t_idx = decode_rescue(num_nodes, num_targets, s)
            target_node = topo.target_nodes[t_idx]
            for a1 in range(act_per_agent):
                for a2 in range(act_per_agent):
                    if a1 == rescue_action or a2 == rescue_action:
                        sp = sink
                    else:
                        v1 = _move(topo, u1, a1)
                        v2 = _move(topo, u2, a2)
                        sp = encode_rescue(num_nodes, num_targets, v1, v2, t_idx)
                    reward = rescue_assist_reward(
                        topo, u1, u2, target_node, a1, a2,
                        rescue_action, assist_action,
                        assist_reward=assist_reward,
                        unassisted_reward=unassisted_reward,
                        wrong_reward=wrong_reward,
                        step_cost=step_cost,
                    )
                    f.write(f"T {a1} {a2} {s} {sp} 1.0\n")
                    f.write(f"R {a1} {a2} {s} * {reward:.1f}\n")

        for a1 in range(act_per_agent):
            for a2 in range(act_per_agent):
                f.write(f"T {a1} {a2} {sink} {sink} 1.0\n")
                f.write(f"R {a1} {a2} {sink} * 0.0\n")

        f.write("\n")
        for a1 in range(act_per_agent):
            for a2 in range(act_per_agent):
                for sp in range(sink):
                    v1, v2, t_idx = decode_rescue(num_nodes, num_targets, sp)
                    target_node = topo.target_nodes[t_idx]
                    for o1, p1 in sensor_outcomes(v1, target_node, detection_prob):
                        for o2, p2 in sensor_outcomes(v2, target_node, detection_prob):
                            prob = p1 * p2
                            if prob > 0.0:
                                f.write(f"O {a1} {a2} {sp} {o1} {o2} {prob:.17g}\n")
                f.write(f"O {a1} {a2} {sink} 0 0 1.0\n")

    return out_path, topo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Rescue/Assist stochastic Labyrinth data.")
    parser.add_argument("bid", help="Labyrinth benchmark id")
    parser.add_argument("detection_prob", nargs="?", type=float, default=0.85)
    parser.add_argument("--assist-reward", type=float, default=DEFAULT_ASSIST_REWARD)
    parser.add_argument("--unassisted-reward", type=float, default=DEFAULT_UNASSISTED_REWARD)
    parser.add_argument("--wrong-reward", type=float, default=DEFAULT_WRONG_REWARD)
    parser.add_argument("--step-cost", type=float, default=DEFAULT_STEP_COST)
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()

    filename, topology = generate_rescue_assist_labyrinth(
        args.bid,
        args.detection_prob,
        force=args.force,
        assist_reward=args.assist_reward,
        unassisted_reward=args.unassisted_reward,
        wrong_reward=args.wrong_reward,
        step_cost=args.step_cost,
    )
    print(filename)
    print(
        f"nodes={topology.num_nodes} targets={topology.num_targets} "
        f"base_actions={topology.base_actions}"
    )
