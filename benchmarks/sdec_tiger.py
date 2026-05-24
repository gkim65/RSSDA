"""
Multi-Agent Tiger Domain Driver for Semi-Decentralized Planning

This module implements the classic multi-agent Tiger problem, a standard benchmark
for Dec-POMDP algorithms. Two agents must coordinate to open a door, but a tiger
hides behind one of two doors with partial observability.

Domain Description:
    - Two doors: Left (L) and Right (R)
    - A tiger is behind one door (hidden state)
    - Agents can: Open Left (OL), Open Right (OR), or Listen (Li)
    - Opening the tiger's door gives large negative reward (-100)
    - Opening the safe door gives positive reward (+10)
    - Listening provides noisy information about tiger location

Action-Based Triggers:
    This domain uses action-based synchronization triggers in RSSDA.py.
    Coordination is triggered when both
    agents choose to Listen (action 8 = Li + Li).

Observation Model (AAMAS "Mod C" / Table 2 convention):
    - Joint-listen (action 8): per-agent at OBS_ACCURACY, conditionally
      independent. Optionally λ-mixed with a uniform 0.25 via LAMBDA.
    - Single-listener (actions 2, 5, 6, 7): informative at OBS_ACCURACY
      ONLY when TRIGGER_MODE == "semi". Uniform 0.25 otherwise.
    - Opening actions: uniform 0.25.

    The regime-conditional single-listener treatment is intentional and
    required for reproducing AAMAS Table 2 — see the OBS_ACCURACY block
    below for the per-regime accuracy values.

To reproduce AAMAS Table 2 SDec-Tiger (h=8):
    TRIGGER_MODE = "decentralized" + OBS_ACCURACY = 0.85  →  val=12.21726
    TRIGGER_MODE = "semi"          + OBS_ACCURACY = 0.75  →  val=27.21518
    TRIGGER_MODE = "centralized"   + OBS_ACCURACY = 0.85  →  val=47.71696

    The published table uses *different* per-agent accuracies across the
    three columns. Holding OBS_ACCURACY constant across regimes will NOT
    reproduce the table — match accuracy to TRIGGER_MODE as above.

Usage:
    python benchmarks/sdec_tiger.py <horizon> [maxit] [ie_min2] [alpha]

Reference:
    Nair et al., "Taming Decentralized POMDPs: Towards Efficient Policy
    Computation for Multiagent Settings"

Important Notes for Stochastic Observations:
    Exact solvers should almost always use tight heuristics (heuristic_type = "POMDP").
    Large approximate domains often benefit from loose heuristics ("HYBRID"/"QMDP").
    The default approximate SDec-Tiger runner intentionally uses the tested
    current profile from the corrected IJCAI Table 1 rerun: mode-matched
    observation accuracy, TI1 enabled, and rollout sampling from the same
    generated T/O/R arrays used by the planner.

Author: [Mahdi Al-Husseini]
License: MIT  (https://opensource.org/license/mit/)
"""

import sys
import os
import time
import random
import math
import numpy as np

# Add parent directories to path for imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_script_dir)  # Parent of benchmarks/
sys.path.insert(0, _root_dir)  # For RSSDA.py
sys.path.insert(0, os.path.join(_root_dir, 'baselines'))  # For decPOMDP.py

try:
    from RSSDA import SDecPOMDP, SDecPOMDPModel, RSSDAConfig, int_tuple, MemoryLimitExceeded
except ImportError:
    print("Warning: Could not import SDecPOMDP solver. Running in test mode.")
    SDecPOMDP = None
    SDecPOMDPModel = None
    int_tuple = tuple

# Import original decPOMDP for fully decentralized RSMAA mode
try:
    from baselines.decPOMDP import DecPOMDP as OriginalDecPOMDP, MemoryLimitExceeded as DecPOMDPMemoryLimitExceeded
except ImportError:
    print("Warning: Could not import original decPOMDP. decentralized_RSMAA mode may not work.")
    OriginalDecPOMDP = None
    DecPOMDPMemoryLimitExceeded = None

# ============================================================================
#                           USER CONFIGURATION
# ============================================================================
# Modify the parameters below to control solver behavior.
# ============================================================================

# --- Problem Settings ---
TRIGGER_MODE = "semi"     # "centralized", "semi", "decentralized", or "decentralized_RSMAA"
                                         # decentralized_RSMAA uses original decPOMDP.py (RS-MAA* algorithm)

DEFAULT_OBS_ACCURACY_BY_MODE = {
    "decentralized": 0.85,
    "decentralized_RSMAA": 0.85,
    "semi": 0.75,
    "centralized": 0.85,
}

# --- Observation Accuracy ---
# Per-agent listening accuracy.
#   - ALWAYS applied to joint-listen (action 8).
#   - APPLIED to single-listener actions (2, 5, 6, 7) ONLY when
#     TRIGGER_MODE == "semi". Single-listener observations are uniform
#     0.25 in every other TRIGGER_MODE.
#
# AAMAS Table 2 reproduction (per column):
#   "decentralized" → OBS_ACCURACY = 0.85   (single-listener uniform)
#   "semi"          → OBS_ACCURACY = 0.75   (single-listener informative)
#   "centralized"   → OBS_ACCURACY = 0.85   (single-listener uniform)
#
# WARNING: applying informative single-listener to "decentralized" or
# "centralized" — i.e. OBS_ACCURACY > 0 with the regime-conditional gate
# removed — makes h=8 search intractable due to LOH branching factor
# combined with a loose dec heuristic. The asymmetry is load-bearing,
# not a tidiness artifact.
# None means "use the AAMAS/IJCAI mode-matched default above".
OBS_ACCURACY = None

# --- Joint-Listen Mixing (λ) ---
# λ ∈ [0, 1] mixes the joint-listen observation between the informative
# OBS_ACCURACY-per-agent product and a uniform 0.25 over joint observations:
#   O(o | a=8, t) = λ · (p1 · p2) + (1 − λ) · 0.25
# λ = 1.0  → fully informative joint-listen (sharp accuracy-product obs)
# λ = 0.0  → uninformative joint-listen (uniform observations, trigger only)
LAMBDA = 1.0

# --- Core Algorithm ---
ALGORITHM = "approximate"       # "exact" or "approximate" (enables TI approximations)

# --- Heuristic Type ---
# Controls upper-bound heuristic for A* search guidance.
#   "QMDP"   - Loose/fast: ignores observations (dot product of belief and state values)
#   "POMDP"  - Tight/exact: full centralized POMDP value function
#   "HYBRID" - Runs exact POMDP for first HYBRID_R steps, then QMDP
# Rule of thumb: Use "POMDP" for exact algorithm, "QMDP" or "HYBRID" for approximate.
HEURISTIC_TYPE = "POMDP"
HYBRID_R = 1                    # Steps of exact POMDP before switching to QMDP (HYBRID mode only)

# --- Decentralized Heuristic Search ---
# When computing decentralized component heuristics, we run a bounded A* search.
# These parameters control early termination of that inner search.
MAXIT = 200                     # Max A* iterations for decentralized heuristic computation
ALPHA = 0.2                     # Early termination: stop if value <= upper - alpha*|upper|
IE_MIN2 = 3                     # Min depth of information-sharing stages for decentralized heuristic

# --- Approximation Techniques (TI Flags) ---
# Enable these for faster but approximate solutions. Requires ALGORITHM = "approximate".
TI1 = True   # Interleaving Planning/Execution: prune branches via consensus voting
TI2 = True  # Progress-based Pruning: limit per-entity exploration budget
TI3 = True  # Tail Approximation: use heuristics for final REC_LIMIT stages
TI4 = True  # Max Clustering: cluster based on L1 distance between beliefs, weighted by probability mass

# --- TI1: Interleaving Parameters ---
# Consensus voting among top nodes to detect centralized stages early.
SCORE_LIMIT = 20                # Number of top policy nodes to sample for voting
CEN_THRESHOLD = 0.6             # Weighted vote threshold to force centralization
SM_TEMPERATURE = 0.6            # Softmax temperature for node weights (lower = focus on best)

# --- TI2: Progress Pruning ---
# Total iteration budget; per-entity budget B = ITER_LIMIT / (nagents + 1).
# Prunes nodes when exploration exceeds entity's fair share.
ITER_LIMIT = 1000

# --- TI3: Tail Approximation ---
# When remaining horizon <= REC_LIMIT, use heuristic value instead of exact expansion.
REC_LIMIT = 2
TAIL_HEURISTIC_TYPE = "POMDP"    # Heuristic for tail (defaults to HEURISTIC_TYPE if None)

# --- TI4: Max Clustering ---
# Cluster into MAX_Clusters based on combination of L1 distance between resulting beliefs and probability mass
MAX_CLUSTERS = 10

# ============================================================================
#                        END USER CONFIGURATION
# ============================================================================

def effective_obs_accuracy(trigger_mode=None):
    """Return the explicit or mode-matched SDec-Tiger listening accuracy."""
    if OBS_ACCURACY is not None:
        return OBS_ACCURACY
    mode = trigger_mode or TRIGGER_MODE
    return DEFAULT_OBS_ACCURACY_BY_MODE[mode]


# ==========================================
# Problem Configuration Class
# ==========================================

class TigerConfig:
    """Holds configuration for the Tiger problem execution."""

    # Trigger definitions (fixed for this domain)
    TRIG_CENTRALIZED = [list(range(9)), [[i, j] for i in range(3) for j in range(3)]]
    TRIG_SEMI = [[8], [[2, 2]]]
    TRIG_DECENTRAL = [[], []]

    def __init__(self):
        # CLI argument parsing (overrides defaults if provided)
        self.horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 6
        self.maxit = int(sys.argv[2]) if len(sys.argv) > 2 else MAXIT
        self.ie_min2 = int(sys.argv[3]) if len(sys.argv) > 3 else IE_MIN2
        self.alpha = float(sys.argv[4]) if len(sys.argv) > 4 else ALPHA
        self.mod = "C"
        self.obs_accuracy = effective_obs_accuracy(TRIGGER_MODE)

        # Set trigger based on TRIGGER_MODE
        if TRIGGER_MODE == "centralized":
            self.active_trigger = self.TRIG_CENTRALIZED
        elif TRIGGER_MODE == "semi":
            self.active_trigger = self.TRIG_SEMI
        else:  # decentralized
            self.active_trigger = self.TRIG_DECENTRAL


# ==========================================
# Problem Factory
# ==========================================

class TigerProblemFactory:
    """Generates the Transition, Observation, and Reward matrices."""
    
    def __init__(self, config):
        self.nagents = 2
        self.nstates = 2  # L=0, R=1
        self.act_per_agent = 3 # OL=0, OR=1, Li=2
        self.obs_per_agent = 2 # HL=0, HR=1
        
        self.nacts = self.act_per_agent ** self.nagents
        self.nobs = self.obs_per_agent ** self.nagents
        self.nsq = self.nstates ** 2
        self.nso = self.nstates * self.nobs
        
        self.config = config
        
        # Matrices
        self.transit = [0.0] * (self.nsq * self.nacts)
        self.obs = [0.0] * (self.nobs * self.nstates * self.nacts)
        self.reward = [0.0] * (self.nstates * self.nacts)
        self.init_beliefs = [0.5, 0.5]

    def _set_obs_val(self, a, t, o, val):
        self.obs[a * self.nso + t * self.nobs + o] = val

    def _set_listen_observations(self):
        """
        Set observation probabilities for the listen actions.

        Joint-listen (action 8) — always applied, regardless of TRIGGER_MODE:
            O_8(o|t) = LAMBDA · (p1 · p2) + (1 − LAMBDA) · 0.25
        with each p_i = OBS_ACCURACY (correct) / 1 − OBS_ACCURACY (wrong).
        Listens are conditionally independent, so the informative kernel
        is the product of per-agent marginals.

        Single-listener (actions 2, 5, 6, 7) — REGIME-CONDITIONAL:
          - TRIGGER_MODE == "semi": informative. Listener at OBS_ACCURACY,
            opener uniform 0.5.
          - Otherwise: uniform 0.25 (left untouched by this method;
            generate() sets it before calling).

        The semi-only treatment is the AAMAS Mod C convention. Applying
        informative single-listener to dec/cen would change the dec/cen
        problem definitions and (per empirical testing) make h=8 search
        intractable due to LOH branching against a loose dec heuristic.

        Opening actions: uniform 0.25, set in generate() and never touched
        here.
        """
        a1 = lambda a: a % 3
        a2 = lambda a: a // 3
        p_corr = effective_obs_accuracy(TRIGGER_MODE)
        p_wrong = 1.0 - p_corr

        # Joint-listen (action 8): λ-mix of informative product and uniform.
        for t in range(self.nstates):
            for o in range(self.nobs):
                b1, b2 = o % 2, o // 2
                p1 = p_corr if b1 == t else p_wrong
                p2 = p_corr if b2 == t else p_wrong
                informative = p1 * p2
                uniform = 0.25
                self._set_obs_val(8, t, o, LAMBDA * informative + (1 - LAMBDA) * uniform)

        # Single-listener (2, 5, 6, 7): informative ONLY in semi mode.
        if TRIGGER_MODE == "semi":
            for a in (2, 5, 6, 7):
                for t in range(self.nstates):
                    for o in range(self.nobs):
                        b1, b2 = o % 2, o // 2
                        if a1(a) == 2 and a2(a) != 2:   # Agent 1 listens
                            p_listen = p_corr if b1 == t else p_wrong
                            self._set_obs_val(a, t, o, p_listen * 0.5)
                        elif a2(a) == 2 and a1(a) != 2: # Agent 2 listens
                            p_listen = p_corr if b2 == t else p_wrong
                            self._set_obs_val(a, t, o, 0.5 * p_listen)
        # else: leave the uniform 0.25 set by generate() in place.

    def generate(self):
        """Constructs and returns the arrays required by DecPOMDP."""
        
        # Standard Formulation Construction
        for a in range(self.nacts):
            if a == 8: # Both Listen
                for s in range(self.nstates):
                    self.reward[a * self.nstates + s] = -2
                    for t in range(self.nstates):
                        self.transit[a * self.nsq + s * self.nstates + t] = 1.0 * (s == t)
            else: # Opening Doors
                for s in range(self.nstates):
                    # Complex Reward Logic for Opening/Listening combinations
                    r_val = -101*((a%3 == s)*(a//3 == 2) + (a//3 == s)*(a%3 == 2)) \
                            -50*((a%3 == s)*(a//3 == s)) \
                            -100*((a%3 != 2)*(a%3 != s)*(a//3 == s) + (a//3 != 2)*(a//3 != s)*(a%3 == s)) \
                            +9*((a%3 != 2)*(a%3 != s)*(a//3 == 2) + (a//3 != 2)*(a//3 != s)*(a%3 == 2)) \
                            +20*((a%3 != 2)*(a%3 != s)*(a//3 != 2)*(a//3 != s))
                    self.reward[a * self.nstates + s] = r_val

                    for t in range(self.nstates):
                        self.transit[a * self.nsq + s * self.nstates + t] = 0.5
                
                # Uniform observations for opening doors (initial pass)
                for t in range(self.nstates):
                    for o in range(self.nobs):
                        self._set_obs_val(a, t, o, 0.25)

        # Set listen-action observation probabilities using the active mode.
        self._set_listen_observations()

        return (self.transit, self.obs, self.reward, self.init_beliefs,
                [self.act_per_agent]*2, [self.obs_per_agent]*2)


# ==========================================
# Rollout Helpers
# ==========================================

def sample_categorical(weights):
    total = sum(weights)
    if total <= 0:
        raise ValueError("Cannot sample from an all-zero distribution.")
    draw = random.random() * total
    acc = 0.0
    for idx, weight in enumerate(weights):
        acc += weight
        if draw <= acc:
            return idx
    return len(weights) - 1


def action_entry(entry):
    if isinstance(entry, (list, tuple)):
        return entry[0] if entry else -1
    return entry


def has_valid_dec_action(dec_actions, clusters):
    if not dec_actions or len(dec_actions) < 2:
        return False
    for agent, cluster in enumerate(clusters):
        if cluster < 0 or cluster >= len(dec_actions[agent]):
            return False
        if action_entry(dec_actions[agent][cluster]) < 0:
            return False
    return True


def find_centralized_cluster(cen_dists_map, step, belief_idx):
    if step >= len(cen_dists_map):
        return 0
    for idx, bid in enumerate(cen_dists_map[step]):
        if bid == belief_idx:
            return idx
    return 0


def select_policy_action(policy, cent_vector, cen_dists_map, step, belief_idx, clusters, act_per_agent):
    stage_policy = policy[step] if step < len(policy) and policy[step] else None
    if stage_policy is None:
        raise IndexError(f"No policy found at step {step}")

    dec_actions = stage_policy[0] if len(stage_policy) > 0 else []
    cen_actions = stage_policy[1] if len(stage_policy) > 1 else []
    use_central = bool(step < len(cent_vector) and cent_vector[step])

    if not use_central and not has_valid_dec_action(dec_actions, clusters):
        use_central = bool(cen_actions)

    if use_central:
        cluster_idx = find_centralized_cluster(cen_dists_map, step, belief_idx)
        if cluster_idx >= len(cen_actions):
            cluster_idx = 0
        joint_action = action_entry(cen_actions[cluster_idx])
        if joint_action < 0:
            raise ValueError(f"Invalid centralized action at step {step}")
        return joint_action, True, cluster_idx

    if not has_valid_dec_action(dec_actions, clusters):
        raise ValueError(f"Invalid decentralized action at step {step}")

    act1 = action_entry(dec_actions[0][clusters[0]])
    act2 = action_entry(dec_actions[1][clusters[1]])
    return act1 + act_per_agent * act2, False, -1


def sample_environment_step(factory, T, O, R, true_state, joint_action):
    reward = R[joint_action * factory.nstates + true_state]

    t_offset = joint_action * factory.nsq + true_state * factory.nstates
    next_state = sample_categorical(T[t_offset:t_offset + factory.nstates])

    o_offset = joint_action * factory.nso + next_state * factory.nobs
    joint_obs = sample_categorical(O[o_offset:o_offset + factory.nobs])

    return next_state, joint_obs, reward


def next_belief_id(solver, belief_idx, joint_action, joint_obs):
    for obs_id, prob, next_belief in solver.get_terminal(belief_idx, joint_action):
        if obs_id == joint_obs and prob > 0:
            return next_belief
    for obs_id, _prob, next_belief in solver.get_terminal(belief_idx, joint_action):
        if obs_id == joint_obs:
            return next_belief
    raise ValueError(f"No transition found for action {joint_action}, observation {joint_obs}")


def advance_clusters(clusters, clustering, clustering_cen, step, used_central, cen_cluster, joint_obs, obs_per_agent):
    o1 = joint_obs % obs_per_agent
    o2 = joint_obs // obs_per_agent
    source = clustering_cen if used_central else clustering
    next_clusters = clusters[:]

    if step >= len(source) or len(source[step]) < 2:
        return next_clusters

    for agent, obs in ((0, o1), (1, o2)):
        parent_cluster = cen_cluster if used_central else clusters[agent]
        mapping = source[step][agent]
        if parent_cluster < 0 or parent_cluster >= len(mapping):
            continue
        if obs < 0 or obs >= len(mapping[parent_cluster]):
            continue
        mapped = mapping[parent_cluster][obs]
        next_clusters[agent] = mapped if mapped >= 0 else 0

    return next_clusters


# ==========================================
# Execution Engine
# ==========================================

def run_interleaved_execution_legacy(config):
    return run_interleaved_execution(config)

    start_time = time.time()

    # 1. Setup Problem
    factory = TigerProblemFactory(config)
    T, O, R, init_b, nacts_fac, nobs_fac = factory.generate()

    # 2. Initialize Model
    # Extract action triggers from config (first element of active_trigger is joint action indices)
    action_triggers = config.active_trigger[0] if config.active_trigger else []

    model = SDecPOMDPModel(
        nagents=factory.nagents,
        nstates=factory.nstates,
        nactions=factory.nacts,
        nobs=factory.nobs,
        transitions=T,
        obs=O,
        rewards=R,
        init_beliefs=init_b,
        nacts_factor=nacts_fac,
        nobs_factor=nobs_fac,
        sync_states=[],
        sync_actions=action_triggers,
        sync_observations=[]
    )

    # 3. Initialize Solver Config (uses top-level USER CONFIGURATION constants)
    solver_config = RSSDAConfig(
        maxh=config.horizon,
        maxit=config.maxit,
        IEmin2=config.ie_min2,
        alpha=config.alpha,
        algorithm=ALGORITHM,
        heuristic_type=HEURISTIC_TYPE,
        tail_heuristic_type=TAIL_HEURISTIC_TYPE,
        TI1=TI1,
        TI2=TI2,
        TI3=TI3,
        TI4=TI4,
        score_limit=SCORE_LIMIT,
        cen_threshold=CEN_THRESHOLD,
        sm_temperature=SM_TEMPERATURE,
        iter_limit=ITER_LIMIT,
        rec_limit=REC_LIMIT,
        hybrid_r=HYBRID_R,
        max_clusters=MAX_CLUSTERS
    )

    # 4. Initialize Solver
    sdec_pomdp = SDecPOMDP(model=model, config=solver_config)

    print(
        f"Config: mode={TRIGGER_MODE}, obs_accuracy={effective_obs_accuracy(TRIGGER_MODE)}, "
        f"lambda={LAMBDA}, algorithm={ALGORITHM}"
    )

    # 5. Initialize Simulation State
    true_state = sample_categorical(init_b)
    print(f"Tiger Initial State: {true_state} (L=0, R=1)")

    current_horizon = config.horizon

    # RSSDA uses integer mapping for beliefs. Get initial index.
    current_belief_idx = sdec_pomdp.dist_dict[int_tuple(sdec_pomdp.init_beliefs)]
    
    total_reward = 0.0
    history_acts = [[], []]
    history_obs = [[], []]
    
    termination_flag = False

    while not termination_flag:
        print(f"\n--- Planning Phase (Horizon: {current_horizon}) ---")

        # Run RSSDA Planner
        # Returns: Value, Policy, Clustering, CentVector, CenDistsMap, ClusteringCen
        sdec_pomdp.cluster_dict.clear()
        try:
            val, policy, clustering, cent_vector, cen_dists_map, clustering_cen = sdec_pomdp.multi_agent_astar(
                current_horizon, init_beliefs=current_belief_idx
            )
        except MemoryLimitExceeded as e:
            print(f"Result: MO")
            print(f"Memory limit exceeded: {e}")
            return "MO"
        # Extract centralized policy from policy structure (policy[step][1] contains centralized actions)
        policy_cen = [p[1] if p and len(p) > 1 else [] for p in policy] if policy else []

        print(f"Expected value: {val:.5f}")
        print("cent_vector: ", cent_vector)

        if val == 0 and not policy and not policy_cen:
            print("Planner returned empty policy. Terminating.")
            break
        
        # ---------------------------------------------------------
        # UPDATED LOGIC: Execute until the FIRST centralized step
        # ---------------------------------------------------------
        if any(cent_vector):
            # Find the FIRST True occurrence
            idx_first_true = cent_vector.index(True)
            # Execute up to AND including that step (index + 1)
            steps_to_execute = idx_first_true + 1
        else:
            # No centralization in horizon, execute full policy
            steps_to_execute = len(policy)

        print(f"Executing {steps_to_execute} steps from generated policy (Stopping after first sync)...")

        # Execution Loop
        for step in range(steps_to_execute):
            # A. Get Joint Action
            # RSSDA Policy Structure:
            #   policy[step] = [dec_actions, cen_actions]
            #   dec_actions[agent][cluster] = action for that agent's cluster
            #   cen_actions[cluster][0]      = joint action for that centralized cluster
            #
            # For CEN steps, each cluster corresponds to a distinct posterior
            # belief (cen_dists_map[step][cluster] = belief_id).  We navigate
            # to the cluster matching the CURRENT belief index so the executed
            # action reflects the observation just received — not just
            # cluster 0 ("optimal path assumption"), which is only correct
            # when the current belief matches cluster 0's belief.
            #
            # For DEC steps at the start of a plan (step 0 of a replan), the
            # agents have no accumulated local history yet, so cluster 0 is
            # the correct choice.  If a driver executes multiple consecutive
            # DEC steps within a single plan (not the case for Tiger under
            # TI1, which stops at the first sync), additional history-based
            # cluster navigation would be needed.

            try:
                step_policy = policy[step] if len(policy) > step and policy[step] else None
                dec_actions = step_policy[0] if step_policy and len(step_policy) > 0 else []
                cen_actions = step_policy[1] if step_policy and len(step_policy) > 1 else []

                # Check if dec_actions has valid actions (not placeholders -1 or -2)
                dec_valid = (dec_actions and len(dec_actions) >= 2 and
                            len(dec_actions[0]) > 0 and len(dec_actions[1]) > 0 and
                            dec_actions[0][0] >= 0 and dec_actions[1][0] >= 0)

                if dec_valid:
                    # DEC step at the start of a replan: cluster 0 for both agents.
                    act1 = dec_actions[0][0]
                    act2 = dec_actions[1][0]

                # Centralized Policy Container — navigate by current belief
                elif cen_actions and len(cen_actions) > 0:
                    # Look up which CEN cluster matches the current belief.
                    cluster_idx = 0  # safe fallback
                    if step < len(cen_dists_map):
                        for c, bid in enumerate(cen_dists_map[step]):
                            if bid == current_belief_idx:
                                cluster_idx = c
                                break

                    if (cluster_idx < len(cen_actions)
                            and len(cen_actions[cluster_idx]) > 0):
                        joint_act = cen_actions[cluster_idx][0]
                    elif len(cen_actions[0]) > 0:
                        # Fallback to cluster 0 if the matched cluster has no action
                        joint_act = cen_actions[0][0]
                    else:
                        raise IndexError(
                            f"Empty CEN cluster at step {step}, cluster {cluster_idx}")

                    act1 = joint_act % factory.act_per_agent
                    act2 = joint_act // factory.act_per_agent

                else:
                    raise IndexError(f"No actions found in policy for step {step}")

            except Exception as e:
                print(f"Error parsing policy structure during execution at step {step}: {e}")
                if len(policy) > step:
                    print(f"Policy[{step}] Dec Actions: {policy[step][0] if policy[step] else 'N/A'}")
                    print(f"Policy[{step}] Cen Actions: {policy[step][1] if len(policy[step]) > 1 else 'N/A'}")
                else:
                    print(f"Policy has only {len(policy)} steps, cannot access step {step}")
                termination_flag = True
                break
                
            # B. Simulation Step (Environment Response)
            step_reward = 0
            
            # Calculate Reward (Tiger Specific Logic for tracking)
            if act1 == 2 and act2 == 2: step_reward = -2
            elif (act1 != 2 and act2 != 2) and (act1 != act2): step_reward = -100 # Different doors
            elif (act1 != 2 and act2 != 2) and (act1 == act2): # Same door
                step_reward = -50 if (act1 == true_state) else 20
            else: # One listens, one opens
                opener = act1 if act1 != 2 else act2
                step_reward = -101 if opener == true_state else 9
            
            total_reward += step_reward

            # Update State
            if act1 != 2 or act2 != 2:
                true_state = random.choice(range(factory.nstates)) # Reset on open

            # Generate Observation — mirrors the planner's regime-conditional
            # O matrix (see TigerProblemFactory._set_listen_observations).
            #   Both-listen: per-agent at OBS_ACCURACY (only matches the
            #     planner exactly at LAMBDA = 1.0; with λ < 1 the joint
            #     distribution is no longer a product of marginals).
            #   Single-listener under TRIGGER_MODE == "semi": listener at
            #     OBS_ACCURACY, opener uniform.
            #   Otherwise (single-listener under dec/cen, or both open):
            #     uniform.
            obs = [0, 0]
            wrong = lambda s: int(s == 0)
            if act1 == 2 and act2 == 2:  # Both listen
                probs = [OBS_ACCURACY, 1.0 - OBS_ACCURACY]
                obs[0] = int(np.random.choice([true_state, wrong(true_state)], p=probs))
                obs[1] = int(np.random.choice([true_state, wrong(true_state)], p=probs))
            elif TRIGGER_MODE == "semi" and ((act1 == 2) != (act2 == 2)):
                # Single-listener under semi: listener informative, opener uniform.
                probs = [OBS_ACCURACY, 1.0 - OBS_ACCURACY]
                listener = 0 if act1 == 2 else 1
                obs[listener] = int(np.random.choice([true_state, wrong(true_state)], p=probs))
                obs[1 - listener] = random.choice(range(factory.nstates))
            else:  # both open, or single-listener under dec/cen — uniform
                obs[0] = random.choice(range(factory.nstates))
                obs[1] = random.choice(range(factory.nstates))

            # Record History
            history_acts[0].append(act1)
            history_acts[1].append(act2)
            history_obs[0].append(obs[0])
            history_obs[1].append(obs[1])

            # C. Belief Update
            flat_action = act1 + (act2 * factory.act_per_agent)
            flat_observation = obs[0] + (obs[1] * factory.obs_per_agent)

            # Direct transition lookup using RSSDA logic
            # get_terminal returns List[Tuple[obs_id, prob, belief_id]]
            terminal_transitions = sdec_pomdp.get_terminal(current_belief_idx, flat_action)
            # Find the belief_id for our observation
            next_belief_idx = None
            for obs_id, prob, belief_id in terminal_transitions:
                if obs_id == flat_observation:
                    next_belief_idx = belief_id
                    break
            if next_belief_idx is None:
                print(f"Warning: No transition found for observation {flat_observation}, keeping current belief")
                next_belief_idx = current_belief_idx
            current_belief_idx = next_belief_idx
            
            # Determine Label for Log
            is_centralized = cent_vector[step] if step < len(cent_vector) else False
            policy_type = "CEN" if is_centralized else "DEC"

            print(f"  Step {step} [Rew: {step_reward}] [{policy_type}]: Actions({act1},{act2}) -> Obs({obs[0]},{obs[1]}) -> BeliefIdx: {current_belief_idx}")

        # Reduce remaining horizon
        current_horizon -= steps_to_execute
        
        # Check Termination
        if current_horizon <= 0:
            termination_flag = True
        else:
            # Setup for next iteration
            sdec_pomdp.maxh = current_horizon

    print("\n=== Simulation Complete ===")
    print(f"Total Reward: {total_reward}")
    print(f"Total Time: {time.time() - start_time:.4f}s")
    
    return total_reward


def run_interleaved_execution(config):
    """Run RSSDA planning/execution with the planner's own T/O/R model arrays."""
    start_time = time.time()

    factory = TigerProblemFactory(config)
    T, O, R, init_b, nacts_fac, nobs_fac = factory.generate()
    action_triggers = config.active_trigger[0] if config.active_trigger else []

    model = SDecPOMDPModel(
        nagents=factory.nagents,
        nstates=factory.nstates,
        nactions=factory.nacts,
        nobs=factory.nobs,
        transitions=T,
        obs=O,
        rewards=R,
        init_beliefs=init_b,
        nacts_factor=nacts_fac,
        nobs_factor=nobs_fac,
        sync_states=[],
        sync_actions=action_triggers,
        sync_observations=[],
    )
    solver_config = RSSDAConfig(
        maxh=config.horizon,
        maxit=config.maxit,
        IEmin2=config.ie_min2,
        alpha=config.alpha,
        algorithm=ALGORITHM,
        heuristic_type=HEURISTIC_TYPE,
        tail_heuristic_type=TAIL_HEURISTIC_TYPE,
        TI1=TI1,
        TI2=TI2,
        TI3=TI3,
        TI4=TI4,
        score_limit=SCORE_LIMIT,
        cen_threshold=CEN_THRESHOLD,
        sm_temperature=SM_TEMPERATURE,
        iter_limit=ITER_LIMIT,
        rec_limit=REC_LIMIT,
        hybrid_r=HYBRID_R,
        max_clusters=MAX_CLUSTERS,
    )

    sdec_pomdp = SDecPOMDP(model=model, config=solver_config)
    current_belief_idx = sdec_pomdp.dist_dict[int_tuple(sdec_pomdp.init_beliefs)]
    current_horizon = config.horizon
    true_state = sample_categorical(init_b)
    total_reward = 0.0
    history_acts = [[], []]
    history_obs = [[], []]

    print(
        f"Config: mode={TRIGGER_MODE}, obs_accuracy={effective_obs_accuracy(TRIGGER_MODE)}, "
        f"lambda={LAMBDA}, algorithm={ALGORITHM}"
    )
    print(f"Tiger Initial State: {true_state} (L=0, R=1)")

    while current_horizon > 0:
        print(f"\n--- Planning Phase (Horizon: {current_horizon}) ---")
        sdec_pomdp.cluster_dict.clear()

        try:
            val, policy, clustering, cent_vector, cen_dists_map, clustering_cen = (
                sdec_pomdp.multi_agent_astar(current_horizon, init_beliefs=current_belief_idx)
            )
        except MemoryLimitExceeded as e:
            print("Result: MO")
            print(f"Memory limit exceeded: {e}")
            return "MO"

        if not policy:
            print("Planner returned empty policy. Terminating.")
            break

        print(f"Expected value: {val:.5f}")
        print("cent_vector: ", cent_vector)

        steps_to_execute = cent_vector.index(True) + 1 if any(cent_vector) else len(policy)
        steps_to_execute = min(steps_to_execute, current_horizon, len(policy))
        print(f"Executing {steps_to_execute} steps from generated policy.")

        current_clusters = [0, 0]
        for step in range(steps_to_execute):
            try:
                joint_act, used_central, cen_cluster = select_policy_action(
                    policy,
                    cent_vector,
                    cen_dists_map,
                    step,
                    current_belief_idx,
                    current_clusters,
                    factory.act_per_agent,
                )
            except Exception as e:
                print(f"Error parsing policy structure during execution at step {step}: {e}")
                if step < len(policy):
                    print(f"Policy[{step}] Dec Actions: {policy[step][0] if policy[step] else 'N/A'}")
                    print(f"Policy[{step}] Cen Actions: {policy[step][1] if len(policy[step]) > 1 else 'N/A'}")
                return total_reward

            act1 = joint_act % factory.act_per_agent
            act2 = joint_act // factory.act_per_agent

            true_state, joint_obs, step_reward = sample_environment_step(
                factory, T, O, R, true_state, joint_act
            )
            obs = [
                joint_obs % factory.obs_per_agent,
                joint_obs // factory.obs_per_agent,
            ]

            total_reward += step_reward
            history_acts[0].append(act1)
            history_acts[1].append(act2)
            history_obs[0].append(obs[0])
            history_obs[1].append(obs[1])

            current_belief_idx = next_belief_id(
                sdec_pomdp, current_belief_idx, joint_act, joint_obs
            )
            current_clusters = advance_clusters(
                current_clusters,
                clustering,
                clustering_cen,
                step,
                used_central,
                cen_cluster,
                joint_obs,
                factory.obs_per_agent,
            )

            policy_type = "CEN" if used_central else "DEC"
            print(
                f"  Step {step} [Rew: {step_reward}] [{policy_type}]: "
                f"Actions({act1},{act2}) -> Obs({obs[0]},{obs[1]}) "
                f"-> BeliefIdx: {current_belief_idx}"
            )

        current_horizon -= steps_to_execute
        sdec_pomdp.maxh = current_horizon

    print("\n=== Simulation Complete ===")
    print(f"Total Reward: {total_reward}")
    print(f"Total Time: {time.time() - start_time:.4f}s")
    return total_reward


# ==========================================
# Decentralized Mode (using original decPOMDP.py / RS-MAA*)
# ==========================================

def run_tiger_decentralized_rsmaa(config, verbose=True):
    """
    Run the Tiger problem using the original decPOMDP.py algorithm (RS-MAA*).
    This is the fully decentralized mode with no sync triggers.
    """
    if OriginalDecPOMDP is None:
        print("Original decPOMDP solver not available. Exiting.")
        return 0

    start_time = time.time()

    # 1. Setup Problem
    factory = TigerProblemFactory(config)
    T, O, R, init_b, nacts_fac, nobs_fac = factory.generate()

    # 2. Convert to pdict format required by decPOMDP.py
    # decPOMDP.py expects flat lists: transitions[act*nstates + s] = (indices, probs)
    from array import array

    T_pdict = []
    for act in range(factory.nacts):
        for s in range(factory.nstates):
            indices = []
            values = []
            for snew in range(factory.nstates):
                val = T[act * factory.nsq + s * factory.nstates + snew]
                if val > 0:
                    indices.append(snew)
                    values.append(val)
            T_pdict.append((array('i', indices), array('d', values)))

    # obs[act*nstates + snew] = (obs_indices, probs)
    O_pdict = []
    for act in range(factory.nacts):
        for snew in range(factory.nstates):
            indices = []
            values = []
            for o in range(factory.nobs):
                val = O[act * factory.nso + snew * factory.nobs + o]
                if val > 0:
                    indices.append(o)
                    values.append(val)
            O_pdict.append((array('i', indices), array('d', values)))

    if verbose:
        print(f"Tiger Problem | Mod: {config.mod} | States: {factory.nstates} | Actions: {factory.nacts} | Obs: {factory.nobs}")

    # 3. Create original DecPOMDP solver (replicates exact optimal performance)
    dec_pomdp = OriginalDecPOMDP(
        nagents=factory.nagents,
        nstates=factory.nstates,
        nactions=factory.nacts,
        nobs=factory.nobs,
        transitions=T_pdict,
        obs=O_pdict,
        rewards=list(R),
        init_beliefs=list(init_b),
        nacts_factor=nacts_fac,
        nobs_factor=nobs_fac,
        maxh=config.horizon,
        cluster_type="lossless",
        maxit=config.maxit,
        q_depth=config.ie_min2,
        alpha=config.alpha,
        iter_limit="inf",
        maxrec="inf",
        memory=None,
        heuristic=None,
        rec_type=None,
        p_threshold_cluster=0,
        p_threshold_expand=0,
        policyvalfound=-math.inf,
        output=verbose
    )
    # Set required attributes not handled by constructor
    dec_pomdp.decentralized = False
    dec_pomdp.onesided = False

    # Sample true state
    true_state = random.choice(range(factory.nstates))
    print(f"Tiger Initial State: {true_state} (L=0, R=1)")
    print(f"\n--- Planning Phase (H: {config.horizon}) ---")

    # 4. Run the solver
    t0 = time.time()
    try:
        val, _, _ = dec_pomdp.multi_agent_astar(config.horizon)
        elapsed = time.time() - t0

        print(f"Planning Time: {elapsed:.4f}s | Exp. Value: {val:.4f}")
        print(f"Expected Value (decentralized RS-MAA*): {val:.5f}")
        print(f"Total Time: {time.time() - start_time:.4f}s")

        return val
    except DecPOMDPMemoryLimitExceeded as e:
        elapsed = time.time() - t0
        print(f"Result: MO")
        print(f"Memory limit exceeded: {e}")
        print(f"Planning Time: {elapsed:.4f}s")
        print(f"Total Time: {time.time() - start_time:.4f}s")
        return "MO"


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmarks/sdec_tiger.py <horizon> [maxit] [IEmin2] [alpha]")
        sys.exit(1)

    config = TigerConfig()

    if TRIGGER_MODE == "decentralized_RSMAA":
        print("Running in DECENTRALIZED mode (RS-MAA* via decPOMDP.py)")
        run_tiger_decentralized_rsmaa(config)
    else:
        run_interleaved_execution(config)
