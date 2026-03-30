#!/usr/bin/env python3
"""
MFQ-style Mean-Field DQN training loop for your AEC TrafficEnvironment WITH JOINT action masking.

Your env observations (per-agent) look like:
  obs = {
      "observation": <1D numpy array>,
      "action_mask": <1D int8 array length B^3>   # B = MAXIMUM_ALLOWED_BID (bids 0..B-1)
  }

Key points:
- Action is MultiDiscrete with D=3 independent bids, each in [0..B-1] (you want 0..9 so B=10).
- Mask is JOINT over the full (a0,a1,a2) combination, so we must apply it on argmax selection AND on target max.
- Mean-field vector mf is concatenated histograms of bids across the population for each dimension:
    mf shape = D*B, e.g. 3*10 = 30.

This script includes:
- The masked MFQ agent class
- The mean-field update per full AEC cycle
- A training + testing loop (very close to your DQN loop)
"""

from tqdm import tqdm
import os
import sys
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from routerl_aec_environment_karma import TrafficEnvironment
from routerl_aec_environment_karma import Keychain as kc
from routerl_aec_environment_karma import PerResourceMeanFieldDQN

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ==========================================================
# ---- Mean-field helpers
# ==========================================================
def action_to_meanfield_dist(action_list, num_dims, num_levels):
    """
    action_list: list of np.array shape (num_dims,), entries in [0..num_levels-1]
    returns: np.array shape (num_dims * num_levels,) -> concatenated per-dim histograms
    """
    if len(action_list) == 0:
        return np.zeros(num_dims * num_levels, dtype=np.float32)

    A = np.stack(action_list, axis=0)  # [N, D]
    out = []
    for d in range(num_dims):
        hist = np.bincount(A[:, d], minlength=num_levels).astype(np.float32)
        hist /= max(hist.sum(), 1.0)
        out.append(hist)
    return np.concatenate(out, axis=0)


# ---- Your experiment config (mostly unchanged)
# ==========================================================
new_machines_after_mutation = 300
human_learning_episodes = 0
training_episodes = 200
testing_episodes = 10
total_episodes = human_learning_episodes + training_episodes + testing_episodes

num_agents = 300
origins = ["E0"]
destinations = ["E17.600"]

seed = 9
torch.manual_seed(seed)
np.random.seed(seed)

records_folder = f"training_records_karma_300_agents_{seed}_mean_field_30_days"
plots_folder = f"plots_karma_300_agents_{seed}_mean_field_30_days"

phases = [1, int(total_episodes)]
phase_names = ["Mutation and AV learning", "Testing phase"]

env_params = {
    "agent_parameters": {
        "new_machines_after_mutation": new_machines_after_mutation,
        "num_agents": num_agents,
        "machine_parameters": {
            "behavior": "selfish",
            "observation_type": "previous_agents_avg_tt_per_route",
            "route_0_fee": 0,
            "travel_time_normalization_value": 100,
        },
    },
    "simulator_parameters": {
        "network_name": "network",
        "custom_network_folder": "../../network_analysis/network_base",
        "sumo_type": "sumo",
        "simulation_timesteps": 100,
    },
    "environment_parameters": {
        "observations_time_window": 10,
        "save_every": 5,
        "number_of_days": 30,
        "centrally_defined_price": 8
    },
    "plotter_parameters": {
        "phases": phases,
        "smooth_by": 50,
        "phase_names": phase_names,
        "records_folder": records_folder,
        "plots_folder": plots_folder,
        "plot_choices": "basic",
    },
    "path_generation_parameters": {
        "origins": origins,
        "destinations": destinations,
        "number_of_paths": 3,
        "beta": -1,
        "visualize_paths": True,
        "all_origins_to_all_destinations": False,
    },
}


# ==========================================================
# ---- Environment init
# ==========================================================
env = TrafficEnvironment(seed=seed, create_agents=True, create_paths=True, karma_pricing=True, **env_params)

print("Number of total agents is:", len(env.all_agents))
print("Number of human agents is:", len(env.human_agents))
print("Number of machine agents (autonomous vehicles) is:", len(env.machine_agents), "\n")

env.start()
env.reset()

# ---- Mutation: humans switch to AVs ----
pre_mutation_agents = env.all_agents.copy()
env.mutation(mutation_start_percentile=0)

machines = env.machine_agents.copy()
mutated_humans = {}
for machine in machines:
    for human in pre_mutation_agents:
        if human.id == machine.id:
            mutated_humans[str(machine.id)] = human
            break

mutated_agent_names = list(mutated_humans.keys())
agent_name_to_idx = {name: i for i, name in enumerate(mutated_agent_names)}

# ---- Observation size ----
obs_spaces = env._observation_spaces
some_agent = next(iter(obs_spaces))
obs_space = obs_spaces[some_agent]
STATE_SIZE = int(np.prod(obs_space["observation"].shape))

# ---- Action sizing (IMPORTANT: you want bids 0..9) ----
# MAXIMUM_ALLOWED_BID is used as number of levels B (10), meaning bids in [0..B-1]
B = int(env.agent_params[kc.MACHINE_PARAMETERS][kc.MAXIMUM_ALLOWED_BID])
D = int(env.simulation_params[kc.NUMBER_OF_PATHS])  # should be 3
assert D == 3, f"This script assumes 3 dims, got D={D}"

MF_DIM = D * B
JOINT_N = B ** D

print(f"[INFO] Using B={B} (bids 0..{B-1}), D={D}, joint action count={JOINT_N}, mf_dim={MF_DIM}")

# ---- MFQ agent ----
mfq = PerResourceMeanFieldDQN(
    state_size=STATE_SIZE,
    num_dims=D,
    num_levels=B,
    num_agents=len(mutated_agent_names),
    widths=(64, 64),
    gamma=0.9,
    learning_rate=0.003,
    memory_size=50_000,
    batch_size=256,
    target_update_freq=1000,
    epsilon=0.99,
    epsilon_min=0.05,
    epsilon_decay_rate=0.001,
    per_agent_epsilon=False,
    seed=seed,
).train()

# Track last (s, a, mf, mask) per learning agent
last_obs = {name: None for name in mutated_agent_names}
last_act = {name: None for name in mutated_agent_names}
last_mf = {name: None for name in mutated_agent_names}
last_mask = {name: None for name in mutated_agent_names}

pbar = tqdm(total=total_episodes, desc="AV learning (MFQ masked)")


# ==========================================================
# ---- TRAINING
# ==========================================================
for episode in range(training_episodes):
    env.reset()

    for name in mutated_agent_names:
        last_obs[name] = None
        last_act[name] = None
        last_mf[name] = None
        last_mask[name] = None

    # mean-field starts as zeros each episode (you can also keep it across episodes if desired)
    mf_current = np.zeros(MF_DIM, dtype=np.float32)

    # Build mean-field per AEC cycle
    cycle_actions = []
    cycle_counter = 0
    cycle_len = len(env.agents) if hasattr(env, "agents") and env.agents else num_agents

    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        done = termination or truncation

        # Current obs mask (used as "next mask" when storing transitions)
        obs_mask = None
        if obs is not None and isinstance(obs, dict) and "action_mask" in obs:
            obs_mask = np.asarray(obs["action_mask"], dtype=np.int8)

        if agent in agent_name_to_idx:
            agent_idx = agent_name_to_idx[agent]

            # Store transition for this agent if we have previous (s,a,mf,mask)
            if last_obs[agent] is not None and last_act[agent] is not None and last_mf[agent] is not None:
                mfq.store_transition(
                    last_obs[agent],
                    last_act[agent],
                    last_mf[agent],
                    reward,
                    obs,
                    mf_current,   # "next" mean-field (updates once per cycle; OK approximation)
                    done,
                    last_mask[agent],
                    obs_mask,
                )

            if done:
                action = None
            else:
                # select masked action conditioned on mean-field
                action = mfq.act(agent_idx, obs, mf_current)

                # update last trackers (for transition storage on next visit of this agent)
                last_obs[agent] = obs
                last_act[agent] = action
                last_mf[agent] = mf_current.copy()
                last_mask[agent] = obs_mask

                # contribute to mean-field for this cycle
                cycle_actions.append(np.array(action, dtype=np.int64))
        else:
            action = None

        env.step(action)

        # Update mean-field at end of full AEC cycle
        cycle_counter += 1
        if cycle_counter % cycle_len == 0:
            mf_current = action_to_meanfield_dist(cycle_actions, D, B)
            cycle_actions = []

    # Learn after episode
    mfq.learn(updates=10)
    pbar.update()


# ==========================================================
# ---- TESTING
# ==========================================================
# ==========================================================
# ---- TESTING (no learning, no replay)
# ==========================================================
mfq.eval()
pbar.set_description("Testing (MFQ masked)")

test_returns = []
test_steps = []

for episode in range(testing_episodes):
    env.reset()

    # mean-field resets each episode (same as training)
    mf_current = np.zeros(MF_DIM, dtype=np.float32)

    cycle_actions = []
    cycle_counter = 0
    cycle_len = len(env.agents) if hasattr(env, "agents") and env.agents else num_agents

    ep_return = 0.0
    ep_steps = 0
    ep_av_steps = 0

    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        done = termination or truncation

        # accumulate reward (only meaningful if your env returns per-agent rewards each turn;
        # this is "sum of rewards over all agent turns" for the episode)
        if reward is not None:
            ep_return += float(reward)

        if agent in agent_name_to_idx:
            agent_idx = agent_name_to_idx[agent]

            if done or obs is None:
                action = None
            else:
                # greedy (eval mode => eps ignored)
                action = mfq.act(agent_idx, obs, mf_current)
                ep_av_steps += 1

                # contribute to mean-field for this cycle
                cycle_actions.append(np.array(action, dtype=np.int64))
        else:
            action = None

        env.step(action)

        ep_steps += 1
        cycle_counter += 1

        # Update mean-field at end of full AEC cycle
        if cycle_counter % cycle_len == 0:
            mf_current = action_to_meanfield_dist(cycle_actions, D, B)
            cycle_actions = []

    test_returns.append(ep_return)
    test_steps.append(ep_steps)

    print(
        f"[TEST] episode={episode+1}/{testing_episodes} "
        f"return(sum of turn rewards)={ep_return:.3f} "
        f"turns={ep_steps} av_turns={ep_av_steps}"
    )

pbar.close()

env.plot_results()
env.stop_simulation()
