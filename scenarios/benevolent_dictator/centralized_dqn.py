from tqdm import tqdm
import torch
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from routerl_aec_environment import TrafficEnvironment
from routerl_aec_environment import Keychain as kc

from routerl_aec_environment import CentralizedDQN

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

new_machines_after_mutation = 300
human_learning_episodes = 0
training_episodes = 500
testing_episodes = 10

total_episodes = human_learning_episodes + training_episodes + testing_episodes

num_agents = 300
origins = ["E0"]
destinations = ["E17.600"]

seed = 15
torch.manual_seed(seed)
np.random.seed(seed)

records_folder = f"training_records_benevolent_dictator_300_agents_{seed}"
plots_folder = f"plots_benevolent_dictator_300_agents_{seed}"

phases = [1, int(total_episodes)]
phase_names = ["Mutation and AV learning", "Testing phase"]

env_params = {
    "agent_parameters": {
        "new_machines_after_mutation": new_machines_after_mutation,
        "num_agents": num_agents,
        "machine_parameters": {
            "behavior": "altruistic",
            "observation_type": "previous_agents_avg_tt_per_route",
            "route_0_fee": 0,
            "travel_time_normalization_value": 100
        }
    },
    "simulator_parameters": {
        "network_name": "network",
        "custom_network_folder": "../../network_analysis/network_base",
        "sumo_type": "sumo",
        "simulation_timesteps": 100,
    },
    "environment_parameters": {
        "observations_time_window": 20,
        "save_every": 2,
        "number_of_days": 4
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
        "all_origins_to_all_destinations": False
    }
}

# ---- Environment initialization ----
env = TrafficEnvironment(seed=seed, create_agents=True, create_paths=True, **env_params)

print("Number of total agents is: ", len(env.all_agents), "\n")
print("Number of human agents is: ", len(env.human_agents), "\n")
print("Number of machine agents (autonomous vehicles) is: ", len(env.machine_agents), "\n")

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

# ---- Infer state/action sizes ----
obs_spaces = env._observation_spaces
some_agent = next(iter(obs_spaces))
obs_space = obs_spaces[some_agent]

STATE_SIZE = int(np.prod(obs_space.shape))
ACTION_SIZE = 3  # your original discrete action count

# ---- Centralized DQN ----
mutated_agent_names = list(mutated_humans.keys())
agent_name_to_idx = {name: i for i, name in enumerate(mutated_agent_names)}
num_mutated = len(mutated_agent_names)

central_dqn = CentralizedDQN(
    state_size=STATE_SIZE,
    action_space_size=ACTION_SIZE,
    num_agents=num_mutated,
    epsilon=0.99,
    epsilon_decay_rate=0.001,
    epsilon_min=0.05,
    memory_size=50_000,
    batch_size=256,
    gamma=0.9,
    learning_rate=0.003,
    widths=(32, 64, 32),
    target_update_freq=1000,
    per_agent_epsilon=False,
).train()

# ---- Buffers for centralized state ----
# rolling matrix of mutated agents' most recent observations: (N_mut, STATE_SIZE)
states_all = np.zeros((num_mutated, STATE_SIZE), dtype=np.float32)

# per-agent last snapshot used for transition storage
last_states_all = {name: None for name in mutated_agent_names}  # stores a COPY of states_all
last_act = {name: None for name in mutated_agent_names}

pbar = tqdm(total=total_episodes, desc="AV learning")
os.makedirs("plots", exist_ok=True)

# -------------------
# TRAINING
# -------------------
for episode in range(training_episodes):
    env.reset()

    # reset centralized rolling state and per-agent trackers
    states_all[:] = 0.0
    for name in mutated_agent_names:
        last_states_all[name] = None
        last_act[name] = None

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        done = termination or truncation

        if agent in agent_name_to_idx:
            agent_idx = agent_name_to_idx[agent]

            # update rolling global state for this agent (even if done, observation is useful for s')
            obs_vec = np.asarray(observation, dtype=np.float32).reshape(-1)
            states_all[agent_idx] = obs_vec

            # store transition for this agent if we have previous (S_global, a)
            if last_states_all[agent] is not None and last_act[agent] is not None:
                central_dqn.store_transition(
                    states_all=last_states_all[agent],         # previous global state snapshot
                    agent_id=agent_idx,
                    action=last_act[agent],
                    reward=reward,
                    next_states_all=states_all.copy(),         # current global state snapshot
                    done=done
                )

            if done:
                action = None
            else:
                # select action using centralized global state
                action = central_dqn.act(agent_idx, states_all)
                last_states_all[agent] = states_all.copy()
                last_act[agent] = action
        else:
            action = None if done else None

        env.step(action)

    central_dqn.learn(updates=10)
    pbar.update()

# -------------------
# TESTING
# -------------------
central_dqn.eval()
pbar.set_description("Testing")

for episode in range(testing_episodes):
    env.reset()

    states_all[:] = 0.0

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        done = termination or truncation

        if agent in agent_name_to_idx and not done:
            agent_idx = agent_name_to_idx[agent]

            # update rolling global state
            obs_vec = np.asarray(observation, dtype=np.float32).reshape(-1)
            states_all[agent_idx] = obs_vec

            # greedy centralized action
            action = central_dqn.act(agent_idx, states_all)
        else:
            action = None

        env.step(action)

    pbar.update()

pbar.close()

env.plot_results()
env.stop_simulation()
