from tqdm import tqdm
import torch
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from routerl_aec_environment_karma import TrafficEnvironment
from routerl_aec_environment_karma import Keychain as kc

from routerl_aec_environment_karma import PerResourceMultiDiscreteDQN

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

new_machines_after_mutation = 300
human_learning_episodes = 0
training_episodes = 300
testing_episodes = 10

total_episodes = human_learning_episodes + training_episodes + testing_episodes

# Number of agents
num_agents = 300

# Origins and destination points in the network
origins = ["E0"]
destinations = ["E17.600"]

seed = 9
torch.manual_seed(seed)
np.random.seed(seed)

records_folder = f"training_records_karma_dqn_factorized_300_agents_{seed}"
plots_folder = f"plots_karma_dqn_factorized_300_agents_{seed}"

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
        "save_every": 1,
        "number_of_days": 30
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
env = TrafficEnvironment(seed=seed, create_agents=True, create_paths=True, karma_pricing=True, **env_params)

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

# ---- Build per-resource DQN ----
obs_spaces = env._observation_spaces
some_agent = next(iter(obs_spaces))
obs_space = obs_spaces[some_agent]

STATE_SIZE = int(np.prod(obs_space['observation'].shape))

# nvec: number of discrete bids per resource/path.
# Your original code used:
#   action_space = np.full(number_of_paths, env.environment_params[kc.MAXIMUM_ALLOWED_BID])
# Make sure this is the correct semantics: each dimension i has Ai discrete values [0..Ai-1].
nvec = np.full(
    env.simulation_params[kc.NUMBER_OF_PATHS],
    env.environment_params[kc.MAXIMUM_ALLOWED_BID],
    dtype=np.int64
)

shared_dqn = PerResourceMultiDiscreteDQN(
    state_size=STATE_SIZE,
    nvec=nvec,
    num_agents=len(mutated_humans),
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
    # helpful if env provides *joint* masks with coupled constraints
    topk_repair=10,
).train()

# Map agent names/ids -> [0..num_mutated-1]
mutated_agent_names = list(mutated_humans.keys())
agent_name_to_idx = {name: i for i, name in enumerate(mutated_agent_names)}

# Track last obs/action per mutated agent for transition storage
last_obs = {name: None for name in mutated_agent_names}
last_act = {name: None for name in mutated_agent_names}

pbar = tqdm(total=total_episodes, desc="AV learning")
os.makedirs("plots", exist_ok=True)

# -------------------
# TRAINING
# -------------------
for episode in range(training_episodes):
    env.reset()

    for name in mutated_agent_names:
        last_obs[name] = None
        last_act[name] = None

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        done = termination or truncation

        if agent in agent_name_to_idx:
            agent_idx = agent_name_to_idx[agent]

            # Store transition if we have previous (s,a)
            if last_obs[agent] is not None and last_act[agent] is not None:
                shared_dqn.store_transition(
                    last_obs[agent],
                    last_act[agent],
                    reward,
                    observation,
                    done
                )

            if done:
                action = None
            else:

                action = shared_dqn.act(agent_idx, observation)
                #print("I am agent: ", agent, action, "\n")
                last_obs[agent] = observation
                last_act[agent] = action
        else:
            action = None if done else None

        env.step(action)

    # Learn
    shared_dqn.learn(updates=10)
    pbar.update()

# -------------------
# TESTING
# -------------------
shared_dqn.eval()
pbar.set_description("Testing")

for episode in range(testing_episodes):
    env.reset()

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        done = termination or truncation

        if agent in agent_name_to_idx and not done:
            agent_idx = agent_name_to_idx[agent]
            action = shared_dqn.act(agent_idx, observation)
        else:
            action = None

        env.step(action)

    pbar.update()

pbar.close()

env.plot_results()
env.stop_simulation()
