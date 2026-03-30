from tqdm import tqdm
import torch
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from routerl_aec_environment import TrafficEnvironment
from routerl_aec_environment import Keychain as kc

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

new_machines_after_mutation = 300
human_learning_episodes = 0
training_episodes = 200
testing_episodes = 10

total_episodes = human_learning_episodes + training_episodes + testing_episodes

# Number of agents
num_agents = 300

# Origins and destination points in the network
origins = ["E0"]
destinations = ["E17.600"]

seed = 10
torch.manual_seed(seed)
np.random.seed(seed)

records_folder = f"training_records_user_equilibrium_300_agents_{seed}"
plots_folder = f"plots_user_equilibrium_300_agents_{seed}"

phases = [1, int(total_episodes)]
phase_names = ["Mutation and AV learning", "Testing phase"]

env_params = {
    "agent_parameters": {
        "new_machines_after_mutation": new_machines_after_mutation,
        "num_agents": num_agents,
        "machine_parameters": {
            "behavior": "selfish",
            "observation_type": "previous_agents_avg_tt_per_route",
            "route_0_fee": 10,
            "travel_time_normalization_value": 100
        }
    },
    "simulator_parameters": {
        "network_name": "network",
        "custom_network_folder": "../network_analysis/network_base",
        "sumo_type": "sumo",
        "simulation_timesteps": 100,
    },
    "environment_parameters": {
        "observations_time_window": 10,
        "save_every": 5,
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
env = TrafficEnvironment(seed=seed, create_agents=True, create_paths=True, monetary_pricing=True, **env_params)

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

# AEC environments use agent names/ids; map them to [0..num_mutated-1]
mutated_agent_names = list(mutated_humans.keys())
mutated_set = set(mutated_agent_names)

def choose_min_tt_route(observation, action_size=3, epsilon=0.10):
    """
    observation[:action_size] = travel times per route
    Returns action in {0,1,...,action_size-1}.
    
    With probability epsilon, chooses a different route than the minimum.
    Handles NaN/inf by treating them as very large.
    """
    if observation is None:
        return None

    tt = np.asarray(observation, dtype=np.float32)[:action_size]
    if tt.shape[0] < action_size:
        return 0

    # Replace NaN and ±inf with +inf
    tt = np.nan_to_num(tt, nan=np.inf, posinf=np.inf, neginf=np.inf)

    # If everything is inf, just pick 0
    if np.all(np.isinf(tt)):
        return 0

    best = int(np.argmin(tt))
    print("tt is: ", tt, "\n\n")

    # With probability epsilon, choose a different valid action
    if np.random.rand() < epsilon:
        other_actions = [a for a in range(action_size) if a != best]
        if other_actions:
            print("random choice is: ", other_actions, "\n\n")
            return int(np.random.choice(other_actions))
    print("best is: ", best)
    return best

pbar = tqdm(total=total_episodes, desc="Min-TT policy")
os.makedirs("plots", exist_ok=True)

# -------------------
# "TRAINING" (really just running episodes with the min-TT policy)
# -------------------
for episode in range(training_episodes):
    env.reset()

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        done = termination or truncation

        if agent in mutated_set and not done:
            action = choose_min_tt_route(observation, action_size=3)  # routes 0/1/2
        else:
            action = None  # let non-mutated agents / done agents be handled by env

        env.step(action)

    pbar.update()

# -------------------
# TESTING
# -------------------
pbar.set_description("Testing (Min-TT policy)")
for episode in range(testing_episodes):
    env.reset()

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        done = termination or truncation

        if agent in mutated_set and not done:
            action = choose_min_tt_route(observation, action_size=3)
        else:
            action = None

        env.step(action)

    pbar.update()

pbar.close()

env.plot_results()
env.stop_simulation()
