from tqdm import tqdm
import torch
import os
import sys
import numpy as np
import itertools

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from routerl_aec_environment import TrafficEnvironment
from routerl_aec_environment import Keychain as kc

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

records_folder = f"training_records_all_possible_paths_approximated"
plots_folder = f"plots_all_possible_paths_approximated"

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
        "custom_network_folder": "../network_analysis/network_base",
        "sumo_type": "sumo",
        "simulation_timesteps": 100,
    },
    "environment_parameters": {
        "observations_time_window": 10,
        "save_every": 1,
        "number_of_days": 1
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


##################### Human Learning #####################
num_episodes = 0

for episode in range(num_episodes):
    env.step()

##################### Mutation #####################
env.mutation(mutation_start_percentile=0)

print("env.human_agents", env.human_agents)
print("env.machine_agents", env.machine_agents)
print("\n\n\n")

env.reset()

"""actions = [0, 1, 2]

for combination in itertools.product(actions, repeat=len(env.possible_agents)):
    print(combination)

    for action in combination:
        env.step(action)

    env.reset()"""

N = len(env.possible_agents)

for n1 in range(N + 1):
    for n2 in range(N - n1 + 1):
        n3 = N - n1 - n2

        print("n1 is: ", n1, "n2 is: ", n2, "n3 is: ", n3, "\n\n")

        actions = ([0] * n1) + ([1] * n2) + ([2] * n3)
        np.random.shuffle(actions)

        env.reset()
        for action in actions:
            env.step(action)


env.stop()