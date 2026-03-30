from tqdm import tqdm
import torch
import os
import sys
import numpy as np
import itertools
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from routerl_aec_environment import TrafficEnvironment
from routerl_aec_environment import Keychain as kc

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

new_machines_after_mutation = 1
human_learning_episodes = 0
training_episodes = 1
testing_episodes = 10

total_episodes = human_learning_episodes + training_episodes + testing_episodes

# Number of agents
num_agents = 1

# Origins and destination points in the network
origins = ["E0"]
destinations = ["E17.600"]

seed = 13
torch.manual_seed(seed)
np.random.seed(seed)

records_folder = f"training_records_all_possible_paths"
plots_folder = f"plots_all_possible_paths"

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
        "custom_network_folder": "../network_analysis/network_new",
        "sumo_type": "sumo",
        "simulation_timesteps": 100,
    },
    "environment_parameters": {
        "observations_time_window": 20,
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

def joint_action_from_csv(csv_path, tie_breaker="id"):
    """
    Returns a list of actions ordered by departure (start_time ascending).
    If multiple agents have the same start_time, break ties by `tie_breaker`
    (default: id) to make the ordering deterministic.
    """
    df = pd.read_csv(csv_path)

    # Drop fully empty rows (like ",,,,,,,,,")
    df = df.dropna(how="all")

    # Ensure numeric types where needed
    df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")
    df["action"] = pd.to_numeric(df["action"], errors="coerce")

    # Drop rows missing start_time or action
    df = df.dropna(subset=["start_time", "action"])

    # Sort by departure time, then by tie_breaker (optional)
    sort_cols = ["start_time"]
    if tie_breaker in df.columns:
        sort_cols.append(tie_breaker)

    df_sorted = df.sort_values(sort_cols, ascending=True)

    # Joint action as a list (in departure order)
    joint_action = df_sorted["action"].astype(int).tolist()
    return joint_action

#joint_action = joint_action_from_csv("../training_records_monetary_pricing_300_agents_9_fee_15/episodes/ep6165.csv")
"""actions = [0, 1, 2]"""
combinations = [0]#*300

for action in combinations:#itertools.product(actions, repeat=len(env.possible_agents)):
    print(action)

    #for action in joint_action:
    #print("action is: ", action, "\n\n")
    env.step(action)

env.reset()

env.stop_simulation()