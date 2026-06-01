import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from tqdm import tqdm
import torch
import os
import sys
import numpy as np
import itertools

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from routerl_aec_environment import TrafficEnvironment
from routerl_aec_environment import Keychain as kc
from collections import defaultdict
import itertools
import numpy as np
import pandas as pd


def get_agent_object(agent_ref, all_agents):
    """
    env.possible_agents may contain agent objects or agent IDs/names.
    This function finds the actual agent object so we can read origin/destination.
    """
    if hasattr(agent_ref, "origin") and hasattr(agent_ref, "destination"):
        return agent_ref

    for agent in all_agents:
        possible_ids = [
            getattr(agent, "id", None),
            getattr(agent, "agent_id", None),
            getattr(agent, "name", None),
            str(getattr(agent, "id", None)),
            str(getattr(agent, "agent_id", None)),
            str(getattr(agent, "name", None)),
        ]

        if agent_ref in possible_ids or str(agent_ref) in possible_ids:
            return agent

    raise ValueError(f"Could not find agent object for possible_agent={agent_ref}")


def two_route_splits(n_agents):
    """
    All possible ways to split n_agents between 2 routes.

    Example for 5 agents:
        (0, 5), (1, 4), (2, 3), ..., (5, 0)
    """
    for n_route_0 in range(n_agents + 1):
        n_route_1 = n_agents - n_route_0
        yield (n_route_0, n_route_1)


def make_actions_for_od_agents(od_agents, route_counts, shuffle=True):
    """
    Converts route counts into individual actions.

    Example:
        route_counts = (3, 2)
        actions = [0, 0, 0, 1, 1]
    """
    actions = []

    for route_id, count in enumerate(route_counts):
        actions.extend([route_id] * count)

    if shuffle:
        np.random.shuffle(actions)

    return dict(zip(od_agents, actions))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

new_machines_after_mutation = 300
human_learning_episodes = 0
training_episodes = 300
testing_episodes = 10

total_episodes = human_learning_episodes + training_episodes + testing_episodes

# Number of agents
num_agents = 300

# Origins and destination points in the network
origins = ["E0", "E25"]
destinations = ["E17.600", "E26"]

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
        "custom_network_folder": "../network_analysis/two_ods_network",
        "sumo_type": "sumo-gui",
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
        "number_of_paths": 2,
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

# Make sure the environment is reset before reading possible_agents
env.reset()

# Group agents by their actual OD pair
agents_by_od = defaultdict(list)

for possible_agent in env.possible_agents:
    agent = get_agent_object(possible_agent, env.all_agents)
    od = (agent.origin, agent.destination)
    agents_by_od[od].append(possible_agent)

print("\nAgents by OD:")
for od, agents in agents_by_od.items():
    print(f"OD {od}: {len(agents)} agents")

# Optional safety check: you expect exactly 2 ODs
if len(agents_by_od) != 2:
    raise ValueError(
        f"Expected exactly 2 OD pairs, but found {len(agents_by_od)}: "
        f"{list(agents_by_od.keys())}"
    )

# Create all 2-route split possibilities for each OD
od_list = list(agents_by_od.keys())

od_to_splits = {
    od: list(two_route_splits(len(agents_by_od[od])))
    for od in od_list
}

total_simulations = 1
for od in od_list:
    total_simulations *= len(od_to_splits[od])

print(f"\nTotal simulations to run: {total_simulations}\n")

results = []

# Cartesian product of OD-specific route splits
#
# Example:
#   OD 0 split = (120, 30)
#   OD 1 split = (80, 70)
#
# means:
#   For OD 0: 120 agents choose route 0, 30 choose route 1
#   For OD 1: 80 agents choose route 0, 70 choose route 1
for sim_id, split_combo in enumerate(
    itertools.product(*[od_to_splits[od] for od in od_list])
):
    print(f"Simulation {sim_id + 1}/{total_simulations}")

    for od, split in zip(od_list, split_combo):
        print(f"  OD {od}: route 0 = {split[0]}, route 1 = {split[1]}")

    env.reset()

    action_by_agent = {}

    for od, route_counts in zip(od_list, split_combo):
        od_agents = agents_by_od[od]

        od_action_dict = make_actions_for_od_agents(
            od_agents,
            route_counts,
            shuffle=True
        )

        action_by_agent.update(od_action_dict)

    # Step through agents in the order expected by the environment
    for possible_agent in env.possible_agents:
        action = action_by_agent[possible_agent]
        env.step(action)

    # Save the tested split
    result_row = {
        "simulation_id": sim_id,
    }

    for od_idx, (od, split) in enumerate(zip(od_list, split_combo)):
        result_row[f"od_{od_idx}_origin"] = od[0]
        result_row[f"od_{od_idx}_destination"] = od[1]
        result_row[f"od_{od_idx}_route_0_count"] = split[0]
        result_row[f"od_{od_idx}_route_1_count"] = split[1]

    # TODO: Add your SO objective here if available.
    # For example:
    # result_row["total_travel_time"] = env.total_travel_time
    # result_row["average_travel_time"] = env.average_travel_time

    results.append(result_row)

# Save all tested combinations
results_df = pd.DataFrame(results)
results_df.to_csv("so_exhaustive_2ods_2routes_results.csv", index=False)

print("\nFinished exhaustive OD-aware search.")
print("Saved results to so_exhaustive_2ods_2routes_results.csv")


env.stop()