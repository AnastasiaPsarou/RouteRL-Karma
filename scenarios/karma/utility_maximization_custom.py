import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from tqdm import tqdm
import torch
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from routerl_aec_environment_karma import TrafficEnvironment
from routerl_aec_environment_karma import Keychain as kc
from routerl_aec_environment_karma import GeneralBiddingModel


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

seed = 9
torch.manual_seed(seed)
np.random.seed(seed)

records_folder = f"training_records_route_choice_300_agents_{seed}_30_days"
plots_folder = f"plots_route_choice_300_agents_{seed}_30_days"

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
            "reward_w3": 0,
        }
    },
    "simulator_parameters": {
        "network_name" : "network",
        "custom_network_folder" : "../../network_analysis/network_base",
        "sumo_type": "sumo",
        "simulation_timesteps": 100,
    },
    "environment_parameters": {
        "observations_time_window": 20,
        "save_every": 1,
        "number_of_days": 30,
        "centrally_defined_price": 4
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
        "origins" : origins,
        "destinations" : destinations,
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

# --- Create ONE shared DQN for all mutated humans ---
# Infer state_size and action_space_size from environment
# You used DQN(3,2) previously, so keep state_size=3 and actions=2 unless your env changed.
obs_spaces = env._observation_spaces

# pick any agent (they should all match for parameter sharing)
some_agent = next(iter(obs_spaces))
obs_space = obs_spaces[some_agent]


for machine in env.machine_agents:
    initial_knowledge = [0, 0, 0]
    machine.utility = GeneralBiddingModel(env.params['agent_parameters']['human_parameters'], initial_knowledge)


# TRAINING
# -------------------
for episode in range(training_episodes):
    env.reset()

    acted_this_day = set()


    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        done = termination or truncation

        machine = next((m for m in env.machine_agents if str(m.id) == agent or getattr(m, "name", None) == agent), None)

        if done: 
            #print("inside done\n\n")
            action = None
        else:
            action = machine.utility.act(observation)
            machine.action = action
            machine.observation = observation
        
        env.step(action)

        acted_this_day.add(agent)

        # if all currently active agents have acted once, learn
        if acted_this_day == set(env.agents):
            #print("acted this day is: ", acted_this_day, "\n\n")
            for machine in env.machine_agents:
                if hasattr(machine, "observation") and hasattr(machine, "action"):
                    #print("learning is happening", machine.urgency, "\n\n")
                    #print("machine.reward is: ", machine.reward, "\n\n")
                    machine.utility.learn(machine.observation, machine.action, machine.reward * machine.urgency, chosen_route=machine.route)

            acted_this_day.clear()


env.plot_results()
env.stop_simulation()
