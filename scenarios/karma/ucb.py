from tqdm import tqdm
import torch
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from routerl_aec_environment_karma import TrafficEnvironment

from routerl_aec_environment_karma import DQN
from routerl_aec_environment_karma import UCB
from routerl_aec_environment_karma import Keychain as kc

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


#########################
## Hyperparameter setting
#########################

new_machines_after_mutation = 300
human_learning_episodes = 0
training_episodes = 300
testing_episodes = 100

total_episodes = human_learning_episodes + training_episodes + testing_episodes


# Number of agents
num_agents = 300

# Origins and destination points in the network
origins = ["E0"]
destinations = ["E17.600"]

seed = 9

records_folder = f"training_records_monetary_pricing_300_agents_new_reward_{seed}_fee_2"
plots_folder = f"plots_monetary_pricing_300_agents_new_reward_{seed}_fee_2"

torch.manual_seed(seed)

# Training phases - needed for the plotting
phases = [1, int(total_episodes)]
phase_names = ["Mutation and AV learning", "Testing phase"]

env_params = {
    "agent_parameters" : {
        "new_machines_after_mutation": new_machines_after_mutation,
        "num_agents": num_agents,
        "machine_parameters" :
        {
            "behavior" : "selfish",
            "observation_type" : "previous_agents_plus_start_time",
            "route_0_fee": 2
        }
    },
    "simulator_parameters" : {
        "network_name" : "network",
        "custom_network_folder" : "../../network_analysis/network_base",
        "sumo_type" : "sumo",
        "simulation_timesteps": 100,
    },  
    "environment_parameters":
    {
        "observations_time_window": 100,
        "save_every" : 1,
    },
    "plotter_parameters" : {
        "phases" : phases,
        "smooth_by" : 50,
        "phase_names" : phase_names,
        "records_folder": records_folder,
        "plots_folder": plots_folder,
        "plot_choices": "basic",
    },
    "path_generation_parameters" : {
        "origins" : origins,
        "destinations" : destinations,
        "number_of_paths" : 3,
        "beta" : -1,
        "visualize_paths" : True,
        "all_origins_to_all_destinations": False
    }
}

## Environment initialization
env = TrafficEnvironment(seed=44, create_agents=True, create_paths=True, karma_pricing = True, **env_params)

print("Number of total agents is: ", len(env.all_agents), "\n")
print("Number of human agents is: ", len(env.human_agents), "\n")
print("Number of machine agents (autonomous vehicles) is: ", len(env.machine_agents), "\n")

## Initialize the connection with SUMO
env.start()
env.reset()


## Mutation -> a number of human agents switch to using AVs 
pre_mutation_agents = env.all_agents.copy()

env.mutation(mutation_start_percentile=0)

print("Number of total agents is: ", len(env.all_agents), "\n")
print("Number of human agents is: ", len(env.human_agents), "\n")
print("Number of machine agents (autonomous vehicles) is: ", len(env.machine_agents), "\n")

## Humans choose the action that will collectively correspond to system optimal (route 0)
for human in env.human_agents:
    human.default_action = 0

machines = env.machine_agents.copy()
mutated_humans = dict()
for machine in machines:
    for human in pre_mutation_agents:
        if human.id == machine.id:
            mutated_humans[str(machine.id)] = human
            break

sorted_mutated_humans = dict(
    sorted(mutated_humans.items(), key=lambda item: item[1].start_time)
)

## Initialize UCB object for each agent
free_flows = env.get_free_flow_times()
i = 1
for h_id, human in sorted_mutated_humans.items():
    num_actions = pow(env.environment_params[kc.MAXIMUM_ALLOWED_BID], env.action_space_size)
    num_states = pow(i, num_actions) 
    alpha = 0.1
    beta = 3
    i = i + 1
    
    mutated_humans[h_id].model = UCB(num_states = num_states, num_actions = num_actions,
                                    num_agents = len(env.all_agents), alpha = alpha, beta=beta)

################
## Training loop
################

pbar = tqdm(total=total_episodes, desc="AV learning")
for episode in range(training_episodes):
    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        
        if termination or truncation:
            obs = [{kc.AGENT_ID : int(agent), kc.TRAVEL_TIME : -reward}]
            last_action = mutated_humans[agent].last_action
            last_observation = mutated_humans[agent].last_obs

            mutated_humans[agent].learn(last_action, obs)
            action = None
        else:
            action = mutated_humans[agent].act(observation)
            mutated_humans[agent].last_action = action

        env.step(action)


    pbar.update()


###############
## Testing loop
###############

pbar.set_description("Testing")
for episode in range(testing_episodes):
    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            action = None
        else:
            action = mutated_humans[agent].act(observation)
        env.step(action)
    pbar.update()

pbar.close()

## Plot results
env.plot_results()

## Close the connection with SUMo
env.stop_simulation()