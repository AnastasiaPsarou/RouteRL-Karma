from tqdm import tqdm
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from routerl_aec_environment import TrafficEnvironment
from routerl_aec_environment import DQN, ReplayBuffer
from routerl_aec_environment import UCB
from routerl_aec_environment import Keychain as kc

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

new_machines_after_mutation = 10
human_learning_episodes = 0
training_episodes = 100
testing_episodes = 20

total_episodes = human_learning_episodes + training_episodes + testing_episodes

# Number of agents
num_agents = 10

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
            "route_0_fee": 2,
            "travel_time_normalization_value": 10
        }
    },
    "simulator_parameters" : {
        "network_name" : "two_route_yield",
        #"custom_network_folder" : "../../network_analysis/network_base",
        "sumo_type" : "sumo",
        "simulation_timesteps": 100,
    },  
    "environment_parameters":
    {
        "observations_time_window": 100,
        "save_every" : 1,
        "number_of_days": 10
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
        #"origins" : origins,
        #"destinations" : destinations,
        "number_of_paths" : 2,
        "beta" : -1,
        "visualize_paths" : True,
        "all_origins_to_all_destinations": False
    }
}

## Environment initialization
env = TrafficEnvironment(seed=44, create_agents=True, create_paths=True, monetary_pricing=False, **env_params)

print("Number of total agents is: ", len(env.all_agents), "\n")
print("Number of human agents is: ", len(env.human_agents), "\n")
print("Number of machine agents (autonomous vehicles) is: ", len(env.machine_agents), "\n")

env.start()
env.reset()

## Mutation -> a number of human agents switch to using AVs 
pre_mutation_agents = env.all_agents.copy()

env.mutation(mutation_start_percentile=0)


machines = env.machine_agents.copy()
mutated_humans = dict()
for machine in machines:
    for human in pre_mutation_agents:
        if human.id == machine.id:
            mutated_humans[str(machine.id)] = human
            break


free_flows = env.get_free_flow_times()
for h_id, human in mutated_humans.items():
    
    mutated_humans[h_id].model = DQN(3, 2, #epsilon=1.0,
                                            epsilon_decay_rate=0.01,   # ~200k steps to 0.2
                                            #epsilon_min=0.3,             # keep more exploration
                                            memory_size=200,           # smaller, tracks recent regime
                                            batch_size=32,
                                            learning_rate=0.1,)
    
    mutated_humans[h_id].memory = ReplayBuffer(capacity = 500)
    

pbar = tqdm(total=total_episodes, desc="AV learning")
os.makedirs("plots", exist_ok=True)
for episode in range(training_episodes):
    print("episode is: ", episode, "\n\n")

    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        
        if termination or truncation:
            mutated_humans[agent].memory.push(mutated_humans[agent].last_obs,
                                                mutated_humans[agent].last_action, 
                                                reward,
                                                observation, 
                                                termination or truncation)
            action = None
        else:
            action = mutated_humans[agent].model.act(observation)

            #print("termination or truncation is: ", termination or truncation, "\n\n")

            if reward != 0:
                #print("I am agent: ", agent, "I had applied action:", mutated_humans[agent].last_action, "my next observation is: ", observation, "my current obs: ", mutated_humans[agent].last_obs, reward)
            
                mutated_humans[agent].memory.push(mutated_humans[agent].last_obs,
                                                mutated_humans[agent].last_action, 
                                                reward,
                                                observation, 
                                                termination or truncation)

            mutated_humans[agent].last_action = action
            mutated_humans[agent].last_obs = observation

        env.step(action)
    
    print("episode is: ", episode, "\n\n")
    #if (episode + 1) % 4 == 0:
    for h_id, _ in mutated_humans.items():
        batch_size = 4
        rollouts = mutated_humans[h_id].memory.sample(batch_size)

        print("rollouts are: ", rollouts, "\n\n")

        mutated_humans[h_id].model.learn(rollouts, batch_size)

    """if episode % 50 == 0:
        env.plot_results()"""
    pbar.update()


for h_id, human in mutated_humans.items():
    mutated_humans[h_id].model.eval()

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
#os.makedirs(plots_folder, exist_ok=True)


env.plot_results()

env.stop_simulation()
