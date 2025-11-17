from tqdm import tqdm
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../')))

#from extended_mutation import MyExtendedMutation
from routerl_aec_environment import TrafficEnvironment
from routerl_aec_environment import MAPPO
from routerl_aec_environment import Keychain as kc
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#### Hyperparameter setting

new_machines_after_mutation = 800
num_agents = 800

human_learning_episodes = 200
training_episodes = 1000
testing_episodes = 100

origins = ["E0"]
destinations = ["E17.600"]

total_episodes = human_learning_episodes + training_episodes

# Training phases - needed for the plotting
phases = [1, int(training_episodes)]
phase_names = ["Mutation and AV learning", "Testing phase"]

records_folder = "training_records_monetary_pricing_300_agents_fee_0_1"
plots_folder = "plots_monetary_pricing_300_agents_fee_0_1"
torch.manual_seed(10)

env_params = {
    "agent_parameters" : {
        "new_machines_after_mutation": new_machines_after_mutation,
        "num_agents": num_agents,
        "machine_parameters" :
        {
            "behavior" : "selfish",
            "observation_type" : "previous_agents_plus_start_time",
            "route_0_fee": 0.1
        }
    },
    "simulator_parameters" : {
        "network_name" : "network",
        "custom_network_folder" : "../../network_analysis/network_base",
        "sumo_type" : "sumo-gui",
        "simulation_timesteps": 100,
    },  
    "environment_parameters":
    {
        "observations_time_window": 100
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

#### Environment initialization

env = TrafficEnvironment(seed=41, create_agents=True, create_paths=True, **env_params)


print("Number of total agents is: ", len(env.all_agents), "\n")
print("Number of human agents is: ", len(env.human_agents), "\n")
print("Number of machine agents (autonomous vehicles) is: ", len(env.machine_agents), "\n")


env.start()
env.reset()

pbar = tqdm(total=total_episodes, desc="Human learning")
    
pre_mutation_agents = env.all_agents.copy()

#### Mutation

env.mutation(mutation_start_percentile=0)


num_agents=len(env.machine_agents)

mappo = MAPPO(
    state_size=5,
    action_space_size=3,
    num_agents=num_agents,
    shared_policy=False,
    share_critic=True,
    policy_arch_kwargs={'num_hidden':2, 'widths':[32,64,32]},
    critic_arch_kwargs={'num_hidden':2, 'widths':[64,64,64]},
    lr_actor=3e-4,
    lr_critic=3e-4
)


machines = env.machine_agents.copy()
mutated_humans = dict()
for machine in machines:
    for human in pre_mutation_agents:
        if human.id == machine.id:
            mutated_humans[str(machine.id)] = human
            break

print(mutated_humans)

raw_ids = sorted(int(k) for k in mutated_humans.keys())   # [3,5,7,9,11,13,15,17,19,21]
id_to_idx = { raw_id: idx for idx, raw_id in enumerate(raw_ids) }


#### AV training loop

os.makedirs("plots", exist_ok=True)
pbar.set_description("AV (MAPPO) learning")
for episode in range(training_episodes):
    print("episode is: ", episode, "\n\n")
    env.reset()
    # local list
    states, actions, rewards, logps, next_states, dones, agent_idxs = [], [], [], [], [], [], []

    for agent in env.agent_iter():
        raw_id = int(agent)
        idx = id_to_idx[raw_id]

        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            #obs = [{kc.AGENT_ID : int(agent), kc.TRAVEL_TIME : -reward}]
            
            last_obs = mappo.get_last_observation(idx)
            last_action = mappo.get_last_action(idx)
            last_logp = mappo.get_last_log_prob(idx)

            states.append(last_obs)
            actions.append(last_action)
            rewards.append(reward)
            logps.append(last_logp)
            next_states.append([0,0,0,0,0])
            agent_idxs.append(idx)
            dones.append(1)
            action = None
        else:
            action = mappo.act(observation, idx)
        #print(f"Agent {agent} action: {action}")
        env.step(action)

    # update MAPPO
    if (episode + 1) % 2 == 0:
        mappo.learn(states, actions, rewards, logps, next_states, dones, agent_idxs)

    if mappo.loss_actor and mappo.loss_critic:
        pbar.set_description(f"Ep {episode} actor={mappo.loss_actor[-1]:.4f} critic={mappo.loss_critic[-1]:.4f}")
    else:
        pbar.set_description(f"Ep {episode} actor=N/A critic=N/A")
    pbar.update()


free_flows = env.get_free_flow_times()
for h_id, human in mutated_humans.items():
    initial_knowledge = free_flows[(human.origin, human.destination)]
    initial_knowledge = [0, 0]
    mutated_humans[h_id].model = mappo.get_policy(id_to_idx[int(h_id)])


mappo.eval()
#### AV testing loop

pbar.set_description("Testing")
for episode in range(testing_episodes):
    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            action = None
        else:
            action = mappo.act(observation, id_to_idx[int(agent)])
        env.step(action)
    pbar.update()

pbar.close()


#### Plots

env.plot_results()

#### Close SUMO connection

env.stop_simulation()