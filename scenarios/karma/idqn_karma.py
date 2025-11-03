import torch
from tqdm import tqdm
import os
import sys
import ast
import pandas as pd
import numpy as np

from tensordict.nn import TensorDictModule
from torchrl.collectors import SyncDataCollector
from tensordict.nn.distributions import CompositeDistribution
from torch.distributions import Categorical
import torch.distributions as d
from tensordict import TensorDict
from torch import nn, distributions as d
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn import TensorDictModule, TensorDictSequential, CompositeDistribution
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs.transforms import TransformedEnv, RewardSum
from torchrl.envs.utils import check_env_specs
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.modules import MultiAgentMLP, ProbabilisticActor
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from pettingzoo.test import parallel_api_test
from tensordict.nn import set_composite_lp_aggregate
set_composite_lp_aggregate(False).set() 


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils import ensure_composite_action_on, ensure_tmp_and_logprob, rebuild_prev_logprob_root

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from routerl_karma import TrafficEnvironment
from routerl_karma import DQN

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


"""TorchRL training parameters"""

# Devices
device = (
    torch.device(0)
    if torch.cuda.is_available()
    else torch.device("cpu")
)

# Sampling
frames_per_batch = 5  # Number of team frames collected per training iteration
n_iters = 100  # Number of sampling and training iterations - the episodes the plotter plots
total_frames = frames_per_batch * n_iters

# Training
num_epochs = 10  # Number of optimization steps per training iteration
minibatch_size = 2  # Size of the mini-batches in each optimization step
lr = 0.0001 # Learning rate
max_grad_norm = 1.0  # Maximum norm for the gradients

# PPO
clip_epsilon = 0.2  # clip value for PPO loss
gamma = 0.85  # discount factor
lmbda = 0.9  # lambda for generalised advantage estimation
entropy_eps = 1e-3  # coefficient of the entropy term in the PPO loss

policy_network_depth=3
policy_network_num_cells = 256

critic_network_depth=4
critic_network_num_cells = 64

""" Environment hyperparameters."""

# Number of agents
num_agents = 10

# Human learning phase
human_learning_episodes = 0
new_machines_after_mutation = 10 # All the agents are AVs, no humans available in the system.

# number of episodes the AV training will take
training_episodes = int((frames_per_batch / new_machines_after_mutation) * n_iters)

# Origins and destination points in the network
origins = ["E0"]
destinations = ["E17.600"]


records_folder = "training_records_karma_10_agents"
plots_folder = "plots_karma_10_agents"

# Training phases - needed for the plotting
phases = [1, int(training_episodes)]
phase_names = ["Mutation and AV learning", "Testing phase"]

env_params = {
    "agent_parameters" : {
        "new_machines_after_mutation": new_machines_after_mutation,
        "num_agents": num_agents,
        "machine_parameters" :
        {
            "behavior" : "selfish",
            "observation_type" : "previous_agents_plus_start_time",
        }
    },
    "simulator_parameters" : {
        "network_name" : "network",
        "custom_network_folder" : "../../network_analysis/network_base",
        "sumo_type" : "sumo",
        "simulation_timesteps": 100,
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

# Environment initialization
env = TrafficEnvironment(seed=42, create_agents=True, create_paths=True, karma_pricing = True, **env_params)

# Initialize the connection with SUMo
env.start()
env.reset()

pre_mutation_agents = env.all_agents.copy()


# Initialize AV agents (mutate all the agents of the system into AVs)
env.mutation(mutation_start_percentile=0)

print("Number of total agents is: ", len(env.all_agents), "\n")
print("Number of human agents is: ", len(env.human_agents), "\n")
print("Number of machine agents (autonomous vehicles) is: ", len(env.machine_agents), "\n")


machines = env.machine_agents.copy()
mutated_humans = dict()
for machine in machines:
    for human in pre_mutation_agents:
        if human.id == machine.id:
            mutated_humans[str(machine.id)] = human
            break


free_flows = env.get_free_flow_times()
for h_id, human in mutated_humans.items():
    initial_knowledge = free_flows[(human.origin, human.destination)]
    initial_knowledge = [0, 0]
    mutated_humans[h_id].model = DQN(3, len(initial_knowledge))

print("mutated humans are: ", mutated_humans, "\n", mutated_humans.items())
observations, infos = env.reset()
actions = {}

print("observations are: ", observations, "\n\n")
for episode in range(training_episodes):
    
    for agent in env.all_agents:
        actions[agent.id] = agent.act(observations[str(agent.id)])
        print("actions are: ", actions, "\n\n")

    env.step(actions)

for episode in range(testing_episodes):
    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            action = None
        else:
            action = mutated_humans[agent].act(observation)
        env.step(action)

#os.makedirs(plots_folder, exist_ok=True)


env.plot_results()

env.stop_simulation()