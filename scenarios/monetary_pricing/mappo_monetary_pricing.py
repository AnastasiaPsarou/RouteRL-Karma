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
from routerl_aec_environment import TrafficEnvironment

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


"""TorchRL training parameters"""

# Devices
device = (
    torch.device(0)
    if torch.cuda.is_available()
    else torch.device("cpu")
)

# Sampling
frames_per_batch = 600  # Number of team frames collected per training iteration
n_iters = 50  # Number of sampling and training iterations - the episodes the plotter plots
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
num_agents = 300

# Human learning phase
human_learning_episodes = 0
new_machines_after_mutation = 300 # All the agents are AVs, no humans available in the system.

# number of episodes the AV training will take
training_episodes = int((frames_per_batch / new_machines_after_mutation) * n_iters)

# Origins and destination points in the network
origins = ["E0"]
destinations = ["E17.600"]


records_folder = "training_records_monetary_pricing_300_agents_fee_0_1"
plots_folder = "plots_monetary_pricing_300_agents_fee_0_1"
torch.manual_seed(10)

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
            "route_0_fee": 0.1
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
env = TrafficEnvironment(seed=42, create_agents=True, create_paths=True, monetary_pricing=True, **env_params)

# Initialize the connection with SUMo
env.start()
env.reset()

# Initialize AV agents (mutate all the agents of the system into AVs)
env.mutation(mutation_start_percentile=0)

print("Number of total agents is: ", len(env.all_agents), "\n")
print("Number of human agents is: ", len(env.human_agents), "\n")
print("Number of machine agents (autonomous vehicles) is: ", len(env.machine_agents), "\n")


# TorchRL PettingZoo environment wrapper
group = {'agents': [str(machine.id) for machine in env.machine_agents]}

env = PettingZooWrapper(
    env=env,
    use_mask=True, # Whether to use the mask in the outputs. It is important for AEC environments to mask out non-acting agents.
    categorical_actions=True,
    done_on_any = False, # Whether the environment’s done keys are set by aggregating the agent keys using any() (when True) or all() (when False).
    group_map=group,
    device=device
)

# TorchRL environment transform to show the episode reward
env = TransformedEnv(
    env,
    RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
)

# Runs some rollouts in the wrapped pettingzoo env to ensure that the spaces are okay.
check_env_specs(env)


share_parameters_policy = False 

#print("value is: ", 2 * np.prod(env.action_spec["agents", "action"].shape))

# Create the MLP for multiple agents
policy_net = nn.Sequential(
    MultiAgentMLP(
        n_agent_inputs = env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs = env.action_spec.space.n,
        n_agents = env.n_agents,
        centralised=False,
        share_params=share_parameters_policy,
        device=device,
        depth=policy_network_depth,
        num_cells=policy_network_num_cells,
        activation_class=nn.Tanh,
    ),
    
)


policy_module = TensorDictModule(
    policy_net,
    in_keys=[("agents", "observation")],
    out_keys=[("agents", "logits")],
) 

policy = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=[("agents", "logits")],
    out_keys=[env.action_key],
    distribution_class=Categorical,
    return_log_prob=True,
    log_prob_key=("agents", "sample_log_prob"),
)



"""Critic network"""

share_parameters_critic = False
mappo = True  # IPPO if False

critic_net = MultiAgentMLP(
    n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
    n_agent_outputs=1, 
    n_agents=env.n_agents,
    centralised=mappo,
    share_params=share_parameters_critic,
    device=device,
    depth=critic_network_depth,
    num_cells=critic_network_num_cells,
    activation_class=torch.nn.ReLU,
)

critic = TensorDictModule(
    module=critic_net,
    in_keys=[("agents", "observation")],
    out_keys=[("agents", "state_value")],
)

"""Collector class"""

collector = SyncDataCollector(
    env,
    policy,
    device=device,
    storing_device=device,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
) 

"""Replay buffer"""

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(
        frames_per_batch, device=device
    ),  
    sampler=SamplerWithoutReplacement(),
    batch_size=minibatch_size,
)

"""Loss module"""

loss_module = ClipPPOLoss(
    actor_network=policy,
    critic_network=critic,
    clip_epsilon=clip_epsilon,
    entropy_coef=entropy_eps,
    normalize_advantage=False,
)


loss_module.set_keys( 
    reward=env.reward_key,  
    action=env.action_key, 
    sample_log_prob=("agents", "sample_log_prob"),
    value=("agents", "state_value"),
    done=("agents", "done"),
    terminated=("agents", "terminated"),
)

loss_module.make_value_estimator(
    ValueEstimators.GAE, gamma=gamma, lmbda=lmbda
) 

GAE = loss_module.value_estimator
optim = torch.optim.Adam(loss_module.parameters(), lr)


pbar = tqdm(total=n_iters, desc="episode_reward_mean = 0")


"""Training loop"""
for tensordict_data in collector: ##loops over frame_per_batch

    ## Generate the rollouts
    tensordict_data.set(
        ("next", "agents", "done"),
        tensordict_data.get(("next", "done"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", env.reward_key))),  # Adjust index to start from 0
    )
    tensordict_data.set(
        ("next", "agents", "terminated"),
        tensordict_data.get(("next", "terminated"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", env.reward_key))),  # Adjust index to start from 0
    )

    # Compute GAE for all agents
    with torch.no_grad():
            GAE(
                tensordict_data,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )

    data_view = tensordict_data.reshape(-1)  
    replay_buffer.extend(data_view)

    ## Update the policies of the learning agents
    for _ in range(num_epochs):
        for _ in range(frames_per_batch // minibatch_size):
            subdata = replay_buffer.sample()
            loss_vals = loss_module(subdata)

            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            loss_value.backward()

            torch.nn.utils.clip_grad_norm_(
                loss_module.parameters(), max_grad_norm
            ) 

            optim.step()
            optim.zero_grad()
   
    collector.update_policy_weights_()
   
    # Logging
    done = tensordict_data.get(("next", "agents", "done"))  # Get done status for the group

    episode_reward_mean = (
        tensordict_data.get(("next", "agents", "episode_reward"))[done].mean().item()
    )

    pbar.set_description(f"episode_reward_mean = {episode_reward_mean}", refresh=False)
    pbar.update()

"""Testing phase"""

policy.eval() # set the policy into evaluation mode

num_episodes = 10

for episode in range(num_episodes):
    env.rollout(len(env.machine_agents), policy=policy)


# Plotting
env.plot_results()

# Stop the connection with SUMO
env.stop_simulation()


