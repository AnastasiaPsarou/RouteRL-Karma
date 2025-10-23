import torch
from tqdm import tqdm
import os
import sys
import ast
import pandas as pd

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
n_iters = 200  # Number of sampling and training iterations - the episodes the plotter plots
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


records_folder = "training_records_google_maps_10_agents"
plots_folder = "plots_google_maps_10_agents"

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
env = TrafficEnvironment(seed=42, create_agents=True, create_paths=True, **env_params)

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

class Splitter(nn.Module):
    def __init__(self, n1=3, n2=100):
        super().__init__()
        self.n1, self.n2 = n1, n2
    def forward(self, x):
        logits1 = x[..., :self.n1]
        logits2 = x[..., self.n1:self.n1+self.n2]
        return logits1, logits2


action_space_one = env.action_spec.space.boxes[0].boxes[0].n
action_space_two = env.action_spec.space.boxes[1].boxes[0].n


"""Actor network"""

# Create the MLP for multiple agents
# The output is the sum of the two action spaces
policy_torch = nn.Sequential(
    MultiAgentMLP(
        n_agent_inputs = env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs = action_space_one + action_space_two,
        n_agents = env.n_agents,
        centralised=True,
        share_params=share_parameters_policy,
        device=device,
        depth=policy_network_depth,
        num_cells=policy_network_num_cells,
        activation_class=nn.Tanh,
    ),
    
)

# Wrap the MultiAgentMlp in a TensorDictModule.
# This is simply a module that will read the in_keys from a tensordict, 
# feed them to the neural networks, and write the outputs in-place at the out_keys.
encoder = TensorDictModule(
    policy_torch,
    in_keys=[("agents", "observation")],
    out_keys=[("agents", "flat_logits")],
)


# We want the MultiAgentMLP to output separately each action. 
# splitter: turns flat -> ("logit1","logit2") 
splitter = TensorDictModule(
    Splitter(action_space_one, action_space_two),
    in_keys=[("agents", "flat_logits")],
    out_keys=[("agents", "a1_logits"), ("agents", "a2_logits")],
)

# Creates the params tree expected by CompositeDistribution.
def _pack_params_only_logits(logit1, logit2):
    return TensorDict(
        {
            "agents": {
                "params": {
                    "a1": {"logits": logit1},   # ONLY 'logits' (no 'probs')
                    "a2": {"logits": logit2},
                }
            }
        },
        batch_size=[],
    )

# The in_keys in the ProbabilisticActor should be 1D.
packer = TensorDictModule(
    _pack_params_only_logits,
    in_keys=[("agents","a1_logits"), ("agents","a2_logits")],
    out_keys=[("agents","params")],
)

# Combine all the components together.
actor_backbone = TensorDictSequential(encoder, splitter, packer)


tmp_action_key = ("agents","_tmp_action")


# Include two categorical distributions (one for each multidiscrete action).
policy_actor = ProbabilisticActor(
    module=actor_backbone,
    in_keys=[("agents","params")],
    distribution_class=CompositeDistribution, # groups multiple distributions together
    distribution_kwargs={
        "distribution_map": {"a1": d.Categorical, "a2": d.Categorical},
        "name_map": {
            "a1": ("agents","_tmp_action","a1"), # where to save each output
            "a2": ("agents","_tmp_action","a2"),
        },
    },
    return_log_prob=False,   # actor does NOT write a summed logprob
)


# Take a1 & a2 as inputs and return the stacked MultiDiscrete vector.
def _join_action(a1, a2):
    return torch.stack((a1.to(torch.long), a2.to(torch.long)), dim=-1)  

joiner = TensorDictModule(
    _join_action,
    in_keys=[(tmp_action_key, "a1"), (tmp_action_key, "a2")],
    out_keys=[env.action_key], # write exactly where env expects
)

# Policy
policy = TensorDictSequential(policy_actor, joiner)  # what the collector uses


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
    actor_network=policy_actor,
    critic_network=critic,
    clip_epsilon=clip_epsilon,
    entropy_coef=entropy_eps,
    normalize_advantage=False,
)


loss_module.set_keys(
    reward=env.reward_key,
    action=("agents","_tmp_action"),
    sample_log_prob=("_prev_log_prob_struct",),  # root-level parent (NOT under "agents")
    value=("agents","state_value"),
    done=("agents","done"),
    terminated=("agents","terminated"),
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

    # Helping function
    tensordict_data = ensure_tmp_and_logprob(tensordict_data)


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

            # Helping functions
            ensure_composite_action_on(subdata)
            rebuild_prev_logprob_root(subdata)
            
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


