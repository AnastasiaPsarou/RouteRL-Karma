from tqdm import tqdm
import torch
import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from routerl_aec_environment import TrafficEnvironment
from routerl_aec_environment import DQN, ReplayBuffer
from routerl_aec_environment import Keychain as kc
from routerl_aec_environment import MeanFieldQLearning

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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

records_folder = f"training_records_monetary_pricing_300_agents_{seed}_fee_10"
plots_folder = f"plots_monetary_pricing_300_agents_{seed}_fee_10"

phases = [1, int(total_episodes)]
phase_names = ["Mutation and AV learning", "Testing phase"]

env_params = {
    "agent_parameters": {
        "new_machines_after_mutation": new_machines_after_mutation,
        "num_agents": num_agents,
        "machine_parameters": {
            "behavior": "selfish",
            "observation_type": "previous_agents_avg_tt_per_route",
            "route_0_fee": 10,
            "travel_time_normalization_value": 1
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
        "save_every": 5,
        "number_of_days": 30
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

def obs_to_vec(observation):
    """
    Convert env observation to a 1D float vector.
    Adjust this if your env returns dicts with nested fields.
    """
    if isinstance(observation, dict) and "observation" in observation:
        observation = observation["observation"]
    return np.asarray(observation, dtype=np.float32).reshape(-1)

def mean_field_onehot(last_actions, self_id, n_actions):
    """
    Mean one-hot distribution of OTHER learning agents' last actions.
    last_actions: dict agent_id -> action (int) or None
    """
    others = [a for aid, a in last_actions.items() if aid != self_id and a is not None]
    if len(others) == 0:
        return np.zeros(n_actions, dtype=np.float32)
    counts = np.bincount(np.asarray(others, dtype=np.int64), minlength=n_actions).astype(np.float32)
    return counts / max(1.0, counts.sum())


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


# Learning agents (mutated humans now machines): keys are strings like "121"
learning_agents = list(mutated_humans.keys())
agent_to_idx = {aid: i for i, aid in enumerate(learning_agents)}

# Determine obs_dim from one sample
env.reset()

obs_spaces = env._observation_spaces
some_agent = next(iter(obs_spaces))
obs_space = obs_spaces[some_agent]
obs_dim = int(np.prod(obs_space.shape))

# Discrete route choice: since number_of_paths=2 -> actions are {0,1}
n_actions = 3

# Mean-field vector: mean one-hot over other agents' last actions -> dimension = n_actions
mf_dim = n_actions

mf_agent = MeanFieldQLearning(
    obs_dim=obs_dim,
    mf_dim=mf_dim,
    n_actions=n_actions,
    num_agents=len(learning_agents),
    widths=(128, 128),
    gamma=0.99,
    learning_rate=3e-4,
    memory_size=50_000,
    batch_size=256,
    target_update_freq=2000,
    epsilon=1.0,
    epsilon_min=0.05,
    epsilon_decay_rate=1e-4,
    per_agent_exploration=True,   # optional: separate eps per agent
    seed=seed,
).train()

# Track previous step info per agent (AEC-style)
last_obs = {aid: None for aid in learning_agents}
last_mf  = {aid: None for aid in learning_agents}
last_act = {aid: None for aid in learning_agents}

# Population "last actions" used to build mean-field
last_actions = {aid: 0 for aid in learning_agents}  # initialize to make mf defined

    

pbar = tqdm(total=training_episodes + testing_episodes, desc="AV learning (MF-DQN)")

for episode in range(training_episodes):
    env.reset()

    # reset per-episode trackers (recommended)
    for aid in learning_agents:
        last_obs[aid] = None
        last_mf[aid] = None
        last_act[aid] = None
        last_actions[aid] = 0

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        done = bool(termination or truncation)

        # Skip agents we are NOT training (e.g., humans)
        if agent not in agent_to_idx:
            env.step(None)
            continue

        obs_vec = obs_to_vec(observation)
        mf_vec = mean_field_onehot(last_actions, agent, n_actions)

        # If we have a previous (s,m,a) for this agent, store transition now
        # (reward corresponds to the previous action in AEC style)
        if last_obs[agent] is not None and last_act[agent] is not None and last_mf[agent] is not None:
            mf_agent.store_transition(
                obs=last_obs[agent],
                action=last_act[agent],
                mf=last_mf[agent],
                reward=reward,
                next_obs=obs_vec,
                next_mf=mf_vec,
                done=done,
            )

        if done:
            action = None
        else:
            idx = agent_to_idx[agent]
            action = mf_agent.act(idx, obs_vec, mf_vec)

            # Update trackers
            last_obs[agent] = obs_vec
            last_mf[agent] = mf_vec
            last_act[agent] = action
            last_actions[agent] = action

        env.step(action)

    # train after each episode (tune updates)
    mf_agent.learn(updates=10)
    pbar.update(1)


mf_agent.eval()
pbar.set_description("Testing (MF-DQN)")

for episode in range(testing_episodes):
    env.reset()

    for aid in learning_agents:
        last_actions[aid] = 0

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        done = bool(termination or truncation)

        if agent not in agent_to_idx:
            env.step(None)
            continue

        if done:
            env.step(None)
            continue

        obs_vec = obs_to_vec(observation)
        mf_vec = mean_field_onehot(last_actions, agent, n_actions)
        idx = agent_to_idx[agent]

        action = mf_agent.act(idx, obs_vec, mf_vec)  # eps=0 in eval()
        last_actions[agent] = action

        env.step(action)

    pbar.update(1)

pbar.close()


env.plot_results()

env.stop_simulation()
