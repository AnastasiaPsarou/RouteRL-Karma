import os

import logging
import numpy as np
import pandas as pd
import random

from routerl_aec_environment_karma.keychain import Keychain as kc
from routerl_aec_environment_karma.environment import HumanAgent
from routerl_aec_environment_karma.environment import MachineAgent


logger = logging.getLogger()
logger.setLevel(logging.WARNING)


def generate_agents(params, free_flow_times, generate_data, seed=23423) -> list:
    """Generates agent objects for the enviornment.
    
    This function creates agents based on predefined parameters, either by 
    generating new data or loading existing data from a CSV file. Each agent 
    is instantiated as either a `MachineAgent` or `HumanAgent`, depending on 
    its type.

    Args:
        params (dict):
            Agent parameters dictionary as specified in `here <https://coexistence-project.github.io/RouteRL/documentation/pz_env.html#>`_.
        free_flow_times (dict[tuple[int], list[float]]):
            Free flow times of route options per OD.
        generate_data (bool):
            If True, generates new agent data. If False, loads existing agent 
            data from a CSV file specified by `params[kc.RECORDS_FOLDER]`.
        seed (int, optional): Random seed for reproducibility. Defaults to 23423.
        
    Returns:
        list: A list of `BaseAgent` objects (either `MachineAgent` or `HumanAgent`).
        
    Raises:
        ValueError: If an unrecognized agent type is encountered in the data.
    """

    set_seed(seed)
    if generate_data:
        # Generate agent data
        agents_data_df = generate_agent_data(params, seed)
    else:
        # Load agent data
        agents_csv_path = os.path.join(params[kc.RECORDS_FOLDER], params[kc.AGENTS_CSV_FILE_NAME])
        agents_data_df = pd.read_csv(agents_csv_path)

    # Getting parameters
    action_space_size = params[kc.ACTION_SPACE_SIZE]
    agents = list()     # Where we will store & return agents

    #income_list = load_unique_incomes("income_samples.csv")
    num_agents = params[kc.NUM_AGENTS]
    income_list = load_shuffled_incomes(
        "income_samples.csv",
        seed=seed,
        col="income",
        n_needed=num_agents      # <- sample n agents, no repeats
    )

    # Generating agent objects from generated agent data
    for i, row in agents_data_df.iterrows():
        row_dict = row.to_dict()

        id, start_time = row_dict[kc.AGENT_ID], row_dict[kc.AGENT_START_TIME]
        origin, destination = row_dict[kc.AGENT_ORIGIN], row_dict[kc.AGENT_DESTINATION]
        income = income_list[i]

        if row_dict[kc.AGENT_KIND] == kc.TYPE_MACHINE:
            agent_params = params[kc.MACHINE_PARAMETERS]
            mutate_to = MachineAgent(id,
                                     start_time,
                                     origin,
                                     destination,
                                     income,
                                     agent_params,
                                     action_space_size)
            agents.append(mutate_to)
        elif row_dict[kc.AGENT_KIND] == kc.TYPE_HUMAN:
            agent_params = params[kc.HUMAN_PARAMETERS]
            initial_knowledge = free_flow_times[(origin, destination)]
            agents.append(HumanAgent(id,
                                     start_time,
                                     origin,
                                     destination,
                                     income,
                                     agent_params,
                                     initial_knowledge))
        else:
            raise ValueError('[AGENT TYPE INVALID] Unrecognized agent type: ' + row_dict[kc.AGENT_KIND])
    return agents

def generate_agent_data(params, seed=23423) -> pd.DataFrame:
    """Generates agent data.

    Constructs a dataframe, where each row is an agent and columns are attributes.

    Args:
        params (dict): Agent parameters dictionary as specified in `here <https://coexistence-project.github.io/RouteRL/documentation/pz_env.html#>`_.
        seed (int): Random seed.
    Returns:
        agents_df (pd.DataFrame): Pandas dataframe with agents attributes.
    """
    
    num_agents = params[kc.NUM_AGENTS]
    simulation_timesteps = params[kc.SIMULATION_TIMESTEPS]
    agent_attributes = [kc.AGENT_ID, kc.AGENT_ORIGIN, kc.AGENT_DESTINATION, kc.AGENT_START_TIME, kc.AGENT_KIND]
    
    num_origins = len(params[kc.ORIGINS])
    num_destinations = len(params[kc.DESTINATIONS])
    
    rng = set_seed(seed)
    agents_df = pd.DataFrame(columns=agent_attributes)  # Where we store our agents
    
    mean_timestep = simulation_timesteps / 2
    std_dev_timestep = simulation_timesteps / 6

    # Generating agent data
    for id in range(num_agents):
        
        # Decide on agent type (temporary solution)
        agent_type = kc.TYPE_HUMAN

        # Randomly assign origin & destination
        #origin, destination = random.randrange(num_origins), random.randrange(num_destinations)
        origin = random.randrange(num_origins)
        destination = origin


        # Randomly assign start time (normal dist)
        start_time = int(rng.normal(mean_timestep, std_dev_timestep))
        start_time = max(0, min(simulation_timesteps, start_time))

        # Registering to the dataframe
        agent_features = [id, origin, destination, start_time, agent_type]
        agent_dict = {attribute : feature for attribute, feature in zip(agent_attributes, agent_features)}
        agents_df.loc[id] = agent_dict
        
    # Saving the generated agents to a csv file
    agents_csv_path = os.path.join(params[kc.RECORDS_FOLDER], params[kc.AGENTS_CSV_FILE_NAME])
    agents_csv_path = str(agents_csv_path)
    agents_df.to_csv(agents_csv_path, index=False)
    
    return agents_df

def load_unique_incomes(samples_path, col="income"):
    """
    Load unique numeric income values from a CSV file.
    Returns a sorted NumPy array of unique values.
    """
    s = pd.read_csv(samples_path, usecols=[col])[col]
    s = pd.to_numeric(s, errors="coerce").dropna()
    uniq = np.unique(s.values)
    if uniq.size == 0:
        raise ValueError(f"No valid numeric '{col}' values found in {samples_path}.")
    return uniq

def load_shuffled_incomes(samples_path, seed, col="income", n_needed=None):
    """
    Load incomes from CSV and return a randomized selection.
    - If n_needed is None: return all incomes in a random order.
    - If n_needed is provided:
        * sample WITHOUT replacement n_needed values (so each is used once).
    """
    s = pd.read_csv(samples_path, usecols=[col])[col]
    s = pd.to_numeric(s, errors="coerce").dropna().to_numpy()

    if s.size == 0:
        raise ValueError(f"No valid numeric '{col}' values found in {samples_path}.")

    rng = np.random.default_rng(seed)

    if n_needed is None:
        rng.shuffle(s)              # full random permutation
        return s

    if s.size < n_needed:
        raise ValueError(
            f"[income sampling] CSV has only {s.size} incomes but {n_needed} agents."
        )

    # sample exactly n_needed rows without replacement
    idx = rng.choice(s.size, size=n_needed, replace=False)
    return s[idx]



def set_seed(seed):
    """Set the seed for random number generation.

    Arguments:
        seed (int): Random seed
    Returns:
        rng: Random default rng for random number generation
    """

    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    return rng
