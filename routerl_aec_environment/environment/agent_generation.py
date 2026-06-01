import os

from itertools import chain
import logging
import numpy as np
import pandas as pd
import random

from routerl_aec_environment.keychain import Keychain as kc
from routerl_aec_environment.environment import HumanAgent
from routerl_aec_environment.environment import MachineAgent


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
    
    # Generating agent objects from generated agent data
    for i, row in agents_data_df.iterrows():
        row_dict = row.to_dict()

        id, start_time, income = row_dict[kc.AGENT_ID], row_dict[kc.AGENT_START_TIME], row_dict[kc.INCOME]
        origin, destination = row_dict[kc.AGENT_ORIGIN], row_dict[kc.AGENT_DESTINATION]

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
    agent_attributes = [kc.AGENT_ID, kc.AGENT_ORIGIN, kc.AGENT_DESTINATION, kc.AGENT_START_TIME, kc.INCOME, kc.AGENT_KIND]
    
    num_origins = len(params[kc.ORIGINS])
    num_destinations = len(params[kc.DESTINATIONS])
    
    rng = set_seed(seed)
    agents_df = pd.DataFrame(columns=agent_attributes)  # Where we store our agents
    
    mean_timestep = simulation_timesteps / 2
    std_dev_timestep = simulation_timesteps / 6

    num_agents = params[kc.NUM_AGENTS]

    # Create quantiles of agents based on income values
    groups = get_income_groups(params, 
                                    "income_samples.csv",
                                    seed=seed,
                                    col="income",
                                    n_needed=num_agents)


    # Create different distributions of departure times for each income group
    start_time_samplers = build_normal_samplers(
        params[kc.QUINTILE_GROUPS],
        simulation_timesteps,
        mean_frac= (mean_timestep / simulation_timesteps),  # earliest groups start earlier
        std_frac= (std_dev_timestep / simulation_timesteps),   # poorer groups more variation
    )

    start_time_stream = iter_sampled_incomes_by_group(
        groups,
        start_time_samplers,
        simulation_timesteps,
        seed=42,
    )

    income_stream = chain.from_iterable(groups)

    # Iterate over id value, start time and income value
    for id, (income, start_time) in zip(range(num_agents), zip(income_stream, start_time_stream)):
        
        # Decide on agent type (temporary solution)
        agent_type = kc.TYPE_HUMAN

        # Randomly assign origin & destination
        #origin, destination = random.randrange(num_origins), random.randrange(num_destinations)
        origin = random.randrange(num_origins)
        destination = origin

        # Registering to the dataframe
        agent_features = [id, origin, destination, start_time, income, agent_type]
        agent_dict = {attribute : feature for attribute, feature in zip(agent_attributes, agent_features)}
        agents_df.loc[id] = agent_dict
        
    # Saving the generated agents to a csv file
    agents_csv_path = os.path.join(params[kc.RECORDS_FOLDER], params[kc.AGENTS_CSV_FILE_NAME])
    agents_csv_path = str(agents_csv_path)
    agents_df.to_csv(agents_csv_path, index=False)
    return agents_df

def iter_sampled_incomes_by_group(groups, samplers, simulation_timesteps, seed=None, sort_each_group=False):
    """
    Yields integer start times sequentially by group.
    Each group's sampler defines its own distribution.
    Clipping ensures all values are within [0, simulation_timesteps].
    """
    if len(samplers) != len(groups):
        raise ValueError("samplers and groups must have the same length.")

    rng = np.random.default_rng(seed)

    for g_idx, group in enumerate(groups):
        size = len(group)
        values = samplers[g_idx](size) # sample floats
        values = np.clip(values, 0, simulation_timesteps)  # bound to [0, T]
        values = np.rint(values).astype(int)               # convert to int (rounded)
        if sort_each_group:
            values = np.sort(values)
        for v in values:
            yield v


def make_normal_sampler(mu, sigma, low, high, rng=None):
    rng = np.random.default_rng(rng)
    def sample(size=None):
        x = rng.normal(mu, sigma, size=size)
        return np.clip(x, low, high)
    return sample


def build_normal_samplers(n_groups, simulation_timesteps,
                          mean_fracs=None, std_fracs=None,
                          mean_frac=None, std_frac=None,
                          mean_start_frac=0.15, mean_end_frac=0.85,
                          std_start_frac=0.10, std_end_frac=0.04,
                          base_seed=None):      # optional: for reproducibility
    T = simulation_timesteps
    ss = np.random.SeedSequence(base_seed)
    child_seeds = ss.spawn(n_groups)

    # broadcast same params to all groups
    if mean_fracs is None:
        mean_fracs = np.full(n_groups, mean_frac if mean_frac is not None else 0.5)
    if std_fracs is None:
        std_fracs  = np.full(n_groups, std_frac if std_frac is not None else 0.08)

    samplers = [
        make_normal_sampler(mu=mf * T, sigma=sf * T, low=0, high=T, rng=np.random.default_rng(cs))
        for mf, sf, cs in zip(mean_fracs, std_fracs, child_seeds)
    ]
    return samplers


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


def get_income_groups(params, samples_path, seed, col="income", n_needed=None):
    """
    Load incomes (using load_shuffled_incomes) and return them split into n_groups lists,
    each sorted ascending. Lower-income values go into the first list, etc.

    Example:
        get_income_groups("incomes.csv", seed=42, n_groups=3)
        -> [[lowest incomes], [middle incomes], [highest incomes]]
    """
    incomes = load_shuffled_incomes(samples_path, seed, col=col, n_needed=n_needed)
    incomes_sorted = np.sort(incomes)

    # split into n_groups as evenly as possible
    groups = np.array_split(incomes_sorted, params[kc.QUINTILE_GROUPS])

    # convert numpy arrays to normal Python lists
    return [g.tolist() for g in groups]


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