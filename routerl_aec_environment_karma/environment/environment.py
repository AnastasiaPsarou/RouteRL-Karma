"""
PettingZoo environment for optimal route choice using SUMO simulator.

"""

import glob
import os
import itertools

from copy import copy
from copy import deepcopy as dc
from gymnasium.spaces import Discrete, MultiDiscrete

import functools
import logging
import numpy as np
import pandas as pd
import random
import threading
import math

from routerl_aec_environment_karma.environment import generate_agents
from routerl_aec_environment_karma.environment import SumoSimulator
from routerl_aec_environment_karma.environment import MachineAgent
from routerl_aec_environment_karma.environment import PreviousAgentStart, PreviousAgentStartPlusStartTime 
from routerl_aec_environment_karma.environment import PreviousAgentStartPlusStartTimeDetectorData, PreviousAgentStartPlusStartTimeMarginalCost, Observations, PreviousAvgTTperRoute
from routerl_aec_environment_karma.keychain import Keychain as kc
from routerl_aec_environment_karma.services import plotter
from routerl_aec_environment_karma.services import Recorder
from routerl_aec_environment_karma.utilities import get_params

from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import agent_selector

logger = logging.getLogger()
logger.setLevel(logging.WARNING)


class TrafficEnvironment(AECEnv):
    """
    A PettingZoo AECEnv interface for optimal route choice using SUMO simulator. 
    This environment is designed for the training of human agents (rational decision-makers) 
    and machine agents (reinforcement learning agents).
    
    See `SUMO <https://sumo.dlr.de/docs/>`_ for details on SUMO. \n
    See `PettingZoo <https://pettingzoo.farama.org/>`_ for details on PettingZoo. 
    
    .. note::
        Users can configure the experiment with keyword arguments, see the structure below. 
        Moreover, users can provide custom demand data in ``training_records/agents.csv``.
        You can refer to the structure of such a file `here <https://github.com/COeXISTENCE-PROJECT/RouteRL/blob/main/docs/_static/agents_example.csv>`_.

    Args:
        seed (int, optional): 
            Random seed for reproducibility. Defaults to ``23423``.
        create_agents (bool, optional):
            Whether to create agent data. Defaults to ``True``.
        create_paths (bool, optional):
            Whether to generate paths. Defaults to ``True``.
        **kwargs (dict, optional): 
            User-defined parameter overrides. These override default values 
            from ``defaults.json`` and allow experiment configuration.
            

    Keyword arguments (see the usage below):
    
        - agent_parameters (dict, optional):
            Agent settings.
            
            - num_agents (int, default=100):
                Total number of agents.
            
            - new_machines_after_mutation (int, default=25):
                Number of humans converted to machines.
            
            - machine_parameters (**dict**):
                Machine agent settings.
                
                - behavior (str, default="selfish"):
                    Route choice behavior.
                    Options: ``selfish``, ``social``, ``altruistic``, ``malicious``, ``competitive``, ``collaborative``.
                    
                - observed_span (int, default=300):
                    Time window considered for observations.
                    
                - observation_type (str, default="previous_agents_plus_start_time"):
                    Type of observation.
                    Options: ``previous_agents``, ``previous_agents_plus_start_time``.

            - human_parameters (**dict**): 
                Human agent settings.
                
                - model (str, default="gawron"):
                    Decision-making model (options: ``random``, ``gawron``, ``weighted``).
                    
                - noise_weight_agent (float, default=0.2):
                    Agent noise weight in the error term composition.
                    
                - noise_weight_path (float, default=0.6):
                    Path noise weight in the error term composition.
                    
                - noise_weight_day (float, default=0.2):
                    Day noise weight in the error term composition.
                    
                - beta (float, default=-1.0):
                    **Negative value**, multiplier of reward (travel time) used in utility, determines sensitivity.
                    
                - beta_k_i_variability (float, default=0):
                    Variance of normal distribution for which ``beta_k_i`` is drawn (computed in utility).
                    
                - epsilon_i_variability (float, default=0):
                    Variance of normal distribution from which error terms are drawn, first term.
                    
                - epsilon_k_i_variability (float, default=0):
                    Variance of normal distribution from which error terms are drawn, second term.
                    
                - epsilon_k_i_t_variability (float, default=0):
                    Variance of normal distribution from which error terms are drawn, third term. These three terms must sum to 1.
                    
                - greedy (float, default=1.0):
                    1 - exploration_rate, probability with which the choice are rational (argmax(U)) and not probabilistic (random).
                    
                - gamma_c (float, default=0):
                    Bounded rationality component. Expressed as relative increase in costs/utilities below which user do not change behaviour (do not notice it).
                    
                - gamma_u (float, default=0):
                    Bounded rationality component. Expressed as relative increase in costs/utilities below which user do not change behaviour (do not notice it).
                
                - remember (int, default=1):
                    Number of days remembered to learn from.
                    
                - alpha_zero (float, default=0.2):
                    Weight with which the recent experience is carried forward in learning.
                    
                - alphas (list[float], default=[0.8]):
                    Vector of weights for historically recorded reward in weighted average, **needs to be size of** ``remember``.

        - environment_parameters (dict, optional):
            Environment settings.
            
            - number_of_days (int, default=1):
                Number of days in the scenario.
                
            - save_every (int, default=1):
                Save the episode data to disk every X days.

        - simulator_parameters (dict, optional): 
            SUMO simulator settings.
            
            - network_name (str, default="csomor"):
                Network name (e.g., ``arterial``, ``cologne``, ``grid``)
                
            - custom_network_folder (str, default="NA"):
                In case of custom network, specify the folder name.
            
            - simulation_timesteps (int, default=180):
                Total simulation time in seconds.
            
            - sumo_type (str, default="sumo"):
                SUMO execution mode (``sumo`` or ``sumo-gui``).
                
            - stuck_time (int, default=600):
                Number of seconds to tolerate before `teleporting` a stopped vehicle to resolve gridlocks.

        - path_generation_parameters (dict, optional):
            Path generation settings.
            
            - number_of_paths (int, default=3):
                Number of routes per OD.
                
            - beta (float, default=-3.0):
                Sensitivity to travel time in path generation.
                
            - weight (str, default="time"):
                Optimization criterion.
                
            - num_samples (int, default=100):
                Number of samples for path generation.
                
            - origins (str | list[str], default="default"):
                Origin points from the network. (e.g., ``["-25166682#0", "-4936412"]``)
                
            - destinations (str | list[str], default="default"):
                Destination points from the network. (e.g., ``["-115604057#1", "-279952229#4"]``)
                
            - visualize_paths (bool, default=True):
                Whether to visualize generated paths. Visuals will be saved in the ``plotter_parameters/plots_folder``.

        - plotter_parameters (dict, optional): 
            Plotting & logging settings.
            
            - records_folder (str, default="training_records"):
                Directory for training records.
                
            - plots_folder (str, default="plots"):
                Directory for plots.
                
            - plot_choices (str, default="all"):
                Selection of plots to be generated. Options: ``none``, ``basic``, ``all``.
                
            - smooth_by (int, default=50): 
                Smoothing parameter for plots.
                
            - phases (list[int], default=[0, 100]):
                X-axis positions for phase markers.
                
            - phase_names (list[str], default=["Human learning", "Mutation - Machine learning"]):
                Phase names for labeling phase markers.
    
    Usage:
        
        .. rubric:: Case 1
        
        .. code-block:: text
        
            % Your file structure in the beginning
            project_directory/
            |-- your_script.py
            
        .. code-block:: python
        
            >>> # Environment initialization
            ... env = TrafficEnvironment(
            ...     seed=42,
            ...     agent_parameters={
            ...         "num_agents": 5, 
            ...         "new_machines_after_mutation": 1, 
            ...         "machine_parameters": {
            ...             "behavior": "selfish"
            ...             }},
            ...     simulator_parameters={"sumo_type": "sumo-gui"},
            ...     path_generation_parameters={"number_of_paths": 2}
            ... )
            
        .. code-block:: text
        
            % File structure after the initialization:
            project_directory/
            |-- your_script.py
            |-- training_records/
            |   |-- agents.csv
            |   |-- paths.csv
            |   |-- detector/
            |   |   |--             % to be populated during simulation
            |   |-- episodes/
            |   |   |--             % to be populated during simulation
            |-- plots/
            |   |-- 0_0.png 
            |   |-- ...             % visuals of generated paths for each OD
            |   |-- ...             % to be populated after the experiment
            
        .. raw:: html

            <hr style="border:1px solid #ccc; margin: 20px 0;">
        
        .. rubric:: Case 2
        
        .. code-block:: text
        
            % Your file structure in the beginning
            project_directory/
            |-- your_script.py
            |-- training_records/
            |   |-- agents.csv      % your custom demand, conforming to the structure
        
        .. warning::
            Demand data in ``agents.csv`` should be aligned with the specified 
            experiment settings (e.g., number of agents, number of origins and destinations, etc.).
            
        .. code-block:: python
        
            >>> env = TrafficEnvironment(
            ...     create_agents=False, # Environment will use your agent data
            ...     agent_parameters={
            ...         "new_machines_after_mutation": 10, 
            ...         "machine_parameters": {
            ...             "behavior": "selfish"
            ...             }},
            ...     simulator_parameters={"network_name": "arterial"},
            ...     path_generation_parameters={"number_of_paths": 3}
            ... )
            
        .. code-block:: text
        
            % File structure after the initialization:
            project_directory/
            |-- your_script.py
            |-- training_records/
            |   |-- agents.csv      % stays the same, used for agent generation
            |   |-- paths.csv
            |   |-- detector/
            |   |   |--             % to be populated during simulation
            |   |-- episodes/
            |   |   |--             % to be populated during simulation
            |-- plots/
            |   |-- 0_0.png 
            |   |-- ...             % visuals of generated paths for each OD
            |   |-- ...             % to be populated after the experiment
            
        .. warning::
            Same approach does not translate to path generation.\n
            ``paths.csv`` is mainly used for visualization purposes. 
            For SUMO to operate correctly, a ``route.rou.xml`` 
            should be generated inside the ``routerl/networks/<net_name>/`` folder.\n
            It is advised to generate paths in each experiment providing a random seed,
            or set ``create_paths=False`` only when above criteria is met.

    Attributes:
        day (int): Current day index in the simulation.
        human_learning (bool): Whether human agents are learning.
        number_of_days (int): Number of days to simulate.
        action_space_size (int): Size of the action space.
        recorder (Recorder): Object for recording simulation data.
        simulator (SumoSimulator): SUMO simulator instance.
        all_agents (list): List of all agent objects.
        machine_agents (list): List of all machine agent objects.
        human_agents (list): List of all human agent objects.
    """
    
    metadata = {
        "render_modes": ["human"],
        "name": "TrafficEnvironment",
    }

    def __init__(self,
                 seed: int = 23423,
                 create_agents: bool = True,
                 create_paths: bool = True,
                 save_detectors_info: bool = False,
                 second_sumo: bool = False,
                 marginal_cost_calculation: bool = False,
                 marginal_cost_calculation_machine_to_all: bool = False,
                 randomize_sumo_seed: bool = False,
                 monetary_pricing: bool = False, 
                 karma_pricing: bool = False,
                 **kwargs) -> None:

        super().__init__()
        self.render_mode = None

        # Read default parameters, update with kwargs
        defaults_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), kc.DEFAULTS_FILE)
        params = get_params(defaults_path, resolve=True, update=kwargs)
        self.kwargs = kwargs

        self.environment_params = params[kc.ENVIRONMENT]
        self.params = params
        self.simulation_params = params[kc.SIMULATOR]
        self.agent_params = params[kc.AGENTS]
        self.plotter_params = params[kc.PLOTTER]
        self.path_gen_params = params[kc.PATH_GEN] if create_paths else None

        self.travel_times_list = []
        self.day = 0
        self.human_learning = True
        self.machine_same_start_time = []
        self.actions_timestep = []
        self.save_detectors_info = save_detectors_info
        self.second_sumo = second_sumo
        self.marginal_cost_calculation = marginal_cost_calculation
        self.monetary_pricing = monetary_pricing
        self.karma_pricing = karma_pricing
        self.marginal_cost_calculation_machine_to_all = marginal_cost_calculation_machine_to_all

        self.number_of_days = self.environment_params[kc.NUMBER_OF_DAYS]
        self.save_every = self.environment_params[kc.SAVE_EVERY]
        self.action_space_size = self.environment_params[kc.ACTION_SPACE_SIZE]
        self._set_seed(seed)
        self.recorder = None
        self.historic_data = []

        if second_sumo == False:
            self.recorder = Recorder(self.plotter_params)
        self.simulator = SumoSimulator(self.simulation_params, self.path_gen_params, seed, not create_agents, save_detectors_info, randomize_sumo_seed)

        self.all_agents = generate_agents(self.agent_params, self.get_free_flow_times(), create_agents, seed)
        self.machine_agents = [agent for agent in self.all_agents if agent.kind == kc.TYPE_MACHINE]
        self.human_agents = [agent for agent in self.all_agents if agent.kind == kc.TYPE_HUMAN]
        self.possible_agents = list()
        
        # Urgency distribution
        self.urgency_distribution = np.random.geometric(0.3, size=len(self.all_agents))
        self.urgency_distribution = np.clip(self.urgency_distribution, 1, 10) / 10 # restrict to 1–10
        self.all_agents_high_urgency = False
        self.testing_urgency=False
        self.urgency_min = 0.1
        self.urgency_max = 1.0
        self.urgency_step = 0.1

        self.urgency_values = np.round(
            np.arange(
                self.urgency_min,
                self.urgency_max + self.urgency_step,
                self.urgency_step
            ),
            1
        )
        self._prepare_urgency_for_day()

        self.marginal_cost_machine_agents_flag() # Initialize marginal cost flag
        self.monetary_pricing_flag() # Initialize monetary pricing flag

        if len(self.machine_agents):
            self._initialize_machine_agents()
        if not self.human_agents:
            self.human_learning = False
        logging.info(f"There are {len(self.human_agents)} human and {len(self.machine_agents)} machine agents.")

        self.episode_actions = dict()

    def __str__(self):
        message = f"TrafficEnvironment with {len(self.all_agents)} agents.\
            \n{len(self.machine_agents)} machines and {len(self.human_agents)} humans.\
            \nMachines: {sorted(self.machine_agents, key=lambda agent: agent.id)}\
            \nHumans: {sorted(self.human_agents, key=lambda agent: agent.id)}"
        return message

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed
        logging.info(f"Seed set to {seed}.")

    def _initialize_machine_agents(self) -> None:

        ## Sort machine agents based on their start_time
        sorted_machine_agents = sorted(self.machine_agents, key=lambda agent: agent.start_time)
        self.possible_agents = [str(agent.id) for agent in sorted_machine_agents]
        self.n_agents = len(self.possible_agents)

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        ## Initialize the observation object
        self.observation_obj = self.get_observation_function()
        self._observation_spaces = self.observation_obj.observation_space()
        
        
        self._action_spaces = {
            agent: Discrete(self.simulation_params[kc.NUMBER_OF_PATHS]) for agent in self.possible_agents
        }

        self.action_spaces = self._action_spaces

        if self.karma_pricing == True:
            self._action_spaces = {
                agent: MultiDiscrete(np.full(self.simulation_params[kc.NUMBER_OF_PATHS], self.environment_params[kc.MAXIMUM_ALLOWED_BID])) for agent in self.possible_agents
            }

            self.action_spaces = self._action_spaces            
            self.centrally_defined_price_route_0 = self.environment_params[kc.CENTRALLY_DEFINED_PRICE_ROUTE_0]
            self.centrally_defined_price_route_1 = self.environment_params[kc.CENTRALLY_DEFINED_PRICE_ROUTE_1]
            self.total_karma_points_used = 0

            for machine in self.machine_agents:
                machine.karma_balance = self.environment_params[kc.MAXIMUM_ALLOWED_BID]


        logging.info("\nMachine's observation space is: %s ", self._observation_spaces)
        logging.info("Machine's action space is: %s", self._action_spaces)

    ################################
    ######## Control methods #######
    ################################

    def start(self, use_subprocess=False) -> None:
        """Start the connection with SUMO.
        
        Returns:
            None
        """

        self.simulator.start(use_subprocess)

    def reset(self, seed: int = None, options: dict = None) -> tuple:
        """Resets the environment.
        
        Args:
            seed (int, optional): Seed for random number generation. Defaults to None.
            options (dict, optional): Additional options for resetting the environment. Defaults to None.
            
        Returns:
            observations (dict): observations.
            infos (dict): dictionary of information for the agents.
        """

        self.episode_actions = dict()
        _, self.sumo_seed = self.simulator.reset()
        self.agents = copy(self.possible_agents)
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}
        self.rewards = {agent: 0 for agent in self.possible_agents}
        self.rewards_humans = {agent.id: 0 for agent in self.human_agents}
        self.travel_times_list = []

        if self.karma_pricing:
            self._reinitialize_karma_balance()

        if len(self.machine_agents) > 0:
            self._agent_selector = agent_selector(self.possible_agents)
            self.agent_selection = self._agent_selector.next()
            self.observations = self.observation_obj.reset_observation()
        else:
            self.observations = {}

        infos = {a: {} for a in self.possible_agents}

        return self.observations, infos

    def step(self, machine_action: int = None) -> None:
        """Step method.

        Takes an action for the current agent (specified by `agent_selection`) and updates
        various parameters including rewards, cumulative rewards, terminations, truncations,
        infos, and agent_selection. Also updates any internal state used by `observe()`.

        Args:
            machine_action (int, optional):
                The action to be taken by the machine agent. Defaults to None.
            
        Returns:
            None
        """

        # If there are machines in the system
        if self.possible_agents:
            if (self.terminations[self.agent_selection]
                    or self.truncations[self.agent_selection]):
                # handles stepping an agent which is already dead
                # accepts a None action for the one agent, and moves the agent_selection to
                # the next dead agent,  or if there are no more dead agents, to the next live agent
                self._was_dead_step(machine_action)
                return

            agent = self.agent_selection
            
            for machine in self.machine_agents:
                if machine.id == int(agent):
                    machine.last_action = machine_action
                    break

            # The cumulative reward of the last agent must be 0
            self._cumulative_rewards[agent] = 0

            if self.karma_pricing == True:
                machine.bid = machine_action
                assigned_route = self.stackelberg_auction(machine, machine_action)
                machine_action = assigned_route

            self.simulation_loop(machine_action, agent)

            # Collect rewards if it is the last agent to act
            if self._agent_selector.is_last():
                # Increase day number
                self.day += 1

                # Calculate marginal cost matrix
                self.marginal_cost()

                # Calculate the rewards
                self._assign_rewards()

                self._redistribute_karma_points()

                # The episode ends when we complete episode_length days
                self.truncations = {agent: not (self.day % self.number_of_days) for agent in self.agents}
                self.terminations = {agent: not (self.day % self.number_of_days) for agent in self.agents}
                self.infos = {agent: {} for agent in self.agents}

                self.observations = self.observation_obj(self.all_agents)

                self._reset_episode()
            else:
                # no rewards are allocated until all players give an action
                self._clear_rewards()
                self.agent_selection = self._agent_selector.next()

            # Adds .rewards to ._cumulative_rewards
            self._accumulate_rewards()

        # If there are only humans in the system
        else:
            self.simulation_loop(machine_action=0, machine_id=0)
            self.day = self.day + 1
            self._assign_rewards()
            if self.second_sumo == False:
                self._reset_episode()

    def close(self) -> None:
        """Not implemented.

        Returns:
            None
        """
        pass
    
    def stop_simulation(self) -> None:
        """End the simulation.

        Returns:
            None
        """

        self.simulator.stop()

    def observe(self, agent: str) -> np.ndarray:
        """Retrieve the observations for a specific agent.

        Args:
            agent (str): The identifier for the agent whose observations are to be retrieved.
            
        Returns:
            self.observation_obj.agent_observations(agent) (np.ndarray): The observations for the specified agent.
        """

        for machine in self.machine_agents:
            if str(machine.id) == agent:
                break

        # If the agent's turn hasn't come and the start time is bigger than the simulator timestep return an "empty observation"
        # The agent hasn't acted yet so only the start time is meaningful
        if agent != self.agent_selection:# and machine.start_time > self.simulator.timestep:
            array = np.zeros(self._observation_spaces[agent].shape[0])
            observation = np.array(array)
            return observation
        
        params = self.agent_params[kc.MACHINE_PARAMETERS]
        observation_type = params[kc.OBSERVATION_TYPE]

        if self.all_agents_high_urgency:
            machine.urgency = np.max(self.urgency_distribution)
        elif self.testing_urgency:
            machine.urgency = np.random.choice(self.urgency_distribution)
        else:
            self._assign_urgency_level_to_an_agent(machine)

        action_mask = self.make_multidiscrete_mask(machine.karma_balance, self._action_spaces[str(machine.id)])

        if observation_type == kc.PREVIOUS_AGENTS_PLUS_START_TIME_MARGINAL_COST or observation_type == kc.PREVIOUS_AVERAGE_TT_PER_ROUTE:
            obs = self.observation_obj.agent_observations(agent, self.all_agents, self.historic_data)
        else:
            obs = self.observation_obj.agent_observations(agent, self.all_agents)

        self.observations[str(machine.id)]["observations"] = obs
        self.observations[str(machine.id)]["action_mask"] = action_mask

        return self.observations[str(machine.id)]

    #########################
    ### Mutation function ###
    #########################

    def mutation(self, disable_human_learning: bool = True, mutation_start_percentile: int = 25) -> None:
        """Perform mutation by converting selected human agents into machine agents.

        This method identifies a subset of human agents that start after the 25th percentile of start times of
        other vehicles, removes a specified number of these agents, and replaces them with machine agents.

        Args:
            disable_human_learning (bool, optional): Boolean flag to disable human agents.
            
        Returns:
            None
            
        Raises:
            ValueError: If there are insufficient human agents available for mutation.
        """

        logging.info("Mutation is about to happen!\n")
        logging.info("There were %s human agents.\n", len(self.human_agents))

        # Mutate to a human that starts after the 25% of the rest of the vehicles
        start_times = [human.start_time for human in self.human_agents]
        percentile = np.percentile(start_times, mutation_start_percentile)

        filtered_human_agents = [human for human in self.human_agents if human.start_time >= percentile]
        #filtered_human_agents = [human for human in self.human_agents]

        number_of_machines_to_be_added = self.agent_params[kc.NEW_MACHINES_AFTER_MUTATION]
        random_humans_deleted = []

        if len(filtered_human_agents) < number_of_machines_to_be_added:
            raise ValueError(
                f"Insufficient human agents for mutation. Required: {number_of_machines_to_be_added}, "
                f"Available: {len(filtered_human_agents)}.\n"
                f"Decrease the number of machines to be added after the mutation.\n"
            )

        for _ in range(0, number_of_machines_to_be_added):
            random_human = random.choice(filtered_human_agents)

            self.human_agents.remove(random_human)
            filtered_human_agents.remove(random_human)

            random_humans_deleted.append(random_human)
            self.machine_agents.append(MachineAgent(random_human.id,
                                                    random_human.start_time,
                                                    random_human.origin,
                                                    random_human.destination,
                                                    random_human.income,
                                                    self.agent_params[kc.MACHINE_PARAMETERS],
                                                    self.action_space_size))
            self.possible_agents.append(str(random_human.id))

        self.n_agents = len(self.possible_agents)
        self.all_agents = self.machine_agents + self.human_agents
        self.marginal_cost_machine_agents_flag() # Initialize marginal cost flag
        self.monetary_pricing_flag()

        if disable_human_learning:  self.human_learning = False

        logging.info(f"Now there are {len(self.human_agents)} human agents.")

        self._initialize_machine_agents()


    def mutation_odd_id_agents(self, disable_human_learning: bool = True, mutation_start_percentile: int = 25) -> None:
        """Perform mutation by converting selected human agents into machine agents.

        This method identifies the human agents that have odd ids, 
        removes a specified number of these agents, and replaces them with machine agents.

        Args:
            disable_human_learning (bool, optional): Boolean flag to disable human agents.
            
        Returns:
            None
            
        Raises:
            ValueError: If there are insufficient human agents available for mutation.
        """

        logging.info("Mutation is about to happen!\n")
        logging.info("There were %s human agents.\n", len(self.human_agents))

        random_humans_deleted = []
        every_two_humans = []
        number_of_machines_to_be_added = self.agent_params[kc.NEW_MACHINES_AFTER_MUTATION]

        for i in range(len(self.human_agents)):
            if len(every_two_humans) >= number_of_machines_to_be_added:
                break

            if i % 2 != 0 and i > 2: ## So that there are 3 human agents before all the AVs
                every_two_humans.append(self.human_agents[i])


        if len(every_two_humans) < number_of_machines_to_be_added:
            raise ValueError(
                f"Insufficient human agents for mutation. Required: {number_of_machines_to_be_added}, "
                f"Available: {len(every_two_humans)}.\n"
                f"Decrease the number of machines to be added after the mutation.\n"
            )

        for human in every_two_humans:
                self.human_agents.remove(human)

                random_humans_deleted.append(human)
                self.machine_agents.append(MachineAgent(human.id,
                                                        human.start_time,
                                                        human.origin, 
                                                        human.destination, 
                                                        human.income, 
                                                        self.agent_params[kc.MACHINE_PARAMETERS], 
                                                        self.simulation_params[kc.NUMBER_OF_PATHS]))
                self.possible_agents.append(str(human.id))

        self.n_agents = len(self.possible_agents)
        self.all_agents = self.machine_agents + self.human_agents
        self.marginal_cost_machine_agents_flag() # Initialize marginal cost flag

        if disable_human_learning:  self.human_learning = False

        logging.info(f"Now there are {len(self.human_agents)} human agents.")

        self._initialize_machine_agents()
    
    """def stackelberg_auction(self, machine, machine_action) -> int:
        
        for route in range(self.simulation_params[kc.NUMBER_OF_PATHS]):
            submitted_bid = machine_action[route]

            if submitted_bid >= self.environment_params[kc.CENTRALLY_DEFINED_PRICE]:
                assigned_route = route
                
                machine.karma_balance -= submitted_bid

                self.total_karma_points_used += submitted_bid

                return assigned_route
   
        # get assigned to the longest route
        assigned_route = self.simulation_params[kc.NUMBER_OF_PATHS] - 1
        
        return assigned_route"""
    
    def stackelberg_auction(self, machine, machine_action) -> int:
        price_route_0 = float(self.environment_params[kc.CENTRALLY_DEFINED_PRICE_ROUTE_0])
        price_route_1 = float(self.environment_params[kc.CENTRALLY_DEFINED_PRICE_ROUTE_1])

        floor_price_route_0 = math.floor(price_route_0)
        decimal_part_route_0 = price_route_0 - floor_price_route_0  # e.g. 0.3 if price=4.3

        floor_price_route_1 = math.floor(price_route_1)
        decimal_part_route_1 = price_route_1 - floor_price_route_1  # e.g. 0.3 if price=4.3

        for route in range(self.simulation_params[kc.NUMBER_OF_PATHS]):
            if route == 0:

                bid = int(machine_action[route])

                # --- probability rule ---
                if bid < floor_price_route_0:
                    win_prob = 0.0

                elif bid == floor_price_route_0:
                    # probability = decimal part of price
                    # (if price is integer, this becomes 1.0)
                    win_prob = decimal_part_route_0 if decimal_part_route_0 > 0 else 1.0

                else:  # bid >= ceil_price
                    win_prob = 1.0

                # --- lottery draw ---
                if random.random() < win_prob:
                    machine.karma_balance -= bid
                    self.total_karma_points_used += bid
                    machine.route = route
                    return route
                
            if route == 1:
                bid = int(machine_action[route])

                # --- probability rule ---
                if bid < floor_price_route_1:
                    win_prob = 0.0

                elif bid == floor_price_route_1:
                    # probability = decimal part of price
                    # (if price is integer, this becomes 1.0)
                    win_prob = decimal_part_route_1 if decimal_part_route_1 > 0 else 1.0

                else:  # bid >= ceil_price
                    win_prob = 1.0

                # --- lottery draw ---
                if random.random() < win_prob:
                    machine.karma_balance -= bid
                    self.total_karma_points_used += bid
                    machine.route = route
                    return route

        # fallback: longest route
        machine.route = self.simulation_params[kc.NUMBER_OF_PATHS] - 1

        return self.simulation_params[kc.NUMBER_OF_PATHS] - 1

    #########################
    ##### Help functions ####
    #########################

    def get_observation(self) -> tuple:
        """Retrieve the current observation from the simulator.

        This method returns the current timestep of the simulation and the values of the episode actions.

        Returns:
            tuple: A tuple containing the current timestep and the episode actions.
        """

        return self.simulator.timestep, self.episode_actions.values()
    
    def all_high_urgency(self):
        self.all_agents_high_urgency = True

    def testing_urgency_function(self):
        self.testing_urgency = True
    

    def _help_step(self, actions: list[tuple]) -> dict:

        for agent, action in actions:

            obs_val = np.atleast_1d(getattr(self, 'observations', {}).get(str(agent.id), [])).ravel().tolist()

            action_dict = {kc.AGENT_ID: agent.id,
                           kc.AGENT_KIND: agent.kind,
                           kc.ACTION: action,
                           kc.BID_ROUTE_0: agent.bid[0],
                           kc.BID_ROUTE_1: agent.bid[1],
                           kc.BID_ROUTE_2: agent.bid[2],
                           kc.KARMA_BALANCE: agent.karma_balance,
                           kc.AGENT_ORIGIN: agent.origin,
                           kc.AGENT_DESTINATION: agent.destination,
                           kc.AGENT_START_TIME: agent.start_time,
                           kc.INCOME: agent.income,
                           kc.URGENCY: agent.urgency,
            }
            self.simulator.add_vehicle(action_dict)
            self.episode_actions[agent.id] = action_dict
        timestep, stopped_vehicles_info, arrivals, teleported = self.simulator.step()

        if self.save_detectors_info == True:
            self._save_detectors_info(stopped_vehicles_info)

        travel_times = dict()
        for veh_id in arrivals:
            if veh_id not in teleported:
                agent_id = int(veh_id)
                travel_times[agent_id] = ({kc.TRAVEL_TIME:
                                            (timestep - self.episode_actions[agent_id][kc.AGENT_START_TIME]) / 60.0})
                travel_times[agent_id].update(self.episode_actions[agent_id])
            
        for veh_id in teleported:
            agent_id = int(veh_id)
            travel_times[agent_id] = ({kc.TRAVEL_TIME: self.simulator.simulation_length / 60.0})
            travel_times[agent_id].update(self.episode_actions[agent_id])

        return travel_times.values()
    
    def _save_detectors_info(self, stopped_vehicles_info):
        folder = self.plotter_params[kc.RECORDS_FOLDER] + '/' + kc.DETECTOR_STOPPED_VEHICLES
        os.makedirs(folder, exist_ok=True)

        if (self.simulator.timestep == 1):
             [os.remove(f) for f in glob.glob(f"{folder}/*.csv")]

        csv_file_path = f"{folder}/stopped_vehicles{self.simulator.timestep - 1}.csv"

        df = pd.DataFrame(stopped_vehicles_info, columns=["time", "detector", "vehicle_id"])
        df.to_csv(csv_file_path, index=False)

    def _reset_episode(self) -> None:
        detectors_dict, self.sumo_seed = self.simulator.reset()

        if (self.day % self.save_every == 0) and (self.second_sumo == False): #In the case where we compute the marginal cost matrix we do not need to record. 
            """record_day = dc(self.day)
            record_travel_times = dc(self.travel_times_list)
            record_agents = dc(self.all_agents)
            record_detectors = dc(detectors_dict)

            recording_task = threading.Thread(
                target=self._record,
                args=(record_day, record_travel_times, record_agents, record_detectors)
            )
            recording_task.start()"""
            record_day = dc(self.day)
            record_travel_times = dc(self.travel_times_list)
            record_agents = dc(self.all_agents)
            record_detectors = dc(detectors_dict)

            self._record(
                record_day,
                record_travel_times,
                record_agents,
                record_detectors,
            )
            #self._record(self.day, self.travel_times_list, self.all_agents, detectors_dict)
        
        # Reset observations
        if len(self.machine_agents) > 0:
            self.observations = self.observation_obj.reset_observation()
            #self._agent_selector = agent_selector(self.possible_agents)
            #self.agent_selection = self._agent_selector.next()
        
        self.historic_data.append(self.travel_times_list)

        MAX_LEN = 20

        if len(self.historic_data) > MAX_LEN:
            del self.historic_data[:10]

        travel_time_by_id = {
            int(record["id"]): float(record["travel_time"])
            for record in self.travel_times_list
        }

        for agent in self.all_agents:
            if agent.id in travel_time_by_id:
                agent.reward = -travel_time_by_id[agent.id]

        """Change the start times of the vehicles"""
        for machine in self.machine_agents:
            self._assign_start_time_to_an_agent(machine)

        self._prepare_urgency_for_day()

        machine_by_id = {str(machine.id): machine for machine in self.machine_agents}

        self.possible_agents = sorted(
            self.possible_agents,
            key=lambda agent_id: machine_by_id[agent_id].start_time
        )

        if self.possible_agents:
            self._agent_selector = agent_selector(self.possible_agents)
            self.agent_selection = self._agent_selector.next()

        self.agents = copy(self.possible_agents)

        self.travel_times_list = []
        self.episode_actions = dict()


    def _assign_rewards(self) -> None:

        for agent in self.all_agents:
            if agent.kind == 'Human':
                reward = agent.get_reward(self.travel_times_list)
            else:
                reward = agent.get_reward(self.travel_times_list, group_vicinity=self.agent_params[kc.MACHINE_PARAMETERS][kc.GROUP_VICINITY])

            # Add the reward in the travel_times_list
            for agent_entry in self.travel_times_list:
                if agent.id == agent_entry[kc.AGENT_ID]:
                    self.travel_times_list.remove(agent_entry)
                    agent_entry[kc.REWARD] = reward
                    self.travel_times_list.append(agent_entry)

            # Save machine's rewards based on PettingZoo standards
            if agent.kind == 'AV':
                self.rewards[str(agent.id)] = reward

            # Human learning
            elif self.human_learning:
                agent.learn(agent.last_action, self.travel_times_list)

    """def _assign_urgency_level_to_an_agent(self, machine) -> None:
        machine.urgency = np.random.choice(self.urgency_distribution)"""

    def _prepare_urgency_for_day(self) -> None:
        """
        Sample the urgency-generation process for the current day.
        Call this once at the beginning of every simulated day.
        """

        max_structured_probability = 1

        # Random proportion of commuters with correlated urgency
        self._p_correlated = np.random.uniform(
            0.0,
            max_structured_probability
        )

        # Random proportion with temporal dependence
        # Ensures p_correlated + p_temporal <= 0.6
        self._p_temporal = np.random.uniform(
            0.0,
            max_structured_probability - self._p_correlated
        )

        # Remaining commuters have independently sampled urgency
        self._p_independent = (
            1.0
            - self._p_correlated
            - self._p_temporal
        )

        # Common urgency for commuters belonging to the
        # correlated group on this day
        self._common_daily_urgency = np.random.choice(
            self.urgency_distribution
        )

        # Temporal relationship also changes between days
        self._temporal_rule = np.random.choice(
            ["same", "increase", "decrease"]
        )

    def _assign_urgency_level_to_an_agent(self, machine) -> None:

        previous_urgency = getattr(machine, "urgency", None)

        mechanism = np.random.choice(
            ["independent", "correlated", "temporal"],
            p=[
                self._p_independent,
                self._p_correlated,
                self._p_temporal,
            ],
        )

        # 1. Independent urgency
        if mechanism == "independent":
            machine.urgency = np.random.choice(
                self.urgency_distribution
            )

        # 2. Correlated with other commuters
        elif mechanism == "correlated":
            machine.urgency = self._common_daily_urgency

        # 3. Depends on previous-day urgency
        elif mechanism == "temporal":

            # No previous observation available
            if previous_urgency is None:
                machine.urgency = np.random.choice(
                    self.urgency_distribution
                )
                return

            levels = np.sort(
                np.unique(np.asarray(self.urgency_distribution))
            )

            # Find the current urgency level
            idx = np.argmin(
                np.abs(levels - previous_urgency)
            )

            if self._temporal_rule == "same":
                machine.urgency = previous_urgency

            elif self._temporal_rule == "increase":
                machine.urgency = levels[
                    min(idx + 1, len(levels) - 1)
                ]

            elif self._temporal_rule == "decrease":
                machine.urgency = levels[
                    max(idx - 1, 0)
                ]

        
    def _assign_start_time_to_an_agent(self, machine) -> None:
        machine.start_time = random.randint(0, self.simulation_params[kc.SIMULATION_TIMESTEPS])

    def _redistribute_karma_points(self) -> None:
        ## Distributed karma points
        new_distributed_karma_points = self.total_karma_points_used // len(self.machine_agents)

        ## Rest of the Karma points 
        remainder_karma_points = self.total_karma_points_used % len(self.machine_agents)

        # Give equal base share to all agents
        for machine in self.machine_agents:
            machine.karma_balance += new_distributed_karma_points

        # Randomly choose 'remainder_karma_points' agents to receive +1
        if remainder_karma_points > 0:
            lucky_agents = random.sample(self.machine_agents, remainder_karma_points)
            for machine in lucky_agents:
                machine.karma_balance += 1

        self.total_karma_points_used = 0

    def _reinitialize_karma_balance(self) -> None:

        for machine in self.machine_agents:
                machine.karma_balance = self.environment_params[kc.MAXIMUM_ALLOWED_BID]

    def make_multidiscrete_mask(self, karma_balance: int, action_space: MultiDiscrete)  ->np.ndarray:
        """
        Creates a mask for a MultiDiscrete action space where the sum
        of action components cannot exceed available karma_balance.

        Args:
            karma_balance (int): available karma_balance
            action_space (MultiDiscrete): e.g., MultiDiscrete([10, 10, 10])

        Returns:
            np.ndarray: shape (product of action dims,), where
                        1 = valid, 0 = invalid
        """
        # Extract per-dimension sizes (e.g., [10, 10, 10])
        dims = action_space.nvec

        # All possible actions: cartesian product
        all_actions = itertools.product(*[range(n) for n in dims])

        mask = np.ones(np.prod(dims), dtype=np.int32)

        for i, action in enumerate(all_actions):

            # Check the bids of each MultiDiscrete action
            for bid in action:
                if bid > karma_balance: # If at least one of the potential bids has higher value than the karma balance -> invalid action 
                    mask[i] = 0

        return mask

    ###################################
    #### Marginal cost calculation ####
    ###################################


    def marginal_cost(self) -> None:
        """Calculate marginal cost matrix"""

        # Save the marginal cost matrices
        if self.marginal_cost_calculation == True and self.machine_agents:
            marginal_cost_calculation = {}

            for machine in self.machine_agents:
                continue

            marginal_cost_calculation = machine.calculate_marginal_cost(self.all_agents, self.travel_times_list, self.sumo_seed, self.marginal_cost_calculation_machine_to_all, self.kwargs)
            self.recorder.remember_marginal_costs(marginal_cost_calculation, self.day)

    def marginal_cost_machine_agents_flag(self) -> None:
        """Enable marginal cost in agent's reward"""

        if self.marginal_cost_calculation == True:
            for agent in self.machine_agents:
                agent.marginal_calculation = True


    def monetary_pricing_flag(self) -> None:
        """Enable monetary cost in agent's reward"""

        if self.monetary_pricing == True:
            for agent in self.all_agents:
                agent.monetary_pricing = True


    ###########################
    ##### Simulation loop #####
    ###########################

    def simulation_loop(self, machine_action: int, machine_id: int) -> None:
        """This function contains the integration of the agent's actions to SUMO.

        We iterate through all the time steps of the simulation.
        For each timestep there are none, one or more than one agents type (humans, machines) that start.
        If more than one machine agents have the same start time, we break from this function because
        we need to take the agent's action from the STEP function.

        Args:
            machine_action (int): The id of the machine agent whose action is to be performed.
            machine_id (int): The id of the machine agent whose action is to be performed.
            
        Returns:
            None
        """
        agent_action = False
        while (
                self.simulator.timestep < self.simulation_params[kc.SIMULATION_TIMESTEPS]
                or len(self.travel_times_list) < len(self.all_agents)
        ):

            # If there are more than one machines with the same start time
            # the humans should act once
            if not self.actions_timestep:
                for human in self.human_agents:
                    if human.start_time == self.simulator.timestep:
                        action = human.act(0)
                        self.actions_timestep.append((human, action))

            for machine in self.machine_agents:
                if machine.start_time == self.simulator.timestep:

                    # In case there are machine agents that have the same start time, but it's not their turn
                    if str(machine.id) != machine_id:

                        # If some machines have the same start time, and they haven't acted yet
                        if (
                                (machine not in self.machine_same_start_time)
                                and not any(machine == item[0] for item in self.actions_timestep)
                        ):
                            self.machine_same_start_time.append(machine)
                        continue
                    else:
                        # Machine acting
                        machine.last_action = machine_action
                        self.actions_timestep.append((machine, machine_action))

                        # The machine acted should be deleted from the self.machine_same_start_time list
                        if machine in self.machine_same_start_time:
                            self.machine_same_start_time.remove(machine)

                        # If the machine isn't the last agent to act then we need to step again for the next agent
                        if not self._agent_selector.is_last():
                            agent_action = True

            # If all machines that have start time as the simulator timestep acted
            if not self.machine_same_start_time:
                travel_times = self._help_step(self.actions_timestep)

                for agent_dict in travel_times:
                    self.travel_times_list.append(agent_dict)

                self.actions_timestep = []
                self.machine_same_start_time = []

            # If the machine agent that had turn acted
            if agent_action:
                agent_action = False
                break

    ###########################
    ##### Free flow times #####
    ###########################

    def get_free_flow_times(self) -> dict:
        """Retrieve free flow times for all origin-destination pairs from the simulator paths data.

        Returns:
            ff_dict (dict): A dictionary where keys are tuples of origin and destination,
                            and values are lists of free flow times.
        """

        paths_df = pd.read_csv(self.simulator.paths_csv_file_path)
        origins = paths_df[kc.ORIGINS].unique()
        destinations = paths_df[kc.DESTINATIONS].unique()
        ff_dict = {(o, d): list() for o in origins for d in destinations}

        for _, row in paths_df.iterrows():
            ff_dict[(row[kc.ORIGINS], row[kc.DESTINATIONS])].append(row[kc.FREE_FLOW_TIME])

        return ff_dict


    ############################
    ##### Disc operations ######
    ############################

    def _record(self, episode: int, ep_observations: dict, agents: list, detectors_dict: dict) -> None:

        dc_episode, dc_ep_observations, dc_agents, dc_detectors = dc(episode), dc(ep_observations), dc(agents), dc(detectors_dict)
        zero_space = [0] * self.action_space_size

        cost_tables = [
            {
                kc.AGENT_ID: agent.id,
                kc.COST_TABLE: getattr(agent.model, 'costs', zero_space) if hasattr(agent, 'model') else zero_space,
            }
            for agent in dc_agents
        ]
        
        observations = [
            {
                kc.AGENT_ID: agent.id,
                kc.OBSERVATION: np.atleast_1d(getattr(self, 'observations', {}).get(str(agent.id), [])).ravel().tolist()
            }
            for agent in dc_agents
        ]

        if self.recorder != None:
            self.recorder.record(dc_episode, dc_ep_observations, observations, cost_tables, dc_detectors)

    def plot_results(self) -> None:
        """Method that plot the results of the simulation.

        Returns:
            None
        """

        plotter(self.plotter_params)

    ############################
    ### PettingZoo functions ###
    ############################

    def render(self) -> None:
        pass

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        """Method that returns the observation space of the agent.

        Args:
            agent (str): The agent name.
        Returns:
            self._observation_spaces[agent] (Any): The observation space of the agent.
        """

        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        """Method that returns the action space of the agent.

        Args:
            agent (str): The agent name.
        Returns:
            self._action_spaces[agent] (Any): The action space of the agent.
        """

        return self._action_spaces[agent]

    #####################################################
    ### Decide on the observation function to be used ###
    #####################################################

    def get_observation_function(self) -> Observations:
        """Returns an observation object based on the provided parameters.

        Returns:
            Observations: An observation object.
        Raises:
            ValueError: If model is unknown.
        """

        params = self.agent_params[kc.MACHINE_PARAMETERS]
        observation_type = params[kc.OBSERVATION_TYPE]
        if observation_type == kc.PREVIOUS_AGENTS_PLUS_START_TIME:
            return PreviousAgentStartPlusStartTime(self.machine_agents,
                                                   self.human_agents,
                                                   self.simulation_params,
                                                   self.agent_params,
                                                   self.environment_params[kc.OBSERVATIONS_TIME_WINDOW])
        elif observation_type == kc.PREVIOUS_AVERAGE_TT_PER_ROUTE:
            return PreviousAvgTTperRoute(self.machine_agents,
                                                   self.human_agents,
                                                   self.simulation_params,
                                                   self.agent_params,
                                                   self.environment_params[kc.OBSERVATIONS_TIME_WINDOW])

        elif observation_type == kc.PREVIOUS_AGENTS:
            return PreviousAgentStart(self.machine_agents,
                                      self.human_agents,
                                      self.simulation_params,
                                      self.agent_params)
        elif observation_type == kc.PREVIOUS_AGENTS_PLUS_START_TIME_DETECTOR_DATA:
            if self.save_detectors_info == False:
                raise Exception("Detector info saving is disabled. Please set 'self.save_detectors_info = True' to proceed or change the observation type.")
            
            return PreviousAgentStartPlusStartTimeDetectorData(self.machine_agents,
                                      self.human_agents,
                                      self.simulation_params,
                                      self.plotter_params,
                                      self.agent_params,
                                      self.simulator)
        elif observation_type == kc.PREVIOUS_AGENTS_PLUS_START_TIME_MARGINAL_COST:
            return PreviousAgentStartPlusStartTimeMarginalCost(self.machine_agents,
                                      self.human_agents,
                                      self.simulation_params,
                                      self.agent_params)
        else:
            raise ValueError('[MODEL INVALID] Unrecognized observation type: ' + observation_type)
