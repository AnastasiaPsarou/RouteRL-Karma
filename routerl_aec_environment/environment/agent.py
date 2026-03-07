"""
This module contains the human and machine agent classes, which represent vehicles driving
from an origin to a destination in the simulation.
"""
from collections import defaultdict
import numpy as np
import os
import pandas as pd
import re
import torch

from abc import ABC, abstractmethod

from routerl_aec_environment.keychain import Keychain as kc
from routerl_aec_environment.human_learning import get_learning_model


class BaseAgent(ABC):
    """This is the abstract base class for the human and machine agent classes.

    Args:
        id (int): 
            The id of the agent.
        kind (str): 
            The kind of the agent (Human or AV).
        start_time (float): 
            The start time of the agent.
        origin (float): 
            The origin of the agent.
        destination (float): 
            The destination value of the agent.
        behavior (float): 
            The behavior of the agent.
    """

    def __init__(self, id, kind, start_time, origin, destination, behavior, income):
        self.id = id
        self.kind = kind
        self.start_time = start_time
        self.origin = origin
        self.destination = destination
        self.behavior = behavior
        self.last_action = 0
        self.default_action = None
        self.income = income 
        
        self.actions = []
        self.observations = []
        self.rewards = []

    @property
    @abstractmethod
    def last_reward(self) -> None:
        """Return the last reward of the agent.

        Returns:
            None
        """

        pass
    
    @last_reward.setter
    @abstractmethod
    def last_reward(self, reward) -> None:
        """Set the last reward of the agent.

        Args:
            reward (float): The reward of the agent.
        Returns:
            None
        """

        pass

    @abstractmethod
    def act(self, observation) -> None:
        """Pick action according to your knowledge, or randomly.

        Args:
            observation (float): The observation of the agent.
        Returns:
            None
        """

        pass

    @abstractmethod
    def learn(self, action, observation) -> None:
        """Pass the applied action and reward once the episode ends, and it will remember the consequences.

        Args:
            action (int): The action taken by the agent.
            observation (float): The observation of the agent.
        Returns:
            None
        """

        pass

    @abstractmethod
    def get_state(self, observation) -> None:
        """Return the state of the agent, given the observation

        Args:
            observation (float): The observation of the agent.
        Returns:
            None
        """

        pass

    @abstractmethod
    def get_reward(self, observation) -> None:
        """Derive the reward of the agent, given the observation

        Args:
            observation (float): The observation of the agent.
        Returns:
            None
        """

        pass


class HumanAgent(BaseAgent):
    """Class representing human drivers, responsible for modeling their learning process
    and decision-making in route selection.

    Args:
        id (int):
            The id of the agent.
        start_time (float):
            The start time of the agent.
        origin (float):
            The origin of the agent.
        destination (float):
            The destination value of the agent.
        params (dict):
            The parameters for the learning model of the agent as specified in `here <https://coexistence-project.github.io/RouteRL/documentation/pz_env.html#>`_.
        initial_knowledge (float):
            The initial knowledge of the agent.
    """

    def __init__(self, id, start_time, origin, destination, income, params, initial_knowledge):
        kind = kc.TYPE_HUMAN
        behavior = kc.SELFISH
        super().__init__(id, kind, start_time, origin, destination, behavior, income)
        self.initial_knowledge = initial_knowledge
        self.model = get_learning_model(params, initial_knowledge)
        self.last_reward = None
        self.last_obs = 0
        self.monetary_pricing = None

    def __repr__(self):
        return f"Human {self.id}"

    @property
    def last_reward(self) -> float:
        """Set the last reward of the agent.

        Returns:
            float: The last reward of the agent.
        """

        return self._last_reward

    @last_reward.setter
    def last_reward(self, reward) -> None:
        """Set the last reward of the agent.

        Args:
            reward (float): The reward of the agent.
        Returns:
            None
        """
        self._last_reward = reward

    def act(self, observation) -> int:  
        """Returns the agent's action (route of choice) based on the current observation from the environment.

        Args:
            observation (list): The observation of the agent.
        Returns:
            int: The action of the agent.
        """
        if self.default_action is not None:
            self.last_action = self.default_action
            return self.default_action
        self.last_action = self.model.act(observation)
        self.last_obs = observation
        return self.last_action

    def learn(self, action, observation) -> None:
        """Updates the agent's knowledge based on the action taken and the resulting observations.

        Args:
            action (int): The action of the agent.
            observation (list[dict]): The observation of the agent.
        Returns:
            None
        """
        reward = self.get_reward(observation)
        self.last_reward = reward
        self.model.learn(self.last_obs, action, reward)

    def get_state(self, _) -> None:
        """Returns the current state of the agent.

        Args:
            _ (Any): The current state of the agent.
        Returns:
            None
        """
        return None

    def get_reward(self, observation: list[dict]) -> float:
        """This function calculated the reward of each individual agent.

        Args:
            observation (list[dict]): The observation of the agent.
        Returns:
            float: Own travel time of the agent.
        """

        own_tt = -1 * next(obs[kc.TRAVEL_TIME] for obs in observation if obs[kc.AGENT_ID] == self.id)
        return own_tt
    

class MachineAgent(BaseAgent):
    """A class that models Autonomous Vehicles (AVs), focusing on their learning mechanisms
    and decision-making processes for selecting optimal routes.

    Args:
        id (int): 
            The id of the agent.
        start_time (float): 
            The start time of the agent.
        origin (float): 
            The origin of the agent.
        destination (float): 
            The destination value of the agent.
        params (dict): 
            The parameters of the machine agent as specified in `here <https://coexistence-project.github.io/RouteRL/documentation/pz_env.html#>`_.
        action_space_size (int): 
            The size of the action space of the agent.
    """

    def __init__(self, id, start_time, origin, destination, income, params, action_space_size):#, marginal_cost_beta):
        kind = kc.TYPE_MACHINE
        behavior = params[kc.BEHAVIOR]
        super().__init__(id, kind, start_time, origin, destination, behavior, income)
        self.observed_span = params[kc.OBSERVED_SPAN]
        self.params = params
        self.action_space_size = action_space_size
        self.state_size = action_space_size * 2
        self.model = None
        self.last_reward = None
        self.rewards_coefs = self._get_reward_coefs()
        self.marginal_calculation = False
        self.monetary_pricing = None
        self.route_0_fee = params[kc.ROUTE_0_FEE]

        self.w1 = params[kc.REWARD_W1]
        self.w2 = params[kc.REWARD_W2]
        self.w3 = params[kc.REWARD_W3]

        self.travel_time_normalization_value = params[kc.TRAVEL_TIME_NORMALIZATION_VALUE]

        self.urgency = None
        #self.marginal_cost_beta = marginal_cost_beta

    def __repr__(self) -> str:
        machine_id = f"Machine {self.id}"

        return machine_id

    @property
    def last_reward(self) -> float:
        """Set the last reward of the agent.

        Returns:
            float: The last reward of the agent.
        """

        return self._last_reward

    @last_reward.setter
    def last_reward(self, reward) -> None:
        """Sets the last reward of the agent.

        Args:
            reward (float): The reward of the agent.
        Returns:
            None
        """

        self._last_reward = reward

    def act(self, _) -> None:
        """**Deprecated**

        Args:
            _ (Any): The current state of the agent.
        Returns:
            None
        """

        return None

    def learn(self, _) -> None:
        """**Deprecated**
        
        Args:
            _ (Any): The current state of the agent.
            
        Returns:
            None
        """

        return None
    
    def get_travel_time_by_id(self, travel_times_list, agent_id):
        for entry in travel_times_list:
            if entry['id'] == agent_id:
                #print("In get travel time by id", entry['id'], agent_id, "entry tt is: ", entry['travel_time'], "\n")
                return entry['travel_time']
        return None  
    
    def get_action_by_id(self, travel_times_list, agent_id):
        for entry in travel_times_list:
            if entry['id'] == agent_id:
                return entry['action']
        return None 
    
    def calculate_marginal_cost(self, all_agents, travel_times_list, sumo_seed, machines_to_all, kwargs):
        from .environment import TrafficEnvironment ## added here because there was circular import problem

        # Marginal cost
        marginal_cost_calculation = defaultdict(dict)
        
        # Calculate the marginal cost on the agent from other AV agents
        agents_to_calculate_marginal_cost = []
        for agent in all_agents:
            if agent.kind == kc.TYPE_MACHINE:
                agents_to_calculate_marginal_cost.append(agent)
        
        ## Delete each agent from the environment
        for machine_agent in agents_to_calculate_marginal_cost:

            # Read the agents already in the simulation
            df = pd.read_csv(os.path.join(self.params[kc.RECORDS_FOLDER], self.params[kc.AGENTS_CSV_FILE_NAME]))

            # Delete from the agents list the specific chosen agent
            df["id"] = df["id"].astype(int)
            df = df[df["id"] != int(machine_agent.id)]

            # Save the new agent list for the new environment run
            df.to_csv(os.path.join(self.params[kc.RECORDS_FOLDER], "agents2.csv"), index=False)

            # Save the actions of the agents that will run in the new simulation
            actions = []
            for index, row in df.iterrows():
                for agent in all_agents: ## all_agents doesn't have the correct actions of the agents
                    if row['id'] == agent.id:
                        actions.append(int(agent.last_action))
                    
            ## Pass the same argument with the difference that the agent data is in agents2.csv
            kwargs["agent_parameters"]["agents_csv_file_name"] = "agents2.csv"   
            #kwargs["simulator_parameters"]["sumo_type"] = "sumo-gui" 
            env = TrafficEnvironment(seed=sumo_seed, create_agents=False, create_paths=False, second_sumo=True, **kwargs)
            
            env.start(use_subprocess=True)
            for agent, action in zip(env.all_agents, actions):
                agent.default_action = action

            env.step()

            for agent in all_agents:
                if machines_to_all == False: #whether to calculate the impact of deleting a machine agent on the other machine agents
                                             #or on human agents as well
                    if agent.id == machine_agent.id or agent.kind == kc.TYPE_HUMAN:
                        if agent.kind != kc.TYPE_HUMAN:
                            marginal_cost_calculation[agent][agent.id] = 0.0
                        continue

                after_step_time = self.get_travel_time_by_id(env.travel_times_list, agent.id)

                initial_time = self.get_travel_time_by_id(travel_times_list, agent.id)

                if initial_time is not None and after_step_time is not None:
                    difference = after_step_time - initial_time
                    marginal_cost_calculation[agent][machine_agent.id] = difference
                else:
                    marginal_cost_calculation[agent][machine_agent.id] = 0.0

            env.stop_simulation() 

        return marginal_cost_calculation


    def get_state(self, observation: list[dict]) -> list[int]:
        """Generates the current state representation based on recent observations of agents navigating
        from the same origin to the same destination.

        Args:
            observation (list[dict]): The recent observations of the agent.
            
        Returns:
            list[int]: The current state representation.
        """

        min_start_time = self.start_time - self.observed_span
        human_prior, machine_prior = list(), list()
        for obs in observation:
            if ((obs[kc.AGENT_ORIGIN], obs[kc.AGENT_DESTINATION]) == (self.origin, self.destination)
                    and (obs[kc.AGENT_START_TIME] > min_start_time)
            ):
                if obs[kc.AGENT_KIND] == kc.TYPE_HUMAN:
                    human_prior.append(obs)
                elif obs[kc.AGENT_KIND] == kc.TYPE_MACHINE:
                    machine_prior.append(obs)

        warmth_human = [0] * (self.state_size // 2)
        warmth_machine = [0] * (self.state_size // 2)
        
        if human_prior:
            for row in human_prior:
                action = row[kc.ACTION]
                start_time = row[kc.AGENT_START_TIME]
                warmth = start_time - min_start_time
                warmth_human[action] += warmth
                
        if machine_prior:
            for row in machine_prior:
                action = row[kc.ACTION]
                start_time = row[kc.AGENT_START_TIME]
                warmth = start_time - min_start_time
                warmth_machine[action] += warmth

        warmth_agents = warmth_human + warmth_machine
        return warmth_agents
    
    def include_impact_in_reward(self) -> float:
        #marginal_folder = r"C:\Users\Anastasia\Documents\RouteRL_exps_benchmark\RouteRL\tutorials\two_route_net\training_records\marginal_cost_matrices"
        marginal_folder = self.params[kc.RECORDS_FOLDER] + '/' + self.params[kc.MARGINAL_MATRICES_FOLDER]
        
        csv_files = [f for f in os.listdir(marginal_folder) if f.endswith('.csv')]
    
        # Extract numbers at the end of filenames
        file_numbers = []
        for f in csv_files:
            match = re.search(r'(\d+)(?=\.csv$)', f)
            if match:
                file_numbers.append((f, int(match.group(1))))

            # Find file with the highest number
        highest_file = max(file_numbers, key=lambda x: x[1])[0]
        
        # Full path to the highest numbered file
        highest_file_path = os.path.join(marginal_folder, highest_file)
        
        # Read the CSV
        df = pd.read_csv(highest_file_path)

        machine_name = f"Machine {self.id}"

        machine_name = f"Machine {self.id}"
        if machine_name not in df.columns:
            raise ValueError(f"No column found for {machine_name}")

        col_values = df[machine_name]
        total_impact = col_values.sum()

        return total_impact


    def get_reward(self, observation: list[dict], group_vicinity: bool = False) -> float:
        """This method calculated the reward of each individual agent, based on the travel time of the agent,
        the group of agents, the other agents, and all agents, weighted according to the agent's behavior.

        Args:
            observation (list[dict]): The current observation of the agent.
                
        Returns:
            float: The reward of the agent.
        """
        vicinity_obs = list()
        if group_vicinity == True:

            min_start_time, max_start_time = self.start_time - self.observed_span, self.start_time + self.observed_span

            for obs in observation:
                if (obs[kc.AGENT_ORIGIN], obs[kc.AGENT_DESTINATION]) == (self.origin, self.destination):
                    if min_start_time <= obs[kc.AGENT_START_TIME] <= max_start_time:
                        vicinity_obs.append(obs)
        else:        
            for obs in observation:
                vicinity_obs.append(obs)

        group_obs, others_obs, all_obs, own_tt = list(), list(), list(), None
        for obs in vicinity_obs:
            all_obs.append(obs[kc.TRAVEL_TIME])
            if obs[kc.AGENT_KIND] == self.kind:
                group_obs.append(obs[kc.TRAVEL_TIME])
            else:
                others_obs.append(obs[kc.TRAVEL_TIME])
            if obs[kc.AGENT_ID] == self.id:
                own_tt = obs[kc.TRAVEL_TIME]
                
        group_tt = np.mean(group_obs) if group_obs else 0
        others_tt = np.mean(others_obs) if others_obs else 0
        all_tt = np.mean(all_obs) if all_obs else 0
        
        a, b, c, d = self.rewards_coefs
        agent_reward  = a * own_tt + b * group_tt + c * others_tt + d * all_tt

        if self.monetary_pricing:
            route_fee = self._calculate_monetary_reward(observation)
            route_length = self._compute_routes_length(observation)
            print("route length is: ", route_length, "\n\n")
            route_length_km = route_length / 1000
            daily_income = self.income / 30 # If we assume that each month has 30 days
            travel_times_norm = -1 * agent_reward / self.travel_time_normalization_value

            component1 = (route_fee / daily_income)
            component2 = travel_times_norm
            component3 = self.urgency

            fuel_fee = 0.094 # euros/km

            #print("component 1", component1, "component 2", component2, "component 3", component3, "\n\n")
            #agent_reward = self.w1 * component1  + self.w2 * component2 #* component3 #+ self.w3 * self.urgency
            agent_reward = self.w1 * route_fee + self.w2 * travel_times_norm * self.urgency * (daily_income / 8) + self.w3 * route_length_km * fuel_fee

        if self.marginal_calculation:
            total_impact = self.include_impact_in_reward()

            beta = self.params[kc.MARGINAL_COST_COEFFICIENT_BETA]

            tahned_impact = torch.tanh(torch.tensor(total_impact))
            agent_reward = agent_reward + beta * tahned_impact.numpy()

        return agent_reward

    def _calculate_monetary_reward(self, observation) -> tuple:
        agent_info = next((entry for entry in observation if entry["id"] == self.id), None)

        if agent_info['action'] == np.int64(0):
            route_fee = self.route_0_fee
        else:
            route_fee = 0

        return route_fee
    
    def _compute_routes_length(self, observation):
        agent_info = next((entry for entry in observation if entry["id"] == self.id), None)

        distance0 = 3423.24
        distance1 = 6403.34
        distance2 = 11603.34

        if agent_info['action'] == np.int64(0):
            distance = distance0
            print("distnace is", distance0)
        if agent_info['action'] == np.int64(1):
            distance = distance1
            print("distnace is", distance1)
        if agent_info['action'] == np.int64(2):
            distance = distance2
            print("distnace is", distance2)

        return distance


    def _get_reward_coefs(self) -> tuple:
        a, b, c, d = 0, 0, 0, 0
        if self.behavior == kc.SELFISH:
            a, b, c, d = -1, 0, 0, 0 
        elif self.behavior == kc.COMPETITIVE:
            a, b, c, d = -2, 0, 1, 0
        elif self.behavior == kc.COLLABORATIVE:
            a, b, c, d = -0.5, -0.5, 0, 0
        elif self.behavior == kc.COOPERATIVE:
            a, b, c, d = 0, -1, 0, 0
        elif self.behavior == kc.SOCIAL:
            a, b, c, d = -0.5, 0, 0, -0.5
        elif self.behavior == kc.ALTRUISTIC:
            a, b, c, d = 0, 0, 0, -1
        elif self.behavior == kc.MALICIOUS:
            a, b, c, d = 0, 0, 1, 0
        return a, b, c, d