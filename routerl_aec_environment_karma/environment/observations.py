"""
Observation functions for RL agents (AVs).
"""

from gymnasium.spaces import Box
from gymnasium import spaces
import numpy as np
from abc import ABC, abstractmethod
import os
import pandas as pd
from typing import List, Dict, Any, Union

from routerl_aec_environment_karma.keychain import Keychain as kc
from .simulator import SumoSimulator


class Observations(ABC):
    """Abstract base class for observation functions.

    Args:
        machine_agents_list (List[Any]): List of machine agents.
        human_agents_list (List[Any]): List of human agents.
    """

    def __init__(self, machine_agents_list: List[Any], human_agents_list: List[Any]) -> None:
        self.machine_agents_list = machine_agents_list
        self.human_agents_list = human_agents_list

    @abstractmethod
    def __call__(self, all_agents: List[Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def observation_space(self) -> Dict[str, Box]:
        """Define the observation space for the observation function.

        Returns:
            Dict[str, Box]: A dictionary where keys are agent IDs and values are Gym spaces.
        """

        pass


class PreviousAgentStart(Observations):
    """Observes the number of agents with the same origin-destination and start time within a threshold.

    Args:
        machine_agents_list (List[Any]): List of machine agents.
        human_agents_list (List[Any]): List of human agents.
        simulation_params (Dict[str, Any]): Simulation parameters.
        agent_params (Dict[str, Any]): Agent parameters.
    Attributes:
        observations (List[Any]): List of observations.
    """

    def __init__(
        self,
        machine_agents_list: List[Any],
        human_agents_list: List[Any],
        simulation_params: Dict[str, Any],
        agent_params: Dict[str, Any],
        ) -> None:

        super().__init__(machine_agents_list, human_agents_list)
        self.simulation_params = simulation_params
        self.agent_params = agent_params
        self.observations = self.reset_observation()

    def __call__(self, all_agents: List[Any]) -> Dict[str, Any]:
        for machine in self.machine_agents_list:
            observation = np.zeros(self.simulation_params[kc.NUMBER_OF_PATHS], dtype=int)

            for agent in all_agents:
                if (machine.id != agent.id and
                    machine.origin == agent.origin and
                    machine.destination == agent.destination and
                    abs(machine.start_time - agent.start_time) < machine.observed_span):
                    
                    observation[agent.last_action] += 1

            self.observations[str(machine.id)] = observation.tolist()

        return self.observations

    def reset_observation(self) -> Dict[str, np.ndarray]:
        """Reset observations to the initial state.

        Returns:
            Dict[str, np.ndarray]: A dictionary of initial observations for all machine agents.
        """

        return {
            str(agent.id): np.zeros(self.simulation_params[kc.NUMBER_OF_PATHS], dtype=np.float32)
            for agent in self.machine_agents_list
        }

    def observation_space(self) -> Dict[str, Box]:
        """Define the observation space for each machine agent.

        Returns:
            Dict[str, Box]: A dictionary where keys are agent IDs and values are Gym spaces.
        """

        return {
            str(agent.id): Box(
                low=0,
                high=len(self.human_agents_list) + len(self.machine_agents_list),
                shape=(self.simulation_params[kc.NUMBER_OF_PATHS],),
                dtype=np.float32
            )
            for agent in self.machine_agents_list
        }

    def agent_observations(self, agent_id: str, all_agents: List[Any]) -> np.ndarray:
        """Retrieve the observation for a specific agent.

        Args:
            agent_id (str): The ID of the agent.
        Returns:
            np.ndarray: The observation array for the specified agent.
        """
        for machine in self.machine_agents_list:
            if machine.id == int(agent_id):
                break
            
        observation = np.zeros(self.simulation_params[kc.NUMBER_OF_PATHS], dtype=np.int32)

        for agent in all_agents:
            if (machine.id != agent.id and
                machine.origin == agent.origin and
                machine.destination == agent.destination and
                machine.start_time > agent.start_time):
                
                observation[agent.last_action] += 1
        
        return observation


class PreviousAgentStartPlusStartTime(Observations):
    """Observes the number of agents with the same origin-destination and start time within a threshold
    and includes the start of the specific agent as well.
    """

    def __init__(
        self,
        machine_agents_list: List[Any],
        human_agents_list: List[Any],
        simulation_params: Dict[str, Any],
        agent_params: Dict[str, Any],
        observations_time_window: int
    ) -> None:
        """Initialize the observation function.

        Args:
            machine_agents_list (List[Any]): List of machine agents.
            human_agents_list (List[Any]): List of human agents.
            simulation_params (Dict[str, Any]): Dictionary of simulation parameters.
            agent_params (Dict[str, Any]): Dictionary of agent parameters.
        Returns:
            None
        """

        super().__init__(machine_agents_list, human_agents_list)
        self.simulation_params = simulation_params
        self.agent_params = agent_params
        self.observations = self.reset_observation()
        self.agent_vectors = {}
        self.time_window = observations_time_window

    def __call__(self, all_agents: List[Any]) -> Dict[str, Any]:
        """Generate observations for all agents.

        Args:
            all_agents (List[Any]): List of all agents.
        Returns:
            Dict[str, Any]: A dictionary of observations keyed by agent IDs.
        """

        """for machine in self.machine_agents_list:
            observation = np.zeros(self.simulation_params[kc.NUMBER_OF_PATHS], dtype=np.int32)

            for agent in all_agents:
                if (machine.id != agent.id and
                    machine.origin == agent.origin and
                    machine.destination == agent.destination and
                    machine.start_time > agent.start_time):
                    
                    observation[agent.last_action] += 1

            self.observations[str(machine.id)] = np.concatenate(
                [
                    np.array([machine.start_time], dtype=np.int32),  # Start time as scalar
                    observation  # Vector of observations
                ]
            )"""

        return self.observations

    def reset_observation(self) -> Dict[str, np.ndarray]:
        """Reset observations to the initial state.

        Returns:
            obs (Dict[str, np.ndarray]): A dictionary of initial observations for all machine agents.
        """
        # Initialize agent vectors as zero arrays
        self.agent_vectors = {
            agent: np.zeros(self.simulation_params[kc.NUMBER_OF_PATHS], dtype=np.int32)
            for agent in self.machine_agents_list
        }
        

        obs = {
            str(agent.id): {
                "observation": np.concatenate(  # Combine start_time and vector into a single array
                    [
                        self.agent_vectors[agent],  # Vector as array
                        np.array([0], dtype=np.int32),  # Urgency as scalar
                        np.array([agent.karma_balance], dtype=np.int32) # Karma points
                    ]
                ),
                "action_mask": np.ones(pow(self.agent_params[kc.MACHINE_PARAMETERS][kc.MAXIMUM_ALLOWED_BID], 3), "int8"),
                }
            for agent in self.machine_agents_list
        }

        self.observations = obs

        return obs

    def observation_space(self) -> Dict[str, Box]:
        """
        Define the observation space for each machine agent.

        Returns:
            Dict[str, Box]: A dictionary where keys are agent IDs and values are Gym spaces.
        """

        total_size = 2 + self.simulation_params[kc.NUMBER_OF_PATHS] # including urgency & start time

        observation_spaces = {
            str(agent.id): spaces.Dict(
                {
                    "observation": Box(
                        low=0, high=np.inf, shape=(total_size,),  dtype=np.float32
                    ),
                    "action_mask": Box(
                        low=0, high=1, shape=(self.agent_params[kc.MACHINE_PARAMETERS][kc.MAXIMUM_ALLOWED_BID],), dtype=np.int8
                    ),
                }
            )
            for agent in self.machine_agents_list
        }

        return observation_spaces
    
    def agent_observations(self, agent_id: str, all_agents: List[Any]) -> np.ndarray:
        """Retrieve the observation for a specific agent.

        Args:
            agent_id (str): The ID of the agent.
        Returns:
            np.ndarray: The observation array for the specified agent.
        """
        for machine in self.machine_agents_list:
            if machine.id == int(agent_id):
                break

        # If the agent has already acted, return the observation that was previously calculated
        observation = np.zeros(self.simulation_params[kc.NUMBER_OF_PATHS], dtype=np.int32)

        for agent in all_agents:
            if (machine.id != agent.id and
                machine.origin == agent.origin and
                machine.destination == agent.destination and
                machine.start_time > agent.start_time):
                
                #dt = abs(agent.start_time - machine.start_time)
                #if 0 < dt < self.time_window:
                observation[agent.last_action] += 1

        observation = np.concatenate((observation, [int(machine.urgency*10)], [machine.karma_balance]))

        self.observations[str(machine.id)]["observation"] = observation
        
        return observation
    
class PreviousAvgTTperRoute(Observations):
    """Observes the number of agents with the same origin-destination and start time within a threshold
    and includes the start of the specific agent as well.
    """

    def __init__(
        self,
        machine_agents_list: List[Any],
        human_agents_list: List[Any],
        simulation_params: Dict[str, Any],
        agent_params: Dict[str, Any],
        observations_time_window: int
    ) -> None:
        """Initialize the observation function.

        Args:
            machine_agents_list (List[Any]): List of machine agents.
            human_agents_list (List[Any]): List of human agents.
            simulation_params (Dict[str, Any]): Dictionary of simulation parameters.
            agent_params (Dict[str, Any]): Dictionary of agent parameters.
        Returns:
            None
        """

        super().__init__(machine_agents_list, human_agents_list)
        self.simulation_params = simulation_params
        self.agent_params = agent_params
        self.observations = self.reset_observation()
        self.agent_vectors = {}
        self.time_window = observations_time_window
        self.historic_travel_times = []

    def __call__(self, all_agents: List[Any]) -> Dict[str, Any]:
        """Generate observations for all agents.

        Args:
            all_agents (List[Any]): List of all agents.
        Returns:
            Dict[str, Any]: A dictionary of observations keyed by agent IDs.
        """

        """for machine in self.machine_agents_list:
            observation = np.zeros(self.simulation_params[kc.NUMBER_OF_PATHS], dtype=np.int32)

            for agent in all_agents:
                if (machine.id != agent.id and
                    machine.origin == agent.origin and
                    machine.destination == agent.destination and
                    machine.start_time > agent.start_time):
                    
                    observation[agent.last_action] += 1

            self.observations[str(machine.id)] = np.concatenate(
                [
                    np.array([machine.start_time], dtype=np.int32),  # Start time as scalar
                    observation  # Vector of observations
                ]
            )"""

        return self.observations

    def reset_observation(self) -> Dict[str, np.ndarray]:
        """Reset observations to the initial state.

        Returns:
            obs (Dict[str, np.ndarray]): A dictionary of initial observations for all machine agents.
        """
        # Initialize agent vectors as zero arrays
        self.agent_vectors = {
            agent: np.zeros(self.simulation_params[kc.NUMBER_OF_PATHS], dtype=np.int32)
            for agent in self.machine_agents_list
        }
        
        # Gather observations in a consistent format
        obs = {
            str(agent.id): {
                "observation": np.concatenate(  # Combine start_time and vector into a single array
                [
                    #np.array([agent.start_time], dtype=np.int32),  # Start time as scalar
                    self.agent_vectors[agent],  # Vector as array
                    np.array([0], dtype=np.int32),  # Start time as scalar
                    #np.eye(300, dtype=np.float32)[agent.id]
                ]
            ),
            "action_mask": np.ones(pow(self.agent_params[kc.MACHINE_PARAMETERS][kc.MAXIMUM_ALLOWED_BID], 3), "int8"),
            }
            for agent in self.machine_agents_list
        }

        self.observations = obs

        return obs

    def observation_space(self) -> Dict[str, Box]:
        """
        Define the observation space for each machine agent.

        Returns:
            Dict[str, Box]: A dictionary where keys are agent IDs and values are Gym spaces.
        """

        total_size = 4 + 2*self.simulation_params[kc.NUMBER_OF_PATHS] #+ 300# including urgency & start time

        return {
            str(agent.id): spaces.Dict(
                {
                    "observation": Box(low=0,
                                        high=np.inf,
                                        shape=(total_size,),  # Combined size for start_time and vector
                                        dtype=np.int32
                    ),
                    "action_mask": Box(
                        low=0, high=1, shape=(self.agent_params[kc.MACHINE_PARAMETERS][kc.MAXIMUM_ALLOWED_BID],), dtype=np.int8
                    ),
                }
            )    
            for agent in self.machine_agents_list
        }

    def _get_agent_assigned_route(self, agent):
        """
        Returns the route the agent actually got assigned to.
        - Human agents: last_action IS the chosen route.
        - Machine agents under karma pricing: last_action is the bid vector;
        the assigned route lives in agent.route (set by stackelberg_auction).
        - Machine agents without karma pricing: last_action IS the chosen route.
        """
        
        # Prefer the post-auction assigned route if it exists and has been set
        route = getattr(agent, "route", None)

        if route is None:
            return None
        if isinstance(route, (list, tuple, np.ndarray)):
            # Shouldn't normally happen once agent.route is used, but guard anyway
            return None
        return int(route)
    
    def agent_observations(self, agent_id: str, all_agents: List[Any], historic_data: Dict) -> np.ndarray:
        """Retrieve the observation for a specific agent.
            Observation:
            [
                mean_tt_route_0,
                mean_tt_route_1,
                mean_tt_route_2,

                previous_agents_route_0,
                previous_agents_route_1,
                previous_agents_route_2,

                urgency,
                normalized_start_time,
                normalized_income,
                karma_balance,
            ]
        """
        for machine in self.machine_agents_list:
            if machine.id == int(agent_id):
                break
        
        #print("travel times list is: ", travel_times_list, len(travel_times_list), "\n\n")
        if historic_data:
            #episode_records = historic_data[-1] if historic_data else []
            mean_tt_before = self.historic_means_before_for_agent_all_episodes(
                agent_id=machine.id,
                historic_data=historic_data,   # full history is OK
                actions=(0,1,2),
                k_closest_before = self.time_window
            )
            mean_tt_before = list(mean_tt_before.values())
            mean_tt_before = np.array(mean_tt_before) / 100
        else:
            mean_tt_before = [0.0, 0.0, 0.0]

        ## If there are nan values -> transform them to zero
        mean_tt_before = np.nan_to_num(mean_tt_before, nan=0.0)

        # ---------------------------------------------------------
        # Count previous agents by their last action / route
        # ---------------------------------------------------------
        previous_action_counts = np.zeros(3, dtype=np.float32)

        for agent in all_agents:
            if (
                machine.id != agent.id
                and machine.origin == agent.origin
                and machine.destination == agent.destination
                and machine.start_time >= agent.start_time
            ):
                route = self._get_agent_assigned_route(agent)

                if route is None:
                    continue  # agent hasn't been assigned a route yet this day

                if 0 <= route < 3:
                    previous_action_counts[route] += 1.0

        previous_action_counts = previous_action_counts / 300
        #Normalize incomes over the highest agent's income        
        richest_agent = max(self.machine_agents_list, key=lambda a: a.income)

        if richest_agent.income != 0:
            normalized_income = (
                machine.income / richest_agent.income
            )
        else:
            normalized_income = 0.0

        one_hot_vec = np.zeros(len(self.machine_agents_list), dtype=np.float32)
        one_hot_vec[machine.id] = 1.0

        # ---------------------------------------------------------
        # Construct final observation
        # ---------------------------------------------------------
        observation = np.concatenate(
            (
                mean_tt_before,
                previous_action_counts,
                [machine.urgency],
                [
                    machine.start_time
                    / self.simulation_params[
                        kc.SIMULATION_TIMESTEPS
                    ]
                ],
                [normalized_income],
                [machine.karma_balance],
            )
        ).astype(np.float32)

        self.observations[str(machine.id)][
            "observation"
        ] = observation

        return observation
    
    def historic_means_before_for_agent_all_episodes(
        self,
        agent_id: int,
        historic_data,
        actions=(0, 1, 2),
        time_key="start_time",
        action_key="action",
        tt_key="travel_time",
        id_key="id",
        fill_value=np.nan,
        k_closest_before=20,
    ):
        if not historic_data:
            return []

        # If someone passes a single iteration as list[dict], wrap it
        if isinstance(historic_data, list) and historic_data and isinstance(historic_data[0], dict):
            historic_data = [historic_data]

        stats_list = []

        total_sum_tt = {a: 0.0 for a in actions}
        total_cnt_tt = {a: 0 for a in actions}

        for idx, records in enumerate(historic_data):
            if not records:
                stats_list.append(None)
                continue

            agent_recs = [
                r for r in records
                if isinstance(r, dict) and r.get(id_key) == agent_id
            ]
            if not agent_recs:
                stats_list.append(None)
                continue

            def safe_time(r):
                t = r.get(time_key)
                try:
                    return float(t) if t is not None else np.inf
                except (TypeError, ValueError):
                    return np.inf

            agent_rec = min(agent_recs, key=safe_time)
            try:
                agent_t = float(agent_rec.get(time_key))
            except (TypeError, ValueError):
                stats_list.append(None)
                continue

            # --- collect valid earlier departures ---
            earlier = []
            for r in records:
                if not isinstance(r, dict):
                    continue

                t = r.get(time_key)
                a = r.get(action_key)
                tt = r.get(tt_key)
                rid = r.get(id_key)

                if t is None or tt is None or rid is None:
                    continue

                try:
                    t = float(t)
                    tt = float(tt)
                except (TypeError, ValueError):
                    continue

                if t < agent_t and a in actions:
                    earlier.append((t, a, tt, rid))

            # pick k closest earlier
            if k_closest_before is not None:
                earlier.sort(key=lambda x: x[0], reverse=True)
                earlier = earlier[:k_closest_before]

            # compute stats
            sum_tt = {a: 0.0 for a in actions}
            cnt_tt = {a: 0 for a in actions}
            selected_agent_ids = []

            for t, a, tt, rid in earlier:
                sum_tt[a] += tt
                cnt_tt[a] += 1
                selected_agent_ids.append(rid)

            mean_tt_before = {
                a: (sum_tt[a] / cnt_tt[a]) if cnt_tt[a] > 0 else fill_value
                for a in actions
            }

            for a in actions:
                total_sum_tt[a] += sum_tt[a]
                total_cnt_tt[a] += cnt_tt[a]

            stats_list.append({
                "index": idx,
                "agent_start_time": agent_t,
                "counts_before": dict(cnt_tt),
                "mean_tt_before": mean_tt_before,
                "k_closest_before": k_closest_before,
                "selected_agent_ids": selected_agent_ids,  # ⭐ NEW
            })

        overall_mean_tt_before = {
            a: (total_sum_tt[a] / total_cnt_tt[a]) if total_cnt_tt[a] > 0 else fill_value
            for a in actions
        }
        overall_counts_before = dict(total_cnt_tt)

        for entry in stats_list:
            if entry is None:
                continue
            entry["overall_counts_before"] = overall_counts_before
            entry["overall_mean_tt_before"] = overall_mean_tt_before

        return entry["overall_mean_tt_before"]


class PreviousAgentStartPlusStartTimeDetectorData(Observations):
    """Observes the number of agents with the same origin-destination and start time within a threshold
    and includes the start of the specific agent as well.
    """

    def __init__(
        self,
        machine_agents_list: List[Any],
        human_agents_list: List[Any],
        simulation_params: Dict[str, Any],
        plotter_params: Dict[str, Any],
        agent_params: Dict[str, Any],
        simulator: SumoSimulator
    ) -> None:
        """Initialize the observation function.

        Args:
            machine_agents_list (List[Any]): List of machine agents.
            human_agents_list (List[Any]): List of human agents.
            simulation_params (Dict[str, Any]): Dictionary of simulation parameters.
            agent_params (Dict[str, Any]): Dictionary of agent parameters.
        Returns:
            None
        """

        super().__init__(machine_agents_list, human_agents_list)
        self.simulation_params = simulation_params
        self.agent_params = agent_params
        self.plotter_params = plotter_params
        self.observations = self.reset_observation()
        self.agent_vectors = {}
        self.simulator = simulator

    def __call__(self, all_agents: List[Any]) -> Dict[str, Any]:
        """Generate observations for all agents.

        Args:
            all_agents (List[Any]): List of all agents.
        Returns:
            Dict[str, Any]: A dictionary of observations keyed by agent IDs.
        """

        return self.observations

    def reset_observation(self) -> Dict[str, np.ndarray]:
        """Reset observations to the initial state.

        Returns:
            obs (Dict[str, np.ndarray]): A dictionary of initial observations for all machine agents.
        """
        # Initialize agent vectors as zero arrays
        self.agent_vectors = {
            agent: np.zeros(self.simulation_params[kc.NUMBER_OF_PATHS], dtype=np.int32)
            for agent in self.machine_agents_list
        }
        
        # Gather observations in a consistent format
        obs = {
            str(agent.id): np.concatenate(  # Combine start_time and vector into a single array
                [
                    np.array([agent.start_time], dtype=np.int32),  # Start time as scalar
                    self.agent_vectors[agent],  # Vector as array
                    np.array([0, 0], dtype=np.int32) # Detectors data is zero
                ]
            )
            for agent in self.machine_agents_list
        }

        self.observations = obs

        return obs

    def observation_space(self) -> Dict[str, Box]:
        """
        Define the observation space for each machine agent.

        Returns:
            Dict[str, Box]: A dictionary where keys are agent IDs and values are Gym spaces.
        """

        total_size = 1 + self.simulation_params[kc.NUMBER_OF_PATHS] + 2 # 2 detectors data are used here

        return {
            str(agent.id): Box(
                low=0,
                high=np.inf,
                shape=(total_size,),  # Combined size for start_time and vector
                dtype=np.float32
            )
            for agent in self.machine_agents_list
        }
    
    def agent_observations(self, agent_id: str, all_agents: List[Any]) -> np.ndarray:
        """Retrieve the observation for a specific agent.

        Args:
            agent_id (str): The ID of the agent.
        Returns:
            np.ndarray: The observation array for the specified agent.
        """
        for machine in self.machine_agents_list:
            if machine.id == int(agent_id):
                break
            
        observation = np.zeros(self.simulation_params[kc.NUMBER_OF_PATHS], dtype=np.int32)

        file_path = f"{self.plotter_params[kc.RECORDS_FOLDER] + '/' + kc.DETECTOR_STOPPED_VEHICLES}/stopped_vehicles{self.simulator.timestep}.csv"

        # If the file doesn't exist, fallback to previous timestep
        if not os.path.isfile(file_path):
            file_path = f"{self.plotter_params[kc.RECORDS_FOLDER] + '/' + kc.DETECTOR_STOPPED_VEHICLES}/stopped_vehicles{self.simulator.timestep - 1}.csv"

        df = pd.read_csv(file_path)

        # Filter and count vehicles per detector
        # In the TRY network I am interested on the detectors E1 and E7
        e7_count = df[df["detector"] == "E7_det"]["vehicle_id"].nunique()
        e1_count = df[df["detector"] == "E1_det"]["vehicle_id"].nunique()

        detector_data_array = np.array([e7_count, e1_count])

        # Calculate the decisions of the vehicles that have start time smaller than the start time of the specific agent.
        for agent in all_agents:
            if (machine.id != agent.id and
                machine.origin == agent.origin and
                machine.destination == agent.destination and
                machine.start_time > agent.start_time):
                
                observation[agent.last_action] += 1

        observation = np.concatenate(([machine.start_time], observation, detector_data_array))

        self.observations[str(machine.id)] = observation

        return observation



################ Under work below ################
##################################################

class PreviousAgentStartPlusStartTimeMarginalCost(Observations):
    """Observes the number of agents with the same origin-destination and start time within a threshold
    and includes the start of the specific agent as well.
    """

    def __init__(
        self,
        machine_agents_list: List[Any],
        human_agents_list: List[Any],
        simulation_params: Dict[str, Any],
        agent_params: Dict[str, Any],
    ) -> None:
        """Initialize the observation function.

        Args:
            machine_agents_list (List[Any]): List of machine agents.
            human_agents_list (List[Any]): List of human agents.
            simulation_params (Dict[str, Any]): Dictionary of simulation parameters.
            agent_params (Dict[str, Any]): Dictionary of agent parameters.
        Returns:
            None
        """

        super().__init__(machine_agents_list, human_agents_list)
        self.simulation_params = simulation_params
        self.agent_params = agent_params
        self.observations = self.reset_observation()
        self.agent_vectors = {}

    def __call__(self, all_agents: List[Any]) -> Dict[str, Any]:
        """Generate observations for all agents.

        Args:
            all_agents (List[Any]): List of all agents.
        Returns:
            Dict[str, Any]: A dictionary of observations keyed by agent IDs.
        """
        return self.observations

    def reset_observation(self) -> Dict[str, np.ndarray]:
        """Reset observations to the initial state.

        Returns:
            obs (Dict[str, np.ndarray]): A dictionary of initial observations for all machine agents.
        """

        # Initialize agent vectors as zero arrays
        self.agent_vectors = {
            agent: np.zeros(self.simulation_params[kc.NUMBER_OF_PATHS], dtype=np.int32)
            for agent in self.machine_agents_list
        }
        
        # Gather observations in a consistent format
        obs = {
            str(agent.id): np.concatenate(  # Combine start_time and vector into a single array
                [
                    np.array([agent.start_time], dtype=np.int32),  # Start time as scalar
                    self.agent_vectors[agent]  # Vector as array
                ]
            )
            for agent in self.machine_agents_list
        }

        self.observations = obs

        return obs
    
    def compute_marginal_cost(self, agent_id, all_agents, travel_times_list):
        from .environment import TrafficEnvironment ## added here because there was circular import problem

        #print(all_agents, "\n\n\n", all_agents[0].start_time, "\n\n\n")
        #print("I am agent ", agent_id, "\n\n\n")

        
        """for agent in all_agents:
            print("all_agents are: ", agent, "\n\n\n")
        print("Travel time list is: ", travel_times_list, "\n\n\n")"""

        for machine in self.machine_agents_list:
            if machine.id == int(agent_id):
                break
        
        agents_to_calculate_marginal_cost = []

        for agent in all_agents:
            if agent.kind == kc.TYPE_MACHINE and agent.start_time < machine.start_time:
                agents_to_calculate_marginal_cost.append(agent)

        print("agents to calculate marginal cost", agents_to_calculate_marginal_cost, "\n\n\n")
        
        ## Delete each agent from the environment
        for machine_agent in agents_to_calculate_marginal_cost:
            df = pd.read_csv(os.path.join(self.agent_params[kc.RECORDS_FOLDER], self.agent_params[kc.AGENTS_CSV_FILE_NAME]))

            df["id"] = df["id"].astype(int)
            df = df[df["id"] != int(machine_agent.id)]

            df.to_csv(os.path.join(self.agent_params[kc.RECORDS_FOLDER], "agents2.csv"), index=False)

            actions = []
            for index, row in df.iterrows():
                for agent in all_agents: ## all_agents doesn't have the correct actions of the agents
                    #if agent.start_time < machine.start_time and row['id'] == agent.id:
                    if row['id'] == agent.id:
                        #print("Append action ", agent.last_action, "for agent", agent.id)
                        actions.append(agent.last_action)
                    
            print("actions are: ", actions, "\n\n\n")

            env_params = {
                "agent_parameters" : {
                    "new_machines_after_mutation": 10,
                    "agents_csv_file_name": "agents2.csv",

                    "human_parameters" :
                    {
                        "model" : "general_model",

                        "noise_weight_agent" : 0,
                        "noise_weight_path" : 0.8,
                        "noise_weight_day" : 0.2,

                        "beta" : -1,
                        "beta_k_i_variability" : 0.1,
                        "epsilon_i_variability" : 0.1,
                        "epsilon_k_i_variability" : 0.1,
                        "epsilon_k_i_t_variability" : 0.1,

                        "greedy" : 0.9,
                        "gamma_c" : 0.0,
                        "gamma_u" : 0.0,
                        "remember" : 1,

                        "alpha_zero" : 0.8,
                        "alphas" : [0.2]  
                    },
                    "machine_parameters" :
                    {
                        "behavior" : "cooperative",
                        "observation_type" : "previous_agents_plus_start_time_marginal_cost",
                    }
                },
                "simulator_parameters" : {
                    "network_name" : "two_route_yield",
                    "sumo_type" : "sumo-gui",
                },  
                "plotter_parameters" : {
                    "smooth_by" : 50,
                    "phase_names" : [
                        "Human learning", 
                        "Mutation - Machine learning",
                        "Testing phase"
                    ]
                },
                "path_generation_parameters":
                {
                    "number_of_paths" : 4,
                    "beta" : -.5,
                    "visualize_paths" : True
                }
            }

            env = TrafficEnvironment(seed=42, create_agents=False, create_paths=True, marginal_cost=True, **env_params)
            env.start(use_subprocess=True)

            for agent, action in zip(env.all_agents, actions):
                agent.default_action = action
                print("agent action that was just used: ", action, agent.kind)
            print("Step inside observations\n\n")
            env.step()

            original_travel_times = {entry['id']: entry for entry in travel_times_list}
            simulated_travel_times = {entry['id']: entry for entry in env.travel_times_list}

            """for entry in travel_times_list:
                if entry['id'] == agent_id:
                    return entry['travel_time']"""
            ## Does not take the correct action
            print("\nafter one step\n", env.travel_times_list, "\n\n\n")
            print("Initial travel_time_list is: ", travel_times_list, "\n\n\n")
            #print("after env.step\n")"""   

            ##

            env.stop_simulation() 
            print("Simulation is over\n\n")

        return 

    def observation_space(self) -> Dict[str, Box]:
        """
        Define the observation space for each machine agent.

        Returns:
            Dict[str, Box]: A dictionary where keys are agent IDs and values are Gym spaces.
        """

        total_size = 1 + self.simulation_params[kc.NUMBER_OF_PATHS]

        return {
            str(agent.id): Box(
                low=0,
                high=np.inf,
                shape=(total_size,),  # Combined size for start_time and vector
                dtype=np.float32
            )
            for agent in self.machine_agents_list
        }
    
    def agent_observations(self, agent_id: str, all_agents: List[Any], travel_times_list: List[Any]) -> np.ndarray:
        """Retrieve the observation for a specific agent.

        Args:
            agent_id (str): The ID of the agent.
        Returns:
            np.ndarray: The observation array for the specified agent.
        """
        self.compute_marginal_cost(agent_id, all_agents, travel_times_list)

        for machine in self.machine_agents_list:
            if machine.id == int(agent_id):
                break

        
        observation = np.zeros(self.simulation_params[kc.NUMBER_OF_PATHS], dtype=np.int32)

        for agent in all_agents:
            if (machine.id != agent.id and
                machine.origin == agent.origin and
                machine.destination == agent.destination and
                machine.start_time > agent.start_time):
                
                observation[agent.last_action] += 1

        observation = np.concatenate(([machine.start_time], observation))

        self.observations[str(machine.id)] = observation
        
        return observation
    