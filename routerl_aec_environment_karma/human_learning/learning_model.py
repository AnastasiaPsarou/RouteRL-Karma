import numpy as np
from collections import deque

from abc import ABC, abstractmethod

from routerl_aec_environment.keychain import Keychain as kc


class BaseLearningModel(ABC):
    """
    This is an abstract base class for the learning models used to model human learning and decision-making.\n
    Users can create their own learning models by inheriting from this class.
    """

    def __init__(self):
        pass

    @abstractmethod
    def act(self, state) -> None:
        """Method to select an action based on the current state and cost.

        Returns:
            None
        """
        pass

    @abstractmethod
    def learn(self, state, action, reward) -> None:
        """Method to learn the model based on the current state and cost.

        Arguments:
            state (Any): The current state of the environment.
            action (Any): The action to take.
            reward (Any): The reward received from the environment.
        Returns:
            None
        """

        pass


class GeneralModel(BaseLearningModel):
    """
    This is the general model of human route-choice behaviour which can accomodate several classic methods.
    It follows the two-step structure of `learn` and `act`.
    The new experience after travelling is used to update expected costs in the `learn`.
    The learned experiences is used to make routing decision (action) in `act`.

    * The variability (non-determinism) can be controlled at several levels:
        - ``epsilon_i_variability``, ``epsilon_k_i_variability``, ``epsilon_k_i_t_variability`` :
                variance of normal distribution from which error terms are drawn.
        - ``beta_k_i_variability`` : 
                variance of normal distribution for which ``$beta_k_i$`` is drawn (computed in utility).
        - ``noise_weight_agent``, ``noise_weight_path``, ``noise_weight_day`` : 
                relative weights for the error term composition.

    * The parameters of the model are:
        - ``beta``                  :   
                **negative value** - multiplier of reward (travel time) used in utility, determines sensitivity
        - ``greedy``                :   
                1 - exploration_rate , probability with which the choice are rational (argmax(U)) and not probabilistic (random)
        - ``gamma_u``, ``gamma_c``  :   
                bounded rationality components. Expressed as relative increase in costs/utilities below which user do not change behaviour (do not notice it)
        - ``alpha_zero``            :   
                weight with which the recent experience is carried forward in learning
        - ``remember``              :   
                number of days remembered to learn
        - ``alphas``                :   
                vector of weights for historically recorded reward in weighted average !!needs to be size of 'remember'!!
    
    """

    def __init__(self, params, initial_knowledge):
        """
        params is the dictionary, exactly like the one used in RouteRL
        
        initial knowledge shall come from SUMO - free flow travel time (or utilities)
        """
        super().__init__()
        self.costs = - np.array(initial_knowledge, dtype=float) # cost matrix
        self.action_space = len(self.costs) # number of paths
        self.mean_time = abs(np.mean(self.costs)) # mean free flow travel time over paths - possibly useful to normalize errors
        
        # weights of respective componenets of error term (should sum to one)
        self.noise_weight_agent = params[kc.NOISE_WEIGHT_AGENT] # drawn once per agent
        self.noise_weight_path = params[kc.NOISE_WEIGHT_PATH] # drawn once per agent per path
        self.noise_weight_day = params[kc.NOISE_WEIGHT_DAY] # drawn daily per agent per path
        assert self.noise_weight_agent + self.noise_weight_path + self.noise_weight_day  == 1 , "Relative weights in error terms do not sum up to 1."

        # \beta_{t,k,i} agent-specific travel time multiplier per path
        self.beta_k_i = params[kc.BETA]*np.random.normal(1, params[kc.BETA_K_I_VARIABILITY], size = self.action_space) # used to compute Utility in `act`
        self.random_term_agent = np.random.normal(0, params[kc.EPSILON_I_VARIABILITY]) # \vareps_{i}
        self.random_term_path = np.random.normal(0, params[kc.EPSILON_K_I_VARIABILITY], size = self.action_space) # \vareps_{k,i}
        self.random_term_day = params[kc.EPSILON_K_I_T_VARIABILITY] # \vareps_{k,i,t}

        self.greedy = params[kc.GREEDY] # probability that agent will make a greedy choice (and not random exploration)

        self.gamma_c = params[kc.GAMMA_C] # bounded rationality on costs (relative)
        self.gamma_u = params[kc.GAMMA_U] # bounded rationality on utilities (relative)
        self.remember = int(params[kc.REMEMBER]) # window for averaging out

        self.alpha_zero = params[kc.ALPHA_ZERO] # weight of recent experience in learning

        self.alphas = list(params[kc.ALPHAS]) # weights for the memory in respective days
        assert len(self.alphas) == self.remember, "weights of history $\alpha_i in 'alphas' and 'remember do not match"
        assert abs(self.alpha_zero + sum(self.alphas) - 1) < 0.01 , "weights for weighted average do not sum up to 1"
        
        self.memory = [deque([cost],maxlen=self.remember) for cost in self.costs] # memories of experiences per path 
        self.first_day = True

    def learn(self, state, action, reward):
        """
        update 'costs' of 'action' after receiving a 'reward'
        """
        self.memory[action].append(reward) #add recent reward to memory (of rewards)
      
        if abs((self.costs[action]-reward)/self.costs[action])>=self.gamma_c: #learn only if relative difference in rewards is above gamma_c
            weight_normalization_factor = 1/(self.alpha_zero+ sum([self.alphas[i] for i,j in enumerate(self.memory[action])])) # needed to make sure weights are always summed to 1
            self.costs[action] = weight_normalization_factor * self.alpha_zero* self.costs[action] #experience weights
            self.costs[action] += sum([weight_normalization_factor * self.alphas[i]*self.memory[action][i] for i,j in enumerate(self.memory[action])]) # weighted average of historical rewards


    def act(self, state):  
        """
        select path from action space based on learned expected costs"""
        # for each path you multiply the expected costs with path-specific beta (drawn at init) and add the noise (computed from 3 components in `get_noises`)
        utilities = [self.beta_k_i[i] * (self.costs[i] + self.mean_time * noise) for i, noise in enumerate(self.get_noises())]
        
        if self.first_day or abs((self.last_action["utility"] - utilities[self.last_action['action']]/self.last_action["utility"])) >= self.gamma_u: #bounded rationality
            if np.random.random() < self.greedy:
                action = int(np.argmax(utilities)) # greedy choice
            else:
                 action = np.random.choice(self.action_space)  # random choice
        else:
            action = self.last_action['action']    
        self.first_day = False
        self.last_action = {"action": action, "utility": utilities[action]}
        return action       
    
    def get_noises(self):
        """"
        compute random term for the utility, composed of 3 parts - two drawn at init and one here, inside

        returns vector of errors
        """
        daily_noise = np.random.normal(0,self.random_term_day, size= self.action_space)

        return [self.noise_weight_agent * self.random_term_agent + 
                self.noise_weight_path * self.random_term_path[k] + 
                self.noise_weight_day * daily_noise[k]
                    for k,_ in enumerate(self.costs)]
    


class GeneralBiddingModel(BaseLearningModel):
    """
    General model for route bidding with MultiDiscrete actions.

    Instead of returning one chosen route, the model returns a bid vector:
        [bid_route_0, bid_route_1, ..., bid_route_K]

    where each entry is a discrete bid level.
    """

    def __init__(self, params, initial_knowledge):
        super().__init__()

        # ------------------------------------------------------------------
        # Bidding-specific parameters
        # ------------------------------------------------------------------
        self.num_bid_levels = params.get("num_bid_levels", 10)

        self.bid_values = np.array(
            params.get("bid_values", list(range(self.num_bid_levels))),
            dtype=int
        )

        assert len(self.bid_values) == self.num_bid_levels, (
            "bid_values must have length num_bid_levels"
        )

        # Map actual bid values to column indices in the route-bid cost table
        self.bid_value_to_col = {
            int(bid): idx for idx, bid in enumerate(self.bid_values)
        }

        self.bid_exploration = params.get("bid_exploration", 0.1)

        # If True, enforce monotone ranking:
        # better utility => bid no smaller than worse utility
        self.enforce_rank_consistency = params.get(
            "enforce_rank_consistency",
            True
        )

        # ------------------------------------------------------------------
        # Route information and initial route-bid cost table
        # ------------------------------------------------------------------
        base_costs = -np.array(initial_knowledge, dtype=float)

        self.num_routes = len(base_costs)
        self.mean_time = abs(np.mean(base_costs))

        # self.costs[k, j] = expected cost/value of submitting bid_values[j]
        # for route k
        self.costs = np.repeat(
            base_costs[:, None],
            self.num_bid_levels,
            axis=1
        )

        # ------------------------------------------------------------------
        # Noise weights
        # ------------------------------------------------------------------
        self.noise_weight_agent = params[kc.NOISE_WEIGHT_AGENT]
        self.noise_weight_path = params[kc.NOISE_WEIGHT_PATH]
        self.noise_weight_day = params[kc.NOISE_WEIGHT_DAY]

        assert abs(
            self.noise_weight_agent
            + self.noise_weight_path
            + self.noise_weight_day
            - 1
        ) < 1e-8, "Relative weights in error terms do not sum to 1."

        # ------------------------------------------------------------------
        # Utility parameters
        # ------------------------------------------------------------------
        self.beta_k_i = params[kc.BETA] * np.random.normal(
            1,
            params[kc.BETA_K_I_VARIABILITY],
            size=self.num_routes
        )

        self.random_term_agent = np.random.normal(
            0,
            params[kc.EPSILON_I_VARIABILITY]
        )

        self.random_term_path = np.random.normal(
            0,
            params[kc.EPSILON_K_I_VARIABILITY],
            size=self.num_routes
        )

        self.random_term_day = params[kc.EPSILON_K_I_T_VARIABILITY]

        # ------------------------------------------------------------------
        # Learning parameters
        # ------------------------------------------------------------------
        self.greedy = params[kc.GREEDY]
        self.gamma_c = params[kc.GAMMA_C]
        self.gamma_u = params[kc.GAMMA_U]
        self.remember = int(params[kc.REMEMBER])
        self.alpha_zero = params[kc.ALPHA_ZERO]

        self.alphas = list(params[kc.ALPHAS])

        assert len(self.alphas) == self.remember, (
            "weights of history in 'alphas' and 'remember' do not match"
        )

        assert abs(self.alpha_zero + sum(self.alphas) - 1) < 0.01, (
            "weights for weighted average do not sum up to 1"
        )

        # ------------------------------------------------------------------
        # Route-bid memory
        # ------------------------------------------------------------------
        self.memory = [
            [
                deque([self.costs[k, j]], maxlen=self.remember)
                for j in range(self.num_bid_levels)
            ]
            for k in range(self.num_routes)
        ]

        # ------------------------------------------------------------------
        # Karma feasibility parameters
        # ------------------------------------------------------------------
        self.karma_balance_key = params.get(
            "karma_balance_key",
            "karma_balance"
        )

        self.route_fixed_karma_costs = np.array(
            params.get(
                "route_fixed_karma_costs",
                [0.0] * self.num_routes
            ),
            dtype=float
        )

        assert len(self.route_fixed_karma_costs) == self.num_routes

        # ------------------------------------------------------------------
        # Last-action storage
        # ------------------------------------------------------------------
        self.first_day = True
        self.last_action = None



    def learn(self, state, action, reward, chosen_route=None):
        """
        Update expected route-bid cost estimates.

        Parameters
        ----------
        state : any
            Environment state / observation.
        action : np.ndarray or list
            Bid vector submitted by the agent.
        reward : float
            Observed reward/cost signal.
        chosen_route : int or None
            Route actually assigned by the mechanism.
            If None, we fall back to the route with the highest submitted bid.
        """
        action = np.asarray(action, dtype=int)

        if chosen_route is None:
            chosen_route = int(np.argmax(action))

        chosen_bid_value = int(action[chosen_route])

        if chosen_bid_value not in self.bid_value_to_col:
            raise ValueError(
                f"Bid value {chosen_bid_value} is not in bid_values: {self.bid_values}"
            )

        chosen_bid_col = self.bid_value_to_col[chosen_bid_value]

        self.memory[chosen_route][chosen_bid_col].append(reward)

        current_estimate = self.costs[chosen_route, chosen_bid_col]

        relative_change = abs(
            (current_estimate - reward) / (abs(current_estimate) + 1e-8)
        )

        if relative_change >= self.gamma_c:
            relevant_memory = self.memory[chosen_route][chosen_bid_col]

            weight_normalization_factor = 1 / (
                self.alpha_zero
                + sum(self.alphas[i] for i, _ in enumerate(relevant_memory))
            )

            updated_cost = (
                weight_normalization_factor
                * self.alpha_zero
                * current_estimate
            )

            updated_cost += sum(
                weight_normalization_factor
                * self.alphas[i]
                * relevant_memory[i]
                for i, _ in enumerate(relevant_memory)
            )

            self.costs[chosen_route, chosen_bid_col] = updated_cost

    def act(self, state):
        """
        Return a karma-feasible bid vector, one bid per route.
        The current karma balance is read from the last entry of the observation.
        """
        karma_balance = self.get_karma_balance_from_state(state)

        utilities = self.compute_route_bid_utilities()

        # exploration: random feasible bids for all routes
        if np.random.random() < self.bid_exploration:
            bids = self.random_feasible_bids(karma_balance)
        else:
            bids = self.utilities_to_bids(
                utilities,
                karma_balance=karma_balance
            )

        # bounded rationality on bid updates
        if not self.first_day and self.last_action is not None:
            last_utilities = self.last_action["utilities"]
            utility_change = np.max(
                np.abs((utilities - last_utilities) / (np.abs(last_utilities) + 1e-8))
            )

            if utility_change < self.gamma_u:
                bids = self.project_bids_to_karma_feasible(
                    self.last_action["bids"].copy(),
                    karma_balance
                )

        self.first_day = False
        self.last_action = {
            "bids": np.array(bids, dtype=int),
            "utilities": utilities.copy(),
            "karma_balance": karma_balance,
        }
        
        return np.array(bids, dtype=int)
    

    def compute_route_bid_utilities(self):
        """
        Compute utility estimates for every route-bid pair.

        Returns
        -------
        utilities : np.ndarray
            Array of shape (num_routes, num_bid_levels).
        """

        noises = np.array(self.get_noises(), dtype=float)

        utilities = np.zeros(
            (self.num_routes, self.num_bid_levels),
            dtype=float
        )

        for k in range(self.num_routes):
            for j in range(self.num_bid_levels):
                utilities[k, j] = self.beta_k_i[k] * (
                    self.costs[k, j] + self.mean_time * noises[k]
                )

        return utilities

    def get_karma_balance_from_state(self, state):
        """
        Reads the current karma balance from the observation.

        In your current observation structure, karma balance is the last entry:
            state["observation"][-1]
        """
        if isinstance(state, dict):
            if "observation" in state:
                obs = np.asarray(state["observation"], dtype=float)
                return float(obs[-1])

            if "observations" in state:
                obs = np.asarray(state["observations"], dtype=float)
                return float(obs[-1])

        # fallback if state is directly an array
        obs = np.asarray(state, dtype=float)
        return float(obs[-1])

    def utilities_to_bids(self, utilities, karma_balance=None):
        """
        Choose one feasible bid per route using route-bid utilities.

        utilities[k, j] is the utility of submitting bid_values[j]
        for route k.
        """

        bids = np.zeros(self.num_routes, dtype=int)

        for k in range(self.num_routes):
            feasible_bid_values = self.feasible_bid_values_for_route(
                karma_balance,
                k
            )

            feasible_cols = [
                self.bid_value_to_col[int(bid)]
                for bid in feasible_bid_values
            ]

            route_utilities = utilities[k, feasible_cols]

            best_local_idx = int(np.argmax(route_utilities))
            best_col = feasible_cols[best_local_idx]

            bids[k] = self.bid_values[best_col]

        return bids
    
    def get_noises(self):
        daily_noise = np.random.normal(0, self.random_term_day, size=self.num_routes)
        return [
            self.noise_weight_agent * self.random_term_agent
            + self.noise_weight_path * self.random_term_path[k]
            + self.noise_weight_day * daily_noise[k]
            for k, _ in enumerate(self.costs)
        ]
    
    def feasible_bid_values_for_route(self, karma_balance, route_idx):
        """
        Return bid values that do not make the karma balance negative.
        """

        max_affordable_bid = (
            karma_balance - self.route_fixed_karma_costs[route_idx]
        )

        feasible = self.bid_values[self.bid_values <= max_affordable_bid]

        if len(feasible) == 0:
            if 0 in self.bid_values:
                return np.array([0], dtype=int)
            return np.array([self.bid_values[0]], dtype=int)

        return feasible.astype(int)

    def random_feasible_bids(self, karma_balance):
        bids = np.zeros(self.num_routes, dtype=int)

        for k in range(self.num_routes):
            feasible = self.feasible_bid_values_for_route(karma_balance, k)
            bids[k] = np.random.choice(feasible)

        return bids


    def project_bids_to_karma_feasible(self, bids, karma_balance):
        """
        If reusing previous bids, clip them to the current karma balance.
        """
        bids = np.asarray(bids, dtype=int).copy()

        for k in range(self.num_routes):
            feasible = self.feasible_bid_values_for_route(karma_balance, k)
            affordable = feasible[feasible <= bids[k]]

            if len(affordable) > 0:
                bids[k] = affordable[-1]
            else:
                bids[k] = feasible[0]

        return bids

"""
Library of specific models implementations
"""

class GawronModel(GeneralModel):
    """
    The Gawron learning model. This model is based on: `Gawron (1998) <https://kups.ub.uni-koeln.de/9257/>`_\n
    In summary, it iteratively shifts the cost expectations towards the received reward.\n
    For decision-making, calculates action utilities based on the ``beta`` parameter and cost expectations, and selects the action with the lowest utility.
    
    Args:
        params (dict): A dictionary containing model parameters.
        initial_knowledge (list or array): Initial knowledge of cost expectations.
    """
    # classic 0.8 - 0.2 exponential smoothing/markov/gawron
    def __init__(self, params, initial_knowledge):
        # params[kc.REMEMBER] = 1
        # params[kc.ALPHA_ZERO] = 0.2
        # params[kc.ALPHAS] = [0.8]
        # params[kc.GREEDY] = 1
        super().__init__(params, initial_knowledge)

class WeightedModel(GeneralModel):
    """
    Weighted Average learning model. Theory based on: `Cascetta (2009) <https://link.springer.com/book/10.1007/978-0-387-75857-2/>`_.\n
    In summary, the model uses the reward and a 5-day weighted average of the past cost expectations with decreasing weights to update the current cost expectation.\n
    For decision-making, calculates action utilities based on the ``beta`` parameter and cost expectations, and selects the action with the lowest utility.
    

    Args:
        params (dict): A dictionary containing model parameters.
        initial_knowledge (list or array): Initial knowledge of cost expectations.
    """
    # 
    def __init__(self, params, initial_knowledge):
        params[kc.REMEMBER] = 5
        params[kc.ALPHA_ZERO] = 0
        params[kc.ALPHAS] = [1/(n+2)/1.45 for n in range(5)]
        params[kc.GREEDY] = 1
        super().__init__(params, initial_knowledge)


class RandomModel(GeneralModel):
    """
    Purely random choice decision model to be used as a baseline. It randomly selects an action from the action space.\n

    Args:
        params (dict): A dictionary containing model parameters.
        initial_knowledge (list or array): Initial knowledge of cost expectations.
    """

    def __init__(self, params, initial_knowledge):
        params[kc.GREEDY] = 0
        params[kc.REMEMBER] = 0
        params[kc.NOISE_WEIGHT_DAY] = 0
        params[kc.NOISE_WEIGHT_PATH] = 0
        params[kc.NOISE_WEIGHT_AGENT] = 0
        super().__init__(params, initial_knowledge)


class AONModel(GeneralModel):
    """
    All-or-Nothing decision model. This model always selects the path with the lowest initial cost,
    meaning it behaves as a shortest-path router based on the given
    initial knowledge.

    Args:
        params (dict): A dictionary containing model parameters.
        initial_knowledge (list or array): Initial knowledge of cost expectations.
    """

    def __init__(self, params, initial_knowledge):
        params[kc.GREEDY] = 1
        params[kc.REMEMBER] = 0
        params[kc.NOISE_WEIGHT_DAY] = 0
        params[kc.NOISE_WEIGHT_PATH] = 0
        params[kc.NOISE_WEIGHT_AGENT] = 0
        params[kc.BETA] = -1
        params[kc.BETA_K_I_VARIABILITY] = 0
        super().__init__(params, initial_knowledge)


class TestModel(GeneralModel):
    """
    This converges to the similar costs on OD pairs - gives quite a lot of variability, but at the expectation it converges

    """

    def __init__(self, params, initial_knowledge):
        params[kc.NOISE_WEIGHT_AGENT] = 0
        params[kc.NOISE_WEIGHT_PATH] = 0.8
        params[kc.NOISE_WEIGHT_DAY] = 0.2

        params[kc.BETA] = -1

        params[kc.BETA_K_I_VARIABILITY] = 0.03

        params[kc.EPSILON_I_VARIABILITY] = 0
        params[kc.EPSILON_K_I_VARIABILITY] = 0.05
        params[kc.EPSILON_K_I_T_VARIABILITY] = 0.05

        params[kc.GREEDY] = 0.7
        params[kc.GAMMA_C] = 0.1
        params[kc.GAMMA_U] = 0
        params[kc.REMEMBER] = 3

        params[kc.ALPHA_ZERO] = 6
        params[kc.ALPHAS] = [0.2,0.1,0.1]

        super().__init__(params, initial_knowledge)

