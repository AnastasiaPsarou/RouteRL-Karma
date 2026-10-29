import numpy as np
from collections import deque
from itertools import product
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
    Utility-based bidding model with a joint discrete choice set.

    One alternative is a complete bid vector:
        option_m = [bid_route_0, bid_route_1, ..., bid_route_K-1]

    With K routes and J bid levels, the model has J**K alternatives. For
    example, three routes and ten bid levels produce 1,000 alternatives.

    The model selects one feasible joint alternative according to

        U_i,m^d = V_i,m^d + epsilon_i,m^d,

    where V is the learned systematic utility and epsilon is different for
    every joint alternative. The selected alternative is returned as a bid
    vector so that the external environment interface does not need to change.
    """

    def __init__(self, params, initial_knowledge):
        super().__init__()

        # ------------------------------------------------------------------
        # Bid levels and joint alternatives
        # ------------------------------------------------------------------
        self.num_bid_levels = int(params.get("num_bid_levels", 10))
        self.bid_values = np.asarray(
            params.get("bid_values", list(range(self.num_bid_levels))),
            dtype=int,
        )

        if len(self.bid_values) != self.num_bid_levels:
            raise ValueError("bid_values must have length num_bid_levels")
        if len(np.unique(self.bid_values)) != len(self.bid_values):
            raise ValueError("bid_values must contain unique values")

        base_costs = -np.asarray(initial_knowledge, dtype=float)
        if base_costs.ndim != 1 or len(base_costs) == 0:
            raise ValueError("initial_knowledge must be a non-empty 1-D sequence")

        self.num_routes = len(base_costs)
        self.mean_time = max(abs(float(np.mean(base_costs))), 1e-8)

        # Every row is one complete bid-vector alternative.
        self.options = np.asarray(
            list(product(self.bid_values.tolist(), repeat=self.num_routes)),
            dtype=int,
        )
        self.num_options = len(self.options)
        self.option_to_index = {
            tuple(option.tolist()): idx for idx, option in enumerate(self.options)
        }

        expected_num_options = self.num_bid_levels ** self.num_routes
        if self.num_options != expected_num_options:
            raise RuntimeError("Failed to construct the complete joint action space")

        self.bid_exploration = float(params.get("bid_exploration", 0.1))
        if not 0.0 <= self.bid_exploration <= 1.0:
            raise ValueError("bid_exploration must be in [0, 1]")

        # If False, affordability is checked independently for each route,
        # which is appropriate when only the price of the allocated route is
        # paid. If True, the sum of all submitted bids must fit the balance.
        self.enforce_joint_budget = bool(
            params.get("enforce_joint_budget", False)
        )

        # ------------------------------------------------------------------
        # Systematic utility V for every joint option
        # ------------------------------------------------------------------
        # initial_knowledge supplies route-level values, not values for complete
        # vectors. The default therefore gives every vector the same neutral
        # prior. It will become option-specific through learning.
        initial_option_value = float(
            params.get("initial_option_value", np.mean(base_costs))
        )
        self.option_values = np.full(
            self.num_options,
            initial_option_value,
            dtype=float,
        )

        # Optional deterministic prior over vectors. A callable receives
        # (option_vector, base_costs) and returns a scalar value.
        option_prior_fn = params.get("option_prior_fn")
        if option_prior_fn is not None:
            self.option_values = np.asarray(
                [option_prior_fn(option.copy(), base_costs.copy())
                 for option in self.options],
                dtype=float,
            )
            if self.option_values.shape != (self.num_options,):
                raise ValueError(
                    "option_prior_fn must return one scalar per joint option"
                )


        # Persistent option-specific error epsilon_i,m.
        self.random_term_option = np.random.normal(
            loc=0.0,
            scale=float(params[kc.EPSILON_K_I_VARIABILITY]),
            size=self.num_options,
        )

        # ------------------------------------------------------------------
        # Learning parameters
        # ------------------------------------------------------------------
        self.greedy = params[kc.GREEDY]
        self.gamma_c = float(params[kc.GAMMA_C])
        self.gamma_u = float(params[kc.GAMMA_U])

        # Agent-specific learning rate alpha_i.
        #
        # Each GeneralBiddingModel instance is assumed to represent one commuter.
        # Therefore alpha_i is sampled once when the model is created and remains
        # fixed for that commuter throughout the simulation.
        #
        # Example:
        #   "learning_rate_mean": 0.20,
        #   "learning_rate_std":  0.05
        #
        # Setting learning_rate_std = 0 gives the same learning rate to all agents.
        self.learning_rate_mean = float(
            params.get("learning_rate_mean", 0.20)
        )
        self.learning_rate_std = float(
            params.get("learning_rate_std", 0.05)
        )

        if not 0.0 <= self.learning_rate_mean <= 1.0:
            raise ValueError("learning_rate_mean must be in [0, 1]")
        if self.learning_rate_std < 0.0:
            raise ValueError("learning_rate_std must be non-negative")

        # Gaussian heterogeneity, clipped to the admissible interval [0, 1].
        self.alpha_i = float(
            np.clip(
                np.random.normal(
                    loc=self.learning_rate_mean,
                    scale=self.learning_rate_std,
                ),
                0.0,
                1.0,
            )
        )

        # If learn() receives one reward per route, this determines how the
        # vector is converted to the scalar outcome of the selected joint
        # alternative. A callable can also be supplied.
        self.reward_aggregation = params.get(
            "joint_reward_aggregation", "mean"
        )

        # ------------------------------------------------------------------
        # Karma feasibility
        # ------------------------------------------------------------------
        self.karma_balance_key = params.get(
            "karma_balance_key", "karma_balance"
        )
        self.route_fixed_karma_costs = np.asarray(
            params.get(
                "route_fixed_karma_costs",
                [0.0] * self.num_routes,
            ),
            dtype=float,
        )
        if len(self.route_fixed_karma_costs) != self.num_routes:
            raise ValueError(
                "route_fixed_karma_costs must have one entry per route"
            )

        # ------------------------------------------------------------------
        # Last-action storage
        # ------------------------------------------------------------------
        self.first_day = True
        self.last_action = None

    def learn(self, state, action, reward, chosen_route=None):
        """
        Update the learned systematic utility V of the selected joint option.

        Parameters
        ----------
        action : array-like, shape (num_routes,)
            Complete bid vector returned by act().
        reward : scalar or array-like
            A scalar directly updates the selected option. If a route-level
            reward vector is supplied, it is converted to a scalar according
            to joint_reward_aggregation. When chosen_route is given, the
            default behavior uses the reward of that route.
        chosen_route : int, optional
            Route actually allocated by the external mechanism.
        """
        del state  # retained for interface compatibility

        option_idx = self.action_to_option_index(action)
        scalar_reward = self._scalarize_reward(reward, chosen_route)
        current_estimate = self.option_values[option_idx]

        relative_change = abs(
            (current_estimate - scalar_reward)
            / (abs(current_estimate) + 1e-8)
        )

        if relative_change < self.gamma_c:
            return

        # Experience-based update of the systematic value V:
        #
        # V_i^{d+1}(a_i^d)
        #   = (1 - alpha_i) V_i^d(a_i^d) + alpha_i r_i^d(x_i^d)
        #
        # Only the selected joint bid vector is updated.
        self.option_values[option_idx] = (
            (1.0 - self.alpha_i) * current_estimate
            + self.alpha_i * scalar_reward
        )

    def act(self, state):
        karma_balance = self.get_karma_balance_from_state(state)
        feasible_mask = self.feasible_option_mask(karma_balance)

        if not np.any(feasible_mask):
            raise RuntimeError(
                "No feasible joint bid option exists for the current balance"
            )

        utilities = self.compute_option_utilities()

        if np.random.random() < self.bid_exploration:
            option_idx = self.random_feasible_option_index(feasible_mask)
        else:
            option_idx = self.select_best_option(utilities, feasible_mask)


        bids = self.options[option_idx].copy()


        return bids

    def compute_option_utilities(self):
        """
        Compute random utility for every complete bid-vector alternative.

        U_m = V_m + epsilon_m

        The option-specific and option-day-specific components differ across
        all alternatives, so the random term can change their ranking.
        """
        epsilon = self.get_option_epsilons()
        systematic_utility = self.option_values
        return systematic_utility + epsilon

    def get_option_epsilons(self):
        """Return the fixed epsilon_i(a) for every joint bid alternative."""
        return self.random_term_option

    def feasible_option_mask(self, karma_balance):
        """Return a Boolean mask over feasible complete bid vectors."""
        balance = float(karma_balance)

        # Each bid must be affordable if its corresponding route is allocated.
        per_route_limits = balance - self.route_fixed_karma_costs
        mask = np.all(self.options <= per_route_limits[None, :], axis=1)

        if self.enforce_joint_budget:
            total_commitment = (
                np.sum(self.options, axis=1)
                + float(np.sum(self.route_fixed_karma_costs))
            )
            mask &= total_commitment <= balance

        return mask

    def select_best_option(self, utilities, feasible_mask):
        """Choose the feasible joint option with maximal random utility."""
        utilities = np.asarray(utilities, dtype=float)
        if utilities.shape != (self.num_options,):
            raise ValueError(
                f"utilities must have shape ({self.num_options},)"
            )

        masked_utilities = np.where(feasible_mask, utilities, -np.inf)
        return int(np.argmax(masked_utilities))

    def random_feasible_option_index(self, feasible_mask):
        feasible_indices = np.flatnonzero(feasible_mask)
        if len(feasible_indices) == 0:
            raise RuntimeError("No feasible joint bidding option")
        return int(np.random.choice(feasible_indices))

    def action_to_option_index(self, action):
        """Map a complete bid vector to its joint-option index."""
        action = np.asarray(action, dtype=int)
        if action.shape != (self.num_routes,):
            raise ValueError(
                f"action must have shape ({self.num_routes},), got {action.shape}"
            )

        key = tuple(action.tolist())
        if key not in self.option_to_index:
            raise ValueError(
                f"Action {key} is not in the joint bidding choice set"
            )
        return self.option_to_index[key]

    def _scalarize_reward(self, reward, chosen_route=None):
        reward_arr = np.asarray(reward, dtype=float)

        if reward_arr.ndim == 0:
            return float(reward_arr)

        flat_reward = reward_arr.reshape(-1)
        if len(flat_reward) != self.num_routes:
            raise ValueError(
                "A non-scalar reward must have one entry per route"
            )

        if chosen_route is not None:
            chosen_route = int(chosen_route)
            if not 0 <= chosen_route < self.num_routes:
                raise ValueError("chosen_route is outside the route index range")
            return float(flat_reward[chosen_route])

        if callable(self.reward_aggregation):
            return float(self.reward_aggregation(flat_reward.copy()))
        if self.reward_aggregation == "mean":
            return float(np.mean(flat_reward))
        if self.reward_aggregation == "sum":
            return float(np.sum(flat_reward))
        if self.reward_aggregation == "min":
            return float(np.min(flat_reward))
        if self.reward_aggregation == "max":
            return float(np.max(flat_reward))

        raise ValueError(
            "joint_reward_aggregation must be 'mean', 'sum', 'min', 'max', "
            "or a callable"
        )

    def get_karma_balance_from_state(self, state):
        """Read the current Karma balance from the observation."""
        if isinstance(state, dict):
            if self.karma_balance_key in state:
                return float(state[self.karma_balance_key])
            if "observation" in state:
                obs = np.asarray(state["observation"], dtype=float)
                return float(obs[-1])
            if "observations" in state:
                obs = np.asarray(state["observations"], dtype=float)
                return float(obs[-1])

        obs = np.asarray(state, dtype=float)
        if obs.size == 0:
            raise ValueError("Cannot read Karma balance from an empty state")
        return float(obs[-1])


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

