import numpy as np
from collections import defaultdict
from .learning_model import BaseLearningModel


class UCB(BaseLearningModel):
    """Tabular UCB agent with MultiDiscrete actions (3 dims, 0..9 each)."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        num_agents: int,
        alpha: float,
        beta: float,
        seed: int | None = None,
    ):
        """
        Args:
            num_states (int): Unused in this tabular implementation, kept for API compatibility.
            num_actions (int): Total number of *flat* actions (should be 1000 here).
            num_agents (int): Unused here, kept for API compatibility.
            alpha (float): Step size for value updates.
            beta (float): UCB exploration coefficient.
            seed (int | None): Random seed for reproducibility.
        """
        # We keep these for consistency, even if not all are used
        self.num_states = num_states
        self.alpha = alpha
        self.beta = beta

        # --- MultiDiscrete action setup ---
        # You said: 3 different actions that can each take values 0..9
        # => MultiDiscrete([10, 10, 10]) => 10 * 10 * 10 = 1000 flat actions
        self.action_nvec = np.array([10, 10, 10], dtype=np.int64)
        self.num_actions = int(np.prod(self.action_nvec))

        # Optional sanity check vs passed num_actions
        if self.num_actions != num_actions:
            print(
                f"[UCB warning] num_actions={num_actions} but MultiDiscrete([10,10,10]) "
                f"implies {self.num_actions}. Using {self.num_actions} internally."
            )

        # Q and counts: one 1D array of length num_actions per state key
        self.sa_counts = defaultdict(lambda: np.zeros(self.num_actions, dtype=np.float32))
        self.Q = defaultdict(lambda: np.zeros(self.num_actions, dtype=np.float32))

        self.global_step = 1  # start at 1 to avoid log(0)

        if seed is not None:
            np.random.seed(seed)

    # --------- Helpers for state & action representations ---------

    def _state_key(self, state) -> tuple:
        """
        Turn the observation (np.ndarray, list, or tuple) into a hashable state key.
        """
        # Ensure it's 1D of ints; if you already have ints, this is cheap.
        return tuple(int(x) for x in state)

    def _action_to_index(self, action) -> int:
        """
        Convert an action to a flat index in [0, num_actions-1].

        Supports:
        - int (already flat)
        - 3-element sequence like (a0, a1, a2) with each in [0, 9]
        """
        # If it's already a scalar, just use it
        if np.isscalar(action):
            return int(action)

        # Otherwise assume it's a MultiDiscrete vector
        # e.g. action = (a0, a1, a2), each in 0..9
        action = tuple(int(x) for x in action)
        return int(np.ravel_multi_index(action, self.action_nvec))

    def _index_to_action(self, idx: int) -> tuple:
        """
        Convert a flat index in [0, num_actions-1] to a MultiDiscrete action vector.
        Returns a 3-tuple (a0, a1, a2), each in [0, 9].
        """
        return tuple(np.unravel_index(int(idx), self.action_nvec))

    # --------- Core UCB API ---------

    def learn(self, state, action, reward: float):
        """
        Update counts and Q-values from a transition (state, action, reward).

        `state` can be any 1D observation compatible with _state_key().
        `action` can be:
          - a flat int action in [0, 999], or
          - a 3-tuple/list MultiDiscrete action (a0, a1, a2).
        """
        raw_state = state["observations"]

        state_key = self._state_key(raw_state)
        a_idx = self._action_to_index(action)

        # Increment visit counts
        self.sa_counts[state_key][a_idx] += 1
        self.global_step += 1

        # Incremental mean update for Q
        old_estimate = self.Q[state_key][a_idx]
        self.Q[state_key][a_idx] = old_estimate + self.alpha * (reward - old_estimate)

    def act(self, obs) -> tuple:
        """
        Select a MultiDiscrete action based on the current value function using UCB.

        Args:
            obs: 1D observation (np.ndarray, list, or tuple).

        Returns:
            tuple[int, int, int]: a MultiDiscrete action (a0, a1, a2), each in [0, 9].
        """
        raw_obs = obs["observations"]
        action_mask = np.asarray(obs["action_mask"], dtype=bool)  # True = allowed

        state_key = self._state_key(raw_obs)
        self.last_obs = state_key
        division_coef = 1e-3  # avoid division by zero

        counts = self.sa_counts[state_key]
        q_values = self.Q[state_key]

        # UCB values for each flat action
        bonus = self.beta * np.sqrt(np.log(self.global_step) / (counts + division_coef))
        values = q_values + bonus

        # --- apply mask: illegal actions → -inf so they are never picked ---
        values = values.copy()
        values[~action_mask] = -np.inf

        # --- choose among legal actions with max UCB ---
        max_val = np.max(values)
        best_actions = np.flatnonzero(values == max_val)
        flat_action = int(np.random.choice(best_actions))

        action = self._index_to_action(flat_action)

        # Convert the chosen flat index back to MultiDiscrete vector
        return action
