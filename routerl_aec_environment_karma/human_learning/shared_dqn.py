import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Sequence


# -------------------------
# Joint Q-network
# -------------------------
class JointQNetwork(nn.Module):
    def __init__(self, state_size: int, flat_action_size: int, widths=(256, 256)):
        super().__init__()
        layers = []
        in_dim = state_size
        for w in widths:
            layers += [nn.Linear(in_dim, w), nn.ReLU()]
            in_dim = w
        layers += [nn.Linear(in_dim, flat_action_size)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, state_size) -> (B, flat_action_size)
        return self.net(x)


# -------------------------
# Replay buffer
# -------------------------
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_flat, reward, next_state, done, next_action_mask):
        self.buffer.append((state, action_flat, reward, next_state, done, next_action_mask))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)

        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),         # (B,)
            np.asarray(rewards, dtype=np.float32),       # (B,)
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),         # (B,)
            np.asarray(next_masks, dtype=np.int32),      # (B, A)
        )

    def __len__(self):
        return len(self.buffer)


# -------------------------
# Shared masked Joint DQN
# -------------------------
class SharedMultiDiscreteDQN:
    """
    DQN for MultiDiscrete actions using a *joint flattened action* representation.

    Env observation must be:
      obs_dict = {"observation": np.array(state), "action_mask": np.array([0/1]*A)}
    where A = prod(nvec) and action_mask is for JOINT actions (flattened index).

    Action returned is MultiDiscrete vector shape (K,).
    """

    def __init__(
        self,
        state_size: int,
        nvec: Sequence[int],
        num_agents: int,
        epsilon: float = 0.99,
        epsilon_decay_rate: float = 0.01,
        epsilon_min: float = 0.0,
        memory_size: int = 100_000,
        batch_size: int = 256,
        gamma: float = 0.99,
        learning_rate: float = 3e-4,
        widths=(256, 256),
        target_update_freq: int = 1000,
        device=None,
        per_agent_epsilon: bool = False,
        mask_invalid_value: float = -1e9,   # used for masking
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = int(state_size)
        self.nvec = np.asarray(list(map(int, nvec)), dtype=np.int64)
        self.num_action_dims = int(len(self.nvec))
        self.flat_action_size = int(np.prod(self.nvec))
        self.num_agents = int(num_agents)

        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_update_freq = int(target_update_freq)
        self.learn_step = 0
        self.mask_invalid_value = float(mask_invalid_value)

        # Networks
        self.q_network = JointQNetwork(self.state_size, self.flat_action_size, widths).to(self.device)
        self.target_q_network = JointQNetwork(self.state_size, self.flat_action_size, widths).to(self.device)
        self.update_target_network()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.replay = ReplayBuffer(capacity=memory_size)

        # Exploration
        self.per_agent_epsilon = bool(per_agent_epsilon)
        if self.per_agent_epsilon:
            self.eps = np.full(shape=(self.num_agents,), fill_value=epsilon, dtype=np.float32)
        else:
            self.epsilon = float(epsilon)
        self.epsilon_decay_rate = float(epsilon_decay_rate)
        self.epsilon_min = float(epsilon_min)

        self.loss_history = []

    def train(self):
        self.q_network.train()
        return self

    def eval(self):
        self.q_network.eval()
        return self

    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def _get_epsilon(self, agent_id: int) -> float:
        if self.per_agent_epsilon:
            return float(self.eps[agent_id])
        return float(self.epsilon)

    def _decay_epsilon(self, agent_id: int = None):
        if self.per_agent_epsilon:
            assert agent_id is not None
            self.eps[agent_id] = max(self.eps[agent_id] - self.epsilon_decay_rate, self.epsilon_min)
        else:
            self.epsilon = max(self.epsilon - self.epsilon_decay_rate, self.epsilon_min)

    # -------------------------
    # Index conversions
    # -------------------------
    def flat_to_multi(self, flat_idx: int) -> np.ndarray:
        return np.asarray(np.unravel_index(int(flat_idx), self.nvec), dtype=np.int64)

    def multi_to_flat(self, action_vec: np.ndarray) -> int:
        action_vec = np.asarray(action_vec, dtype=np.int64)
        assert action_vec.shape == (self.num_action_dims,)
        return int(np.ravel_multi_index(action_vec, self.nvec))

    # -------------------------
    # Action selection with mask
    # -------------------------
    @torch.no_grad()
    def act(self, agent_id: int, obs_dict: dict) -> np.ndarray:
        obs = np.asarray(obs_dict["observation"], dtype=np.float32)
        mask = np.asarray(obs_dict["action_mask"], dtype=np.int32)
        assert mask.shape == (self.flat_action_size,)

        valid = np.flatnonzero(mask == 1)
        if valid.size == 0:
            # fallback: all zeros (shouldn't happen ideally)
            return np.zeros(self.num_action_dims, dtype=np.int64)

        eps = self._get_epsilon(agent_id)

        # explore (only among valid)
        if self.q_network.training and np.random.rand() < eps:
            flat = int(np.random.choice(valid))
            return self.flat_to_multi(flat)

        # exploit (masked argmax)
        s = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,S)
        q = self.q_network(s).squeeze(0)  # (A,)

        mask_t = torch.as_tensor(mask, device=self.device)
        q = q.masked_fill(mask_t == 0, self.mask_invalid_value)

        flat = int(torch.argmax(q).item())
        return self.flat_to_multi(flat)

    # -------------------------
    # Store transition
    # -------------------------
    def store_transition(self, obs_dict, action_vec, reward, next_obs_dict, done):
        state = np.asarray(obs_dict["observation"], dtype=np.float32)
        next_state = np.asarray(next_obs_dict["observation"], dtype=np.float32)

        action_flat = self.multi_to_flat(action_vec)

        next_mask = np.asarray(next_obs_dict["action_mask"], dtype=np.int32)
        assert next_mask.shape == (self.flat_action_size,)

        self.replay.push(state, action_flat, float(reward), next_state, float(done), next_mask)

    # -------------------------
    # Learning (masked target max)
    # -------------------------
    def learn(self, updates: int = 1):
        if len(self.replay) < self.batch_size:
            return

        for _ in range(int(updates)):
            states, actions, rewards, next_states, dones, next_masks = self.replay.sample(self.batch_size)

            states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)        # (B,S)
            next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)  # (B,S)

            actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)  # (B,1)
            rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)  # (B,1)
            dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)      # (B,1)

            next_masks_t = torch.as_tensor(next_masks, dtype=torch.int32, device=self.device)  # (B,A)

            # Q(s,a)
            q_all = self.q_network(states_t)               # (B,A)
            q_sa = q_all.gather(1, actions_t)             # (B,1)

            # target: r + gamma*(1-done)*max_{valid a'} Q_target(s',a')
            with torch.no_grad():
                next_q_all = self.target_q_network(next_states_t)  # (B,A)
                next_q_all = next_q_all.masked_fill(next_masks_t == 0, self.mask_invalid_value)

                # guard: if some rows have no valid actions, max would be -1e9. replace with 0.
                valid_counts = (next_masks_t == 1).sum(dim=1, keepdim=True)  # (B,1)
                next_max = next_q_all.max(dim=1, keepdim=True)[0]            # (B,1)
                next_max = torch.where(valid_counts > 0, next_max, torch.zeros_like(next_max))

                target = rewards_t + self.gamma * (1.0 - dones_t) * next_max

            loss = self.loss_fn(q_sa, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.loss_history.append(float(loss.item()))
            self.learn_step += 1

            if self.learn_step % self.target_update_freq == 0:
                self.update_target_network()

        if not self.per_agent_epsilon:
            self._decay_epsilon()
