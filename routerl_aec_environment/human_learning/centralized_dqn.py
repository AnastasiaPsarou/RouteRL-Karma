import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Optional, Tuple


class CentralizedNetwork(nn.Module):
    """
    Q(s_global, agent_id, a) -> Q-values over actions for that agent.
    """
    def __init__(
        self,
        global_state_size: int,
        action_space_size: int,
        num_agents: int,
        widths=(64, 128, 64),
        agent_id_embed_dim: int = 16,
    ):
        super().__init__()
        self.num_agents = int(num_agents)

        # Learnable embedding for agent id (more efficient than one-hot for many agents)
        self.agent_embed = nn.Embedding(self.num_agents, agent_id_embed_dim)

        in_dim = global_state_size + agent_id_embed_dim

        layers = []
        layers += [nn.Linear(in_dim, widths[0]), nn.ReLU()]
        for i in range(len(widths) - 1):
            layers += [nn.Linear(widths[i], widths[i + 1]), nn.ReLU()]
        layers += [nn.Linear(widths[-1], action_space_size)]
        self.net = nn.Sequential(*layers)

    def forward(self, global_state: torch.Tensor, agent_ids: torch.Tensor) -> torch.Tensor:
        """
        global_state: (B, G)
        agent_ids:    (B,)  int64
        returns:      (B, A)
        """
        emb = self.agent_embed(agent_ids)  # (B, E)
        x = torch.cat([global_state, emb], dim=1)  # (B, G+E)
        return self.net(x)


class ReplayBuffer:
    """
    Stores centralized transitions per-agent:
      (global_state, agent_id, action, reward, next_global_state, done)
    """
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, global_state, agent_id, action, reward, next_global_state, done):
        self.buffer.append((global_state, agent_id, action, reward, next_global_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        gs, aids, acts, rews, ngs, dones = zip(*batch)
        return (
            np.asarray(gs, dtype=np.float32),      # (B, G)
            np.asarray(aids, dtype=np.int64),      # (B,)
            np.asarray(acts, dtype=np.int64),      # (B,)
            np.asarray(rews, dtype=np.float32),    # (B,)
            np.asarray(ngs, dtype=np.float32),     # (B, G)
            np.asarray(dones, dtype=np.float32),   # (B,)
        )

    def __len__(self):
        return len(self.buffer)


class CentralizedDQN:
    """
    Centralized DQN with shared parameters, conditioning on global state + agent id.

    - Input: global_state = concat(all agents' states), shape (num_agents*state_size,)
    - Action: discrete action for ONE agent (0..A-1)
    - Output: Q-values over that agent's actions

    You call:
      act(agent_id, states_all)  where states_all shape (num_agents, state_size)
    or:
      act_batch(states_all) -> actions for all agents
    """

    def __init__(
        self,
        state_size: int,
        action_space_size: int,
        num_agents: int,
        epsilon: float = 0.99,
        epsilon_decay_rate: float = 0.01,
        epsilon_min: float = 0.0,
        memory_size: int = 100_000,
        batch_size: int = 256,
        gamma: float = 0.99,
        learning_rate: float = 3e-4,
        widths=(64, 128, 64),
        target_update_freq: int = 1000,
        device=None,
        per_agent_epsilon: bool = False,
        agent_id_embed_dim: int = 16,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_size = int(state_size)
        self.num_agents = int(num_agents)
        self.action_space_size = int(action_space_size)

        self.global_state_size = self.num_agents * self.state_size

        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_update_freq = int(target_update_freq)
        self.learn_step = 0

        self.q_network = CentralizedNetwork(
            global_state_size=self.global_state_size,
            action_space_size=self.action_space_size,
            num_agents=self.num_agents,
            widths=widths,
            agent_id_embed_dim=agent_id_embed_dim,
        ).to(self.device)

        self.target_q_network = CentralizedNetwork(
            global_state_size=self.global_state_size,
            action_space_size=self.action_space_size,
            num_agents=self.num_agents,
            widths=widths,
            agent_id_embed_dim=agent_id_embed_dim,
        ).to(self.device)

        self.update_target_network()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.replay = ReplayBuffer(capacity=memory_size)

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
        return float(self.eps[agent_id]) if self.per_agent_epsilon else float(self.epsilon)

    def _decay_epsilon(self, agent_id: Optional[int] = None):
        if self.per_agent_epsilon:
            assert agent_id is not None
            self.eps[agent_id] = max(self.eps[agent_id] - self.epsilon_decay_rate, self.epsilon_min)
        else:
            self.epsilon = max(self.epsilon - self.epsilon_decay_rate, self.epsilon_min)

    @staticmethod
    def _to_global_state(states_all: np.ndarray) -> np.ndarray:
        """
        states_all: (N, S) -> global_state: (N*S,)
        """
        return np.asarray(states_all, dtype=np.float32).reshape(-1)

    @torch.no_grad()
    def act(self, agent_id: int, states_all: np.ndarray) -> int:
        """
        Choose action for one agent, conditioning on global state.
        states_all shape: (num_agents, state_size)
        """
        eps = self._get_epsilon(agent_id)

        if self.q_network.training and np.random.rand() < eps:
            return int(np.random.randint(self.action_space_size))

        global_state = self._to_global_state(states_all)
        gs_t = torch.as_tensor(global_state, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,G)
        aid_t = torch.as_tensor([agent_id], dtype=torch.long, device=self.device)                   # (1,)

        q = self.q_network(gs_t, aid_t)  # (1,A)
        return int(torch.argmax(q, dim=1).item())

    @torch.no_grad()
    def act_batch(self, states_all: np.ndarray) -> np.ndarray:
        """
        Choose actions for all agents at once from the same global state.
        states_all shape: (N,S)
        returns: (N,)
        """
        global_state = self._to_global_state(states_all)
        gs_t = torch.as_tensor(global_state, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,G)

        agent_ids = torch.arange(self.num_agents, device=self.device, dtype=torch.long)  # (N,)
        gs_rep = gs_t.repeat(self.num_agents, 1)                                         # (N,G)

        q = self.q_network(gs_rep, agent_ids)      # (N,A)
        greedy = torch.argmax(q, dim=1).cpu().numpy()

        actions = greedy.copy()
        if self.q_network.training:
            if self.per_agent_epsilon:
                explore_mask = np.random.rand(self.num_agents) < self.eps
            else:
                explore_mask = np.random.rand(self.num_agents) < self.epsilon

            random_actions = np.random.randint(self.action_space_size, size=self.num_agents)
            actions[explore_mask] = random_actions[explore_mask]

        return actions

    def store_transition(
        self,
        states_all: np.ndarray,
        agent_id: int,
        action: int,
        reward: float,
        next_states_all: np.ndarray,
        done: bool,
    ):
        """
        Store one agent's experience, but with centralized (global) state and next state.
        """
        gs = self._to_global_state(states_all)
        ngs = self._to_global_state(next_states_all)
        self.replay.push(gs, int(agent_id), int(action), float(reward), ngs, float(done))

    def learn(self, updates: int = 1):
        if len(self.replay) < self.batch_size:
            return

        for _ in range(int(updates)):
            gs, agent_ids, actions, rewards, ngs, dones = self.replay.sample(self.batch_size)

            gs_t = torch.as_tensor(gs, dtype=torch.float32, device=self.device)                 # (B,G)
            ngs_t = torch.as_tensor(ngs, dtype=torch.float32, device=self.device)               # (B,G)
            aid_t = torch.as_tensor(agent_ids, dtype=torch.long, device=self.device)            # (B,)
            act_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1) # (B,1)
            rew_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1) # (B,1)
            done_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)   # (B,1)

            # Q(s,a) for the specific agent in the transition
            q_all = self.q_network(gs_t, aid_t)            # (B,A)
            q_sa = q_all.gather(1, act_t)                  # (B,1)

            # target: r + gamma*(1-done)*max_a' Q_target(s', agent_id, a')
            with torch.no_grad():
                next_q_all = self.target_q_network(ngs_t, aid_t)     # (B,A)
                next_max = next_q_all.max(1, keepdim=True)[0]        # (B,1)
                target = rew_t + self.gamma * (1.0 - done_t) * next_max

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
