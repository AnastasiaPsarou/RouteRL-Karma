import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


class Network(nn.Module):
    def __init__(self, state_size, action_space_size, widths):
        super().__init__()
        self.input_layer = nn.Linear(state_size, widths[0])
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(widths[i], widths[i + 1]) for i in range(len(widths) - 1)]
        )
        self.out_layer = nn.Linear(widths[-1], action_space_size)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return self.out_layer(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class SharedDQN:
    """
    One shared DQN for all agents:
      - shared Q-network
      - shared target network
      - shared optimizer
      - shared replay buffer
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
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_space_size = action_space_size
        self.num_agents = num_agents

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learn_step = 0

        # Shared networks
        self.q_network = Network(state_size, action_space_size, widths).to(self.device)
        self.target_q_network = Network(state_size, action_space_size, widths).to(self.device)
        self.update_target_network()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Shared replay buffer across all agents
        self.replay = ReplayBuffer(capacity=memory_size)

        # Exploration
        self.per_agent_epsilon = per_agent_epsilon
        if per_agent_epsilon:
            self.eps = np.full(shape=(num_agents,), fill_value=epsilon, dtype=np.float32)
        else:
            self.epsilon = float(epsilon)

        self.epsilon_decay_rate = float(epsilon_decay_rate)
        self.epsilon_min = float(epsilon_min)

        # Optional: store last (state, action) per agent if your env code uses it
        self.last_state = [None] * num_agents
        self.last_action = [None] * num_agents

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

    @torch.no_grad()
    def act(self, agent_id: int, state: np.ndarray) -> int:
        """
        Choose action for a specific agent using the shared policy.
        """
        eps = self._get_epsilon(agent_id)

        if self.q_network.training and np.random.rand() < eps:
            action = np.random.randint(self.action_space_size)
        else:
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q_network(s)
            action = int(torch.argmax(q, dim=1).item())

        self.last_state[agent_id] = state
        self.last_action[agent_id] = action
        return action

    @torch.no_grad()
    def act_batch(self, states: np.ndarray) -> np.ndarray:
        """
        Vectorized action selection for all agents at once.
        `states`: shape (num_agents, state_size)
        Returns actions: shape (num_agents,)
        """
        assert states.shape[0] == self.num_agents

        # Greedy actions from network
        s = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        q = self.q_network(s)
        greedy = torch.argmax(q, dim=1).cpu().numpy()

        # Epsilon exploration (shared or per-agent)
        actions = greedy.copy()
        if self.q_network.training:
            if self.per_agent_epsilon:
                rand = np.random.rand(self.num_agents)
                explore_mask = rand < self.eps
            else:
                explore_mask = np.random.rand(self.num_agents) < self.epsilon

            random_actions = np.random.randint(self.action_space_size, size=self.num_agents)
            actions[explore_mask] = random_actions[explore_mask]

        return actions

    def store_transition(self, state, action, reward, next_state, done):
        """
        Add experience from ANY agent into the shared replay.
        """
        self.replay.push(state, action, reward, next_state, done)

    def learn(self, updates: int = 1):
        """
        Perform `updates` gradient steps from the shared replay buffer.
        Call this every environment step (or every few steps), after storing transitions.
        """
        if len(self.replay) < self.batch_size:
            return  # not enough data yet

        for _ in range(updates):
            states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

            states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
            rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
            next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
            dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

            # Q(s,a)
            q_sa = self.q_network(states_t).gather(1, actions_t)

            # target: r + gamma * max_a' Q_target(s',a') * (1-done)
            with torch.no_grad():
                next_q = self.target_q_network(next_states_t).max(1, keepdim=True)[0]
                target = rewards_t + self.gamma * (1.0 - dones_t) * next_q

            loss = self.loss_fn(q_sa, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.loss_history.append(float(loss.item()))

            self.learn_step += 1
            if self.learn_step % self.target_update_freq == 0:
                self.update_target_network()

        # Decay exploration (choose one)
        # Option A: shared epsilon decay each learn call
        if not self.per_agent_epsilon:
            self._decay_epsilon()
        # Option B: if per-agent epsilon, decay externally when you want (e.g., per episode)

