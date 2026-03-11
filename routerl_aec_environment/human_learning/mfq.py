# mean_field_q_learning_discrete.py
# ------------------------------------------------------------
# Neural Mean-Field Q-Learning for single Discrete action space.
#
# Q(s, m, a) is approximated with a neural network.
# Mean-field m is provided externally (e.g. mean one-hot of others' actions).
#
# Differences vs DQN:
# - Still Q-learning update, but exploration can be softmax (Boltzmann) as in MFQ
# - Target network is optional (recommended for stability)
# ------------------------------------------------------------

import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class MFQNet(nn.Module):
    """Q-network: Q(obs, mf) -> Q-values for all actions."""
    def __init__(self, obs_dim: int, mf_dim: int, n_actions: int, widths=(128, 128)):
        super().__init__()
        in_dim = obs_dim + mf_dim
        layers = []
        prev = in_dim
        for w in widths:
            layers += [nn.Linear(prev, w), nn.ReLU()]
            prev = w
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_actions)

    def forward(self, obs_t: torch.Tensor, mf_t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs_t, mf_t], dim=-1)
        z = self.backbone(x)
        return self.head(z)


class MeanFieldQLearning:
    """
    Neural MFQ (Mean-Field Q-learning) for discrete actions.

    - Uses mean-field vector m as part of state.
    - Supports:
        * epsilon-greedy
        * or Boltzmann (softmax) exploration commonly used in MFQ
    - Uses target network optionally for stability.
    """

    def __init__(
        self,
        obs_dim: int,
        mf_dim: int,
        n_actions: int,
        num_agents: int,
        widths=(128, 128),
        gamma=0.99,
        learning_rate=3e-4,
        memory_size=100_000,
        batch_size=256,
        target_update_freq=2000,
        use_target_net=True,
        # exploration
        exploration="boltzmann",   # "boltzmann" or "epsilon"
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay_rate=1e-4,
        temperature=1.0,          # for boltzmann
        temperature_min=0.05,
        temperature_decay=1e-4,
        per_agent_exploration=False,
        device=None,
        seed=0,
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.obs_dim = int(obs_dim)
        self.mf_dim = int(mf_dim)
        self.n_actions = int(n_actions)
        self.num_agents = int(num_agents)

        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_update_freq = int(target_update_freq)
        self.use_target_net = bool(use_target_net)

        self.exploration = str(exploration).lower()
        if self.exploration not in ("epsilon", "boltzmann"):
            raise ValueError("exploration must be 'epsilon' or 'boltzmann'")

        self.per_agent_exploration = bool(per_agent_exploration)
        if self.exploration == "epsilon":
            if self.per_agent_exploration:
                self.eps = np.full(self.num_agents, float(epsilon), dtype=np.float32)
            else:
                self.eps = float(epsilon)
            self.epsilon_min = float(epsilon_min)
            self.epsilon_decay_rate = float(epsilon_decay_rate)
        else:
            if self.per_agent_exploration:
                self.temp = np.full(self.num_agents, float(temperature), dtype=np.float32)
            else:
                self.temp = float(temperature)
            self.temperature_min = float(temperature_min)
            self.temperature_decay = float(temperature_decay)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.q = MFQNet(self.obs_dim, self.mf_dim, self.n_actions, widths=widths).to(self.device)
        self.q_tgt = MFQNet(self.obs_dim, self.mf_dim, self.n_actions, widths=widths).to(self.device)
        self.q_tgt.load_state_dict(self.q.state_dict())
        self.q_tgt.eval()

        self.opt = optim.Adam(self.q.parameters(), lr=float(learning_rate))
        self.replay = deque(maxlen=int(memory_size))

        self.train_steps = 0
        self._is_train = True

    def train(self):
        self._is_train = True
        self.q.train()
        return self

    def eval(self):
        self._is_train = False
        self.q.eval()
        return self

    @staticmethod
    def _as_float_1d(x):
        return np.asarray(x, dtype=np.float32).reshape(-1)

    def store_transition(self, obs, action, mf, reward, next_obs, next_mf, done):
        self.replay.append((
            self._as_float_1d(obs),
            int(action),
            self._as_float_1d(mf),
            float(reward),
            self._as_float_1d(next_obs),
            self._as_float_1d(next_mf),
            bool(done),
        ))

    def _decay_exploration(self):
        if not self._is_train:
            return
        if self.exploration == "epsilon":
            if self.per_agent_exploration:
                self.eps = np.maximum(self.epsilon_min, self.eps - self.epsilon_decay_rate)
            else:
                self.eps = max(self.epsilon_min, self.eps - self.epsilon_decay_rate)
        else:
            if self.per_agent_exploration:
                self.temp = np.maximum(self.temperature_min, self.temp - self.temperature_decay)
            else:
                self.temp = max(self.temperature_min, self.temp - self.temperature_decay)

    def act(self, agent_idx: int, obs, mf) -> int:
        obs = self._as_float_1d(obs)
        mf = self._as_float_1d(mf)

        with torch.no_grad():
            o = torch.tensor(obs, device=self.device).unsqueeze(0)
            m = torch.tensor(mf, device=self.device).unsqueeze(0)
            qvals = self.q(o, m).squeeze(0)  # [A]

        if not self._is_train:
            return int(torch.argmax(qvals).item())

        if self.exploration == "epsilon":
            eps = float(self.eps[agent_idx]) if self.per_agent_exploration else float(self.eps)
            if np.random.rand() < eps:
                return int(np.random.randint(self.n_actions))
            return int(torch.argmax(qvals).item())

        # Boltzmann (softmax) exploration
        temp = float(self.temp[agent_idx]) if self.per_agent_exploration else float(self.temp)
        temp = max(temp, 1e-6)
        probs = torch.softmax(qvals / temp, dim=0).cpu().numpy()
        return int(np.random.choice(self.n_actions, p=probs))

    def learn(self, updates=1):
        if not self._is_train:
            return
    
        if len(self.replay) < self.batch_size:
            return

        for _ in range(int(updates)):
            batch = random.sample(self.replay, self.batch_size)
            obs, act, mf, rew, obs2, mf2, done = map(list, zip(*batch))

            obs_t  = torch.tensor(np.asarray(obs, dtype=np.float32), device=self.device)
            mf_t   = torch.tensor(np.asarray(mf, dtype=np.float32), device=self.device)
            act_t  = torch.tensor(np.asarray(act, dtype=np.int64), device=self.device).unsqueeze(-1)

            rew_t  = torch.tensor(np.asarray(rew, dtype=np.float32), device=self.device).unsqueeze(-1)
            obs2_t = torch.tensor(np.asarray(obs2, dtype=np.float32), device=self.device)
            mf2_t  = torch.tensor(np.asarray(mf2, dtype=np.float32), device=self.device)
            done_t = torch.tensor(np.asarray(done, dtype=np.float32), device=self.device).unsqueeze(-1)

            # Q(s,m,a)
            qvals = self.q(obs_t, mf_t)      # [B, A]
            q_sa = qvals.gather(1, act_t)    # [B, 1]

            with torch.no_grad():
                q_next_all = (self.q_tgt if self.use_target_net else self.q)(obs2_t, mf2_t)  # [B, A]
                q_next = torch.max(q_next_all, dim=1, keepdim=True).values                  # [B, 1]
                y = rew_t + (1.0 - done_t) * self.gamma * q_next

            loss = nn.MSELoss()(q_sa, y)

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
            self.opt.step()

            self.train_steps += 1
            self._decay_exploration()

            if self.use_target_net and self.train_steps % self.target_update_freq == 0:
                self.q_tgt.load_state_dict(self.q.state_dict())
