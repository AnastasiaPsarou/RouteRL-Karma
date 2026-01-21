# mean_field_dqn_masked.py
# ------------------------------------------------------------
# Mean-Field DQN (MFQ-style) for MultiDiscrete(3) with JOINT action masking.
#
# Your env provides obs as:
#   obs = {"observation": <1D array>, "action_mask": <1D int8 array length (B^3)>}
#
# Where:
#   B = MAXIMUM_ALLOWED_BID   (since you want bids 0..9 and MAXIMUM_ALLOWED_BID=10)
#   D = 3
#   joint_n = B**D
#
# This agent:
#   - conditions on mean-field vector m (size D*B; per-dim histogram of bids)
#   - uses 3 Q-heads (one per dimension) but enforces JOINT mask by enumerating joint actions
#   - applies mask both in act() and in the target-max during learning
#
# Notes:
#   - Enumeration cost is O(B^3). With B=10 => 1000, totally fine.
#   - Target computation loops over batch elements for correctness/simple integration.
#     If you want, we can vectorize it later.
# ------------------------------------------------------------

import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ==========================================================
# ---- Network + MFQ DQN with JOINT masking
# ==========================================================
class MFQNet(nn.Module):
    def __init__(self, obs_dim, mf_dim, num_dims, num_levels, widths=(64, 64)):
        super().__init__()
        self.num_dims = num_dims
        self.num_levels = num_levels

        in_dim = obs_dim + mf_dim
        layers = []
        prev = in_dim
        for w in widths:
            layers += [nn.Linear(prev, w), nn.ReLU()]
            prev = w
        self.backbone = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(prev, num_levels) for _ in range(num_dims)])

    def forward(self, obs_flat, mf_vec):
        x = torch.cat([obs_flat, mf_vec], dim=-1)
        z = self.backbone(x)
        return [head(z) for head in self.heads]  # list[D], each [B, num_levels]


def decode_joint_index(idx: int, num_levels: int, num_dims: int = 3) -> np.ndarray:
    """Convert joint index (base-num_levels) -> MultiDiscrete action vector length num_dims."""
    a = np.zeros(num_dims, dtype=np.int64)
    x = int(idx)
    for d in range(num_dims - 1, -1, -1):
        a[d] = x % num_levels
        x //= num_levels
    return a

class PerResourceMeanFieldDQN:
    """
    MFQ-style DQN:
      Q(s,m,a) ≈ sum_d Q_d(s,m,a_d)
    and applies JOINT action mask (length B^3) when selecting actions and computing target max.
    """

    def __init__(
        self,
        state_size: int,
        num_dims: int,
        num_levels: int,
        num_agents: int,
        widths=(64, 64),
        gamma=0.9,
        learning_rate=3e-3,
        memory_size=50_000,
        batch_size=256,
        target_update_freq=1000,
        epsilon=0.99,
        epsilon_min=0.05,
        epsilon_decay_rate=0.001,
        per_agent_epsilon=False,
        device=None,
        seed=0,
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.num_dims = int(num_dims)        # 3
        self.num_levels = int(num_levels)    # B = MAXIMUM_ALLOWED_BID (10 -> bids 0..9)
        self.num_agents = int(num_agents)

        self.mf_dim = self.num_dims * self.num_levels
        self.joint_n = self.num_levels ** self.num_dims  # B^3

        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_update_freq = int(target_update_freq)

        self.per_agent_epsilon = bool(per_agent_epsilon)
        if self.per_agent_epsilon:
            self.eps = np.full(self.num_agents, float(epsilon), dtype=np.float32)
        else:
            self.eps = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay_rate = float(epsilon_decay_rate)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.q = MFQNet(state_size, self.mf_dim, self.num_dims, self.num_levels, widths=widths).to(self.device)
        self.q_tgt = MFQNet(state_size, self.mf_dim, self.num_dims, self.num_levels, widths=widths).to(self.device)
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
    def _flatten_obs(obs_dict):
        return np.asarray(obs_dict["observation"], dtype=np.float32).reshape(-1)

    def _get_mask(self, obs_dict):
        m = obs_dict.get("action_mask", None)
        if m is None:
            return None
        m = np.asarray(m, dtype=np.int8).reshape(-1)
        if m.size != self.joint_n:
            # safer to ignore than crash
            return None
        return m

    def _decay_eps(self):
        if not self._is_train:
            return
        if self.per_agent_epsilon:
            self.eps = np.maximum(self.epsilon_min, self.eps - self.epsilon_decay_rate)
        else:
            self.eps = max(self.epsilon_min, self.eps - self.epsilon_decay_rate)

    def _joint_q_values(self, q_heads_list):
        """
        q_heads_list: list of tensors [1, L] for each dim
        returns tensor [joint_n] with summed Q over all joint actions.
        """
        qs = [q.squeeze(0) for q in q_heads_list]  # each [L]
        q_joint = qs[0].view(self.num_levels, *([1] * (self.num_dims - 1)))
        for d in range(1, self.num_dims):
            shape = [1] * self.num_dims
            shape[d] = self.num_levels
            q_joint = q_joint + qs[d].view(*shape)
        return q_joint.reshape(-1)  # [joint_n]

    def _masked_argmax_joint(self, q_heads_list, action_mask_1d):
        q_joint = self._joint_q_values(q_heads_list)

        if action_mask_1d is None:
            best_idx = int(torch.argmax(q_joint).item())
            return best_idx, float(q_joint[best_idx].item())

        mask_t = torch.tensor(action_mask_1d.astype(bool), device=q_joint.device)
        q_masked = torch.where(mask_t, q_joint, torch.tensor(-1e9, device=q_joint.device))
        best_idx = int(torch.argmax(q_masked).item())
        return best_idx, float(q_masked[best_idx].item())

    def act(self, agent_idx, obs_dict, mf_vec):
        obs_flat = self._flatten_obs(obs_dict)
        mf_vec = np.asarray(mf_vec, dtype=np.float32).reshape(-1)
        mask = self._get_mask(obs_dict)

        eps = 0.0
        if self._is_train:
            eps = float(self.eps[agent_idx]) if self.per_agent_epsilon else float(self.eps)

        # Masked exploration: sample uniformly from valid joint actions
        if np.random.rand() < eps:
            if mask is None:
                return np.random.randint(0, self.num_levels, size=(self.num_dims,), dtype=np.int64)
            valid = np.flatnonzero(mask)
            if valid.size == 0:
                return np.zeros(self.num_dims, dtype=np.int64)
            j = int(np.random.choice(valid))
            return decode_joint_index(j, self.num_levels, self.num_dims)

        with torch.no_grad():
            o = torch.tensor(obs_flat, device=self.device).unsqueeze(0)
            m = torch.tensor(mf_vec, device=self.device).unsqueeze(0)
            q_heads = self.q(o, m)
            best_j, _ = self._masked_argmax_joint(q_heads, mask)
            return decode_joint_index(best_j, self.num_levels, self.num_dims)

    def store_transition(self, s, a, mf, r, s2, mf2, done, mask, mask2):
        self.replay.append((
            self._flatten_obs(s),
            np.asarray(a, dtype=np.int64),
            np.asarray(mf, dtype=np.float32),
            float(r),
            self._flatten_obs(s2),
            np.asarray(mf2, dtype=np.float32),
            bool(done),
            None if mask is None else np.asarray(mask, dtype=np.int8),
            None if mask2 is None else np.asarray(mask2, dtype=np.int8),
        ))

    def learn(self, updates=10):
        if len(self.replay) < self.batch_size:
            return

        for _ in range(int(updates)):
            batch = random.sample(self.replay, self.batch_size)
            s, a, mf, r, s2, mf2, done, mask, mask2 = map(list, zip(*batch))

            s_t  = torch.tensor(np.asarray(s,  dtype=np.float32), device=self.device, dtype=torch.float32)
            a_t  = torch.tensor(np.asarray(a,  dtype=np.int64),   device=self.device, dtype=torch.long)
            mf_t = torch.tensor(np.asarray(mf, dtype=np.float32), device=self.device, dtype=torch.float32)

            r_t  = torch.tensor(np.asarray(r,  dtype=np.float32), device=self.device, dtype=torch.float32).unsqueeze(-1)
            s2_t = torch.tensor(np.asarray(s2, dtype=np.float32), device=self.device, dtype=torch.float32)
            mf2_t= torch.tensor(np.asarray(mf2,dtype=np.float32), device=self.device, dtype=torch.float32)
            d_t  = torch.tensor(np.asarray(done, dtype=np.float32), device=self.device, dtype=torch.float32).unsqueeze(-1)

            # Current Q(s,m,a) = sum_d Q_d(s,m,a_d)
            q_heads = self.q(s_t, mf_t)
            q_sa = 0.0
            for d in range(self.num_dims):
                q_sa = q_sa + q_heads[d].gather(1, a_t[:, d:d+1])

            with torch.no_grad():
                q_heads2 = self.q_tgt(s2_t, mf2_t)
                q_next_vals = []
                for i in range(self.batch_size):
                    heads_i = [qh[i:i+1] for qh in q_heads2]
                    m2 = mask2[i]
                    if m2 is not None and np.asarray(m2).size != self.joint_n:
                        m2 = None
                    _, best_val = self._masked_argmax_joint(heads_i, m2)
                    q_next_vals.append(best_val)

                q_next = torch.tensor(q_next_vals, device=self.device, dtype=torch.float32).unsqueeze(-1)
                y = r_t + (1.0 - d_t) * self.gamma * q_next  # all float32 now

            loss = nn.MSELoss()(q_sa, y)

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
            self.opt.step()

            self.train_steps += 1
            self._decay_eps()

            if self.train_steps % self.target_update_freq == 0:
                self.q_tgt.load_state_dict(self.q.state_dict())
