import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Sequence, List, Optional, Tuple


# -------------------------
# Per-resource Q-network
# -------------------------
class ResourceQNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, widths=(256, 256)):
        super().__init__()
        layers = []
        in_dim = state_size
        for w in widths:
            layers += [nn.Linear(in_dim, w), nn.ReLU()]
            in_dim = w
        layers += [nn.Linear(in_dim, action_size)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, S) -> (B, Ai)
        return self.net(x)


# -------------------------
# Replay buffer (stores MultiDiscrete actions + per-dim masks)
# -------------------------
class FactorReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_vec, reward, next_state, done, next_masks_per_dim):
        # action_vec: (K,)
        # next_masks_per_dim: list length K, each (Ai,) or a single array (K, maxAi) with padding
        self.buffer.append((state, action_vec, reward, next_state, done, next_masks_per_dim))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)

        states = np.asarray(states, dtype=np.float32)           # (B,S)
        actions = np.asarray(actions, dtype=np.int64)           # (B,K)
        rewards = np.asarray(rewards, dtype=np.float32)         # (B,)
        next_states = np.asarray(next_states, dtype=np.float32) # (B,S)
        dones = np.asarray(dones, dtype=np.float32)             # (B,)

        return states, actions, rewards, next_states, dones, list(next_masks)

    def __len__(self):
        return len(self.buffer)


# -------------------------
# Shared MultiDiscrete DQN with per-resource networks
# -------------------------
class PerResourceMultiDiscreteDQN:
    """
    One independent Q-network per action dimension (resource).
    Joint value is approximated as sum_i Q_i(s, a_i).

    Expected obs_dict:
      obs_dict["observation"] -> np.array(state) shape (S,)
      obs_dict["action_mask"] -> EITHER
         (Aflat,) flattened joint mask (your current setup), OR
         factorized per-dim masks:
            - list length K, each array (Ai,) of 0/1
            - OR np.array shape (K, max(nvec)) with 0/1 and padding zeros past Ai

    Action returned: np.ndarray shape (K,) (MultiDiscrete bids)
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
        mask_invalid_value: float = -1e9,
        # If you only have joint masks and constraints can couple dims,
        # this tries to "repair" invalid combos by searching top-k per dim.
        topk_repair: int = 10,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = int(state_size)
        self.nvec = np.asarray(list(map(int, nvec)), dtype=np.int64)
        self.K = int(len(self.nvec))
        self.num_agents = int(num_agents)

        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_update_freq = int(target_update_freq)
        self.mask_invalid_value = float(mask_invalid_value)
        self.topk_repair = int(topk_repair)

        # One Q-net per resource
        self.q_nets: List[ResourceQNetwork] = []
        self.tgt_nets: List[ResourceQNetwork] = []
        for i in range(self.K):
            qi = ResourceQNetwork(self.state_size, int(self.nvec[i]), widths).to(self.device)
            ti = ResourceQNetwork(self.state_size, int(self.nvec[i]), widths).to(self.device)
            self.q_nets.append(qi)
            self.tgt_nets.append(ti)

        self.update_target_network()

        # One optimizer over all params (so loss on sum backprops into all nets)
        params = []
        for net in self.q_nets:
            params += list(net.parameters())
        self.optimizer = optim.Adam(params, lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.replay = FactorReplayBuffer(capacity=memory_size)
        self.learn_step = 0
        self.loss_history = []

        # Exploration
        self.per_agent_epsilon = bool(per_agent_epsilon)
        if self.per_agent_epsilon:
            self.eps = np.full(shape=(self.num_agents,), fill_value=epsilon, dtype=np.float32)
        else:
            self.epsilon = float(epsilon)
        self.epsilon_decay_rate = float(epsilon_decay_rate)
        self.epsilon_min = float(epsilon_min)

    def train(self):
        for net in self.q_nets:
            net.train()
        return self

    def eval(self):
        for net in self.q_nets:
            net.eval()
        return self

    def update_target_network(self):
        for q, t in zip(self.q_nets, self.tgt_nets):
            t.load_state_dict(q.state_dict())

    def _get_epsilon(self, agent_id: int) -> float:
        return float(self.eps[agent_id]) if self.per_agent_epsilon else float(self.epsilon)

    def _decay_epsilon(self, agent_id: Optional[int] = None):
        if self.per_agent_epsilon:
            assert agent_id is not None
            self.eps[agent_id] = max(self.eps[agent_id] - self.epsilon_decay_rate, self.epsilon_min)
        else:
            self.epsilon = max(self.epsilon - self.epsilon_decay_rate, self.epsilon_min)

    # -------------------------
    # Mask handling
    # -------------------------
    def _is_joint_mask(self, mask: np.ndarray) -> bool:
        return mask.ndim == 1 and mask.size == int(np.prod(self.nvec))

    def _joint_to_factor_masks(self, joint_mask_flat: np.ndarray) -> List[np.ndarray]:
        """
        Convert flattened joint mask (prod(nvec),) to per-dim masks by marginalizing:
          mask_i[a_i] = 1 iff exists some combination of other dims that is valid.
        NOTE: This does NOT guarantee that independently chosen bids combine into a valid joint action
              if constraints couple dimensions.
        """
        joint = np.asarray(joint_mask_flat, dtype=np.int32).reshape(tuple(self.nvec.tolist()))
        factor_masks = []
        for i in range(self.K):
            # reduce over all dims except i
            axes = tuple(ax for ax in range(self.K) if ax != i)
            mi = joint.max(axis=axes).astype(np.int32)  # (Ai,)
            factor_masks.append(mi)
        return factor_masks

    def _normalize_factor_masks(self, mask_any) -> List[np.ndarray]:
        """
        Accept list-of-arrays or (K, maxAi) and return list length K with shapes (Ai,).
        """
        if isinstance(mask_any, list) or isinstance(mask_any, tuple):
            out = []
            for i in range(self.K):
                mi = np.asarray(mask_any[i], dtype=np.int32)
                assert mi.shape == (int(self.nvec[i]),)
                out.append(mi)
            return out

        mask_any = np.asarray(mask_any, dtype=np.int32)
        # (K, maxAi) padded
        assert mask_any.ndim == 2 and mask_any.shape[0] == self.K
        out = []
        for i in range(self.K):
            Ai = int(self.nvec[i])
            out.append(mask_any[i, :Ai].copy())
        return out

    def _get_factor_masks(self, obs_dict: dict) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        """
        Returns (factor_masks, joint_mask_flat_or_None)
        """
        mask = np.asarray(obs_dict["action_mask"], dtype=np.int32)
        if self._is_joint_mask(mask):
            return self._joint_to_factor_masks(mask), mask
        else:
            # factorized masks
            return self._normalize_factor_masks(mask), None

    # -------------------------
    # Act: one argmax per resource, then combine
    # -------------------------
    @torch.no_grad()
    def act(self, agent_id: int, obs_dict: dict) -> np.ndarray:
        obs = np.asarray(obs_dict["observation"], dtype=np.float32)
        factor_masks, joint_mask_flat = self._get_factor_masks(obs_dict)

        eps = self._get_epsilon(agent_id)

        s = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,S)

        # explore: pick random valid bid per resource
        if self.q_nets[0].training and np.random.rand() < eps:
            action = np.zeros(self.K, dtype=np.int64)
            for i in range(self.K):
                valid = np.flatnonzero(factor_masks[i] == 1)
                action[i] = int(np.random.choice(valid)) if valid.size > 0 else 0
            # optional repair if you have a joint mask and constraints couple dims
            if joint_mask_flat is not None:
                action = self._repair_with_joint_mask(s, factor_masks, joint_mask_flat, action)
            return action

        # exploit: masked argmax per resource
        action = np.zeros(self.K, dtype=np.int64)
        per_dim_q = []
        for i in range(self.K):
            q = self.q_nets[i](s).squeeze(0)  # (Ai,)
            mi = torch.as_tensor(factor_masks[i], device=self.device)
            q = q.masked_fill(mi == 0, self.mask_invalid_value)
            per_dim_q.append(q)
            action[i] = int(torch.argmax(q).item()) if int(mi.sum().item()) > 0 else 0

        # if only factor masks exist, we're done
        if joint_mask_flat is None:
            return action

        # if constraints are coupled, independently optimal bids may form invalid joint action
        action = self._repair_with_joint_mask(s, factor_masks, joint_mask_flat, action, per_dim_q=per_dim_q)
        return action

    @torch.no_grad()
    def _repair_with_joint_mask(
        self,
        s: torch.Tensor,
        factor_masks: List[np.ndarray],
        joint_mask_flat: np.ndarray,
        action: np.ndarray,
        per_dim_q: Optional[List[torch.Tensor]] = None,
    ) -> np.ndarray:
        """
        If chosen action is invalid under joint mask, search top-k per dim by summed Q to find a valid combo.
        If still not found, fall back to any valid joint action (random).
        """
        # check if valid
        flat = int(np.ravel_multi_index(action, self.nvec))
        if joint_mask_flat[flat] == 1:
            return action

        # compute q vectors if not provided
        if per_dim_q is None:
            per_dim_q = []
            for i in range(self.K):
                q = self.q_nets[i](s).squeeze(0)  # (Ai,)
                mi = torch.as_tensor(factor_masks[i], device=self.device)
                q = q.masked_fill(mi == 0, self.mask_invalid_value)
                per_dim_q.append(q)

        # get top-k indices per dim
        topk = []
        for i in range(self.K):
            Ai = int(self.nvec[i])
            k = min(self.topk_repair, Ai)
            vals, idx = torch.topk(per_dim_q[i], k=k)
            topk.append(idx.detach().cpu().numpy().astype(np.int64))

        # brute-force over cartesian product of top-k lists (k^K, small when K=3)
        best = None
        best_score = -np.inf
        for a0 in topk[0]:
            for a1 in topk[1]:
                for a2 in topk[2] if self.K == 3 else [0]:
                    cand = np.array([a0, a1, a2], dtype=np.int64) if self.K == 3 else None
                    if self.K != 3:
                        # generic product for K!=3
                        cand = np.array([topk[j][0] for j in range(self.K)], dtype=np.int64)

                    flat = int(np.ravel_multi_index(cand, self.nvec))
                    if joint_mask_flat[flat] != 1:
                        continue

                    score = 0.0
                    for i in range(self.K):
                        score += float(per_dim_q[i][cand[i]].item())
                    if score > best_score:
                        best_score = score
                        best = cand

        if best is not None:
            return best

        # fallback: pick any valid joint action
        valid = np.flatnonzero(joint_mask_flat == 1)
        if valid.size == 0:
            return action  # nothing to do
        flat = int(np.random.choice(valid))
        return np.asarray(np.unravel_index(flat, self.nvec), dtype=np.int64)

    # -------------------------
    # Store transition
    # -------------------------
    def store_transition(self, obs_dict, action_vec, reward, next_obs_dict, done):
        state = np.asarray(obs_dict["observation"], dtype=np.float32)
        next_state = np.asarray(next_obs_dict["observation"], dtype=np.float32)

        action_vec = np.asarray(action_vec, dtype=np.int64)
        assert action_vec.shape == (self.K,)

        # store *factorized* next masks (even if env gave joint mask)
        next_factor_masks, _ = self._get_factor_masks(next_obs_dict)
        self.replay.push(state, action_vec, float(reward), next_state, float(done), next_factor_masks)

    # -------------------------
    # Learn: TD on summed Q
    # -------------------------
    def learn(self, updates: int = 1):
        if len(self.replay) < self.batch_size:
            return

        for _ in range(int(updates)):
            states, actions, rewards, next_states, dones, next_masks_list = self.replay.sample(self.batch_size)

            states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)         # (B,S)
            next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device) # (B,S)
            actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)         # (B,K)
            rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)  # (B,1)
            dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)      # (B,1)

            # Build (K) masks tensors, each (B,Ai)
            masks_t: List[torch.Tensor] = []
            for i in range(self.K):
                Ai = int(self.nvec[i])
                mi = np.stack([np.asarray(next_masks_list[b][i], dtype=np.int32) for b in range(self.batch_size)], axis=0)
                assert mi.shape == (self.batch_size, Ai)
                masks_t.append(torch.as_tensor(mi, dtype=torch.int32, device=self.device))

            # Q_total(s,a) = sum_i Q_i(s, a_i)
            q_total = torch.zeros((self.batch_size, 1), device=self.device)
            for i in range(self.K):
                q_all = self.q_nets[i](states_t)                      # (B,Ai)
                ai = actions_t[:, i].unsqueeze(1)                     # (B,1)
                q_sa = q_all.gather(1, ai)                            # (B,1)
                q_total = q_total + q_sa

            # target = r + gamma*(1-done)*sum_i max_{valid a'i} Q_tgt_i(s', a'i)
            with torch.no_grad():
                next_total = torch.zeros((self.batch_size, 1), device=self.device)
                for i in range(self.K):
                    next_q = self.tgt_nets[i](next_states_t)          # (B,Ai)
                    mi = masks_t[i]                                   # (B,Ai)
                    next_q = next_q.masked_fill(mi == 0, self.mask_invalid_value)

                    valid_counts = (mi == 1).sum(dim=1, keepdim=True) # (B,1)
                    next_max = next_q.max(dim=1, keepdim=True)[0]     # (B,1)
                    next_max = torch.where(valid_counts > 0, next_max, torch.zeros_like(next_max))
                    next_total = next_total + next_max

                target = rewards_t + self.gamma * (1.0 - dones_t) * next_total

            loss = self.loss_fn(q_total, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.loss_history.append(float(loss.item()))
            self.learn_step += 1

            if self.learn_step % self.target_update_freq == 0:
                self.update_target_network()

        if not self.per_agent_epsilon:
            self._decay_epsilon()
