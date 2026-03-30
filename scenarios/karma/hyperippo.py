"""
HyperIPPO (MLP Hypernetworks, shared weights) for an AEC PettingZoo-style TrafficEnvironment
with MultiDiscrete actions + JOINT action masking.

Key design choices (matching your MFQ loop assumptions):
- Env action space is MultiDiscrete with D=3 dims, each in [0..B-1].
- Env provides a JOINT action mask of length (B**D) over the full (a0,a1,a2) combination.
- We train IPPO with a single Categorical policy over the JOINT action index in [0..B**D - 1].
  (This is the simplest way to respect JOINT masking in PPO.)
- Masking is applied at:
  - action selection (sampling / greedy)
  - PPO update (log-probs, entropy, ratio)

AEC correctness:
- Reward for (s_t, a_t) is observed when the agent is visited again.
- We store transitions (s, mask, a, logp, v, r, done, s_next) when next obs arrives.

PyTorch only.

You should be able to drop this into your repo next to your other scripts.
"""

from __future__ import annotations

import os
import sys
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


# ------------------------------------------------------------
# Your imports
# ------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from routerl_aec_environment_karma import TrafficEnvironment
from routerl_aec_environment_karma import Keychain as kc


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
@dataclass
class Args:
    seed: int = 10
    device: str = "cpu"

    training_episodes: int = 200
    testing_episodes: int = 10

    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    entropy_coef: float = 0.001
    value_coef: float = 0.5

    lr: float = 8e-4
    optimizer: str = "Adam"
    epochs: int = 4
    minibatch_size: int = 4096
    max_grad_norm: float = 0.5
    log_every_episodes: int = 10

    # Hypernet architecture
    use_agent_id_embeddings: bool = True
    embedding_dim: int = 32
    hypernet_hidden_dims: Tuple[int, ...] = (64, 64)
    target_activation: str = "tanh"

    # Target actor/critic MLP sizes
    actor_layers: Tuple[int, ...] = (64, 64)
    critic_layers: Tuple[int, ...] = (64, 64)

    # MultiDiscrete assumptions
    num_action_dims: int = 3  # D
    # B inferred from env: env.agent_params[...][MAXIMUM_ALLOWED_BID]
    # joint action count = B**D

    # Entropy scheduling
    entropy_schedule: str = "linear"   # "linear" | "exp" | "constant"
    entropy_start: float = 0.01        # higher early exploration
    entropy_end: float = 0.0           # near-deterministic late


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def flatten_obs(obs: Any) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32)
    return arr.reshape(-1)


def extract_obs_and_mask(env_obs: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    env_obs expected:
      {"observation": <np array>, "action_mask": <np int8 array length JOINT_N>}
    """
    if env_obs is None:
        return None, None
    if isinstance(env_obs, dict) and "observation" in env_obs:
        o = flatten_obs(env_obs["observation"])
        m = None
        if "action_mask" in env_obs and env_obs["action_mask"] is not None:
            m = np.asarray(env_obs["action_mask"], dtype=np.int8).reshape(-1)
        return o, m
    # fallback: treat as plain array obs with no mask
    return flatten_obs(env_obs), None


def activation_fn(name: str):
    name = name.lower()
    if name == "relu":
        return torch.relu
    if name == "tanh":
        return torch.tanh
    raise ValueError(f"Unsupported activation: {name}")

def entropy_coef_at_episode(ep: int, total_eps: int, args: Args) -> float:
    if args.entropy_schedule == "constant":
        return args.entropy_coef

    frac = min(max(ep / max(total_eps - 1, 1), 0.0), 1.0)

    if args.entropy_schedule == "linear":
        return args.entropy_start + frac * (args.entropy_end - args.entropy_start)

    if args.entropy_schedule == "exp":
        # exponential decay from start -> end
        # choose a decay rate so that at the end we are close to entropy_end
        if args.entropy_end <= 0:
            # if you want to end at 0, use a small floor
            end = 1e-6
        else:
            end = args.entropy_end
        start = max(args.entropy_start, 1e-8)
        # alpha(t) = start * r^t with r chosen so alpha(T)=end
        r = (end / start) ** (1.0 / max(total_eps - 1, 1))
        return start * (r ** ep)

    raise ValueError(f"Unknown entropy_schedule: {args.entropy_schedule}")


def joint_index_to_action(j: int, B: int, D: int) -> List[int]:
    """
    Map joint index in [0..B**D-1] to MultiDiscrete action [a0,a1,...,a{D-1}]
    using base-B digits (a0 is most-significant digit).
    """
    out = [0] * D
    x = int(j)
    for d in reversed(range(D)):
        out[d] = x % B
        x //= B
    return out


def action_to_joint_index(a: List[int], B: int, D: int) -> int:
    """
    Inverse mapping. a length D, each in [0..B-1].
    """
    j = 0
    for d in range(D):
        j = j * B + int(a[d])
    return j


def masked_categorical_from_logits(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.distributions.Categorical:
    """
    logits: (B, A)
    mask:   (B, A) with {0,1} (or bool). 1 means valid.
    """
    if mask is None:
        return torch.distributions.Categorical(logits=logits)

    # Ensure boolean
    valid = mask.to(dtype=torch.bool)
    # Set invalid logits to a very negative number so prob ~ 0.
    # (Avoid -inf to keep things numerically stable in edge cases.)
    masked_logits = logits.masked_fill(~valid, -1e9)
    return torch.distributions.Categorical(logits=masked_logits)


# ------------------------------------------------------------
# MLP Hypernetwork building blocks (PyTorch)
# ------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Tuple[int, ...], out_dim: int, use_bias: bool = True):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h, bias=use_bias), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, out_dim, bias=use_bias)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPHyperNetwork(nn.Module):
    """
    Generates weights and biases for each layer of a target MLP.
    For each target layer (in_dim -> out_dim), generates:
      W: (B, in_dim, out_dim)
      b: (B, out_dim)
    """
    def __init__(
        self,
        embedding_dim: int,
        layer_dims: List[Tuple[int, int]],
        hypernet_hidden_dims: Tuple[int, ...],
        use_bias_in_hypernet: bool = True,
        final_gain_style: str = "actor",  # "actor" or "critic"
    ):
        super().__init__()
        self.layer_dims = layer_dims
        self.weight_mlps = nn.ModuleList()
        self.bias_mlps = nn.ModuleList()

        for (in_dim, out_dim) in layer_dims:
            self.weight_mlps.append(MLP(embedding_dim, hypernet_hidden_dims, in_dim * out_dim, use_bias=use_bias_in_hypernet))
            self.bias_mlps.append(MLP(embedding_dim, hypernet_hidden_dims, out_dim, use_bias=use_bias_in_hypernet))

        self.final_gain_style = final_gain_style
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        last_w_mlp = self.weight_mlps[-1].net[-1]
        if isinstance(last_w_mlp, nn.Linear):
            if self.final_gain_style == "actor":
                nn.init.orthogonal_(last_w_mlp.weight, gain=0.01)
            else:
                nn.init.orthogonal_(last_w_mlp.weight, gain=1.0)
            if last_w_mlp.bias is not None:
                nn.init.zeros_(last_w_mlp.bias)

    def forward(self, emb: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        weights: List[torch.Tensor] = []
        biases: List[torch.Tensor] = []
        for (in_dim, out_dim), w_mlp, b_mlp in zip(self.layer_dims, self.weight_mlps, self.bias_mlps):
            w_flat = w_mlp(emb)                  # (B, in_dim*out_dim)
            b = b_mlp(emb)                       # (B, out_dim)
            w = w_flat.view(-1, in_dim, out_dim) # (B, in_dim, out_dim)
            weights.append(w)
            biases.append(b)
        return weights, biases


class ActorCriticHyperIPPO(nn.Module):
    """
    Hypernetwork-generated per-agent actor & critic MLPs.

    Actor outputs logits over JOINT actions (size = B**D).
    Critic outputs scalar value.
    """
    def __init__(
        self,
        obs_dim: int,
        joint_action_dim: int,
        num_agents: int,
        actor_layers: Tuple[int, ...],
        critic_layers: Tuple[int, ...],
        embedding_dim: int,
        hypernet_hidden_dims: Tuple[int, ...],
        use_agent_id_embeddings: bool,
        target_activation: str,
        use_bias_in_hypernet: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.joint_action_dim = joint_action_dim
        self.num_agents = num_agents
        self.use_agent_id_embeddings = use_agent_id_embeddings
        self.act_fn = activation_fn(target_activation)

        if use_agent_id_embeddings:
            self.agent_emb = nn.Embedding(num_agents, embedding_dim)
            nn.init.orthogonal_(self.agent_emb.weight, gain=np.sqrt(2))
            eff_emb_dim = embedding_dim
        else:
            self.agent_emb = None
            eff_emb_dim = num_agents

        self.actor_layer_dims = self._compute_layer_dims(obs_dim, list(actor_layers), joint_action_dim)
        self.critic_layer_dims = self._compute_layer_dims(obs_dim, list(critic_layers), 1)

        self.actor_hnet = MLPHyperNetwork(
            embedding_dim=eff_emb_dim,
            layer_dims=self.actor_layer_dims,
            hypernet_hidden_dims=hypernet_hidden_dims,
            use_bias_in_hypernet=use_bias_in_hypernet,
            final_gain_style="actor",
        )
        self.critic_hnet = MLPHyperNetwork(
            embedding_dim=eff_emb_dim,
            layer_dims=self.critic_layer_dims,
            hypernet_hidden_dims=hypernet_hidden_dims,
            use_bias_in_hypernet=use_bias_in_hypernet,
            final_gain_style="critic",
        )

    @staticmethod
    def _compute_layer_dims(input_dim: int, hidden_layers: List[int], output_dim: int) -> List[Tuple[int, int]]:
        dims: List[Tuple[int, int]] = []
        last = input_dim
        for h in hidden_layers:
            dims.append((last, h))
            last = h
        dims.append((last, output_dim))
        return dims

    def _get_embedding(self, agent_idx: torch.Tensor) -> torch.Tensor:
        if self.use_agent_id_embeddings:
            return self.agent_emb(agent_idx)
        return F.one_hot(agent_idx, num_classes=self.num_agents).float()

    def _apply_generated_mlp(
        self,
        x: torch.Tensor,
        weights: List[torch.Tensor],
        biases: List[torch.Tensor],
        final_no_activation: bool = True,
    ) -> torch.Tensor:
        for i, (W, b) in enumerate(zip(weights, biases)):
            x = torch.bmm(x.unsqueeze(1), W).squeeze(1) + b
            is_last = (i == len(weights) - 1)
            if not is_last or not final_no_activation:
                x = self.act_fn(x)
        return x

    def forward(self, obs: torch.Tensor, agent_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self._get_embedding(agent_idx)
        aW, ab = self.actor_hnet(emb)
        cW, cb = self.critic_hnet(emb)
        logits = self._apply_generated_mlp(obs, aW, ab, final_no_activation=True)  # (B, joint_action_dim)
        value = self._apply_generated_mlp(obs, cW, cb, final_no_activation=True)   # (B, 1)
        return logits, value.squeeze(-1)


@torch.no_grad()
def select_action(
    model: nn.Module,
    obs_np: np.ndarray,
    mask_np: Optional[np.ndarray],
    agent_idx: int,
    device: torch.device,
    B: int,
    D: int,
    greedy: bool = False,
):
    obs_t = torch.from_numpy(obs_np).to(device=device, dtype=torch.float32).unsqueeze(0)
    aidx_t = torch.tensor([agent_idx], device=device, dtype=torch.long)

    logits, value = model(obs_t, aidx_t)  # logits: (1, JOINT_N)
    mask_t = None
    if mask_np is not None:
        mask_t = torch.from_numpy(mask_np.astype(np.int8)).to(device=device).unsqueeze(0)

    dist = masked_categorical_from_logits(logits, mask_t)

    if greedy:
        # Greedy w.r.t masked logits
        if mask_t is None:
            act_joint = int(torch.argmax(logits, dim=-1).item())
        else:
            masked_logits = logits.masked_fill(~mask_t.to(torch.bool), -1e9)
            act_joint = int(torch.argmax(masked_logits, dim=-1).item())
    else:
        act_joint = int(dist.sample().item())

    logp = float(dist.log_prob(torch.tensor([act_joint], device=device)).item())
    act_multi = joint_index_to_action(act_joint, B=B, D=D)

    return act_multi, act_joint, logp, float(value.item())


# ------------------------------------------------------------
# Buffer: store AEC transitions (s, mask, a_joint, logp_old, V(s), r, done, s_next)
# ------------------------------------------------------------
class RolloutBuffer:
    def __init__(self):
        self.agent_idx: List[int] = []
        self.obs: List[np.ndarray] = []
        self.mask: List[np.ndarray] = []
        self.act_joint: List[int] = []
        self.logp: List[float] = []
        self.val: List[float] = []
        self.rew: List[float] = []
        self.done: List[float] = []
        self.next_obs: List[np.ndarray] = []

    def clear(self):
        self.agent_idx.clear()
        self.obs.clear()
        self.mask.clear()
        self.act_joint.clear()
        self.logp.clear()
        self.val.clear()
        self.rew.clear()
        self.done.clear()
        self.next_obs.clear()

    def add(
        self,
        agent_idx: int,
        obs: np.ndarray,
        mask: np.ndarray,
        act_joint: int,
        logp: float,
        val: float,
        rew: float,
        done: bool,
        next_obs: np.ndarray,
    ):
        self.agent_idx.append(int(agent_idx))
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.mask.append(np.asarray(mask, dtype=np.int8))
        self.act_joint.append(int(act_joint))
        self.logp.append(float(logp))
        self.val.append(float(val))
        self.rew.append(float(rew))
        self.done.append(float(done))
        self.next_obs.append(np.asarray(next_obs, dtype=np.float32))

    def __len__(self):
        return len(self.act_joint)


# ------------------------------------------------------------
# Correct GAE using V(next_obs)
# ------------------------------------------------------------
@torch.no_grad()
def compute_gae_from_transitions(
    rew: torch.Tensor,
    done: torch.Tensor,
    val: torch.Tensor,
    next_val: torch.Tensor,
    gamma: float,
    lam: float,
):
    T = rew.shape[0]
    adv = torch.zeros_like(rew)
    last_gae = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - done[t]
        delta = rew[t] + gamma * next_val[t] * nonterminal - val[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
    ret = adv + val
    return adv, ret


def ppo_update(model: nn.Module, optimizer: optim.Optimizer, buf: RolloutBuffer, args: Args,  entropy_coef: float):
    device = torch.device(args.device)
    N = len(buf)
    if N == 0:
        return {}

    agent_idx = torch.tensor(buf.agent_idx, device=device, dtype=torch.long)
    obs = torch.tensor(np.stack(buf.obs), device=device, dtype=torch.float32)
    mask = torch.tensor(np.stack(buf.mask), device=device, dtype=torch.int8)  # (N, JOINT_N)
    act_joint = torch.tensor(buf.act_joint, device=device, dtype=torch.long)
    logp_old = torch.tensor(buf.logp, device=device, dtype=torch.float32)
    val = torch.tensor(buf.val, device=device, dtype=torch.float32)
    rew = torch.tensor(buf.rew, device=device, dtype=torch.float32)
    done = torch.tensor(buf.done, device=device, dtype=torch.float32)
    next_obs = torch.tensor(np.stack(buf.next_obs), device=device, dtype=torch.float32)

    with torch.no_grad():
        _, next_val = model(next_obs, agent_idx)
        next_val = next_val.squeeze(-1)

    adv, ret = compute_gae_from_transitions(rew, done, val, next_val, args.gamma, args.gae_lambda)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    metrics = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}
    n_mb = 0

    for _ in range(args.epochs):
        perm = torch.randperm(N, device=device)
        for start in range(0, N, args.minibatch_size):
            mb = perm[start : start + args.minibatch_size]

            mb_obs = obs[mb]
            mb_mask = mask[mb]
            mb_aidx = agent_idx[mb]
            mb_act = act_joint[mb]
            mb_logp_old = logp_old[mb]
            mb_adv = adv[mb]
            mb_ret = ret[mb]

            logits, value = model(mb_obs, mb_aidx)  # logits: (MB, JOINT_N)
            dist = masked_categorical_from_logits(logits, mb_mask)
            logp = dist.log_prob(mb_act)
            entropy = dist.entropy().mean()

            ratio = torch.exp(logp - mb_logp_old)
            pg1 = ratio * mb_adv
            pg2 = torch.clamp(ratio, 1.0 - args.ppo_clip, 1.0 + args.ppo_clip) * mb_adv
            actor_loss = -(torch.min(pg1, pg2)).mean() - entropy_coef * entropy

            value = value.squeeze(-1)
            critic_loss = F.mse_loss(value, mb_ret) * args.value_coef

            optimizer.zero_grad(set_to_none=True)
            (actor_loss + critic_loss).backward()
            if args.max_grad_norm and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            metrics["actor_loss"] += float(actor_loss.item())
            metrics["critic_loss"] += float(critic_loss.item())
            metrics["entropy"] += float(entropy.item())
            n_mb += 1

    for k in metrics:
        metrics[k] /= max(n_mb, 1)
    return metrics


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    args = Args()
    set_seed(args.seed)
    device = torch.device(args.device)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # -------------------------
    # Env params (copied from your MFQ script; adjust as needed)
    # -------------------------
    new_machines_after_mutation = 300
    training_episodes = args.training_episodes
    testing_episodes = args.testing_episodes
    total_episodes = training_episodes + testing_episodes

    num_agents = 300
    origins = ["E0"]
    destinations = ["E17.600"]

    seed = args.seed
    records_folder = f"training_records_hyperippo_masked_300_agents_seed_{seed}_karma_pricing"
    plots_folder = f"plots_hyperippo_masked_300_agents_seed_{seed}_karma_pricing"

    phases = [1, int(total_episodes)]
    phase_names = ["Training", "Testing"]

    env_params = {
        "agent_parameters": {
            "new_machines_after_mutation": new_machines_after_mutation,
            "num_agents": num_agents,
            "machine_parameters": {
                "behavior": "selfish",
                "observation_type": "previous_agents_avg_tt_per_route",
                "route_0_fee": 0,
                "travel_time_normalization_value": 1,
            },
        },
        "simulator_parameters": {
            "network_name": "network",
            "custom_network_folder": "../../network_analysis/network_base",
            "sumo_type": "sumo",
            "simulation_timesteps": 100,
        },
        "environment_parameters": {
            "observations_time_window": 10,
            "save_every": 1,
            "number_of_days": 30,
            "centrally_defined_price": 4.8
        },
        "plotter_parameters": {
            "phases": phases,
            "smooth_by": 50,
            "phase_names": phase_names,
            "records_folder": records_folder,
            "plots_folder": plots_folder,
            "plot_choices": "basic",
        },
        "path_generation_parameters": {
            "origins": origins,
            "destinations": destinations,
            "number_of_paths": 3,
            "beta": -1,
            "visualize_paths": True,
            "all_origins_to_all_destinations": False,
        },
    }

    env = TrafficEnvironment(seed=seed, create_agents=True, create_paths=True, karma_pricing=True, **env_params)

    print("Number of total agents is:", len(env.all_agents))
    print("Number of human agents is:", len(env.human_agents))
    print("Number of machine agents (autonomous vehicles) is:", len(env.machine_agents), "\n")

    env.start()
    env.reset()

    # ---- Mutation: humans switch to AVs ----
    pre_mutation_agents = env.all_agents.copy()
    env.mutation(mutation_start_percentile=0)

    machines = env.machine_agents.copy()
    mutated_humans = {}
    for machine in machines:
        for human in pre_mutation_agents:
            if human.id == machine.id:
                mutated_humans[str(machine.id)] = human
                break

    learning_agents = list(mutated_humans.keys())
    agent_to_idx = {name: i for i, name in enumerate(learning_agents)}
    num_learning_agents = len(learning_agents)
    print("Controlled (mutated) agents:", num_learning_agents)

    # -------------------------
    # Infer obs dim + action sizes
    # -------------------------
    obs_spaces = env._observation_spaces
    some_agent = learning_agents[0]
    obs_space = obs_spaces[some_agent]

    # obs_space is likely a Dict space; we only use ["observation"]
    # If your env exposes differently, adjust this.
    if isinstance(obs_space, dict) or hasattr(obs_space, "__getitem__"):
        # Try the same key you used in MFQ
        try:
            obs_dim = int(np.prod(obs_space["observation"].shape))
        except Exception:
            # fallback: sample one obs to infer
            o, _ = extract_obs_and_mask(env.observe(some_agent))
            if o is None:
                raise RuntimeError("Could not infer obs_dim; please adjust observation extraction.")
            obs_dim = int(o.size)
    else:
        obs_dim = int(np.prod(obs_space.shape))

    B = int(env.agent_params[kc.MACHINE_PARAMETERS][kc.MAXIMUM_ALLOWED_BID])
    D = int(env.simulation_params[kc.NUMBER_OF_PATHS])
    assert D == args.num_action_dims, f"Expected D={args.num_action_dims}, env has D={D}"

    joint_action_dim = B ** D
    print(f"[INFO] B={B} (0..{B-1}), D={D}, joint_action_dim={joint_action_dim}, obs_dim={obs_dim}")

    # -------------------------
    # Build model + optimizer
    # -------------------------
    model = ActorCriticHyperIPPO(
        obs_dim=obs_dim,
        joint_action_dim=joint_action_dim,
        num_agents=num_learning_agents,
        actor_layers=args.actor_layers,
        critic_layers=args.critic_layers,
        embedding_dim=args.embedding_dim,
        hypernet_hidden_dims=args.hypernet_hidden_dims,
        use_agent_id_embeddings=args.use_agent_id_embeddings,
        target_activation=args.target_activation,
        use_bias_in_hypernet=True,
    ).to(device)

    Optimizer = getattr(optim, args.optimizer)
    optimizer = Optimizer(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.01,
        total_iters=args.training_episodes
    )

    buf = RolloutBuffer()

    # Pending per agent (AEC): store last (obs, mask, act_joint, logp, val)
    pending: Dict[str, Optional[Dict[str, Any]]] = {a: None for a in learning_agents}

    # -------------------------
    # Training
    # -------------------------
    pbar = tqdm(total=training_episodes + testing_episodes, desc="HyperIPPO (masked MultiDiscrete) training")
    running_returns: List[float] = []
    running_turns: List[int] = []

    for ep in range(training_episodes):
        print("training episode is: ", ep, "\n\n")
        env.reset()
        buf.clear()
        for a in learning_agents:
            pending[a] = None

        ep_return = {a: 0.0 for a in learning_agents}
        turn_count = 0

        for agent in env.agent_iter():
            env_obs, reward, termination, truncation, info = env.last()
            done = bool(termination or truncation)

            # skip non-learning agents
            if agent not in agent_to_idx:
                env.step(None)
                turn_count += 1
                continue

            aidx = agent_to_idx[agent]
            obs_np, mask_np = extract_obs_and_mask(env_obs)

            # Close previous step for this agent: reward corresponds to previous action
            if pending[agent] is not None:
                prev = pending[agent]

                # In AEC: reward/termination/truncation you see now correspond to the *previous action* of this agent.
                done_for_transition = float(termination or truncation)

                if obs_np is not None:
                    next_obs_np = obs_np
                    if done_for_transition == 1.0:
                        next_obs_np = np.zeros_like(prev["obs"], dtype=np.float32)
                else:
                    next_obs_np = np.zeros_like(prev["obs"], dtype=np.float32)
                    done_for_transition = 1.0  # if no obs, it must be terminal for our purposes

                #print("action is: ", aidx, "\n\n")
                buf.add(
                    agent_idx=aidx,
                    obs=prev["obs"],
                    mask=prev["mask"],
                    act_joint=prev["act_joint"],
                    logp=prev["logp"],
                    val=prev["val"],
                    rew=float(reward),
                    done=done_for_transition,
                    next_obs=next_obs_np,
                )

                ep_return[agent] += float(reward)
                pending[agent] = None

            # Select action unless done
            if done or obs_np is None or mask_np is None:
                action = None
                pending[agent] = None
            else:
                # NOTE: sampling during training
                act_multi, act_joint, logp, val = select_action(
                    model=model,
                    obs_np=obs_np,
                    mask_np=mask_np,
                    agent_idx=aidx,
                    device=device,
                    B=B,
                    D=D,
                    greedy=False,
                )
                action = act_multi  # MultiDiscrete action for env
                pending[agent] = {"obs": obs_np, "mask": mask_np, "act_joint": act_joint, "logp": logp, "val": val}

            env.step(action)
            turn_count += 1

        # Close leftover pending steps conservatively (if agent never revisited at end)
        for agent in learning_agents:
            pending[agent] = None
            """if pending[agent] is not None:
                prev = pending[agent]
                buf.add(
                    agent_idx=agent_to_idx[agent],
                    obs=prev["obs"],
                    mask=prev["mask"],
                    act_joint=prev["act_joint"],
                    logp=prev["logp"],
                    val=prev["val"],
                    rew=0.0,
                    done=True,
                    next_obs=np.zeros_like(prev["obs"], dtype=np.float32),
                )"""
                

        ent_coef = entropy_coef_at_episode(ep, training_episodes, args)
        metrics = ppo_update(model, optimizer, buf, args, entropy_coef=ent_coef)

        #metrics = ppo_update(model, optimizer, buf, args)
        scheduler.step()

        mean_return = float(np.mean(list(ep_return.values())))
        running_returns.append(mean_return)
        running_turns.append(turn_count)

        if (ep + 1) % args.log_every_episodes == 0:
            print(
                f"[EP {ep+1}] mean_return_over_agents={np.mean(running_returns):.4f} "
                f"mean_turns={np.mean(running_turns):.1f} "
                f"actor_loss={metrics.get('actor_loss', 0.0):.4f} "
                f"critic_loss={metrics.get('critic_loss', 0.0):.4f} "
                f"entropy={metrics.get('entropy', 0.0):.4f}"
            )
            running_returns.clear()
            running_turns.clear()

        pbar.update(1)

    # -------------------------
    # Testing (greedy wrt masked logits)
    # -------------------------
    model.eval()
    pbar.set_description("HyperIPPO (masked MultiDiscrete) testing")

    for ep in range(testing_episodes):
        env.reset()
        for a in learning_agents:
            pending[a] = None

        ep_return = {a: 0.0 for a in learning_agents}

        for agent in env.agent_iter():
            env_obs, reward, termination, truncation, info = env.last()
            done = bool(termination or truncation)

            if agent not in agent_to_idx:
                env.step(None)
                continue

            # add reward from previous action when revisited
            if pending[agent] is not None:
                ep_return[agent] += float(reward)
                pending[agent] = None

            obs_np, mask_np = extract_obs_and_mask(env_obs)
            if done or obs_np is None or mask_np is None:
                env.step(None)
                continue

            aidx = agent_to_idx[agent]
            act_multi, act_joint, logp, val = select_action(
                model=model,
                obs_np=obs_np,
                mask_np=mask_np,
                agent_idx=aidx,
                device=device,
                B=B,
                D=D,
                greedy=False,
            )

            # mark pending so we collect reward at next visit
            pending[agent] = {"dummy": True}
            env.step(act_multi)

        # Close leftover pending steps
        for agent in learning_agents:
            pending[agent] = None

        mean_return = float(np.mean(list(ep_return.values())))
        print(f"[TEST EP {ep+1}] mean_return_over_agents={mean_return:.4f}")
        pbar.update(1)

    pbar.close()

    env.plot_results()
    env.stop_simulation()


if __name__ == "__main__":
    main()
