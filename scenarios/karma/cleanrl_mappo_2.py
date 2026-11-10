from __future__ import annotations

from tqdm import tqdm
import os
import itertools
import sys
import random
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import argparse
from collections import defaultdict


# ------------------------------------------------------------
# Your imports
# ------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from routerl_aec_environment_karma import TrafficEnvironment


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
@dataclass
class Args:
    seed: int = 10
    device: str = "cpu"

    training_episodes: int = 104
    testing_episodes: int = 10

    gamma: float = 0.99
    gae_lambda: float = 0.9
    ppo_clip: float = 0.2
    entropy_coef: float = 0.0
    value_coef: float = 0.5

    lr_actor: float = 0.0003
    lr_critic: float = 0.0003
    optimizer: str = "Adam"

    epochs: int = 4
    minibatch_size: int = 4096
    max_grad_norm: float = 0.5

    log_every_episodes: int = 10
    masked_logits_neg_inf: float = -1e9

    # --- MAPPO centralized critic config ---
    num_routes: int = 3                 # number_of_paths
    route_action_index: int = 0         # which component of act_vec corresponds to route choice
    include_timestep_feature: bool = True
    checkpoint_dir: str = f"/scratch/tmp/psarou_karma_part_c/checkpoints_mappo_seed_{seed}"

def parse_args() -> Args:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=Args.seed,
        help="Random seed for the experiment.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=Args.device,
        help="Device to use, e.g. cpu or cuda.",
    )

    parsed = parser.parse_args()

    args = Args()
    args.seed = parsed.seed
    args.device = parsed.device

    # Important: update checkpoint_dir after changing the seed.
    args.checkpoint_dir = (
        f"/scratch/tmp/psarou_karma_part_c/checkpoints_mappo_masked_joint_300_agents_seed_{args.seed}"
        f"_route_0_4_route_1_3_max_bid_5_dr"
    )

    return args

# ------------------------------------------------------------
# Networks
# ------------------------------------------------------------
class MLPActorJoint(nn.Module):
    """
    Shared actor over JOINT index of MultiDiscrete:
      action_dim = prod(nvec)
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden=(64, 64)):
        super().__init__()
        layers: List[nn.Module] = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, action_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class MLPCriticCentral(nn.Module):
    """
    Centralized critic:
      input = concat(local_obs, global_features)
    """
    def __init__(self, central_obs_dim: int, hidden=(256, 256)):
        super().__init__()
        layers: List[nn.Module] = []
        last = central_obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, central_obs: torch.Tensor) -> torch.Tensor:
        return self.net(central_obs)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def extract_obs_and_mask(obs: Any) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if isinstance(obs, dict):
        obs_vec = np.asarray(obs.get("observation", []), dtype=np.float32).reshape(-1)
        mask = obs.get("action_mask", None)
        if mask is None:
            return obs_vec, None
        mask_arr = np.asarray(mask)
        if mask_arr.dtype != np.bool_:
            mask_arr = mask_arr.astype(np.int8)
        return obs_vec, mask_arr
    else:
        return np.asarray(obs, dtype=np.float32).reshape(-1), None


def compute_action_strides(nvec: np.ndarray) -> np.ndarray:
    nvec = np.asarray(nvec, dtype=np.int64)
    strides = np.ones_like(nvec)
    for i in range(len(nvec) - 2, -1, -1):
        strides[i] = strides[i + 1] * nvec[i + 1]
    return strides


def joint_index_to_multidiscrete(index: int, nvec: np.ndarray, strides: np.ndarray) -> List[int]:
    idx = int(index)
    out = []
    for i in range(len(nvec)):
        val = idx // int(strides[i])
        idx = idx % int(strides[i])
        out.append(int(val))
    return out


# -------------------------
# Centralized critic features (NO urgency added here)
# -------------------------
def build_global_features(
    *,
    args: Args,
    step_in_episode: int,
    max_steps_hint: int,
    route_counts: np.ndarray,
) -> np.ndarray:
    """
    Global features used ONLY by critic. No urgency is added here.

    - route_counts_norm: (num_routes,)
    - optional timestep: (1,)
    """
    counts = route_counts.astype(np.float32)
    counts_norm = counts / (counts.sum() + 1e-6)

    feats = [counts_norm]

    if args.include_timestep_feature:
        t_norm = np.float32(step_in_episode / float(max_steps_hint)) if max_steps_hint > 0 else np.float32(0.0)
        feats.append(np.array([t_norm], dtype=np.float32))

    return np.concatenate(feats, axis=0).astype(np.float32)


def make_central_obs(local_obs: np.ndarray, global_feats: np.ndarray) -> np.ndarray:
    return np.concatenate([local_obs.astype(np.float32), global_feats.astype(np.float32)], axis=0).astype(np.float32)


@torch.no_grad()
def select_action_masked_joint_mappo(
    actor: nn.Module,
    critic: nn.Module,
    obs_vec: np.ndarray,
    central_obs_vec: np.ndarray,
    action_mask: Optional[np.ndarray],
    device: torch.device,
    args: Args,
) -> Tuple[int, float, float]:
    """
    - actor uses local obs only
    - critic uses central obs (local + global features)
    """
    obs_t = torch.from_numpy(obs_vec).to(device=device, dtype=torch.float32)
    cobs_t = torch.from_numpy(central_obs_vec).to(device=device, dtype=torch.float32)

    logits = actor(obs_t)
    if action_mask is not None:
        mask_t = torch.from_numpy(action_mask).to(device=device)
        valid = mask_t != 0
        if valid.any():
            logits = logits.masked_fill(~valid, args.masked_logits_neg_inf)

    dist = torch.distributions.Categorical(logits=logits)
    act = dist.sample()
    logp = dist.log_prob(act)

    value = critic(cobs_t).squeeze(-1)
    return int(act.item()), float(logp.item()), float(value.item())

@torch.no_grad()
def compute_mappo_policy_snapshot(
    actor,
    eval_obs,
    eval_masks,
    device,
    args,
):
    """
    eval_obs:
        (num_samples, obs_dim)

    eval_masks:
        (num_samples, action_dim)

    returns:
        (num_samples, action_dim)
    """

    was_training = actor.training
    actor.eval()

    obs_t = torch.as_tensor(
        eval_obs,
        dtype=torch.float32,
        device=device,
    )

    mask_t = torch.as_tensor(
        eval_masks,
        device=device,
    )

    logits = actor(obs_t)

    valid = mask_t != 0
    row_has_any_valid = valid.any(dim=1)

    safe_logits = logits.clone()

    safe_logits[row_has_any_valid] = safe_logits[
        row_has_any_valid
    ].masked_fill(
        ~valid[row_has_any_valid],
        args.masked_logits_neg_inf,
    )

    probs = torch.softmax(safe_logits, dim=-1)

    if was_training:
        actor.train()

    return probs.cpu().numpy().astype(np.float32)

def save_mappo_policy_checkpoint(
    *,
    actor,
    eval_obs,
    eval_masks,
    eval_karma,
    eval_urgency,
    eval_agent_ids,
    nvec,
    device,
    args,
    checkpoint_dir,
    step,
    phase,
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    policy = compute_mappo_policy_snapshot(
        actor=actor,
        eval_obs=eval_obs,
        eval_masks=eval_masks,
        device=device,
        args=args,
    )

    # Explicitly store what each flattened joint-action index means.
    joint_actions = np.asarray(
        list(
            itertools.product(
                *[range(int(n)) for n in nvec]
            )
        ),
        dtype=np.int16,
    )

    output_path = checkpoint_dir / f"step_{step:08d}.npz"

    np.savez_compressed(
        output_path,

        step=np.asarray(step, dtype=np.int64),
        phase=np.asarray(phase),

        # Exact observation -> probabilities
        eval_obs=np.asarray(eval_obs, dtype=np.float32),
        policy=np.asarray(policy, dtype=np.float32),
        eval_action_masks=np.asarray(eval_masks, dtype=bool),

        # Explicit state metadata
        karma_balance=np.asarray(eval_karma, dtype=np.float32),
        urgency=np.asarray(eval_urgency, dtype=np.float32),
        agent_ids=np.asarray(eval_agent_ids),

        # Explicit action encoding
        nvec=np.asarray(nvec, dtype=np.int64),
        joint_actions=joint_actions,
    )

    print(
        f"[checkpoint] Saved {phase} policy checkpoint: "
        f"{output_path}"
    )

def compute_gae(rews: np.ndarray, vals: np.ndarray, dones: np.ndarray, gamma: float, lam: float):
    T = len(rews)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    next_value = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        delta = rews[t] + gamma * next_value * nonterminal - vals[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
        next_value = vals[t]
    ret = adv + vals
    return adv, ret


def ppo_update_masked_joint_mappo(
    actor: nn.Module,
    critic: nn.Module,
    actor_opt: optim.Optimizer,
    critic_opt: optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    args: Args,
):
    """
    batch keys:
      obs: (M, obs_dim)
      central_obs: (M, central_obs_dim)
      act: (M,)
      logp_old: (M,)
      adv: (M,)
      ret: (M,)
      mask: (M, action_dim)
    """
    obs = batch["obs"]
    central_obs = batch["central_obs"]
    act = batch["act"]
    logp_old = batch["logp_old"]
    adv = batch["adv"]
    ret = batch["ret"]
    mask = batch["mask"]

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    M = obs.shape[0]
    idx = torch.randperm(M, device=obs.device)

    total_actor_loss = 0.0
    total_critic_loss = 0.0
    total_entropy = 0.0
    n_mb = 0

    for start in range(0, M, args.minibatch_size):
        mb_idx = idx[start : start + args.minibatch_size]
        mb_obs = obs[mb_idx]
        mb_cobs = central_obs[mb_idx]
        mb_act = act[mb_idx]
        mb_logp_old = logp_old[mb_idx]
        mb_adv = adv[mb_idx]
        mb_ret = ret[mb_idx]
        mb_mask = mask[mb_idx]

        logits = actor(mb_obs)

        valid = mb_mask != 0
        row_has_any_valid = valid.any(dim=1)
        if row_has_any_valid.all():
            logits = logits.masked_fill(~valid, args.masked_logits_neg_inf)
        else:
            safe_logits = logits.clone()
            safe_logits[row_has_any_valid] = safe_logits[row_has_any_valid].masked_fill(
                ~valid[row_has_any_valid], args.masked_logits_neg_inf
            )
            logits = safe_logits

        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(mb_act)
        entropy = dist.entropy().mean()

        ratio = torch.exp(logp - mb_logp_old)
        pg1 = ratio * mb_adv
        pg2 = torch.clamp(ratio, 1.0 - args.ppo_clip, 1.0 + args.ppo_clip) * mb_adv
        actor_loss = -(torch.min(pg1, pg2)).mean() - args.entropy_coef * entropy

        value = critic(mb_cobs).squeeze(-1)
        critic_loss = F.mse_loss(value, mb_ret) * args.value_coef

        actor_opt.zero_grad()
        critic_opt.zero_grad()
        actor_loss.backward()
        critic_loss.backward()

        if args.max_grad_norm and args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)

        actor_opt.step()
        critic_opt.step()

        total_actor_loss += float(actor_loss.item())
        total_critic_loss += float(critic_loss.item())
        total_entropy += float(entropy.item())
        n_mb += 1

    return {
        "actor_loss": total_actor_loss / max(n_mb, 1),
        "critic_loss": total_critic_loss / max(n_mb, 1),
        "entropy": total_entropy / max(n_mb, 1),
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    new_machines_after_mutation = 400
    human_learning_episodes = 0
    training_episodes = args.training_episodes
    testing_episodes = args.testing_episodes
    total_episodes = human_learning_episodes + training_episodes + testing_episodes

    num_agents = 400
    origins = ["E0"]
    destinations = ["E17.600"]

    seed = args.seed
    records_folder = f"training_records_mappo_{seed}_route_0_4_route_1_3_max_bid_5_dr"
    plots_folder = f"plots_mappo_{seed}_route_0_4_route_1_3_max_bid_5_dr"

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
                "travel_time_normalization_value": 16.5,
                "max_allowed_bid": 5,
            }
        },
        "simulator_parameters": {
            "network_name": "network",
            "custom_network_folder": "../../network_analysis/network_base",
            "sumo_type": "sumo",
            "simulation_timesteps": 100,
        },
        "environment_parameters": {
            "demand_mode": "variable",
            "demand_levels": [200, 250, 300, 350, 400],
            "observations_time_window": 10,
            "save_every": 1,
            "number_of_days": 30,
            "centrally_defined_price_route_0": 4,
            "centrally_defined_price_route_1": 3,
            "total_karma_supply": 2000,
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
            "all_origins_to_all_destinations": False
        }
    }

    env = TrafficEnvironment(seed=seed, create_agents=True, create_paths=True, karma_pricing=True, **env_params)

    print("Number of total agents is: ", len(env.all_agents), "\n")
    print("Number of human agents is: ", len(env.human_agents), "\n")
    print("Number of machine agents (autonomous vehicles) is: ", len(env.machine_agents), "\n")

    env.start()
    #env.reset()

    
    # Mutate the permanent population
    env.mutation(mutation_start_percentile=0)

    # Include every machine agent in the permanent pool,
    # not only the agents active on Day 0.
    pool_machines = [
        agent for agent in env.agent_pool
        if agent.kind == "AV"
    ]

    mutated_agent_names = [
        str(agent.id) for agent in pool_machines
    ]

    agent_name_to_idx = {
        name: i
        for i, name in enumerate(mutated_agent_names)
    }

    # Persistent lookup covering ALL potential machine agents
    machine_by_name = {
        str(machine.id): machine
        for machine in pool_machines
    }

    print("Controlled agents:", len(mutated_agent_names))
    print("Currently active:", len(env.possible_agents))

    assert len(mutated_agent_names) == 400
    assert set(env.possible_agents).issubset(
        set(mutated_agent_names)
    )


    # -------------------------
    # Infer obs/action sizes
    # -------------------------
    obs_spaces = env._observation_spaces
    act_spaces = env._action_spaces

    if not env.possible_agents:
        raise RuntimeError(
            "No active machine agents available to infer spaces."
        )

    some_agent = env.possible_agents[0]

    obs_space = obs_spaces[some_agent]
    act_space = act_spaces[some_agent]

    try:
        obs_shape = obs_space["observation"].shape
    except Exception:
        obs_shape = obs_space.shape

    obs_dim = int(np.prod(obs_shape))

    nvec = np.asarray(act_space.nvec, dtype=np.int64)
    strides = compute_action_strides(nvec)
    action_dim = int(np.prod(nvec))

    print("Using active agent:", some_agent)
    print("Observation dimension:", obs_dim)
    print("Action dimension:", action_dim)


    try:
        obs_shape = obs_space["observation"].shape
    except Exception:
        obs_shape = obs_space.shape
    obs_dim = int(np.prod(obs_shape))

    act_spaces = env._action_spaces
    act_space = act_spaces[some_agent]
    nvec = np.asarray(act_space.nvec, dtype=np.int64)
    strides = compute_action_strides(nvec)
    action_dim = int(np.prod(nvec))

    # Central critic gets local obs + route_counts_norm + optional t_norm
    global_dim = args.num_routes + (1 if args.include_timestep_feature else 0)
    central_obs_dim = obs_dim + global_dim

    actor = MLPActorJoint(obs_dim, action_dim, hidden=(64, 64)).to(device)
    critic = MLPCriticCentral(central_obs_dim, hidden=(256, 256)).to(device)

    Optimizer = getattr(optim, args.optimizer)
    actor_opt = Optimizer(actor.parameters(), lr=args.lr_actor)
    critic_opt = Optimizer(critic.parameters(), lr=args.lr_critic)

    # AEC steps are agent-by-agent, so this is just a normalization hint
    max_steps_hint = max(1, int(num_agents * env_params["environment_parameters"]["number_of_days"]))
    
    # -------------------------
    # Policy checkpoint setup
    # -------------------------
    checkpoint_dir = (
        f"/scratch/tmp/psarou_karma_part_c/checkpoints_mappo_masked_joint_300_agents_seed_{seed}"
        f"_route_0_4_route_1_3_max_bid_5_dr"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_interval = 2  # save every 2 training episodes

    eval_obs_by_agent: Dict[str, np.ndarray] = {}
    eval_masks_by_agent: Dict[str, np.ndarray] = {}

    eval_obs: Optional[np.ndarray] = None
    eval_masks: Optional[np.ndarray] = None


    eval_records = []
    eval_counts = defaultdict(int)

    SAMPLES_PER_KARMA_URGENCY = 5

    machine_by_name = {
        str(machine.id): machine
        for machine in env.agent_pool
        if machine.kind == "AV"
    }

    saved_initial_policy = False

    # -------------------------
    # Training loop
    # -------------------------
    pbar = tqdm(total=training_episodes + testing_episodes, desc="MAPPO AEC training (masked joint)")

    # ============================================================
    # TRAINING
    # ============================================================

    running_ep_returns = []
    running_ep_lens = []

    # These become the FIXED evaluation bank used for every checkpoint.
    eval_obs = None
    eval_masks = None
    eval_karma = None
    eval_urgency = None
    eval_agent_ids = None

    for ep in range(training_episodes):
        env.reset()
        last_collected_day = env.day
        route_counts = np.zeros(args.num_routes, dtype=np.int32)
        step_in_episode = 0
        turn_count = 0

        pending = {a: None for a in mutated_agent_names}
        traj = {
            a: {k: [] for k in (
                "obs", "central_obs", "mask", "act", "logp", "val", "rew", "done"
            )}
            for a in mutated_agent_names
        }
        ep_return = {a: 0.0 for a in mutated_agent_names}

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            done = termination or truncation
            acted_agent = False

            if agent not in agent_name_to_idx:
                if not done:
                    raise RuntimeError(f"Active agent {agent} is not controlled")
                action = None
            elif done:
                action = None
            else:
                if pending[agent] is not None:
                    raise RuntimeError(
                        f"Agent {agent} has an unresolved action before its next turn"
                    )

                obs_vec, mask_arr = extract_obs_and_mask(obs)
                if obs_vec.size != obs_dim:
                    raise ValueError(
                        f"Agent {agent}: observation length {obs_vec.size} != {obs_dim}"
                    )
                if mask_arr is None:
                    mask_arr = np.ones(action_dim, dtype=np.int8)
                else:
                    mask_arr = np.asarray(mask_arr, dtype=np.int8).reshape(-1)
                if mask_arr.size != action_dim or not np.any(mask_arr):
                    raise ValueError(
                        f"Agent {agent}: invalid action mask, size={mask_arr.size}"
                    )

                machine = machine_by_name[agent]
                karma = float(machine.karma_balance)
                urgency = round(float(machine.urgency), 1)

                if not saved_initial_policy:
                    key = (karma, urgency)
                    if eval_counts[key] < SAMPLES_PER_KARMA_URGENCY:
                        eval_records.append({
                            "obs": obs_vec.copy(),
                            "mask": mask_arr.copy(),
                            "karma": karma,
                            "urgency": urgency,
                            "agent_id": agent,
                        })
                        eval_counts[key] += 1

                global_features = build_global_features(
                    args=args,
                    step_in_episode=step_in_episode,
                    max_steps_hint=max_steps_hint,
                    route_counts=route_counts,
                )
                central_obs_vec = make_central_obs(obs_vec, global_features)
                act_idx, logp, value = select_action_masked_joint_mappo(
                    actor=actor,
                    critic=critic,
                    obs_vec=obs_vec,
                    central_obs_vec=central_obs_vec,
                    action_mask=mask_arr,
                    device=device,
                    args=args,
                )
                action = joint_index_to_multidiscrete(act_idx, nvec, strides)
                pending[agent] = {
                    "obs": obs_vec.copy(),
                    "central_obs": central_obs_vec.copy(),
                    "mask": mask_arr.copy(),
                    "act": act_idx,
                    "logp": logp,
                    "val": value,
                }
                acted_agent = True
                step_in_episode += 1

            env.step(action)

            # Read the assigned route for the action just taken, before
            # resetting route counts for the following commuting day.
            if acted_agent:
                assigned_route = int(machine_by_name[agent].route)
                if not (0 <= assigned_route < args.num_routes):
                    raise ValueError(
                        f"Invalid assigned route {assigned_route} for {agent}"
                    )
                route_counts[assigned_route] += 1

            # The environment writes this snapshot after _assign_rewards()
            # and before replacing the active vehicle population.
            if env.completed_day_index > last_collected_day:
                day_rewards = dict(env.completed_day_rewards)
                for completed_agent, day_reward in day_rewards.items():
                    if completed_agent not in pending:
                        raise RuntimeError(
                            f"Unknown completed agent {completed_agent} on "
                            f"day {env.completed_day_index}"
                        )
                    prev = pending[completed_agent]
                    if prev is None:
                        raise RuntimeError(
                            f"No pending action for {completed_agent} "
                            f"on completed day {env.completed_day_index}"
                        )
                    for key in ("obs", "central_obs", "mask", "act", "logp", "val"):
                        traj[completed_agent][key].append(prev[key])
                    traj[completed_agent]["rew"].append(float(day_reward))
                    # Daily independent transitions (no bootstrapping across days).
                    traj[completed_agent]["done"].append(0.0)
                    ep_return[completed_agent] += float(day_reward)
                    pending[completed_agent] = None

                print(
                    f"[TRAIN EP {ep + 1}] Completed day "
                    f"{env.completed_day_index}: {len(day_rewards)} rewards"
                )
                last_collected_day = env.completed_day_index
                route_counts[:] = 0

            turn_count += 1

        unresolved = [a for a, item in pending.items() if item is not None]
        if unresolved:
            raise RuntimeError(
                f"Training episode {ep + 1} ended with unresolved actions: "
                f"{unresolved[:10]}"
            )

        # ========================================================
        # MARK THE END OF THE 30-DAY EPISODE
        # ========================================================
        for agent in mutated_agent_names:
            if traj[agent]["done"]:
                traj[agent]["done"][-1] = 1.0

        # Freeze exactly the first episode's evaluation states and reuse them
        # for all policy checkpoints.
        if not saved_initial_policy:
            if not eval_records:
                raise RuntimeError("No evaluation states collected in episode 1")
            eval_obs = np.stack([r["obs"] for r in eval_records]).astype(np.float32)
            eval_masks = np.stack([r["mask"] for r in eval_records]).astype(np.int8)
            eval_karma = np.asarray([r["karma"] for r in eval_records], dtype=np.float32)
            eval_urgency = np.asarray([r["urgency"] for r in eval_records], dtype=np.float32)
            eval_agent_ids = np.asarray([r["agent_id"] for r in eval_records])

            print(
                f"[evaluation bank] states={len(eval_obs)}, "
                f"obs_dim={obs_dim}, action_dim={action_dim}"
            )
            save_mappo_policy_checkpoint(
                actor=actor,
                eval_obs=eval_obs,
                eval_masks=eval_masks,
                eval_karma=eval_karma,
                eval_urgency=eval_urgency,
                eval_agent_ids=eval_agent_ids,
                nvec=nvec,
                device=device,
                args=args,
                checkpoint_dir=args.checkpoint_dir,
                step=0,
                phase="initial",
            )
            saved_initial_policy = True

        # Assemble PPO data across every agent who participated.
        active_trajectories = [
            traj[a] for a in mutated_agent_names if traj[a]["obs"]
        ]
        if not active_trajectories:
            print(f"[WARNING] No transitions in training episode {ep + 1}")
            pbar.update(1)
            continue

        obs_list, central_list, mask_list = [], [], []
        act_list, logp_list, adv_list, ret_list = [], [], [], []

        for t in active_trajectories:
            rewards = np.asarray(t["rew"], dtype=np.float32)
            values = np.asarray(t["val"], dtype=np.float32)
            dones = np.asarray(t["done"], dtype=np.float32)
            adv, returns = compute_gae(
                rewards, values, dones, args.gamma, args.gae_lambda
            )
            obs_list.append(np.asarray(t["obs"], dtype=np.float32))
            central_list.append(np.asarray(t["central_obs"], dtype=np.float32))
            mask_list.append(np.asarray(t["mask"], dtype=np.int8))
            act_list.append(np.asarray(t["act"], dtype=np.int64))
            logp_list.append(np.asarray(t["logp"], dtype=np.float32))
            adv_list.append(adv)
            ret_list.append(returns)

        batch = {
            "obs": torch.as_tensor(np.concatenate(obs_list), dtype=torch.float32, device=device),
            "central_obs": torch.as_tensor(np.concatenate(central_list), dtype=torch.float32, device=device),
            "mask": torch.as_tensor(np.concatenate(mask_list), dtype=torch.int8, device=device),
            "act": torch.as_tensor(np.concatenate(act_list), dtype=torch.int64, device=device),
            "logp_old": torch.as_tensor(np.concatenate(logp_list), dtype=torch.float32, device=device),
            "adv": torch.as_tensor(np.concatenate(adv_list), dtype=torch.float32, device=device),
            "ret": torch.as_tensor(np.concatenate(ret_list), dtype=torch.float32, device=device),
        }

        metrics = {}
        for _ in range(args.epochs):
            metrics = ppo_update_masked_joint_mappo(
                actor=actor,
                critic=critic,
                actor_opt=actor_opt,
                critic_opt=critic_opt,
                batch=batch,
                args=args,
            )

        if ((ep + 1) % checkpoint_interval == 0 or
                (ep + 1) == training_episodes):
            save_mappo_policy_checkpoint(
                actor=actor,
                eval_obs=eval_obs,
                eval_masks=eval_masks,
                eval_karma=eval_karma,
                eval_urgency=eval_urgency,
                eval_agent_ids=eval_agent_ids,
                nvec=nvec,
                device=device,
                args=args,
                checkpoint_dir=args.checkpoint_dir,
                step=ep + 1,
                phase="training",
            )

        mean_return = float(np.mean(list(ep_return.values())))
        running_ep_returns.append(mean_return)
        running_ep_lens.append(turn_count)
        if ((ep + 1) % args.log_every_episodes == 0 or
                (ep + 1) == training_episodes):
            print(
                f"[EP {ep + 1}] mean_return={np.mean(running_ep_returns):.4f} "
                f"mean_turns={np.mean(running_ep_lens):.1f} "
                f"actor_loss={metrics.get('actor_loss', 0):.4f} "
                f"critic_loss={metrics.get('critic_loss', 0):.4f} "
                f"entropy={metrics.get('entropy', 0):.4f}"
            )
            running_ep_returns.clear()
            running_ep_lens.clear()
        pbar.update(1)

    # ============================================================
    # TESTING
    # ============================================================
    actor.eval()
    critic.eval()
    pbar.set_description("MAPPO AEC testing (masked joint)")
    env.testing_urgency_function()

    for ep in range(testing_episodes):
        env.reset()
        last_collected_day = env.day
        route_counts = np.zeros(args.num_routes, dtype=np.int32)
        step_in_episode = 0
        pending = {a: None for a in mutated_agent_names}
        ep_return = {a: 0.0 for a in mutated_agent_names}

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            done = termination or truncation
            acted_agent = False

            if agent not in agent_name_to_idx:
                if not done:
                    raise RuntimeError(f"Active test agent {agent} is not controlled")
                action = None
            elif done:
                action = None
            else:
                if pending[agent] is not None:
                    raise RuntimeError(f"Unresolved test action for agent {agent}")
                obs_vec, mask_arr = extract_obs_and_mask(obs)
                if obs_vec.size != obs_dim:
                    raise ValueError(f"Test observation size {obs_vec.size} != {obs_dim}")
                if mask_arr is None:
                    mask_arr = np.ones(action_dim, dtype=np.int8)
                else:
                    mask_arr = np.asarray(mask_arr, dtype=np.int8).reshape(-1)
                if mask_arr.size != action_dim or not np.any(mask_arr):
                    raise ValueError(f"Invalid test action mask for {agent}")

                with torch.no_grad():
                    logits = actor(torch.as_tensor(
                        obs_vec, dtype=torch.float32, device=device
                    ))
                    valid = torch.as_tensor(mask_arr != 0, device=device)
                    logits = logits.masked_fill(~valid, args.masked_logits_neg_inf)
                    dist = torch.distributions.Categorical(logits=logits)
                    act_idx = int(dist.sample().item())

                action = joint_index_to_multidiscrete(act_idx, nvec, strides)
                pending[agent] = True
                acted_agent = True
                step_in_episode += 1

            env.step(action)

            if acted_agent:
                assigned_route = int(machine_by_name[agent].route)
                if not (0 <= assigned_route < args.num_routes):
                    raise ValueError(f"Invalid test route {assigned_route} for {agent}")
                route_counts[assigned_route] += 1

            if env.completed_day_index > last_collected_day:
                day_rewards = dict(env.completed_day_rewards)
                for completed_agent, day_reward in day_rewards.items():
                    if completed_agent not in pending or pending[completed_agent] is None:
                        raise RuntimeError(
                            f"Missing test action for {completed_agent} on "
                            f"day {env.completed_day_index}"
                        )
                    ep_return[completed_agent] += float(day_reward)
                    pending[completed_agent] = None
                print(
                    f"[TEST EP {ep + 1}] Completed day "
                    f"{env.completed_day_index}: {len(day_rewards)} rewards"
                )
                last_collected_day = env.completed_day_index
                route_counts[:] = 0

        unresolved = [a for a, item in pending.items() if item is not None]
        if unresolved:
            raise RuntimeError(
                f"Testing episode {ep + 1} ended with unresolved actions: "
                f"{unresolved[:10]}"
            )

        global_step = training_episodes + ep + 1
        save_mappo_policy_checkpoint(
            actor=actor,
            eval_obs=eval_obs,
            eval_masks=eval_masks,
            eval_karma=eval_karma,
            eval_urgency=eval_urgency,
            eval_agent_ids=eval_agent_ids,
            nvec=nvec,
            device=device,
            args=args,
            checkpoint_dir=args.checkpoint_dir,
            step=global_step,
            phase="testing",
        )
        print(
            f"[TEST EP {ep + 1}] "
            f"mean_return_over_agents={np.mean(list(ep_return.values())):.4f}"
        )
        pbar.update(1)

    pbar.close()
    env.plot_results()
    env.stop_simulation()


if __name__ == "__main__":
    main()
