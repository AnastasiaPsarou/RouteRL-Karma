from __future__ import annotations

from tqdm import tqdm
import os
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

    training_episodes: int = 34
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
    actor: nn.Module,
    eval_obs: np.ndarray,
    eval_masks: np.ndarray,
    device: torch.device,
    args: Args,
) -> np.ndarray:
    """
    Computes masked-softmax policy probabilities for a fixed evaluation set.

    The MAPPO actor is decentralized, so the policy snapshot only needs
    the local observations and action masks. The centralized critic is not
    used for these policy-probability checkpoints.

    eval_obs shape:
        (num_agents, num_eval_states, obs_dim)

    eval_masks shape:
        (num_agents, num_eval_states, action_dim)

    returns:
        policy_probs shape:
            (num_agents, num_eval_states, action_dim)
    """
    was_training = actor.training
    actor.eval()

    num_agents, num_eval_states, _obs_dim = eval_obs.shape
    action_dim = eval_masks.shape[-1]

    policy_probs = np.zeros(
        (num_agents, num_eval_states, action_dim),
        dtype=np.float32,
    )

    for agent_idx in range(num_agents):
        obs_t = torch.tensor(
            eval_obs[agent_idx],
            device=device,
            dtype=torch.float32,
        )

        mask_t = torch.tensor(
            eval_masks[agent_idx],
            device=device,
        )

        logits = actor(obs_t)

        valid = mask_t != 0
        row_has_any_valid = valid.any(dim=1)

        if row_has_any_valid.all():
            logits = logits.masked_fill(~valid, args.masked_logits_neg_inf)
        else:
            safe_logits = logits.clone()
            safe_logits[row_has_any_valid] = safe_logits[row_has_any_valid].masked_fill(
                ~valid[row_has_any_valid],
                args.masked_logits_neg_inf,
            )
            logits = safe_logits

        probs = torch.softmax(logits, dim=-1)
        policy_probs[agent_idx] = probs.cpu().numpy()

    if was_training:
        actor.train()

    return policy_probs


def save_mappo_policy_checkpoint(
    *,
    actor: nn.Module,
    eval_obs: np.ndarray,
    eval_masks: np.ndarray,
    device: torch.device,
    args: Args,
    checkpoint_dir: str | Path,
    step: int,
    phase: str,
) -> None:
    """
    Saves a MAPPO policy-probability checkpoint.

    The saved .npz has the same structure expected by your diagnostics script:
        step
        policy
        eval_action_masks

    policy shape:
        agents x eval_states x joint_actions
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    policy = compute_mappo_policy_snapshot(
        actor=actor,
        eval_obs=eval_obs,
        eval_masks=eval_masks,
        device=device,
        args=args,
    )

    output_path = checkpoint_dir / f"step_{step:08d}.npz"

    np.savez_compressed(
        output_path,
        step=np.array(step, dtype=np.int64),
        policy=policy.astype(np.float32),
        eval_action_masks=eval_masks.astype(bool),
        phase=np.array(phase),
    )

    print(f"[checkpoint] Saved {phase} policy checkpoint: {output_path}")


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

    new_machines_after_mutation = 300
    human_learning_episodes = 0
    training_episodes = args.training_episodes
    testing_episodes = args.testing_episodes
    total_episodes = human_learning_episodes + training_episodes + testing_episodes

    num_agents = 300
    origins = ["E0"]
    destinations = ["E17.600"]

    seed = args.seed
    records_folder = f"/scratch/tmp/psarou_karma_part_c/training_records_mappo_{seed}_route_0_4_route_1_3_max_bid_5_dr"
    plots_folder = f"/scratch/tmp/psarou_karma_part_c/plots_mappo_{seed}_route_0_4_route_1_3_max_bid_5_dr"

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
            "custom_network_folder": "network_analysis/network_base",
            "sumo_type": "sumo",
            "simulation_timesteps": 100,
        },
        "environment_parameters": {
            "observations_time_window": 10,
            "save_every": 5,
            "number_of_days": 30,
            "centrally_defined_price_route_0": 4,
            "centrally_defined_price_route_1": 3,
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
    env.reset()

    pre_mutation_agents = env.all_agents.copy()
    env.mutation(mutation_start_percentile=0)

    machines = env.machine_agents.copy()
    mutated_humans = {}
    for machine in machines:
        for human in pre_mutation_agents:
            if human.id == machine.id:
                mutated_humans[str(machine.id)] = human
                break

    mutated_agent_names = list(mutated_humans.keys())
    agent_name_to_idx = {name: i for i, name in enumerate(mutated_agent_names)}
    print("Controlled (mutated) agents:", len(mutated_agent_names))

    # -------------------------
    # Infer obs/action sizes
    # -------------------------
    obs_spaces = env._observation_spaces
    some_agent = mutated_agent_names[0]
    obs_space = obs_spaces[some_agent]

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

    saved_initial_policy = False

    # -------------------------
    # Training loop
    # -------------------------
    pbar = tqdm(total=training_episodes + testing_episodes, desc="MAPPO AEC training (masked joint)")

    running_ep_returns = []
    running_ep_lens = []

    for ep in range(training_episodes):
        env.reset()

        route_counts = np.zeros(args.num_routes, dtype=np.int32)
        step_in_episode = 0

        pending: Dict[str, Dict[str, Any] | None] = {a: None for a in mutated_agent_names}
        traj: Dict[str, Dict[str, List[Any]]] = {
            a: {"obs": [], "central_obs": [], "mask": [], "act": [], "logp": [], "val": [], "rew": [], "done": []}
            for a in mutated_agent_names
        }

        ep_return = {a: 0.0 for a in mutated_agent_names}
        turn_count = 0

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            done = termination or truncation

            if agent in agent_name_to_idx:
                # 1) Close previous pending step
                if pending[agent] is not None:
                    prev = pending[agent]
                    traj[agent]["obs"].append(prev["obs"])
                    traj[agent]["central_obs"].append(prev["central_obs"])
                    traj[agent]["mask"].append(prev["mask"])
                    traj[agent]["act"].append(prev["act"])
                    traj[agent]["logp"].append(prev["logp"])
                    traj[agent]["val"].append(prev["val"])
                    traj[agent]["rew"].append(float(reward))
                    traj[agent]["done"].append(float(done))
                    ep_return[agent] += float(reward)
                    pending[agent] = None

                # 2) Choose next action unless done
                if done:
                    action = None
                else:
                    obs_vec, mask_arr = extract_obs_and_mask(obs)

                    if mask_arr is None:
                        mask_arr = np.ones(action_dim, dtype=np.int8)
                    else:
                        mask_arr = mask_arr.reshape(-1)
                        if mask_arr.shape[0] != action_dim:
                            raise ValueError(
                                f"action_mask length {mask_arr.shape[0]} != action_dim {action_dim}. "
                                f"Expected prod(nvec)={action_dim}, nvec={nvec.tolist()}"
                            )

                    # Collect the first valid observation/mask for each learning agent.
                    # These fixed states are used for policy-change diagnostics.
                    if agent not in eval_obs_by_agent:
                        eval_obs_by_agent[agent] = obs_vec.copy().astype(np.float32)
                        eval_masks_by_agent[agent] = mask_arr.copy().astype(np.int8)

                    gfeats = build_global_features(
                        args=args,
                        step_in_episode=step_in_episode,
                        max_steps_hint=max_steps_hint,
                        route_counts=route_counts,
                    )
                    central_obs_vec = make_central_obs(obs_vec, gfeats)

                    act_idx, logp, val = select_action_masked_joint_mappo(
                        actor, critic, obs_vec, central_obs_vec, mask_arr, device, args
                    )
                    act_vec = joint_index_to_multidiscrete(act_idx, nvec, strides)
                    action = act_vec

                    # Update global aggregate with chosen route
                    route = int(act_vec[args.route_action_index])
                    if 0 <= route < args.num_routes:
                        route_counts[route] += 1

                    pending[agent] = {
                        "obs": obs_vec,
                        "central_obs": central_obs_vec,
                        "mask": mask_arr.astype(np.int8),
                        "act": act_idx,
                        "logp": logp,
                        "val": val,
                    }
                    step_in_episode += 1
            else:
                action = None

            env.step(action)
            turn_count += 1

        # Close any remaining pending
        for a in mutated_agent_names:
            if pending[a] is not None:
                prev = pending[a]
                traj[a]["obs"].append(prev["obs"])
                traj[a]["central_obs"].append(prev["central_obs"])
                traj[a]["mask"].append(prev["mask"])
                traj[a]["act"].append(prev["act"])
                traj[a]["logp"].append(prev["logp"])
                traj[a]["val"].append(prev["val"])
                traj[a]["rew"].append(0.0)
                traj[a]["done"].append(1.0)
                pending[a] = None

        # -------------------------
        # Build fixed eval set and save initial policy once
        # -------------------------
        if not saved_initial_policy:
            missing_agents = [
                agent for agent in mutated_agent_names
                if agent not in eval_obs_by_agent
            ]

            if missing_agents:
                print(
                    f"[WARNING] Missing eval observations for {len(missing_agents)} agents. "
                    f"Using zero observations and all-valid masks for them."
                )

            eval_obs_list = []
            eval_masks_list = []

            for agent in mutated_agent_names:
                if agent in eval_obs_by_agent:
                    eval_obs_list.append(eval_obs_by_agent[agent])
                    eval_masks_list.append(eval_masks_by_agent[agent])
                else:
                    eval_obs_list.append(np.zeros(obs_dim, dtype=np.float32))
                    eval_masks_list.append(np.ones(action_dim, dtype=np.int8))

            eval_obs = np.stack(eval_obs_list, axis=0)[:, None, :]
            eval_masks = np.stack(eval_masks_list, axis=0)[:, None, :]

            save_mappo_policy_checkpoint(
                actor=actor,
                eval_obs=eval_obs,
                eval_masks=eval_masks,
                device=device,
                args=args,
                checkpoint_dir=args.checkpoint_dir,
                step=0,
                phase="training",
            )

            saved_initial_policy = True



        # -------------------------
        # Build PPO batch from all agents
        # -------------------------
        obs_list = []
        central_obs_list = []
        mask_list = []
        act_list = []
        logp_list = []
        adv_list = []
        ret_list = []

        for a in mutated_agent_names:
            if len(traj[a]["obs"]) == 0:
                continue

            rews = np.asarray(traj[a]["rew"], dtype=np.float32)
            vals = np.asarray(traj[a]["val"], dtype=np.float32)
            dones = np.asarray(traj[a]["done"], dtype=np.float32)
            adv, ret = compute_gae(rews, vals, dones, args.gamma, args.gae_lambda)

            obs_list.append(np.asarray(traj[a]["obs"], dtype=np.float32))
            central_obs_list.append(np.asarray(traj[a]["central_obs"], dtype=np.float32))
            mask_list.append(np.asarray(traj[a]["mask"], dtype=np.int8))
            act_list.append(np.asarray(traj[a]["act"], dtype=np.int64))
            logp_list.append(np.asarray(traj[a]["logp"], dtype=np.float32))
            adv_list.append(adv)
            ret_list.append(ret)

        if len(obs_list) == 0:
            pbar.update(1)
            continue

        obs_np = np.concatenate(obs_list, axis=0)
        central_obs_np = np.concatenate(central_obs_list, axis=0)
        mask_np = np.concatenate(mask_list, axis=0)
        act_np = np.concatenate(act_list, axis=0)
        logp_np = np.concatenate(logp_list, axis=0)
        adv_np = np.concatenate(adv_list, axis=0)
        ret_np = np.concatenate(ret_list, axis=0)

        batch = {
            "obs": torch.from_numpy(obs_np).to(device=device, dtype=torch.float32),
            "central_obs": torch.from_numpy(central_obs_np).to(device=device, dtype=torch.float32),
            "mask": torch.from_numpy(mask_np).to(device=device),
            "act": torch.from_numpy(act_np).to(device=device, dtype=torch.int64),
            "logp_old": torch.from_numpy(logp_np).to(device=device, dtype=torch.float32),
            "adv": torch.from_numpy(adv_np).to(device=device, dtype=torch.float32),
            "ret": torch.from_numpy(ret_np).to(device=device, dtype=torch.float32),
        }

        metrics = {}
        for _ in range(args.epochs):
            metrics = ppo_update_masked_joint_mappo(actor, critic, actor_opt, critic_opt, batch, args)

        # Save policy checkpoint after PPO update
        if (
            eval_obs is not None
            and eval_masks is not None
            and ((ep + 1) % checkpoint_interval == 0 or (ep + 1) == training_episodes)
        ):
            save_mappo_policy_checkpoint(
                actor=actor,
                eval_obs=eval_obs,
                eval_masks=eval_masks,
                device=device,
                args=args,
                checkpoint_dir=args.checkpoint_dir,
                step=ep+1,
                phase="training",
            )

        mean_return = float(np.mean(list(ep_return.values())))
        running_ep_returns.append(mean_return)
        running_ep_lens.append(turn_count)

        if (ep + 1) % args.log_every_episodes == 0:
            print(
                f"[EP {ep+1}] mean_return_over_agents={np.mean(running_ep_returns):.4f} "
                f"mean_turns={np.mean(running_ep_lens):.1f} "
                f"actor_loss={metrics.get('actor_loss', 0):.4f} "
                f"critic_loss={metrics.get('critic_loss', 0):.4f} "
                f"entropy={metrics.get('entropy', 0):.4f} "
                f"nvec={nvec.tolist()} action_dim={action_dim} central_obs_dim={central_obs_dim}"
            )
            running_ep_returns.clear()
            running_ep_lens.clear()

        pbar.update(1)

    # -------------------------
    # Testing (greedy, masked)
    # -------------------------
    actor.eval()
    critic.eval()


    pbar.set_description("MAPPO AEC testing (masked joint)")
    env.testing_urgency_function()

    for ep in range(testing_episodes):
        env.reset()

        route_counts = np.zeros(args.num_routes, dtype=np.int32)
        step_in_episode = 0

        pending: Dict[str, Dict[str, Any] | None] = {a: None for a in mutated_agent_names}
        ep_return = {a: 0.0 for a in mutated_agent_names}

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            done = termination or truncation

            if agent in agent_name_to_idx:
                if pending[agent] is not None:
                    ep_return[agent] += float(reward)
                    pending[agent] = None

                if done:
                    action = None
                else:
                    obs_vec, mask_arr = extract_obs_and_mask(obs)
                    if mask_arr is None:
                        mask_arr = np.ones(action_dim, dtype=np.int8)
                    else:
                        mask_arr = mask_arr.reshape(-1)

                    # Greedy action from actor (critic not needed here)
                    obs_t = torch.from_numpy(obs_vec).to(device=device, dtype=torch.float32)
                    logits = actor(obs_t)

                    mask_t = torch.from_numpy(mask_arr).to(device=device)
                    valid = mask_t != 0
                    if valid.any():
                        logits = logits.masked_fill(~valid, args.masked_logits_neg_inf)

                    dist = torch.distributions.Categorical(logits=logits)
                    act_idx = int(dist.sample().item())
                    #act_idx = int(torch.argmax(logits).item())
                    act_vec = joint_index_to_multidiscrete(act_idx, nvec, strides)
                    action = act_vec

                    # update global aggregates
                    route = int(act_vec[args.route_action_index])
                    if 0 <= route < args.num_routes:
                        route_counts[route] += 1

                    pending[agent] = {"dummy": True}
                    step_in_episode += 1
            else:
                action = None

            env.step(action)

        global_step = args.training_episodes + ep + 1

        save_mappo_policy_checkpoint(
            actor=actor,
            eval_obs=eval_obs,
            eval_masks=eval_masks,
            device=device,
            args=args,
            checkpoint_dir=args.checkpoint_dir,
            step=global_step,
            phase="testing",
        )

        mean_return = float(np.mean(list(ep_return.values())))
        print(f"[TEST EP {ep+1}] mean_return_over_agents={mean_return:.4f}")
        pbar.update(1)

    pbar.close()

    env.plot_results()
    env.stop_simulation()


if __name__ == "__main__":
    main()