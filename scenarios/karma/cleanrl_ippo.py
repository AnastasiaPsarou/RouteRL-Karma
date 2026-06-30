from __future__ import annotations

from tqdm import tqdm
import os
import sys
import random
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ------------------------------------------------------------
# Your imports
# ------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Use the environment that has action masking / karma pricing
from routerl_aec_environment_karma import TrafficEnvironment

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
@dataclass
class Args:
    seed: int = 25
    device: str = "cpu"

    # episodes
    training_episodes: int = 34
    testing_episodes: int = 10

    # PPO hyperparams
    gamma: float = 0.99
    gae_lambda: float = 0.90
    ppo_clip: float = 0.2
    entropy_coef: float = 0.001
    value_coef: float = 0.5

    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    optimizer: str = "Adam"

    epochs: int = 4
    minibatch_size: int = 4096  # total (agent-steps) per minibatch
    max_grad_norm: float = 0.5  # set <=0 to disable

    # Logging
    log_every_episodes: int = 10

    # Masking
    masked_logits_neg_inf: float = -1e9  # what to set invalid logits to
    checkpoint_dir: str = f"/scratch/tmp/psarou_karma_part_c/checkpoints_ippo_seed_{seed}"


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
        f"/scratch/tmp/psarou_karma_part_c/checkpoints_ippo_masked_joint_300_agents_seed_{args.seed}"
        f"_route_0_4_route_1_3_max_bid_5"
    )

    return args

# ------------------------------------------------------------
# Networks
# ------------------------------------------------------------
class MLPActorJoint(nn.Module):
    """
    Actor over the JOINT action space:
      action_dim = prod(nvec) = number of all joint combinations

    We'll apply action_mask to these logits before sampling / logprob.
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
        # obs: (..., obs_dim) -> logits (..., action_dim)
        return self.net(obs)


class MLPCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden=(128, 128)):
        super().__init__()
        layers: List[nn.Module] = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def extract_obs_and_mask(obs: Any) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Your env returns per-agent observation dict like:
      {"observation": <vec>, "action_mask": <mask>}
    This returns:
      obs_vec: float32 flattened
      mask: int8/bool array shape (action_dim,) or None
    """
    if isinstance(obs, dict):
        obs_vec = np.asarray(obs.get("observation", []), dtype=np.float32).reshape(-1)
        mask = obs.get("action_mask", None)
        if mask is None:
            return obs_vec, None
        mask_arr = np.asarray(mask)
        # allow int8 {0,1} or bool
        if mask_arr.dtype != np.bool_:
            mask_arr = mask_arr.astype(np.int8)
        return obs_vec, mask_arr
    else:
        # no mask
        return np.asarray(obs, dtype=np.float32).reshape(-1), None


def compute_action_strides(nvec: np.ndarray) -> np.ndarray:
    """
    For decoding joint index -> MultiDiscrete vector.
    Example nvec=[n0,n1,n2], strides=[n1*n2, n2, 1]
    """
    nvec = np.asarray(nvec, dtype=np.int64)
    strides = np.ones_like(nvec)
    for i in range(len(nvec) - 2, -1, -1):
        strides[i] = strides[i + 1] * nvec[i + 1]
    return strides


def joint_index_to_multidiscrete(index: int, nvec: np.ndarray, strides: np.ndarray) -> List[int]:
    """
    Decode joint action index (0..prod(nvec)-1) into a length-K list.
    """
    idx = int(index)
    out = []
    for i in range(len(nvec)):
        val = idx // int(strides[i])
        idx = idx % int(strides[i])
        out.append(int(val))
    return out


@torch.no_grad()
def select_action_masked_joint(
    actor: nn.Module,
    critic: nn.Module,
    obs_vec: np.ndarray,
    action_mask: Optional[np.ndarray],
    device: torch.device,
    args: Args,
) -> Tuple[int, float, float]:
    """
    Samples a JOINT action index from masked categorical.

    Returns:
      act_idx: int
      logp: float
      value: float
    """
    obs_t = torch.from_numpy(obs_vec).to(device=device, dtype=torch.float32)
    logits = actor(obs_t)  # (action_dim,)

    if action_mask is not None:
        mask_t = torch.from_numpy(action_mask).to(device=device)
        # mask_t can be int8/bool. Valid where mask_t != 0.
        valid = mask_t != 0
        if valid.any():
            logits = logits.masked_fill(~valid, args.masked_logits_neg_inf)
        else:
            # all invalid: fall back to unmasked (or could pick 0)
            pass

    dist = torch.distributions.Categorical(logits=logits)
    act = dist.sample()
    logp = dist.log_prob(act)
    value = critic(obs_t).squeeze(-1)

    return int(act.item()), float(logp.item()), float(value.item())


@torch.no_grad()
def compute_ippo_policy_snapshot(
    actor: nn.Module,
    eval_obs: np.ndarray,
    eval_masks: np.ndarray,
    device: torch.device,
    args: Args,
) -> np.ndarray:
    """
    Computes masked-softmax policy probabilities for a fixed evaluation set.

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


def save_ippo_policy_checkpoint(
    checkpoint_path: str,
    step: int,
    actor: nn.Module,
    eval_obs: np.ndarray,
    eval_masks: np.ndarray,
    device: torch.device,
    args: Args,
    nvec: np.ndarray,
    strides: np.ndarray,
):
    """
    Saves the current shared IPPO policy on a fixed evaluation set.
    """
    policy_probs = compute_ippo_policy_snapshot(
        actor=actor,
        eval_obs=eval_obs,
        eval_masks=eval_masks,
        device=device,
        args=args,
    )

    np.savez_compressed(
        checkpoint_path,
        step=np.array(step, dtype=np.int64),
        policy=policy_probs.astype(np.float32),
        aggregate_policy=policy_probs.mean(axis=0).astype(np.float32),
        eval_obs=eval_obs.astype(np.float32),
        eval_action_masks=eval_masks.astype(np.int8),
        nvec=nvec.astype(np.int64),
        strides=strides.astype(np.int64),
        action_dim=np.array(policy_probs.shape[-1], dtype=np.int64),
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


def ppo_update_masked_joint(
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
      act: (M,)
      logp_old: (M,)
      adv: (M,)
      ret: (M,)
      mask: (M, action_dim)  (0/1 or bool)
    """
    obs = batch["obs"]
    act = batch["act"]
    logp_old = batch["logp_old"]
    adv = batch["adv"]
    ret = batch["ret"]
    mask = batch["mask"]  # (M, action_dim)

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
        mb_act = act[mb_idx]
        mb_logp_old = logp_old[mb_idx]
        mb_adv = adv[mb_idx]
        mb_ret = ret[mb_idx]
        mb_mask = mask[mb_idx]  # (mb, action_dim)

        logits = actor(mb_obs)  # (mb, action_dim)

        # Apply mask: invalid logits -> very negative
        valid = mb_mask != 0
        # If some rows have all zeros, we leave them unmasked to avoid NaNs
        row_has_any_valid = valid.any(dim=1)
        if row_has_any_valid.all():
            logits = logits.masked_fill(~valid, args.masked_logits_neg_inf)
        else:
            # mixed: mask only rows that have at least one valid
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

        value = critic(mb_obs).squeeze(-1)
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

    # -------------------------
    # Env setup (based on your IDQN config)
    # -------------------------
    new_machines_after_mutation = 300
    human_learning_episodes = 0
    training_episodes = args.training_episodes
    testing_episodes = args.testing_episodes
    total_episodes = human_learning_episodes + training_episodes + testing_episodes

    num_agents = 300
    origins = ["E0"]
    destinations = ["E17.600"]

    seed = args.seed
    records_folder = f"/scratch/tmp/psarou_karma_part_c/training_records_ippo_{seed}_max_bid_5_route_0_4_route_1_3"
    plots_folder = f"/scratch/tmp/psarou_karma_part_c/plots_ippo_{seed}_max_bid_5_route_0_4_route_1_3"

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
            "save_every": 1,
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

    mutated_agent_names = list(mutated_humans.keys())
    agent_name_to_idx = {name: i for i, name in enumerate(mutated_agent_names)}
    print("Controlled (mutated) agents:", len(mutated_agent_names))

    # -------------------------
    # Infer obs/action sizes
    # -------------------------
    obs_spaces = env._observation_spaces
    some_agent = mutated_agent_names[0]
    obs_space = obs_spaces[some_agent]

    # dict obs space: use 'observation'
    try:
        obs_shape = obs_space["observation"].shape
    except Exception:
        obs_shape = obs_space.shape
    obs_dim = int(np.prod(obs_shape))

    # MultiDiscrete action space is still useful to decode joint index to (3,)
    act_spaces = env._action_spaces
    act_space = act_spaces[some_agent]
    nvec = np.asarray(act_space.nvec, dtype=np.int64)  # length 3
    strides = compute_action_strides(nvec)
    action_dim = int(np.prod(nvec))

    actor = MLPActorJoint(obs_dim, action_dim, hidden=(64, 64)).to(device)
    critic = MLPCritic(obs_dim, hidden=(256, 256)).to(device)

    Optimizer = getattr(optim, args.optimizer)
    actor_opt = Optimizer(actor.parameters(), lr=args.lr_actor)
    critic_opt = Optimizer(critic.parameters(), lr=args.lr_critic)

    # -------------------------
    # Policy checkpoint setup
    # -------------------------
    checkpoint_dir = (
        f"/scratch/tmp/psarou_karma_part_c/checkpoints_ippo_{seed}"
        f"_max_bid_5_route_0_4_route_1_3"
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
    pbar = tqdm(total=training_episodes + testing_episodes, desc="IPPO AEC training (masked joint)")

    running_ep_returns = []
    running_ep_lens = []

    for ep in range(training_episodes):
        env.reset()

        # pending: last action waiting for reward
        pending: Dict[str, Dict[str, Any] | None] = {a: None for a in mutated_agent_names}
        traj: Dict[str, Dict[str, List[Any]]] = {
            a: {"obs": [], "mask": [], "act": [], "logp": [], "val": [], "rew": [], "done": []}
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

                    # Ensure mask is present and correct length
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

                    act_idx, logp, val = select_action_masked_joint(
                        actor, critic, obs_vec, mask_arr, device, args
                    )
                    # decode joint index to MultiDiscrete vector action for env.step
                    act_vec = joint_index_to_multidiscrete(act_idx, nvec, strides)
                    action = act_vec

                    pending[agent] = {
                        "obs": obs_vec,
                        "mask": mask_arr.astype(np.int8),
                        "act": act_idx,
                        "logp": logp,
                        "val": val,
                    }
            else:
                action = None

            env.step(action)
            turn_count += 1

        # Close any remaining pending with terminal reward=0 fallback
        for a in mutated_agent_names:
            if pending[a] is not None:
                prev = pending[a]
                traj[a]["obs"].append(prev["obs"])
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

            save_ippo_policy_checkpoint(
                checkpoint_path=os.path.join(checkpoint_dir, "step_00000000.npz"),
                step=0,
                actor=actor,
                eval_obs=eval_obs,
                eval_masks=eval_masks,
                device=device,
                args=args,
                nvec=nvec,
                strides=strides,
            )

            saved_initial_policy = True

        # -------------------------
        # Build PPO batch from all agents
        # -------------------------
        obs_list = []
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
            print("vals is: ", vals, "\n\n")
            dones = np.asarray(traj[a]["done"], dtype=np.float32)
            adv, ret = compute_gae(rews, vals, dones, args.gamma, args.gae_lambda)

            obs_list.append(np.asarray(traj[a]["obs"], dtype=np.float32))            # (T, obs_dim)
            mask_list.append(np.asarray(traj[a]["mask"], dtype=np.int8))             # (T, action_dim)
            act_list.append(np.asarray(traj[a]["act"], dtype=np.int64))              # (T,)
            logp_list.append(np.asarray(traj[a]["logp"], dtype=np.float32))          # (T,)
            adv_list.append(adv)
            ret_list.append(ret)

        if len(obs_list) == 0:
            pbar.update(1)
            continue

        obs_np = np.concatenate(obs_list, axis=0)     # (M, obs_dim)
        mask_np = np.concatenate(mask_list, axis=0)   # (M, action_dim)
        act_np = np.concatenate(act_list, axis=0)     # (M,)
        logp_np = np.concatenate(logp_list, axis=0)   # (M,)
        adv_np = np.concatenate(adv_list, axis=0)     # (M,)
        ret_np = np.concatenate(ret_list, axis=0)     # (M,)

        batch = {
            "obs": torch.from_numpy(obs_np).to(device=device, dtype=torch.float32),
            "mask": torch.from_numpy(mask_np).to(device=device),
            "act": torch.from_numpy(act_np).to(device=device, dtype=torch.int64),
            "logp_old": torch.from_numpy(logp_np).to(device=device, dtype=torch.float32),
            "adv": torch.from_numpy(adv_np).to(device=device, dtype=torch.float32),
            "ret": torch.from_numpy(ret_np).to(device=device, dtype=torch.float32),
        }

        metrics = {}
        for _ in range(args.epochs):
            metrics = ppo_update_masked_joint(actor, critic, actor_opt, critic_opt, batch, args)

        # Save policy checkpoint after PPO update
        if (
            eval_obs is not None
            and eval_masks is not None
            and ((ep + 1) % checkpoint_interval == 0 or (ep + 1) == training_episodes)
        ):
            save_ippo_policy_checkpoint(
                checkpoint_path=os.path.join(checkpoint_dir, f"step_{ep + 1:08d}.npz"),
                step=ep + 1,
                actor=actor,
                eval_obs=eval_obs,
                eval_masks=eval_masks,
                device=device,
                args=args,
                nvec=nvec,
                strides=strides,
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
                f"nvec={nvec.tolist()} action_dim={action_dim}"
            )
            running_ep_returns.clear()
            running_ep_lens.clear()

        pbar.update(1)

    # Save final policy explicitly
    if eval_obs is not None and eval_masks is not None:
        save_ippo_policy_checkpoint(
            checkpoint_path=os.path.join(checkpoint_dir, "final_policy.npz"),
            step=training_episodes,
            actor=actor,
            eval_obs=eval_obs,
            eval_masks=eval_masks,
            device=device,
            args=args,
            nvec=nvec,
            strides=strides,
        )
    else:
        print("[WARNING] No fixed eval set was built, so final_policy.npz was not saved.")

    # -------------------------
    # Testing (greedy, masked)
    # -------------------------
    actor.eval()
    critic.eval()
    pbar.set_description("IPPO AEC testing (masked joint)")

    for ep in range(testing_episodes):
        env.reset()

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

                    obs_t = torch.from_numpy(obs_vec).to(device=device, dtype=torch.float32)
                    logits = actor(obs_t)

                    mask_t = torch.from_numpy(mask_arr).to(device=device)
                    valid = mask_t != 0
                    if valid.any():
                        logits = logits.masked_fill(~valid, args.masked_logits_neg_inf)

                    #act_idx = int(torch.argmax(logits).item())
                    dist = torch.distributions.Categorical(logits=logits)
                    act_idx = int(dist.sample().item())
                    act_vec = joint_index_to_multidiscrete(act_idx, nvec, strides)
                    action = act_vec
                    pending[agent] = {"dummy": True}
            else:
                action = None

            env.step(action)

        """if ep == 1:
            return"""
        mean_return = float(np.mean(list(ep_return.values())))
        print(f"[TEST EP {ep+1}] mean_return_over_agents={mean_return:.4f}")

         # Save policy checkpoint during testing
        if (
            eval_obs is not None
            and eval_masks is not None
            and ((ep + 1) % testing_checkpoint_interval == 0 or (ep + 1) == testing_episodes)
        ):
            testing_step = training_episodes + ep + 1

            save_ippo_policy_checkpoint(
                checkpoint_path=os.path.join(
                    checkpoint_dir,
                    f"testing_step_{testing_step:08d}.npz"
                ),
                step=testing_step,
                actor=actor,
                eval_obs=eval_obs,
                eval_masks=eval_masks,
                device=device,
                args=args,
                nvec=nvec,
                strides=strides,
            )

        pbar.update(1)

    pbar.close()

    env.plot_results()
    env.stop_simulation()


if __name__ == "__main__":
    main()
