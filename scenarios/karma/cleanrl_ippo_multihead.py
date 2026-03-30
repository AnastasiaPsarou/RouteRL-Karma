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
    seed: int = 9
    device: str = "cpu"

    # episodes
    training_episodes: int = 100
    testing_episodes: int = 1

    # PPO hyperparams
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    entropy_coef: float = 0.001
    value_coef: float = 0.5

    lr_actor: float = 8e-4
    lr_critic: float = 8e-4
    optimizer: str = "Adam"

    epochs: int = 4
    minibatch_size: int = 4096  # total (agent-steps) per minibatch
    max_grad_norm: float = 0.5  # set <=0 to disable

    # Logging
    log_every_episodes: int = 10

    # Masking
    masked_logits_neg_inf: float = -1e9  # what to set invalid logits to


# ------------------------------------------------------------
# Networks
# ------------------------------------------------------------
class MLPActorMultiHead(nn.Module):
    """
    Factorized policy for MultiDiscrete: shared trunk + one categorical head per component.
    If nvec = [10,10,10], you get 3 heads, each outputs 10 logits.
    """
    def __init__(self, obs_dim: int, nvec: np.ndarray, hidden=(64, 64)):
        super().__init__()
        self.nvec = list(map(int, nvec.tolist()))

        layers: List[nn.Module] = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(last, n) for n in self.nvec])

    def forward(self, obs: torch.Tensor) -> List[torch.Tensor]:
        """
        obs: (..., obs_dim)
        returns: list of logits per head, each (..., nvec[i])
        """
        z = self.trunk(obs)
        return [head(z) for head in self.heads]


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
      mask: int8/bool array shape (prod(nvec),) or None
    """
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


def joint_mask_to_head_masks_torch(mask_joint: torch.Tensor, nvec: np.ndarray) -> List[torch.Tensor]:
    """
    Convert joint action mask of shape (..., prod(nvec)) into per-head masks:
      returns list of tensors with shapes (..., nvec[i])

    Rule: head action value a_i is valid if there exists at least one combination
    of other components that yields a valid joint action.

    mask_joint: torch tensor of 0/1 or bool with shape (..., prod(nvec))
    """
    nvec_list = list(map(int, nvec.tolist()))
    *batch_shape, joint_dim = mask_joint.shape
    expected = int(np.prod(nvec_list))
    if joint_dim != expected:
        raise ValueError(f"joint mask length {joint_dim} != prod(nvec) {expected} (nvec={nvec_list})")

    m = (mask_joint != 0).reshape(*batch_shape, *nvec_list)  # (..., n0, n1, ..., nK)

    head_masks: List[torch.Tensor] = []
    K = len(nvec_list)
    for i in range(K):
        # reduce over all action dims except i
        reduce_dims = [len(batch_shape) + d for d in range(K) if d != i]
        hm = m.any(dim=reduce_dims)  # (..., nvec[i])
        head_masks.append(hm)
    return head_masks


@torch.no_grad()
def select_action_masked_multhead(
    actor: nn.Module,
    critic: nn.Module,
    obs_vec: np.ndarray,
    action_mask_joint: Optional[np.ndarray],
    device: torch.device,
    args: Args,
    nvec: np.ndarray,
) -> Tuple[List[int], float, float]:
    """
    Samples a MultiDiscrete action vector from factorized masked categorical policy.

    Returns:
      act_vec: list[int] (len = len(nvec))
      logp: float (sum of per-head log probs)
      value: float
    """
    obs_t = torch.from_numpy(obs_vec).to(device=device, dtype=torch.float32)

    logits_list = actor(obs_t)  # list of (nvec[i],)

    head_masks = None
    if action_mask_joint is not None:
        mask_t = torch.from_numpy(action_mask_joint).to(device=device)
        mask_t = mask_t.reshape(1, -1)  # (1, prod)
        head_masks = joint_mask_to_head_masks_torch(mask_t, nvec)
        head_masks = [hm.squeeze(0) for hm in head_masks]  # each (nvec[i],)

    act_vec: List[int] = []
    logp_sum = 0.0

    for i, logits in enumerate(logits_list):
        if head_masks is not None:
            valid = head_masks[i]
            if valid.any():
                logits = logits.masked_fill(~valid, args.masked_logits_neg_inf)

        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        act_vec.append(int(a.item()))
        logp_sum += float(dist.log_prob(a).item())

    value = critic(obs_t).squeeze(-1)
    return act_vec, float(logp_sum), float(value.item())


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


def ppo_update_masked_multhead(
    actor: nn.Module,
    critic: nn.Module,
    actor_opt: optim.Optimizer,
    critic_opt: optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    args: Args,
    nvec: np.ndarray,
):
    """
    batch keys:
      obs: (M, obs_dim)                 -> actor & critic input
      act: (M, K)                       -> MultiDiscrete components
      logp_old: (M,)
      adv: (M,)
      ret: (M,)
      mask_joint: (M, prod(nvec))       -> env joint mask
    """
    obs = batch["obs"]
    act = batch["act"]  # (M, K)
    logp_old = batch["logp_old"]
    adv = batch["adv"]
    ret = batch["ret"]
    mask_joint = batch["mask_joint"]  # (M, prod)

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    M = obs.shape[0]
    idx = torch.randperm(M, device=obs.device)

    total_actor_loss = 0.0
    total_critic_loss = 0.0
    total_entropy = 0.0
    n_mb = 0

    K = act.shape[1]

    for start in range(0, M, args.minibatch_size):
        mb_idx = idx[start: start + args.minibatch_size]
        mb_obs = obs[mb_idx]
        mb_act = act[mb_idx]  # (mb, K)
        mb_logp_old = logp_old[mb_idx]
        mb_adv = adv[mb_idx]
        mb_ret = ret[mb_idx]
        mb_mask_joint = mask_joint[mb_idx]  # (mb, prod)

        logits_list = actor(mb_obs)  # list of (mb, nvec[i])

        head_masks = joint_mask_to_head_masks_torch(mb_mask_joint, nvec)  # list of (mb, nvec[i])

        # Compute summed log-prob and summed entropy across heads
        logp = torch.zeros(mb_obs.shape[0], device=mb_obs.device, dtype=torch.float32)
        entropy = torch.zeros(mb_obs.shape[0], device=mb_obs.device, dtype=torch.float32)

        for i in range(K):
            logits = logits_list[i]     # (mb, nvec[i])
            valid = head_masks[i]       # (mb, nvec[i])

            row_any = valid.any(dim=1)
            safe_logits = logits.clone()
            safe_logits[row_any] = safe_logits[row_any].masked_fill(
                ~valid[row_any], args.masked_logits_neg_inf
            )

            dist = torch.distributions.Categorical(logits=safe_logits)
            ai = mb_act[:, i]  # (mb,)
            logp = logp + dist.log_prob(ai)
            entropy = entropy + dist.entropy()

        entropy_mean = entropy.mean()

        ratio = torch.exp(logp - mb_logp_old)
        pg1 = ratio * mb_adv
        pg2 = torch.clamp(ratio, 1.0 - args.ppo_clip, 1.0 + args.ppo_clip) * mb_adv
        actor_loss = -(torch.min(pg1, pg2)).mean() - args.entropy_coef * entropy_mean

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
        total_entropy += float(entropy_mean.item())
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
    args = Args()
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
    records_folder = f"training_records_ippo_multihead_300_agents_seed_{seed}_3000_days"
    plots_folder = f"plots_ippo_multihead_300_agents_seed_{seed}_300_days"

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
                "travel_time_normalization_value": 100,
            }
        },
        "simulator_parameters": {
            "network_name": "network",
            "custom_network_folder": "../../network_analysis/network_base",
            "sumo_type": "sumo",
            "simulation_timesteps": 100,
        },
        "environment_parameters": {
            "observations_time_window": 20,
            "save_every": 1,
            "number_of_days": 30,
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

    try:
        obs_shape = obs_space["observation"].shape
    except Exception:
        obs_shape = obs_space.shape
    obs_dim = int(np.prod(obs_shape))

    act_spaces = env._action_spaces
    act_space = act_spaces[some_agent]
    nvec = np.asarray(act_space.nvec, dtype=np.int64)
    action_dim_joint = int(np.prod(nvec))  # expected length of joint action mask
    K = len(nvec)

    actor = MLPActorMultiHead(obs_dim, nvec, hidden=(64, 64)).to(device)
    critic = MLPCritic(obs_dim, hidden=(128, 128)).to(device)

    Optimizer = getattr(optim, args.optimizer)
    actor_opt = Optimizer(actor.parameters(), lr=args.lr_actor)
    critic_opt = Optimizer(critic.parameters(), lr=args.lr_critic)

    # -------------------------
    # Training loop
    # -------------------------
    pbar = tqdm(total=training_episodes + testing_episodes, desc="IPPO AEC training (masked multi-head)")

    running_ep_returns = []
    running_ep_lens = []

    for ep in range(training_episodes):
        env.reset()

        pending: Dict[str, Dict[str, Any] | None] = {a: None for a in mutated_agent_names}
        traj: Dict[str, Dict[str, List[Any]]] = {
            a: {"obs": [], "mask_joint": [], "act": [], "logp": [], "val": [], "rew": [], "done": []}
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
                    traj[agent]["mask_joint"].append(prev["mask_joint"])
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
                        mask_arr = np.ones(action_dim_joint, dtype=np.int8)
                    else:
                        mask_arr = mask_arr.reshape(-1)
                        if mask_arr.shape[0] != action_dim_joint:
                            raise ValueError(
                                f"action_mask length {mask_arr.shape[0]} != prod(nvec) {action_dim_joint}. "
                                f"nvec={nvec.tolist()}"
                            )

                    act_vec, logp, val = select_action_masked_multhead(
                        actor, critic, obs_vec, mask_arr, device, args, nvec
                    )
                    action = act_vec  # MultiDiscrete vector directly

                    pending[agent] = {
                        "obs": obs_vec,
                        "mask_joint": mask_arr.astype(np.int8),
                        "act": np.asarray(act_vec, dtype=np.int64),  # (K,)
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
                traj[a]["mask_joint"].append(prev["mask_joint"])
                traj[a]["act"].append(prev["act"])
                traj[a]["logp"].append(prev["logp"])
                traj[a]["val"].append(prev["val"])
                traj[a]["rew"].append(0.0)
                traj[a]["done"].append(1.0)
                pending[a] = None

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
            dones = np.asarray(traj[a]["done"], dtype=np.float32)
            adv, ret = compute_gae(rews, vals, dones, args.gamma, args.gae_lambda)

            obs_list.append(np.asarray(traj[a]["obs"], dtype=np.float32))               # (T, obs_dim)
            mask_list.append(np.asarray(traj[a]["mask_joint"], dtype=np.int8))          # (T, prod)
            act_list.append(np.asarray(traj[a]["act"], dtype=np.int64))                 # (T, K)
            logp_list.append(np.asarray(traj[a]["logp"], dtype=np.float32))             # (T,)
            adv_list.append(adv)
            ret_list.append(ret)

        if len(obs_list) == 0:
            pbar.update(1)
            continue

        obs_np = np.concatenate(obs_list, axis=0)       # (M, obs_dim)
        mask_np = np.concatenate(mask_list, axis=0)     # (M, prod)
        act_np = np.concatenate(act_list, axis=0)       # (M, K)
        logp_np = np.concatenate(logp_list, axis=0)     # (M,)
        adv_np = np.concatenate(adv_list, axis=0)       # (M,)
        ret_np = np.concatenate(ret_list, axis=0)       # (M,)

        if act_np.ndim != 2 or act_np.shape[1] != K:
            raise RuntimeError(f"Expected act_np shape (M,{K}), got {act_np.shape}")

        batch = {
            "obs": torch.from_numpy(obs_np).to(device=device, dtype=torch.float32),
            "mask_joint": torch.from_numpy(mask_np).to(device=device),
            "act": torch.from_numpy(act_np).to(device=device, dtype=torch.int64),
            "logp_old": torch.from_numpy(logp_np).to(device=device, dtype=torch.float32),
            "adv": torch.from_numpy(adv_np).to(device=device, dtype=torch.float32),
            "ret": torch.from_numpy(ret_np).to(device=device, dtype=torch.float32),
        }

        metrics = {}
        for _ in range(args.epochs):
            metrics = ppo_update_masked_multhead(actor, critic, actor_opt, critic_opt, batch, args, nvec)

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
                f"nvec={nvec.tolist()} joint_mask_dim={action_dim_joint}"
            )
            running_ep_returns.clear()
            running_ep_lens.clear()

        pbar.update(1)

    # -------------------------
    # Testing (greedy, masked)
    # -------------------------
    actor.eval()
    critic.eval()
    pbar.set_description("IPPO AEC testing (masked multi-head)")

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
                        mask_arr = np.ones(action_dim_joint, dtype=np.int8)
                    else:
                        mask_arr = mask_arr.reshape(-1)

                    obs_t = torch.from_numpy(obs_vec).to(device=device, dtype=torch.float32)
                    logits_list = actor(obs_t)  # list of (nvec[i],)

                    head_masks = None
                    if mask_arr is not None:
                        mask_t = torch.from_numpy(mask_arr).to(device=device).reshape(1, -1)
                        head_masks = joint_mask_to_head_masks_torch(mask_t, nvec)
                        head_masks = [hm.squeeze(0) for hm in head_masks]

                    act_vec: List[int] = []
                    for i, logits in enumerate(logits_list):
                        if head_masks is not None:
                            valid = head_masks[i]
                            if valid.any():
                                logits = logits.masked_fill(~valid, args.masked_logits_neg_inf)
                        act_vec.append(int(torch.argmax(logits).item()))

                    action = act_vec
                    pending[agent] = {"dummy": True}
            else:
                action = None

            env.step(action)

        mean_return = float(np.mean(list(ep_return.values())))
        print(f"[TEST EP {ep+1}] mean_return_over_agents={mean_return:.4f}")
        pbar.update(1)

    pbar.close()

    env.plot_results()
    env.stop_simulation()


if __name__ == "__main__":
    main()
