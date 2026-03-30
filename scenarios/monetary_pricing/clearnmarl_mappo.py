from __future__ import annotations

from tqdm import tqdm
import os
import sys
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ------------------------------------------------------------
# Your imports
# ------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from routerl_aec_environment import TrafficEnvironment


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
@dataclass
class Args:
    seed: int = 9
    device: str = "cpu"

    # episodes
    training_episodes: int = 300
    testing_episodes: int = 10

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


# ------------------------------------------------------------
# Networks
# ------------------------------------------------------------
class MLPActor(nn.Module):
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
        # obs: (..., obs_dim)
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
        # returns (..., 1)
        return self.net(obs)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def flatten_obs(obs: Any) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32)
    return arr.reshape(-1)


@torch.no_grad()
def select_action(actor: nn.Module, critic: nn.Module, obs_np: np.ndarray, device: torch.device):
    obs_t = torch.from_numpy(obs_np).to(device=device, dtype=torch.float32)
    logits = actor(obs_t)
    dist = torch.distributions.Categorical(logits=logits)
    act = dist.sample()
    logp = dist.log_prob(act)
    value = critic(obs_t).squeeze(-1)
    return int(act.item()), float(logp.item()), float(value.item())


def compute_gae(rews: np.ndarray, vals: np.ndarray, dones: np.ndarray, gamma: float, lam: float):
    """
    Per-agent trajectory GAE.
    Inputs are 1D arrays of length T (for one agent):
      rews[t] reward assigned to the action at t
      vals[t] V(o_t) predicted at the time action was chosen
      dones[t] 1.0 if terminal after that action else 0.0
    Returns:
      adv (T,), ret (T,)
    """
    T = len(rews)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    next_value = 0.0  # bootstrap 0 at episode end
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        delta = rews[t] + gamma * next_value * nonterminal - vals[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
        next_value = vals[t]
    ret = adv + vals
    return adv, ret


def ppo_update(
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
    """
    obs = batch["obs"]
    act = batch["act"]
    logp_old = batch["logp_old"]
    adv = batch["adv"]
    ret = batch["ret"]

    # normalize advantage (common PPO trick)
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

        logits = actor(mb_obs)
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
# Main training script
# ------------------------------------------------------------
def main():
    args = Args()
    set_seed(args.seed)
    device = torch.device(args.device)

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # -------------------------
    # Your env setup (paste from your script)
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
    records_folder = f"training_records_ippo_300_agents_seed_{seed}_monetary_pricing"
    plots_folder = f"plots_ippo_300_agents_seed_{seed}_monetary_pricing"

    phases = [1, int(total_episodes)]
    phase_names = ["Training", "Testing"]

    env_params = {
        "agent_parameters": {
            "new_machines_after_mutation": new_machines_after_mutation,
            "num_agents": num_agents,
            "machine_parameters": {
                "behavior": "selfish",
                "observation_type": "previous_agents_avg_tt_per_route",
                "route_0_fee": 5,
                "travel_time_normalization_value": 100
            }
        },
        "simulator_parameters": {
            "network_name": "network",
            "custom_network_folder": "../../network_analysis/network_base",
            "sumo_type": "sumo-gui",
            "simulation_timesteps": 100,
        },
        "environment_parameters": {
            "observations_time_window": 20,
            "save_every": 5,
            "number_of_days": 30
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

    env = TrafficEnvironment(seed=seed, create_agents=True, create_paths=True, monetary_pricing=True, **env_params)

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
    obs_dim = int(np.prod(obs_space.shape))

    # You used ACTION_SIZE = 3 in DQN
    action_dim = 3

    actor = MLPActor(obs_dim, action_dim, hidden=(64, 64)).to(device)
    critic = MLPCritic(obs_dim, hidden=(128, 128)).to(device)

    Optimizer = getattr(optim, args.optimizer)
    actor_opt = Optimizer(actor.parameters(), lr=args.lr_actor)
    critic_opt = Optimizer(critic.parameters(), lr=args.lr_critic)

    # -------------------------
    # Training loop
    # -------------------------
    pbar = tqdm(total=training_episodes + testing_episodes, desc="IPPO AEC training")

    running_ep_returns = []
    running_ep_lens = []

    for ep in range(training_episodes):
        env.reset()

        # For PPO in AEC, we store per-agent trajectories of completed steps.
        # pending[agent] holds the last (obs, act, logp, value) waiting for its reward on next turn.
        pending: Dict[str, Dict[str, Any] | None] = {a: None for a in mutated_agent_names}
        traj: Dict[str, Dict[str, List[Any]]] = {
            a: {"obs": [], "act": [], "logp": [], "val": [], "rew": [], "done": []}
            for a in mutated_agent_names
        }

        # For logging per episode (per-agent returns)
        ep_return = {a: 0.0 for a in mutated_agent_names}
        turn_count = 0

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            done = termination or truncation
            print("reward is: ", reward, "\n\n")

            # Only for controlled agents
            if agent in agent_name_to_idx:
                # 1) Close previous pending step for this agent (if any)
                if pending[agent] is not None:
                    prev = pending[agent]
                    traj[agent]["obs"].append(prev["obs"])
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
                    obs_np = flatten_obs(obs)
                    print("obs_np is: ", obs_np, "\n\n")
                    act, logp, val = select_action(actor, critic, obs_np, device)
                    action = act
                    pending[agent] = {"obs": obs_np, "act": act, "logp": logp, "val": val}

            else:
                # uncontrolled agents
                action = None if done else None

            env.step(action)
            turn_count += 1

        # Episode ended. Some agents might still have a pending action that never got revisited.
        # We cannot reliably recover their final reward without env-specific bookkeeping,
        # so we close them with reward=0, done=1 as a safe fallback.
        # If your env *does* revisit them after termination, pending will usually be empty anyway.
        for a in mutated_agent_names:
            if pending[a] is not None:
                prev = pending[a]
                traj[a]["obs"].append(prev["obs"])
                traj[a]["act"].append(prev["act"])
                traj[a]["logp"].append(prev["logp"])
                traj[a]["val"].append(prev["val"])
                traj[a]["rew"].append(0.0)
                traj[a]["done"].append(1.0)
                pending[a] = None

        # -------------------------
        # Build PPO batch from all agents' trajectories
        # -------------------------
        obs_list = []
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
            act_list.append(np.asarray(traj[a]["act"], dtype=np.int64))
            logp_list.append(np.asarray(traj[a]["logp"], dtype=np.float32))
            adv_list.append(adv)
            ret_list.append(ret)

        if len(obs_list) == 0:
            # no data collected (shouldn't happen), skip update
            pbar.update(1)
            continue

        obs_np = np.concatenate(obs_list, axis=0)               # (M, obs_dim)
        act_np = np.concatenate(act_list, axis=0)               # (M,)
        logp_np = np.concatenate(logp_list, axis=0)             # (M,)
        adv_np = np.concatenate(adv_list, axis=0)               # (M,)
        ret_np = np.concatenate(ret_list, axis=0)               # (M,)

        batch = {
            "obs": torch.from_numpy(obs_np).to(device=device, dtype=torch.float32),
            "act": torch.from_numpy(act_np).to(device=device, dtype=torch.int64),
            "logp_old": torch.from_numpy(logp_np).to(device=device, dtype=torch.float32),
            "adv": torch.from_numpy(adv_np).to(device=device, dtype=torch.float32),
            "ret": torch.from_numpy(ret_np).to(device=device, dtype=torch.float32),
        }

        # Multiple PPO epochs over the same on-policy batch
        metrics = {}
        for _ in range(args.epochs):
            metrics = ppo_update(actor, critic, actor_opt, critic_opt, batch, args)

        # Logging (mean return per agent this episode)
        mean_return = float(np.mean(list(ep_return.values())))
        running_ep_returns.append(mean_return)
        running_ep_lens.append(turn_count)

        if (ep + 1) % args.log_every_episodes == 0:
            print(
                f"[EP {ep+1}] mean_return_over_agents={np.mean(running_ep_returns):.4f} "
                f"mean_turns={np.mean(running_ep_lens):.1f} "
                f"actor_loss={metrics.get('actor_loss', 0):.4f} "
                f"critic_loss={metrics.get('critic_loss', 0):.4f} "
                f"entropy={metrics.get('entropy', 0):.4f}"
            )
            running_ep_returns.clear()
            running_ep_lens.clear()

        pbar.update(1)

    # -------------------------
    # Testing (greedy)
    # -------------------------
    actor.eval()
    critic.eval()
    pbar.set_description("IPPO AEC testing")

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
                    obs_np = flatten_obs(obs)
                    obs_t = torch.from_numpy(obs_np).to(device=device, dtype=torch.float32)
                    with torch.no_grad():
                        logits = actor(obs_t)
                        action = int(torch.argmax(logits).item())
                    pending[agent] = {"dummy": True}  # just to count reward next time
            else:
                action = None if done else None

            env.step(action)

        mean_return = float(np.mean(list(ep_return.values())))
        print(f"[TEST EP {ep+1}] mean_return_over_agents={mean_return:.4f}")
        pbar.update(1)

    pbar.close()

    # Your plotting & shutdown
    env.plot_results()
    env.stop_simulation()


if __name__ == "__main__":
    main()
