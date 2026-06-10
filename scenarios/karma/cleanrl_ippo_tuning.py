#!/usr/bin/env python3
"""
Tune IPPO hyperparameters with Grid Search and/or Optuna.

This script is adapted from the MAPPO tuning script, but uses the IPPO
training logic from cleanrl_ippo.py:

- shared IPPO actor
- local IPPO critic
- masked joint action selection
- local observations only
- no centralized critic
- no centralized/global observation features

Output:
    ippo_tuning_results.csv
"""

from __future__ import annotations

import csv
import importlib.util
import itertools
import json
import os
import random
import sys
import time
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm


# ============================================================
# HYPERPARAMETER GRIDS
# ============================================================

IPPO_LR_ACTOR_GRID = [3e-4, 8e-4, 1e-3]
IPPO_LR_CRITIC_GRID = [3e-4, 8e-4, 1e-3]

IPPO_GAMMA_GRID = [0.99]
IPPO_GAE_LAMBDA_GRID = [0.90, 0.95]
IPPO_PPO_CLIP_GRID = [0.2]

IPPO_ENTROPY_COEF_GRID = [0.0, 0.001, 0.01]
IPPO_VALUE_COEF_GRID = [0.5]

IPPO_EPOCHS_GRID = [4]
IPPO_MINIBATCH_SIZE_GRID = [4096]
IPPO_MAX_GRAD_NORM_GRID = [0.5]


# ============================================================
# SCRIPT SETTINGS
# ============================================================

# Change this to the filename/path of your IPPO script.
IPPO_SCRIPT = "cleanrl_ippo.py"

MODE = "grid"  # "grid", "optuna", or "both"

EPISODES = 30
EVAL_WINDOW = 10

DEVICE = "cpu"
NUM_AGENTS = 300
NUM_ROUTES = 3

MAX_ALLOWED_BID = 5
ROUTE_0_PRICE = 4
ROUTE_1_PRICE = 3

# Use the same path style as your IPPO script.
CUSTOM_NETWORK_FOLDER = "../../network_analysis/network_base"

OUTPUT_CSV = "ippo_tuning_results.csv"
VERBOSE = True

# Use this for a quick test, e.g. MAX_GRID_TRIALS = 2.
MAX_GRID_TRIALS = None

# Optuna settings
N_OPTUNA_TRIALS = 40
OPTUNA_STUDY_NAME = "ippo_optuna"
OPTUNA_STORAGE = None

# Evaluation seeds
ENV_SEEDS = [12, 13]
TORCH_SEEDS = [101, 102]

# Network architectures
ACTOR_HIDDEN = (64, 64)
CRITIC_HIDDEN = (128, 128)


# ============================================================
# MODULE LOADING
# ============================================================

def load_ippo_module(script_path: str):
    script_path = os.path.abspath(script_path)
    module_dir = os.path.dirname(script_path)

    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    module_name = "ippo_source"

    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)

    # Required for dataclasses inside dynamically loaded modules.
    sys.modules[module_name] = module

    if spec.loader is None:
        raise RuntimeError(f"Could not load IPPO script: {script_path}")

    spec.loader.exec_module(module)
    return module


def make_args(ippo, overrides: Dict[str, Any]):
    """
    Create Args() while only passing fields that actually exist in ippo.Args.

    This avoids errors if you later add tuning-only parameters such as
    actor_hidden or critic_hidden.
    """
    valid_arg_names = {f.name for f in fields(ippo.Args)}
    arg_overrides = {
        k: v
        for k, v in overrides.items()
        if k in valid_arg_names
    }

    return replace(ippo.Args(), **arg_overrides)


# ============================================================
# ENVIRONMENT SETTINGS
# ============================================================

def default_env_params(args, seed: int, run_name: str, num_agents: int):
    total_episodes = int(args.training_episodes) + int(args.testing_episodes)

    return {
        "agent_parameters": {
            "new_machines_after_mutation": num_agents,
            "num_agents": num_agents,
            "machine_parameters": {
                "behavior": "selfish",
                "observation_type": "previous_agents_avg_tt_per_route",
                "route_0_fee": 0,
                "travel_time_normalization_value": 16.5,
                "max_allowed_bid": MAX_ALLOWED_BID,
            },
        },
        "simulator_parameters": {
            "network_name": "network",
            "custom_network_folder": CUSTOM_NETWORK_FOLDER,
            "sumo_type": "sumo",
            "simulation_timesteps": 100,
        },
        "environment_parameters": {
            "observations_time_window": 10,
            "save_every": 5,
            "number_of_days": 30,
            "centrally_defined_price_route_0": ROUTE_0_PRICE,
            "centrally_defined_price_route_1": ROUTE_1_PRICE,
        },
        "plotter_parameters": {
            "phases": [1, max(1, total_episodes)],
            "smooth_by": 50,
            "phase_names": ["Training", "Testing"],
            "records_folder": f"tuning_records/ippo/{run_name}_seed_{seed}",
            "plots_folder": f"tuning_plots/ippo/{run_name}_seed_{seed}",
            "plot_choices": "basic",
        },
        "path_generation_parameters": {
            "origins": ["E0"],
            "destinations": ["E17.600"],
            "number_of_paths": NUM_ROUTES,
            "beta": -1,
            "visualize_paths": False,
            "all_origins_to_all_destinations": False,
        },
    }


# ============================================================
# SINGLE TRAINING RUN
# ============================================================

def train_once(
    ippo,
    overrides: Dict[str, Any],
    trial_name: str,
    episodes: int,
    env_seed: int,
    torch_seed: int,
    num_agents: int,
    eval_window: int,
    device: str,
    verbose: bool = True,
):
    args = make_args(ippo, overrides)

    args.training_episodes = int(episodes)
    args.testing_episodes = 0
    args.seed = int(env_seed)
    args.device = device

    # Seed environment randomness.
    ippo.set_seed(env_seed)
    random.seed(env_seed)
    np.random.seed(env_seed)

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    torch_device = torch.device(args.device)

    env_params = default_env_params(
        args=args,
        seed=env_seed,
        run_name=trial_name,
        num_agents=num_agents,
    )

    env = None
    ep_returns = []

    try:
        env = ippo.TrafficEnvironment(
            seed=env_seed,
            create_agents=True,
            create_paths=True,
            karma_pricing=True,
            **env_params,
        )

        env.start()
        env.reset()

        # Mutation: humans switch to AVs.
        pre_mutation_agents = env.all_agents.copy()
        env.mutation(mutation_start_percentile=0)

        mutated_humans = {}

        for machine in env.machine_agents.copy():
            for human in pre_mutation_agents:
                if human.id == machine.id:
                    mutated_humans[str(machine.id)] = human
                    break

        mutated_agent_names = list(mutated_humans.keys())

        if not mutated_agent_names:
            raise RuntimeError("No controlled agents found after mutation.")

        # ----------------------------------------------------
        # Infer observation and action dimensions
        # ----------------------------------------------------
        obs_space = env._observation_spaces[mutated_agent_names[0]]

        try:
            obs_shape = obs_space["observation"].shape
        except Exception:
            obs_shape = obs_space.shape

        obs_dim = int(np.prod(obs_shape))

        act_space = env._action_spaces[mutated_agent_names[0]]
        nvec = np.asarray(act_space.nvec, dtype=np.int64)
        strides = ippo.compute_action_strides(nvec)
        action_dim = int(np.prod(nvec))

        actor_hidden = tuple(overrides.get("actor_hidden", ACTOR_HIDDEN))
        critic_hidden = tuple(overrides.get("critic_hidden", CRITIC_HIDDEN))

        # Seed neural-network initialization separately.
        torch.manual_seed(torch_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)

        actor = ippo.MLPActorJoint(
            obs_dim,
            action_dim,
            hidden=actor_hidden,
        ).to(torch_device)

        critic = ippo.MLPCritic(
            obs_dim,
            hidden=critic_hidden,
        ).to(torch_device)

        Optimizer = getattr(optim, args.optimizer)

        actor_opt = Optimizer(
            actor.parameters(),
            lr=args.lr_actor,
        )

        critic_opt = Optimizer(
            critic.parameters(),
            lr=args.lr_critic,
        )

        iterator = (
            tqdm(
                range(args.training_episodes),
                desc=trial_name,
                leave=False,
            )
            if verbose
            else range(args.training_episodes)
        )

        # ----------------------------------------------------
        # Training loop
        # ----------------------------------------------------
        for ep in iterator:
            env.reset()

            pending = {
                a: None
                for a in mutated_agent_names
            }

            ep_return = {
                a: 0.0
                for a in mutated_agent_names
            }

            traj = {
                a: {
                    "obs": [],
                    "mask": [],
                    "act": [],
                    "logp": [],
                    "val": [],
                    "rew": [],
                    "done": [],
                }
                for a in mutated_agent_names
            }

            for agent in env.agent_iter():
                obs, reward, termination, truncation, info = env.last()
                done = termination or truncation

                if agent in pending:
                    # Close previous pending transition.
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

                    # Choose next action unless done.
                    if done:
                        action = None
                    else:
                        obs_vec, mask_arr = ippo.extract_obs_and_mask(obs)

                        if mask_arr is None:
                            mask_arr = np.ones(action_dim, dtype=np.int8)
                        else:
                            mask_arr = mask_arr.reshape(-1)

                            if mask_arr.shape[0] != action_dim:
                                raise ValueError(
                                    f"action_mask length {mask_arr.shape[0]} "
                                    f"!= action_dim {action_dim}. "
                                    f"Expected prod(nvec)={action_dim}, "
                                    f"nvec={nvec.tolist()}."
                                )

                        act_idx, logp, val = ippo.select_action_masked_joint(
                            actor=actor,
                            critic=critic,
                            obs_vec=obs_vec,
                            action_mask=mask_arr,
                            device=torch_device,
                            args=args,
                        )

                        act_vec = ippo.joint_index_to_multidiscrete(
                            act_idx,
                            nvec,
                            strides,
                        )

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

            # Close any remaining pending steps with terminal reward fallback.
            for agent in mutated_agent_names:
                if pending[agent] is not None:
                    prev = pending[agent]

                    traj[agent]["obs"].append(prev["obs"])
                    traj[agent]["mask"].append(prev["mask"])
                    traj[agent]["act"].append(prev["act"])
                    traj[agent]["logp"].append(prev["logp"])
                    traj[agent]["val"].append(prev["val"])
                    traj[agent]["rew"].append(0.0)
                    traj[agent]["done"].append(1.0)

                    pending[agent] = None

            # ------------------------------------------------
            # Build PPO batch
            # ------------------------------------------------
            obs_list = []
            mask_list = []
            act_list = []
            logp_list = []
            adv_list = []
            ret_list = []

            for agent in mutated_agent_names:
                if len(traj[agent]["obs"]) == 0:
                    continue

                rews = np.asarray(traj[agent]["rew"], dtype=np.float32)
                vals = np.asarray(traj[agent]["val"], dtype=np.float32)
                dones = np.asarray(traj[agent]["done"], dtype=np.float32)

                adv, ret = ippo.compute_gae(
                    rews=rews,
                    vals=vals,
                    dones=dones,
                    gamma=args.gamma,
                    lam=args.gae_lambda,
                )

                obs_list.append(np.asarray(traj[agent]["obs"], dtype=np.float32))
                mask_list.append(np.asarray(traj[agent]["mask"], dtype=np.int8))
                act_list.append(np.asarray(traj[agent]["act"], dtype=np.int64))
                logp_list.append(np.asarray(traj[agent]["logp"], dtype=np.float32))
                adv_list.append(adv)
                ret_list.append(ret)

            if obs_list:
                batch = {
                    "obs": torch.from_numpy(
                        np.concatenate(obs_list)
                    ).to(torch_device, dtype=torch.float32),

                    "mask": torch.from_numpy(
                        np.concatenate(mask_list)
                    ).to(torch_device),

                    "act": torch.from_numpy(
                        np.concatenate(act_list)
                    ).to(torch_device, dtype=torch.int64),

                    "logp_old": torch.from_numpy(
                        np.concatenate(logp_list)
                    ).to(torch_device, dtype=torch.float32),

                    "adv": torch.from_numpy(
                        np.concatenate(adv_list)
                    ).to(torch_device, dtype=torch.float32),

                    "ret": torch.from_numpy(
                        np.concatenate(ret_list)
                    ).to(torch_device, dtype=torch.float32),
                }

                for _ in range(int(args.epochs)):
                    ippo.ppo_update_masked_joint(
                        actor=actor,
                        critic=critic,
                        actor_opt=actor_opt,
                        critic_opt=critic_opt,
                        batch=batch,
                        args=args,
                    )

            mean_ep_return = float(np.mean(list(ep_return.values())))
            ep_returns.append(mean_ep_return)

        window = ep_returns[-min(eval_window, len(ep_returns)):]

        score = float(np.mean(window)) if window else float("-inf")

        return {
            "score": score,
            "last_return": ep_returns[-1] if ep_returns else None,
            "mean_return_all": float(np.mean(ep_returns)) if ep_returns else None,
            "params": overrides,
            "env_seed": env_seed,
            "torch_seed": torch_seed,
            "episodes": episodes,
        }

    finally:
        if env is not None:
            try:
                env.stop_simulation()
            except Exception:
                pass


# ============================================================
# RESULT LOGGING
# ============================================================

def append_result(csv_path, row):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    flat = dict(row)
    params = flat.pop("params", {})

    for k, v in params.items():
        if isinstance(v, (list, tuple, dict)):
            flat[f"param_{k}"] = json.dumps(v)
        else:
            flat[f"param_{k}"] = v

    write_header = not csv_path.exists()

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))

        if write_header:
            writer.writeheader()

        writer.writerow(flat)


# ============================================================
# PARAMETER EVALUATION
# ============================================================

def evaluate_param_set(ippo, params, trial_name):
    results = []

    seed_pairs = list(itertools.product(ENV_SEEDS, TORCH_SEEDS))

    for env_seed, torch_seed in tqdm(
        seed_pairs,
        desc=f"Seeds for {trial_name}",
        leave=False,
    ):
        result = train_once(
            ippo=ippo,
            overrides=params,
            trial_name=f"{trial_name}_env{env_seed}_torch{torch_seed}",
            episodes=EPISODES,
            env_seed=env_seed,
            torch_seed=torch_seed,
            num_agents=NUM_AGENTS,
            eval_window=EVAL_WINDOW,
            device=DEVICE,
            verbose=VERBOSE,
        )

        results.append(result["score"])

    return {
        "score": float(np.mean(results)),
        "score_std": float(np.std(results)),
        "score_min": float(np.min(results)),
        "score_max": float(np.max(results)),
        "num_seed_runs": len(results),
        "params": params,
    }


# ============================================================
# GRID SEARCH
# ============================================================

def run_grid(ippo):
    grid = list(
        itertools.product(
            IPPO_LR_ACTOR_GRID,
            IPPO_LR_CRITIC_GRID,
            IPPO_GAMMA_GRID,
            IPPO_GAE_LAMBDA_GRID,
            IPPO_PPO_CLIP_GRID,
            IPPO_ENTROPY_COEF_GRID,
            IPPO_VALUE_COEF_GRID,
            IPPO_EPOCHS_GRID,
            IPPO_MINIBATCH_SIZE_GRID,
            IPPO_MAX_GRAD_NORM_GRID,
        )
    )

    if MAX_GRID_TRIALS is not None:
        grid = grid[:MAX_GRID_TRIALS]

    best = {"score": float("-inf")}

    for i, combo in enumerate(tqdm(grid, desc="Grid experiments"), start=1):
        params = {
            "lr_actor": combo[0],
            "lr_critic": combo[1],
            "gamma": combo[2],
            "gae_lambda": combo[3],
            "ppo_clip": combo[4],
            "entropy_coef": combo[5],
            "value_coef": combo[6],
            "epochs": combo[7],
            "minibatch_size": combo[8],
            "max_grad_norm": combo[9],
        }

        print(f"\n[grid {i}/{len(grid)}] {params}")

        result = evaluate_param_set(
            ippo=ippo,
            params=params,
            trial_name=f"grid_{i:04d}",
        )

        result["trial"] = i
        result["method"] = "grid"

        append_result(OUTPUT_CSV, result)

        print(
            f"mean_score={result['score']:.6f}, "
            f"std={result['score_std']:.6f}"
        )

        if result["score"] > best["score"]:
            best = result
            print(f"new best score={best['score']:.6f}")

    return best


# ============================================================
# OPTUNA
# ============================================================

def run_optuna(ippo):
    try:
        import optuna
    except ImportError:
        raise SystemExit("Optuna is not installed. Run: pip install optuna")

    def objective(trial):
        params = {
            "lr_actor": trial.suggest_categorical(
                "lr_actor",
                IPPO_LR_ACTOR_GRID,
            ),
            "lr_critic": trial.suggest_categorical(
                "lr_critic",
                IPPO_LR_CRITIC_GRID,
            ),
            "gamma": trial.suggest_categorical(
                "gamma",
                IPPO_GAMMA_GRID,
            ),
            "gae_lambda": trial.suggest_categorical(
                "gae_lambda",
                IPPO_GAE_LAMBDA_GRID,
            ),
            "ppo_clip": trial.suggest_categorical(
                "ppo_clip",
                IPPO_PPO_CLIP_GRID,
            ),
            "entropy_coef": trial.suggest_categorical(
                "entropy_coef",
                IPPO_ENTROPY_COEF_GRID,
            ),
            "value_coef": trial.suggest_categorical(
                "value_coef",
                IPPO_VALUE_COEF_GRID,
            ),
            "epochs": trial.suggest_categorical(
                "epochs",
                IPPO_EPOCHS_GRID,
            ),
            "minibatch_size": trial.suggest_categorical(
                "minibatch_size",
                IPPO_MINIBATCH_SIZE_GRID,
            ),
            "max_grad_norm": trial.suggest_categorical(
                "max_grad_norm",
                IPPO_MAX_GRAD_NORM_GRID,
            ),
        }

        result = evaluate_param_set(
            ippo=ippo,
            params=params,
            trial_name=f"optuna_{trial.number:04d}",
        )

        result["trial"] = trial.number
        result["method"] = "optuna"

        append_result(OUTPUT_CSV, result)

        return result["score"]

    study = optuna.create_study(
        direction="maximize",
        study_name=OPTUNA_STUDY_NAME,
        storage=OPTUNA_STORAGE,
        load_if_exists=bool(OPTUNA_STORAGE),
    )

    study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

    return {
        "score": study.best_value,
        "params": study.best_params,
        "trial": study.best_trial.number,
        "method": "optuna",
    }


# ============================================================
# MAIN
# ============================================================

def main():
    ippo = load_ippo_module(IPPO_SCRIPT)

    started = time.time()
    best_results = []

    if MODE in {"grid", "both"}:
        best_results.append(run_grid(ippo))

    if MODE in {"optuna", "both"}:
        best_results.append(run_optuna(ippo))

    if not best_results:
        raise RuntimeError(f"Unknown MODE: {MODE}")

    best = max(best_results, key=lambda r: r["score"])

    print("\n================ BEST RESULT ================")
    print(json.dumps(best, indent=2, default=str))
    print(f"Results CSV: {OUTPUT_CSV}")
    print(f"Elapsed seconds: {time.time() - started:.1f}")


if __name__ == "__main__":
    main()