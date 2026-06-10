"""
Tune MAPPO hyperparameters with Grid Search.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import os
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm


# =========================
# HYPERPARAMETER GRIDS
# =========================

MAPPO_LR_ACTOR_GRID = [3e-4, 8e-4, 1e-3]
MAPPO_LR_CRITIC_GRID = [3e-4, 8e-4, 1e-3]
MAPPO_GAMMA_GRID = [0.99]
MAPPO_GAE_LAMBDA_GRID = [0.90]
MAPPO_PPO_CLIP_GRID = [0.2]
MAPPO_ENTROPY_COEF_GRID = [0.0, 0.001, 0.01]
MAPPO_VALUE_COEF_GRID = [0.5]
MAPPO_EPOCHS_GRID = [4]
MAPPO_MINIBATCH_SIZE_GRID = [4096]
MAPPO_MAX_GRAD_NORM_GRID = [0.5]


MAPPO_SCRIPT = "cleanrl_mappo.py"

MODE = "grid"  # "grid", "optuna", or "both"

EPISODES = 30
EVAL_WINDOW = 10
DEVICE = "cpu"
NUM_AGENTS = 300
OUTPUT_CSV = "mappo_tuning_results.csv"
VERBOSE = True

MAX_GRID_TRIALS = None

N_OPTUNA_TRIALS = 40
OPTUNA_STUDY_NAME = "mappo_optuna"
OPTUNA_STORAGE = None

ENV_SEEDS = [12, 13]
TORCH_SEEDS = [101, 102]


def load_mappo_module(script_path: str):
    script_path = os.path.abspath(script_path)
    module_dir = os.path.dirname(script_path)

    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    module_name = "mappo_source"

    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = module  # required for @dataclass

    spec.loader.exec_module(module)
    return module


def product_grid(grid: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(grid.keys())
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


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
                "max_allowed_bid": 5,
            },
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
            "phases": [1, max(1, total_episodes)],
            "smooth_by": 50,
            "phase_names": ["Training", "Testing"],
            "records_folder": f"tuning_records/{run_name}_seed_{seed}",
            "plots_folder": f"tuning_plots/{run_name}_seed_{seed}",
            "plot_choices": "basic",
        },
        "path_generation_parameters": {
            "origins": ["E0"],
            "destinations": ["E17.600"],
            "number_of_paths": int(args.num_routes),
            "beta": -1,
            "visualize_paths": False,
            "all_origins_to_all_destinations": False,
        },
    }


def train_once(
    mappo,
    overrides,
    trial_name,
    episodes,
    env_seed,
    torch_seed,
    num_agents,
    eval_window,
    device,
    verbose=True,
):
    
    args = replace(mappo.Args(), **overrides)
    
    args.training_episodes = int(episodes)
    args.testing_episodes = 0
    args.seed = int(env_seed)
    args.device = device

    # Seed environment randomness first
    mappo.set_seed(env_seed)
    random.seed(env_seed)
    np.random.seed(env_seed)

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    torch_device = torch.device(args.device)
    env_params = default_env_params(
        args,
        seed=env_seed,
        run_name=trial_name,
        num_agents=num_agents,
    )

    env = None
    ep_returns = []

    try:
        env = mappo.TrafficEnvironment(
            seed=env_seed,
            create_agents=True,
            create_paths=True,
            karma_pricing=True,
            **env_params,
        )
        env.start()
        env.reset()

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

        obs_space = env._observation_spaces[mutated_agent_names[0]]
        try:
            obs_shape = obs_space["observation"].shape
        except Exception:
            obs_shape = obs_space.shape
        obs_dim = int(np.prod(obs_shape))

        act_space = env._action_spaces[mutated_agent_names[0]]
        nvec = np.asarray(act_space.nvec, dtype=np.int64)
        strides = mappo.compute_action_strides(nvec)
        action_dim = int(np.prod(nvec))

        global_dim = int(args.num_routes) + (1 if args.include_timestep_feature else 0)
        central_obs_dim = obs_dim + global_dim

        actor_hidden = tuple(overrides.get("actor_hidden", (64, 64)))
        critic_hidden = tuple(overrides.get("critic_hidden", (256, 256)))

        # Seed neural network initialization separately
        torch.manual_seed(torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)

        actor = mappo.MLPActorJoint(obs_dim, action_dim, hidden=actor_hidden).to(torch_device)
        critic = mappo.MLPCriticCentral(central_obs_dim, hidden=critic_hidden).to(torch_device)

        Optimizer = getattr(optim, args.optimizer)
        actor_opt = Optimizer(actor.parameters(), lr=args.lr_actor)
        critic_opt = Optimizer(critic.parameters(), lr=args.lr_critic)

        max_steps_hint = max(1, int(num_agents * env_params["environment_parameters"]["number_of_days"]))
        iterator = tqdm(range(args.training_episodes), desc=trial_name, leave=False) if verbose else range(args.training_episodes)

        for ep in iterator:
            env.reset()
            route_counts = np.zeros(args.num_routes, dtype=np.int32)
            step_in_episode = 0

            pending = {a: None for a in mutated_agent_names}
            ep_return = {a: 0.0 for a in mutated_agent_names}

            traj = {
                a: {
                    "obs": [], "central_obs": [], "mask": [],
                    "act": [], "logp": [], "val": [], "rew": [], "done": []
                }
                for a in mutated_agent_names
            }

            for agent in env.agent_iter():
                obs, reward, termination, truncation, info = env.last()
                done = termination or truncation

                if agent in pending:
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

                    if done:
                        action = None
                    else:
                        obs_vec, mask_arr = mappo.extract_obs_and_mask(obs)

                        if mask_arr is None:
                            mask_arr = np.ones(action_dim, dtype=np.int8)
                        else:
                            mask_arr = mask_arr.reshape(-1)

                        gfeats = mappo.build_global_features(
                            args=args,
                            step_in_episode=step_in_episode,
                            max_steps_hint=max_steps_hint,
                            route_counts=route_counts,
                        )

                        central_obs_vec = mappo.make_central_obs(obs_vec, gfeats)

                        act_idx, logp, val = mappo.select_action_masked_joint_mappo(
                            actor, critic, obs_vec, central_obs_vec, mask_arr, torch_device, args
                        )

                        act_vec = mappo.joint_index_to_multidiscrete(act_idx, nvec, strides)
                        action = act_vec

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

            obs_list, central_obs_list, mask_list = [], [], []
            act_list, logp_list, adv_list, ret_list = [], [], [], []

            for a in mutated_agent_names:
                if len(traj[a]["obs"]) == 0:
                    continue

                rews = np.asarray(traj[a]["rew"], dtype=np.float32)
                vals = np.asarray(traj[a]["val"], dtype=np.float32)
                dones = np.asarray(traj[a]["done"], dtype=np.float32)

                adv, ret = mappo.compute_gae(rews, vals, dones, args.gamma, args.gae_lambda)

                obs_list.append(np.asarray(traj[a]["obs"], dtype=np.float32))
                central_obs_list.append(np.asarray(traj[a]["central_obs"], dtype=np.float32))
                mask_list.append(np.asarray(traj[a]["mask"], dtype=np.int8))
                act_list.append(np.asarray(traj[a]["act"], dtype=np.int64))
                logp_list.append(np.asarray(traj[a]["logp"], dtype=np.float32))
                adv_list.append(adv)
                ret_list.append(ret)

            if obs_list:
                batch = {
                    "obs": torch.from_numpy(np.concatenate(obs_list)).to(torch_device, dtype=torch.float32),
                    "central_obs": torch.from_numpy(np.concatenate(central_obs_list)).to(torch_device, dtype=torch.float32),
                    "mask": torch.from_numpy(np.concatenate(mask_list)).to(torch_device),
                    "act": torch.from_numpy(np.concatenate(act_list)).to(torch_device, dtype=torch.int64),
                    "logp_old": torch.from_numpy(np.concatenate(logp_list)).to(torch_device, dtype=torch.float32),
                    "adv": torch.from_numpy(np.concatenate(adv_list)).to(torch_device, dtype=torch.float32),
                    "ret": torch.from_numpy(np.concatenate(ret_list)).to(torch_device, dtype=torch.float32),
                }

                for _ in range(int(args.epochs)):
                    mappo.ppo_update_masked_joint_mappo(
                        actor, critic, actor_opt, critic_opt, batch, args
                    )

            ep_returns.append(float(np.mean(list(ep_return.values()))))

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


def append_result(csv_path, row):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    flat = dict(row)
    params = flat.pop("params", {})

    for k, v in params.items():
        flat[f"param_{k}"] = json.dumps(v) if isinstance(v, (list, tuple, dict)) else v

    write_header = not csv_path.exists()

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(flat)


def run_grid(mappo):
    grid = list(
        itertools.product(
            MAPPO_LR_ACTOR_GRID,
            MAPPO_LR_CRITIC_GRID,
            MAPPO_GAMMA_GRID,
            MAPPO_GAE_LAMBDA_GRID,
            MAPPO_PPO_CLIP_GRID,
            MAPPO_ENTROPY_COEF_GRID,
            MAPPO_VALUE_COEF_GRID,
            MAPPO_EPOCHS_GRID,
            MAPPO_MINIBATCH_SIZE_GRID,
            MAPPO_MAX_GRAD_NORM_GRID,
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
            mappo=mappo,
            params=params,
            trial_name=f"grid_{i:04d}",
        )

        result["trial"] = i
        result["method"] = "grid"
        append_result(OUTPUT_CSV, result)

        print(f"mean_score={result['score']:.6f}, std={result['score_std']:.6f}")

        if result["score"] > best["score"]:
            best = result
            print(f"new best score={best['score']:.6f}")

    return best

def evaluate_param_set(mappo, params, trial_name):
    results = []

    seed_pairs = list(itertools.product(ENV_SEEDS, TORCH_SEEDS))

    for env_seed, torch_seed in tqdm(seed_pairs, desc=f"Seeds for {trial_name}", leave=False):
        result = train_once(
            mappo=mappo,
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



def run_optuna(mappo):
    try:
        import optuna
    except ImportError:
        raise SystemExit("Optuna is not installed. Run: pip install optuna")

    def objective(trial):
        params = {
            "lr_actor": trial.suggest_categorical("lr_actor", MAPPO_LR_ACTOR_GRID),
            "lr_critic": trial.suggest_categorical("lr_critic", MAPPO_LR_CRITIC_GRID),
            "gamma": trial.suggest_categorical("gamma", MAPPO_GAMMA_GRID),
            "gae_lambda": trial.suggest_categorical("gae_lambda", MAPPO_GAE_LAMBDA_GRID),
            "ppo_clip": trial.suggest_categorical("ppo_clip", MAPPO_PPO_CLIP_GRID),
            "entropy_coef": trial.suggest_categorical("entropy_coef", MAPPO_ENTROPY_COEF_GRID),
            "value_coef": trial.suggest_categorical("value_coef", MAPPO_VALUE_COEF_GRID),
            "epochs": trial.suggest_categorical("epochs", MAPPO_EPOCHS_GRID),
            "minibatch_size": trial.suggest_categorical("minibatch_size", MAPPO_MINIBATCH_SIZE_GRID),
            "max_grad_norm": trial.suggest_categorical("max_grad_norm", MAPPO_MAX_GRAD_NORM_GRID),
        }

        result = evaluate_param_set(
            mappo=mappo,
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




# =========================
# SETTINGS
# =========================

def main():
    mappo = load_mappo_module(MAPPO_SCRIPT)

    started = time.time()
    best_results = []

    if MODE in {"grid", "both"}:
        best_results.append(run_grid(mappo))

    if MODE in {"optuna", "both"}:
        best_results.append(run_optuna(mappo))

    best = max(best_results, key=lambda r: r["score"])

    print("\n================ BEST RESULT ================")
    print(json.dumps(best, indent=2, default=str))
    print(f"Results CSV: {OUTPUT_CSV}")
    print(f"Elapsed seconds: {time.time() - started:.1f}")


if __name__ == "__main__":
    main()