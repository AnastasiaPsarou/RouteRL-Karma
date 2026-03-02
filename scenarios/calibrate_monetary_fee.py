"""
Fee calibration for a 3-action choice model where ONLY action 0 has a fee F.

Agents choose action j that minimizes:
    C_ij(F) = t_ij(F) + fee_j / income_i
with:
    fee_0 = F
    fee_1 = fee_2 = 0

This script:
  1) Tests a list/grid of candidate fees
  2) Runs EACH fee at least 5 times using 5 different seeds
  3) Computes mean/std of:
        - choice shares (s0,s1,s2)
        - total travel time (optional metric)
        - loss vs system-optimum shares (and optional travel-time target)
  4) Picks the fee with the smallest mean loss

IMPORTANT:
- You must implement `run_one_simulation(F, seed)` for your SUMO setup.
- Everything else is plug-and-play.
"""

from __future__ import annotations

import os
import json
import time
import shutil
import subprocess
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from routerl_aec_environment import TrafficEnvironment
from routerl_aec_environment import Keychain as kc

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



# -----------------------------
# User-configurable parameters
# -----------------------------

# 5 different seeds (you can change these)
SEEDS: List[int] = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

# Candidate fees to test (edit to your preferred range / spacing)
FEE_VALUES: np.ndarray = np.linspace(10, 50, 9)

# System-optimum target shares (must sum to 1)
S_STAR: np.ndarray = np.array([0.1325, 0.5875, 0.28], dtype=float)

NUMBER_OF_INDIVIDUALS_EACH_ROUTE: np.ndarray = np.array([40, 176, 84], dtype=int)

TT_PER_ROUTE = [21.468, 26.1609, 34.08]


# Loss weights: share mismatch always included; travel-time term optional.
W_SHARES: float = 1.0
W_TOTAL_TT: float = 0.0  # set >0 to include travel-time matching
TOTAL_TT_STAR: Optional[float] = None  # set if you want to match a target total travel time

# If your simulation is noisy, consider larger reps than 5
REPS_PER_FEE: int = len(SEEDS)  # you asked for at least 5; keep at 5 unless needed

# Output file for results
RESULTS_JSON_PATH: str = "fee_search_results.json"


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class FeeEvaluation:
    fee: float
    loss_mean: float
    loss_std: float
    shares_mean: List[float]
    shares_std: List[float]
    total_tt_mean: Optional[float]
    total_tt_std: Optional[float]
    # Keep raw per-seed results if you want to debug
    per_seed: List[Dict]

def compute_shares_and_total_tt_from_records(records):
    """
    records: env.travel_times or env.historic_data-like structure.
    Expected: list-of-dicts OR list-of-list-of-dicts (like your example).

    Returns:
        shares: np.array([s0, s1, s2]) floats summing to 1
        total_tt: float (sum of travel_time over all records with travel_time)
    """
    # --- flatten to a list of dicts ---
    flat = records
    # Flatten while it's a list with a single element that is also a list
    while isinstance(flat, list) and len(flat) == 1 and isinstance(flat[0], list):
        flat = flat[0]

    if not isinstance(flat, list):
        raise TypeError(f"Expected list or nested list, got: {type(flat)}")
    if len(flat) == 0:
        return np.array([0.0, 0.0, 0.0]), 0.0
    if not isinstance(flat[0], dict):
        raise TypeError(f"Expected list of dicts, got element type: {type(flat[0])}")

    # --- extract actions ---
    actions = []
    travel_times = []

    for d in flat:
        if not isinstance(d, dict):
            continue

        # action
        if "action" in d and d["action"] is not None:
            try:
                actions.append(int(d["action"]))
            except Exception:
                pass

        # travel_time
        if "travel_time" in d and d["travel_time"] is not None:
            try:
                travel_times.append(float(d["travel_time"]))
            except Exception:
                pass

    if len(actions) == 0:
        raise ValueError("No valid 'action' fields found in records.")
    if len(travel_times) == 0:
        raise ValueError("No valid 'travel_time' fields found in records.")

    actions = np.asarray(actions, dtype=int)
    counts = np.bincount(actions, minlength=3)[:3]
    total = counts.sum()
    shares = counts / total if total > 0 else np.array([0.0, 0.0, 0.0])

    total_tt = float(np.sum(travel_times))
    return shares.astype(float), total_tt

def choose_action_from_utility(tt_per_route, income, F):
    fees = np.array([F, 0.0, 0.0])
    #tt_per_route = tt_per_route / 16.5
    tt_per_route = [travel_time / 16.5 for travel_time in tt_per_route]
    C = np.array(tt_per_route) + fees / income
    #print("C is: ", C, fees, income, "\n\n")
    return int(np.argmin(C))


# -------------------------------------
# Run a TrafficEnvironment() simulation
# -------------------------------------

def run_one_simulation(F: float, seed: int) -> Tuple[np.ndarray, Optional[float]]:
    """
    Run ONE SUMO simulation with fee F and random seed.

    You MUST implement this function for your environment.

    It must return:
        shares: np.array([s0, s1, s2])  # fractions adding to 1
        total_tt: Optional[float]       # total travel time, or None if not available

    Notes:
    - Make sure that the simulation uses the provided seed.
    - Make sure the fee F is applied ONLY when an agent chooses action 0.
    - You can compute shares from logs of agent decisions, or from route IDs, etc.
    - You can compute total travel time from tripinfo.xml (sum of durations),
      or any other metric you prefer.
    """
    print("fee is: ", F, "\n\n")
    new_machines_after_mutation = 300
    human_learning_episodes = 0
    training_episodes = 200
    testing_episodes = 10

    total_episodes = human_learning_episodes + training_episodes + testing_episodes

    # Number of agents
    num_agents = 300

    # Origins and destination points in the network
    origins = ["E0"]
    destinations = ["E17.600"]

    records_folder = f"training_records_user_equilibrium_300_agents_{seed}"
    plots_folder = f"plots_user_equilibrium_300_agents_{seed}"

    phases = [1, int(total_episodes)]
    phase_names = ["Mutation and AV learning", "Testing phase"]

    env_params = {
        "agent_parameters": {
            "new_machines_after_mutation": new_machines_after_mutation,
            "num_agents": num_agents,
            "machine_parameters": {
                "behavior": "selfish",
                "observation_type": "previous_agents_avg_tt_per_route",
                "route_0_fee": 0,
                "travel_time_normalization_value": 1
            }
        },
        "simulator_parameters": {
            "network_name": "network",
            "custom_network_folder": "../network_analysis/network_base",
            "sumo_type": "sumo",
            "simulation_timesteps": 100,
        },
        "environment_parameters": {
            "observations_time_window": 10,
            "save_every": 5,
            "number_of_days": 1
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

    # ---- Environment initialization ----
    env = TrafficEnvironment(seed=seed, create_agents=True, create_paths=True, **env_params)

    env.start()
    env.reset()

    ##################### Mutation #####################
    env.mutation(mutation_start_percentile=0)

    env.reset()
    
    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()

        if termination or truncation:
            # agent is dead -> only valid action is None
            env.step(None)
            continue

        machine_agent = next((a for a in env.machine_agents if str(a.id) == agent))

        action = choose_action_from_utility(TT_PER_ROUTE, machine_agent.income / 30, F)
        env.step(action)

    
    shares, total_tt = compute_shares_and_total_tt_from_records(env.historic_data)
    env.stop_simulation()

    return shares, total_tt


# -----------------------------
# Evaluation & loss
# -----------------------------

def compute_loss(
    shares: np.ndarray,
    s_star: np.ndarray,
    total_tt: Optional[float],
    w_shares: float,
    w_total_tt: float,
    total_tt_star: Optional[float],
) -> float:
    if total_tt is None:
        raise ValueError("total_tt is None, but travel-time-only loss requires total_tt.")
    return float(total_tt)


def evaluate_fee(
    F: float,
    seeds: Sequence[int],
    reps: int,
    s_star: np.ndarray,
    w_shares: float,
    w_total_tt: float,
    total_tt_star: Optional[float],
    simulate: Callable[[float, int], Tuple[np.ndarray, Optional[float]]],
) -> FeeEvaluation:
    """
    Evaluate one fee across multiple seeds (and optionally multiple reps).
    If reps == len(seeds), we just run each seed once.
    If reps > len(seeds), seeds are cycled.
    """
    losses: List[float] = []
    shares_all: List[np.ndarray] = []
    tt_all: List[float] = []
    per_seed: List[Dict] = []

    for r in range(reps):
        seed = seeds[r % len(seeds)]
        shares, total_tt = simulate(F, seed)

        shares = np.asarray(shares, dtype=float)
        if shares.shape != (3,):
            raise ValueError(f"Expected shares shape (3,), got {shares.shape}")

        # Normalize defensively (in case of rounding issues)
        ssum = shares.sum()
        if not np.isfinite(ssum) or ssum <= 0:
            raise ValueError(f"Invalid shares returned: {shares}")
        shares = shares / ssum

        loss = compute_loss(shares, s_star, total_tt, w_shares, w_total_tt, total_tt_star)

        losses.append(loss)
        shares_all.append(shares)
        if total_tt is not None:
            tt_all.append(float(total_tt))

        per_seed.append({
            "seed": int(seed),
            "shares": shares.tolist(),
            "total_tt": None if total_tt is None else float(total_tt),
            "loss": float(loss),
        })

    losses_arr = np.array(losses, dtype=float)
    shares_arr = np.stack(shares_all, axis=0)  # (reps, 3)

    total_tt_mean: Optional[float] = None
    total_tt_std: Optional[float] = None
    if len(tt_all) > 0:
        tt_arr = np.array(tt_all, dtype=float)
        total_tt_mean = float(tt_arr.mean())
        total_tt_std = float(tt_arr.std(ddof=0))

    return FeeEvaluation(
        fee=float(F),
        loss_mean=float(losses_arr.mean()),
        loss_std=float(losses_arr.std(ddof=0)),
        shares_mean=shares_arr.mean(axis=0).tolist(),
        shares_std=shares_arr.std(axis=0, ddof=0).tolist(),
        total_tt_mean=total_tt_mean,
        total_tt_std=total_tt_std,
        per_seed=per_seed,
    )


# -----------------------------
# Search over fees
# -----------------------------

def search_over_fees(
    fee_values: Sequence[float],
    seeds: Sequence[int],
    reps_per_fee: int,
    s_star: np.ndarray,
    w_shares: float,
    w_total_tt: float,
    total_tt_star: Optional[float],
    simulate: Callable[[float, int], Tuple[np.ndarray, Optional[float]]],
) -> Tuple[FeeEvaluation, List[FeeEvaluation]]:
    results: List[FeeEvaluation] = []

    for idx, F in enumerate(fee_values):
        print(f"[{idx+1}/{len(fee_values)}] Testing fee F={F:.6g} ...")
        res = evaluate_fee(
            F=float(F),
            seeds=seeds,
            reps=reps_per_fee,
            s_star=s_star,
            w_shares=w_shares,
            w_total_tt=w_total_tt,
            total_tt_star=total_tt_star,
            simulate=simulate,
        )
        print(
            f"    loss_mean={res.loss_mean:.6g} (std={res.loss_std:.3g}) | "
            f"shares_mean={np.array(res.shares_mean)} | "
            f"total_tt_mean={res.total_tt_mean}"
        )
        results.append(res)

    best = min(results, key=lambda r: r.loss_mean)
    return best, results


def save_results_json(path: str, best: FeeEvaluation, results: List[FeeEvaluation]) -> None:
    payload = {
        "best": asdict(best),
        "results": [asdict(r) for r in results],
        "meta": {
            "seeds": SEEDS,
            "fee_values": [float(x) for x in FEE_VALUES],
            "s_star": S_STAR.tolist(),
            "w_shares": W_SHARES,
            "w_total_tt": W_TOTAL_TT,
            "total_tt_star": TOTAL_TT_STAR,
            "reps_per_fee": REPS_PER_FEE,
        }
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved results to: {path}")


# -----------------------------
# Main
# -----------------------------

def main():
    # Sanity checks
    if len(SEEDS) < 5:
        raise ValueError("Provide at least 5 seeds.")
    if REPS_PER_FEE < 5:
        raise ValueError("REPS_PER_FEE must be at least 5 as requested.")
    if S_STAR.shape != (3,):
        raise ValueError("S_STAR must be a length-3 vector.")
    if not np.isclose(S_STAR.sum(), 1.0):
        raise ValueError("S_STAR must sum to 1.")

    best, results = search_over_fees(
        fee_values=FEE_VALUES,
        seeds=SEEDS,
        reps_per_fee=REPS_PER_FEE,
        s_star=S_STAR,
        w_shares=W_SHARES,
        w_total_tt=W_TOTAL_TT,
        total_tt_star=TOTAL_TT_STAR,
        simulate=run_one_simulation,
    )

    print("\n=== BEST FEE (by mean loss) ===")
    print(f"Fee: {best.fee}")
    print(f"Mean shares: {best.shares_mean}  (std: {best.shares_std})")
    print(f"Mean loss: {best.loss_mean} (std: {best.loss_std})")
    print(f"Mean total travel time: {best.total_tt_mean} (std: {best.total_tt_std})")

    save_results_json(RESULTS_JSON_PATH, best, results)


if __name__ == "__main__":
    main()