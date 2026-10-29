
import os
import sys

import numpy as np
import torch
from tqdm import tqdm
import argparse

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)

from routerl_aec_environment_karma import TrafficEnvironment

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
NUM_AGENTS = 300
TRAINING_EPISODES = 200
TESTING_EPISODES = 0

TOTAL_EPISODES = TRAINING_EPISODES + TESTING_EPISODES

# Karma prices for routes 0, 1, and 2
ROUTE_PRICES = (7, 2, 0)

MAX_ALLOWED_BID = 5

ORIGINS = ["E0"]
DESTINATIONS = ["E17.600"]


# ------------------------------------------------------------
# Rule-based Karma bidding
# ------------------------------------------------------------
def choose_karma_bid(karma_balance, nvec, action_mask=None):
    """
    Submit the route price for every route the commuter can afford.

    balance < 2:
        [0, 0, 0]

    2 <= balance < 7:
        [0, 2, 0]

    balance >= 7:
        [7, 2, 0]

    If the full bid vector is not permitted by the environment's
    action mask, submit a bid only for the fastest affordable route.

    The result is a BID VECTOR, not an assigned route.
    """

    balance = float(karma_balance)

    if not np.isfinite(balance) or balance < 0:
        raise ValueError(
            f"Invalid Karma balance: {karma_balance}"
        )

    # Bid the required price for each affordable route.
    preferred = [
        price if balance >= price else 0
        for price in ROUTE_PRICES
    ]

    nvec = np.asarray(nvec, dtype=np.int64)

    if nvec.shape != (3,):
        raise ValueError(
            f"Expected 3 bid components, got {nvec.tolist()}"
        )

    # Candidate bid vectors, ordered by preference.
    candidates = [preferred]

    if balance >= ROUTE_PRICES[0]:
        candidates.append([7, 0, 0])

    if balance >= ROUTE_PRICES[1]:
        candidates.append([0, 2, 0])

    candidates.append([0, 0, 0])

    # 10% probability of bidding nothing
    if np.random.random() < 0.10:
        candidates = [[0, 0, 0]]


    # Read the flattened joint-action mask.
    mask = None

    if action_mask is not None:
        mask = np.asarray(action_mask).reshape(-1)

        expected_size = int(np.prod(nvec))

        if mask.size != expected_size:
            raise ValueError(
                f"Action mask size {mask.size} != "
                f"expected size {expected_size}"
            )

    # Return the first feasible candidate.
    for bids in candidates:

        # Check MultiDiscrete action bounds.
        if any(
            bid >= upper
            for bid, upper in zip(bids, nvec)
        ):
            continue

        # Check the environment's action mask.
        if mask is not None:

            joint_index = int(
                np.ravel_multi_index(
                    tuple(bids),
                    tuple(nvec)
                )
            )

            if not bool(mask[joint_index]):
                continue

        return bids

    raise ValueError(
        f"No feasible bid vector for Karma balance {balance}"
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    records_folder = (
        f"training_records_karma_rule_based_"
        f"300_agents_{seed}_7_2_0"
    )

    plots_folder = (
        f"plots_karma_rule_based_"
        f"300_agents_{seed}_7_2_0"
    )

    phases = [1, TOTAL_EPISODES]
    phase_names = ["Rule-based bidding", "Testing"]

    env_params = {
        "agent_parameters": {
            "new_machines_after_mutation": NUM_AGENTS,
            "num_agents": NUM_AGENTS,
            "machine_parameters": {
                "behavior": "selfish",
                "observation_type": "previous_agents_avg_tt_per_route",
                "route_0_fee": 0,
                "travel_time_normalization_value": 100,
                "max_allowed_bid": MAX_ALLOWED_BID,
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
            "save_every": 5,
            "number_of_days": 30,
            "centrally_defined_price_route_0": 7,
            "centrally_defined_price_route_1": 2,
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
            "origins": ORIGINS,
            "destinations": DESTINATIONS,
            "number_of_paths": 3,
            "beta": -1,
            "visualize_paths": True,
            "all_origins_to_all_destinations": False,
        },
    }

    # --------------------------------------------------------
    # Initialize Karma auction environment
    # --------------------------------------------------------
    env = TrafficEnvironment(
        seed=seed,
        create_agents=True,
        create_paths=True,
        karma_pricing=True,
        **env_params,
    )

    print("Number of total agents:", len(env.all_agents))
    print("Number of human agents:", len(env.human_agents))
    print("Number of machine agents:", len(env.machine_agents))

    env.start()
    env.reset()

    # --------------------------------------------------------
    # Mutation: humans become machine agents
    # --------------------------------------------------------
    env.mutation(mutation_start_percentile=0)

    machine_by_name = {
        str(machine.id): machine
        for machine in env.machine_agents
    }

    if not machine_by_name:
        raise RuntimeError("No machine agents after mutation.")

    # --------------------------------------------------------
    # Extract Karma bid action space
    # --------------------------------------------------------
    sample_name = next(iter(machine_by_name))

    action_space = env._action_spaces[sample_name]

    nvec = np.asarray(
        action_space.nvec,
        dtype=np.int64
    )

    print("Karma bid action space:", nvec.tolist())

    if nvec.shape != (3,):
        raise ValueError(
            f"Expected 3-dimensional bid vector, got {nvec}"
        )

    if nvec[0] <= 7 or nvec[1] <= 2:
        raise ValueError(
            "The action space cannot represent bids of 7 and 2. "
            "Check max_allowed_bid."
        )

    pbar = tqdm(
        total=TOTAL_EPISODES,
        desc="Rule-based Karma bidding"
    )

    # ========================================================
    # TRAINING / WARM-UP
    # ========================================================
    # No learning takes place.
    # Agents use the same rule-based strategy throughout.
    # ========================================================

    for episode in range(TRAINING_EPISODES):
        print("episode is: ", episode, "\n\n")

        env.reset()

        for agent in env.agent_iter():

            observation, reward, termination, truncation, info = (
                env.last()
            )

            done = termination or truncation

            machine = machine_by_name.get(str(agent))

            if machine is not None and not done:

                # Current Karma balance BEFORE bidding.
                karma_balance = machine.karma_balance

                # Read action mask if provided.
                action_mask = (
                    observation.get("action_mask")
                    if isinstance(observation, dict)
                    else None
                )

                # Submit a BID VECTOR, not a route number.
                action = choose_karma_bid(
                    karma_balance=karma_balance,
                    nvec=nvec,
                    action_mask=action_mask,
                )
                print("My karma balance is, and my action is: ", karma_balance, action)

            else:
                action = None

            # The auction determines the actual assigned route.
            env.step(action)

        pbar.update(1)

    # ========================================================
    # TESTING
    # ========================================================

    pbar.set_description("Testing rule-based Karma bidding")

    # Same urgency evaluation protocol as MAPPO.
    env.testing_urgency_function()

    for episode in range(TESTING_EPISODES):

        env.reset()

        for agent in env.agent_iter():

            observation, reward, termination, truncation, info = (
                env.last()
            )

            done = termination or truncation

            machine = machine_by_name.get(str(agent))

            if machine is not None and not done:

                karma_balance = machine.karma_balance

                action_mask = (
                    observation.get("action_mask")
                    if isinstance(observation, dict)
                    else None
                )

                action = choose_karma_bid(
                    karma_balance=karma_balance,
                    nvec=nvec,
                    action_mask=action_mask,
                )

            else:
                action = None

            env.step(action)

        pbar.update(1)

    # ========================================================
    # FINISH
    # ========================================================

    pbar.close()

    env.plot_results()
    env.stop_simulation()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="Random seed for the experiment"
    )

    args = parser.parse_args()

    main(args.seed)

