
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
TESTING_EPISODES = 10

TOTAL_EPISODES = TRAINING_EPISODES + TESTING_EPISODES

# Karma prices for routes 0, 1, and 2
ROUTE_PRICES = (7, 2, 0)

MAX_ALLOWED_BID = 5

ORIGINS = ["E0"]
DESTINATIONS = ["E17.600"]


# ------------------------------------------------------------
# Rule-based Karma bidding
# ------------------------------------------------------------
def choose_karma_bid(karma_balance, urgency, nvec, action_mask=None):
    """
    Return a three-route bid vector based on urgency and Karma balance.

    Route 0: bid 7 when urgency > 0.7 and Karma >= 7.
    Route 1: bid 2 when 0.3 < urgency < 0.7 and Karma >= 2.
    Otherwise, bid nothing.

    The auction determines the assigned route.
    """

    balance = float(karma_balance)
    urgency = float(urgency)

    if not np.isfinite(balance) or balance < 0:
        raise ValueError(f"Invalid Karma balance: {karma_balance}")

    if not np.isfinite(urgency):
        raise ValueError(f"Invalid urgency: {urgency}")

    nvec = np.asarray(nvec, dtype=np.int64)

    if nvec.shape != (3,):
        raise ValueError(
            f"Expected three bid components, got {nvec.tolist()}"
        )

    # --------------------------------------------------------
    # Choose bid vector based on urgency and available Karma
    # --------------------------------------------------------
    if urgency > 0.7 and balance >= ROUTE_PRICES[0]:
        bids = [7, 0, 0]

    elif 0.3 < urgency < 0.7 and balance >= ROUTE_PRICES[1]:
        bids = [0, 2, 0]

    else:
        bids = [0, 0, 0]

    # --------------------------------------------------------
    # Check MultiDiscrete action bounds
    # --------------------------------------------------------
    if any(
        bid < 0 or bid >= int(limit)
        for bid, limit in zip(bids, nvec)
    ):
        raise ValueError(
            f"Bid vector {bids} is outside "
            f"MultiDiscrete action space {nvec.tolist()}"
        )

    # --------------------------------------------------------
    # Validate against the environment action mask
    # --------------------------------------------------------
    if action_mask is not None:

        mask = np.asarray(action_mask).reshape(-1)
        expected_size = int(np.prod(nvec))

        if mask.size != expected_size:
            raise ValueError(
                f"Action mask size {mask.size} "
                f"!= expected {expected_size}"
            )

        joint_index = int(
            np.ravel_multi_index(
                tuple(bids),
                tuple(nvec)
            )
        )

        if not bool(mask[joint_index]):
            raise ValueError(
                f"Bid vector {bids} is not permitted "
                f"(balance={balance}, urgency={urgency})"
            )

    return bids


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(seed):

    torch.manual_seed(seed)
    np.random.seed(seed)

    records_folder = (
        f"training_records_karma_urgency_rule_"
        f"300_agents_{seed}_7_2_0"
    )

    plots_folder = (
        f"plots_karma_urgency_rule_"
        f"300_agents_{seed}_7_2_0"
    )

    phases = [1, TOTAL_EPISODES]
    phase_names = ["Urgency-based bidding", "Testing"]

    env_params = {
        "agent_parameters": {
            "new_machines_after_mutation": NUM_AGENTS,
            "num_agents": NUM_AGENTS,
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
    #env.reset()

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
        desc="Urgency-based Karma bidding"
    )

    # ========================================================
    # TRAINING / WARM-UP
    # ========================================================
    # No learning takes place.
    # Agents use the same urgency-based bidding rule throughout.
    # ========================================================

    for episode in range(TRAINING_EPISODES):

        print("Episode:", episode)

        env.reset()

        for agent in env.agent_iter():

            observation, reward, termination, truncation, info = (
                env.last()
            )

            done = termination or truncation

            machine = machine_by_name.get(str(agent))

            if machine is not None and not done:

                # Access the current machine-agent attributes.
                karma_balance = machine.karma_balance
                urgency = machine.urgency

                # Extract action mask if provided.
                action_mask = (
                    observation.get("action_mask")
                    if isinstance(observation, dict)
                    else None
                )

                # Submit a BID VECTOR, not a route index.
                action = choose_karma_bid(
                    karma_balance=karma_balance,
                    urgency=urgency,
                    nvec=nvec,
                    action_mask=action_mask,
                )

                print(
                    "Karma:", karma_balance,
                    "Urgency:", urgency,
                    "Bids:", action
                )

            else:
                action = None

            # The auction determines the assigned route.
            env.step(action)

        pbar.update(1)

    # ========================================================
    # TESTING
    # ========================================================

    pbar.set_description("Testing urgency-based Karma bidding")

    # Use the same testing urgency protocol as MAPPO.
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
                urgency = machine.urgency

                action_mask = (
                    observation.get("action_mask")
                    if isinstance(observation, dict)
                    else None
                )

                action = choose_karma_bid(
                    karma_balance=karma_balance,
                    urgency=urgency,
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

