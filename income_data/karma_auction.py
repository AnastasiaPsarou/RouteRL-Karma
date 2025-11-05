import torch

def karma_auction(x: int, agent_bids: dict):
    """

    Conducts a karma-based auction among agents based on their bid tensors.

    Args:

        x (int): Number of top bids (agents) per route-timeslot combination to assign.

        agent_bids (dict): Dictionary where keys are agent IDs and values are torch.Tensor (n x m),

                           representing bids for each timeslot (n) and route (m).


    Returns:

        dict: Assignment dictionary where each key is an agent_id, and the value is

              a tuple (route, timeslot, fee).

              

    Process:

        1. Flatten and sum bids to find the most attractive route-timeslot pairs.

        2. Sort all (timeslot, route) combinations by total bid value.

        3. For each combination, assign it to the top 'x' unassigned agents based on their bids.

        4. Each agent is assigned only once.

    """

    assignment = {}

    assigned_agents = set()

    # Determine shape (assumes consistent shapes across agents)
    sample_agent = next(iter(agent_bids.values()))

    n, m = sample_agent.shape

    # Sum bids over all agents to find attractiveness per (timeslot, route)
    total_bids = torch.zeros((n, m))

    for bids in agent_bids.values():
        total_bids += bids

    total_bids_flat = total_bids.flatten()

    # Get sorted indices by attractiveness (descending)
    sorted_indices = torch.argsort(total_bids_flat, descending=True)

    # Proceed in order of most attractive combinations
    for idx in sorted_indices:

        timeslot = idx // m

        route = idx % m

        # Collect all agents' bids for this (timeslot, route)
        agent_rankings = []

        for agent_id, bids in agent_bids.items():


            if agent_id not in assigned_agents:

                agent_rankings.append((agent_id, bids[timeslot, route].item()))
 
        if not agent_rankings:

            continue

        # Sort agents for this slot-route combination by highest bids
        agent_rankings.sort(key=lambda x: x[1], reverse=True)

        # Assign top x agents
        for agent_id, fee in agent_rankings[:x]:

            if agent_id not in assigned_agents:

                assignment[agent_id] = (route.item() if hasattr(route, 'item') else route,

                                        timeslot.item() if hasattr(timeslot, 'item') else timeslot,

                                        fee)

                assigned_agents.add(agent_id)

        # Stop if all agents are assigned
        if len(assigned_agents) == len(agent_bids):

            break

    return assignment


agent_bids = {
    "A1": torch.tensor([[10, 20, 15],
                        [8, 25, 5]], dtype=torch.float32),
    "A2": torch.tensor([[12, 18, 9],
                        [20, 15, 7]], dtype=torch.float32),
    "A3": torch.tensor([[5, 8, 19],
                        [30, 10, 12]], dtype=torch.float32)
}
 
# Run auction (assign to top 1 bidder per route-timeslot)
# Assignment {'agent_id': (route, timeslot, fee)}
assignment = karma_auction(x=1, agent_bids=agent_bids)
 
# Show results
print(assignment)