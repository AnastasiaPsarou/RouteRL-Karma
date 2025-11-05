import numpy as np
import pandas as pd

def load_unique_incomes(samples_path, col="income"):
    """
    Load unique numeric income values from a CSV file.
    Returns a sorted NumPy array of unique values.
    """
    s = pd.read_csv(samples_path, usecols=[col])[col]
    s = pd.to_numeric(s, errors="coerce").dropna()
    uniq = np.unique(s.values)
    if uniq.size == 0:
        raise ValueError(f"No valid numeric '{col}' values found in {samples_path}.")
    return uniq


def assign_incomes_to_agents(
    agents_df,
    income_samples_path,
    samples_col="income",
    income_col_out="income",
    mode="random",
    seed=42,
):
    """
    Assign incomes from sampled distribution to AV agents.

    Parameters
    ----------
    agents_df : pd.DataFrame
        DataFrame of agents to which incomes should be assigned.
    income_samples_path : str or Path
        Path to CSV file containing sampled incomes.
    samples_col : str
        Column name in the samples CSV (default: 'income').
    income_col_out : str
        Name of the income column to create/overwrite in agents_df.
    mode : str
        Assignment mode: 'random', 'cycle', or 'match_by_rank'.
    seed : int or None
        Random number generator seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Copy of `agents_df` with a new/updated income column.
    """
    # Load unique incomes
    unique_incomes = load_unique_incomes(income_samples_path, col=samples_col)
    n_agents = len(agents_df)
    rng = np.random.default_rng(seed)

    if mode == "random":
        assigned = rng.choice(unique_incomes, size=n_agents, replace=True)

    elif mode == "cycle":
        reps = int(np.ceil(n_agents / unique_incomes.size))
        assigned = np.tile(unique_incomes, reps)[:n_agents]

    elif mode == "match_by_rank":
        # If agents already have an income-like column, use it for ranking;
        # otherwise, use their index.
        base_col_candidates = [income_col_out, "income", "base_income", "salary"]
        base_col = next((c for c in base_col_candidates if c in agents_df.columns), None)

        if base_col is not None:
            base_vals = pd.to_numeric(agents_df[base_col], errors="coerce").fillna(-np.inf)
            order = np.argsort(base_vals.values)  # ascending
        else:
            order = np.arange(n_agents)

        # Map ranks to unique incomes by quantiles
        ranks = np.empty(n_agents, dtype=int)
        ranks[order] = np.arange(n_agents)
        q = (ranks + 0.5) / n_agents
        k = unique_incomes.size
        cdf_grid = (np.arange(k) + 0.5) / k
        assigned = np.interp(q, cdf_grid, unique_incomes)

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose from 'random', 'cycle', or 'match_by_rank'.")

    result = agents_df.copy()
    result[income_col_out] = assigned
    return result


print(load_unique_incomes("income_samples.csv"))