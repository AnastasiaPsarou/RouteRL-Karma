import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Times New Roman']


def smooth_data(data, window_size):
    """Smooth data using a moving average with padding at the edges."""
    if len(data) < window_size:
        return data  # If the data is too short, return it unchanged
    half_window = window_size // 2
    # Pad the data at the edges to reduce edge effects
    padded_data = np.pad(data, (half_window, half_window), mode='edge')
    smoothed_data = np.convolve(
        padded_data, 
        np.ones(window_size) / window_size, 
        mode='valid'
    )
    # Ensure the result has the same length as the original data
    return smoothed_data[:len(data)]

# Function to process CSV files for a single algorithm
def process_csv_files(directory, step=10):
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return [], []

    csv_files = sorted(
        [file for file in os.listdir(directory) if file.endswith('.csv')],
        key=lambda x: int(''.join(filter(str.isdigit, os.path.splitext(x)[0]))))
    
    # Select every `step`th file
    csv_files = csv_files[::step]
    
    if not csv_files:
        print(f"No CSV files found in: {directory}")
        return [], []

    action_proportions = []
    file_numbers = []
    
    for csv_file in csv_files:
        file_number = int(''.join(filter(str.isdigit, os.path.splitext(csv_file)[0])))
        file_numbers.append(file_number)
        
        file_path = os.path.join(directory, csv_file)
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue
        
        if "kind" not in df.columns or "action" not in df.columns:
            print(f"Missing expected columns in {file_path}")
            continue
        
        # Calculate the proportion of action 0 for kind = "AV"
        av_actions = df[df["kind"] == "AV"]["action"].value_counts(normalize=True).sort_index()
        action_proportions.append(av_actions.get(0, 0))  # Get proportion of action 0, default to 0
    
    return file_numbers, action_proportions

colors = ["firebrick", "teal", "peru", "navy", "salmon", "slategray", "darkviolet"]


def plot_actions_with_error(
    ax, directories, algorithm_names, process_function, colors, 
    smoothing_window=10, show_legend=False, extend_x=10000, 
    show_vertical_line=True, legend_location='lower left'
):
    added_labels = set()  # Keep track of added legend labels
    handles = []  # Store line handles for the legend
    labels = []   # Store corresponding labels for the legend

    grouped_data = {}  # Dictionary to group data by algorithm

    # Group data by algorithm name
    for i, directory in enumerate(directories):
        # Use the passed `process_function` instead of a hardcoded function
        file_numbers, action_proportions = process_function(directory)

        if not action_proportions:
            continue

        # Smooth data using the smoothing function
        smoothed_proportions = smooth_data(action_proportions, smoothing_window)

        # Ensure `file_numbers` and `smoothed_proportions` have the same length
        if len(file_numbers) > len(smoothed_proportions):
            file_numbers = file_numbers[:len(smoothed_proportions)]

        # Add data to grouped_data dictionary
        algorithm_name = algorithm_names[i]
        if algorithm_name not in grouped_data:
            grouped_data[algorithm_name] = {}
        for x, y in zip(file_numbers, smoothed_proportions):
            if x not in grouped_data[algorithm_name]:
                grouped_data[algorithm_name][x] = []
            grouped_data[algorithm_name][x].append(y)

    # Process grouped data for plotting
    for algorithm_name, data in grouped_data.items():
        sorted_episodes = sorted(data.keys())
        means = []
        stds = []
        for episode in sorted_episodes:
            values = data[episode]
            mean = np.mean(values)
            above = [v for v in values if v > mean]
            below = [v for v in values if v < mean]
            std_above = np.std(above) if above else 0
            std_below = np.std(below) if below else 0

            means.append(mean)
            stds.append((std_below, std_above))


        # Filter data to start from x=200
        filtered_episodes = [ep for ep in sorted_episodes if ep >= 0]
        filtered_means = [mean for ep, mean in zip(sorted_episodes, means) if ep >= 0]
        filtered_stds = [std for ep, std in zip(sorted_episodes, stds) if ep >= 0]

        # Extend the last value of the plot with error bars
        if filtered_episodes[-1] < extend_x:
            last_x = filtered_episodes[-1]
            last_mean = filtered_means[-1]
            last_std = filtered_stds[-1]
            extended_x = list(range(last_x + 1, extend_x + 1))
            extended_mean = [last_mean] * len(extended_x)
            extended_std = [last_std] * len(extended_x)

            filtered_episodes = list(filtered_episodes) + extended_x
            filtered_means = list(filtered_means) + extended_mean
            filtered_stds = list(filtered_stds) + extended_std

        # Determine line style for specific algorithms
        if "adapt" in algorithm_name or "centralized" in algorithm_name:
            linestyle = ':'  # Dashed line for adaptation cases
        else:
            linestyle = '-'   # Solid line for other cases

        # Plot the mean line with error bars
        color = colors[len(added_labels) % len(colors)]  # Select color for the plot
        ax.plot(
            filtered_episodes,
            filtered_means,
            color=color,
            linewidth=2.5,
            linestyle=linestyle,  # Apply the determined line style
            label=algorithm_name
        )
        lower_bounds = [m - std[0] for m, std in zip(filtered_means, filtered_stds)]
        upper_bounds = [m + std[1] for m, std in zip(filtered_means, filtered_stds)]

        ax.fill_between(
            filtered_episodes,
            lower_bounds,
            upper_bounds,
            color=color,
            alpha=0.2
        )

        added_labels.add(algorithm_name)

    # Add the horizontal line at y=1
    optimal_line = ax.axhline(
        y=1,
        color='black',
        linestyle='-',
        linewidth=6,  # Make the line thicker
        label='Optimal Choice'
    )
    handles.append(optimal_line)  # Add horizontal line to legend
    labels.append('Optimal choice')  # Add label for the horizontal line

    # Optionally add the vertical line at episode 8200
    """if show_vertical_line:
        vertical_line = ax.axvline(
            x=8200,
            color='black',
            linestyle='--',
            linewidth=4,
            label='End of AV learning phase'
        )
        handles.append(vertical_line)  # Add vertical line to legend
        labels.append('End of AV learning phase')  # Add label for the vertical line"""

    desired_ticks = [0, 0.5, 1]  # Values you want on the y-axis
    plt.yticks(desired_ticks)
    plt.ylim(0, 1)

    ax.set_xlabel("Episodes (days)", fontsize=24, labelpad=10)  # Move xlabel outside the plot
    if show_legend:
        # Create legend using handles and labels
        ax.legend(fontsize=12, loc=legend_location, ncol=2)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(visible=True, linestyle='--', linewidth=0.5)

def process_action_counts(directory, step=10, kind_filter="AV"):
    """
    Read ep*.csv in `directory`, sample every `step`-th file,
    and return:
      episodes:  sorted list of episode numbers (aligned)
      action_counts: dict[action] -> list of counts aligned to `episodes`
    Only rows with df['kind'] == kind_filter are used, if that column is present.
    If `kind` is missing, all rows are counted.
    """
    if not os.path.exists(directory):
        print(f"[WARN] Directory does not exist: {directory}")
        return [], {}

    csv_files = sorted(
        [fn for fn in os.listdir(directory) if fn.endswith(".csv") and fn.startswith("ep")],
        key=lambda x: int(''.join(filter(str.isdigit, os.path.splitext(x)[0]))),
    )
    csv_files = csv_files[::step]  # sample
    if not csv_files:
        print(f"[WARN] No CSV files found in: {directory}")
        return [], {}

    # First pass: collect all actions present (to keep columns consistent)
    actions_seen = set()
    eps_order = []
    per_file_counts = []
    for fn in csv_files:
        ep = int(''.join(filter(str.isdigit, os.path.splitext(fn)[0])))
        path = os.path.join(directory, fn)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[WARN] Error reading {path}: {e}")
            continue

        if kind_filter is not None and "kind" in df.columns:
            df = df[df["kind"] == kind_filter]

        if "action" not in df.columns:
            print(f"[WARN] Missing 'action' in {path}")
            continue

        counts = df["action"].value_counts()
        if counts.empty:
            continue

        eps_order.append(ep)
        per_file_counts.append(counts)
        actions_seen.update(counts.index.tolist())

    if not eps_order:
        return [], {}

    actions_sorted = sorted(actions_seen)
    # Align into arrays
    episodes = []
    action_counts = {a: [] for a in actions_sorted}

    for ep, counts in sorted(zip(eps_order, per_file_counts), key=lambda t: t[0]):
        episodes.append(ep)
        for a in actions_sorted:
            action_counts[a].append(int(counts.get(a, 0)))

    return episodes, action_counts


def plot_action_counts(
    ax, directories, algorithm_names, process_function=process_action_counts,
    kind_filter="AV", step=10, smoothing_window=1, show_legend=True,
    legend_location="best"
):
    """
    Group directories by algorithm name (replications).
    For each algorithm:
      - compute per-episode action counts for each replication,
      - align on common episode numbers,
      - average counts across replications (mean),
      - plot one line per action of the replication-mean counts.
    No error bands (you asked for counts only).
    """
    # 1) Gather per-replication series
    grouped = {}  # algo -> list of (episodes, {action -> counts})
    for d, algo in zip(directories, algorithm_names):
        eps, act_counts = process_function(d, step=step, kind_filter=kind_filter)
        if not eps or not act_counts:
            print(f"[INFO] Skipping empty result for: {d}")
            continue
        grouped.setdefault(algo, []).append((eps, act_counts))

    if not grouped:
        raise ValueError("No usable data to plot. Check paths, CSVs, or columns.")

    # 2) For each algorithm, align episodes across reps and compute mean per action
    for algo, rep_list in grouped.items():
        # union of episodes across reps
        all_eps = sorted(set().union(*[set(eps) for eps, _ in rep_list]))
        if not all_eps:
            continue

        # union of actions across reps
        all_actions = sorted(set().union(*[
            set(act_counts.keys()) for _, act_counts in rep_list
        ]))

        # Build per-action matrix: rows=replications, cols=episodes
        means_per_action = {}
        for a in all_actions:
            # Make a matrix of shape (n_reps, n_eps) with NaNs
            M = np.full((len(rep_list), len(all_eps)), np.nan, dtype=float)
            for r, (eps, act_counts) in enumerate(rep_list):
                # Map episode -> count for this action
                cnts = act_counts.get(a, [])
                if not cnts:
                    continue
                # Build mapping for this replication
                rep_map = dict(zip(eps, cnts))
                for c, ep in enumerate(all_eps):
                    if ep in rep_map:
                        M[r, c] = rep_map[ep]
            # mean across reps (ignore NaNs)
            mean_counts = np.nanmean(M, axis=0)
            # Optional smoothing
            if smoothing_window and smoothing_window > 1:
                mean_counts = smooth_data(mean_counts, smoothing_window)
            means_per_action[a] = mean_counts

        # 3) Plot one line per action for this algorithm
        # Different algorithms -> different linestyles; actions -> colors
        if "adapt" in algo or "centralized" in algo:
            linestyle = ":"
        else:
            linestyle = "-"

        # define a color cycle for actions
        action_palette = ["firebrick", "teal", "peru", "navy", "salmon", "slategray", "darkviolet",
                          "goldenrod", "olive", "tab:pink", "tab:purple", "tab:cyan"]

        for idx, a in enumerate(sorted(means_per_action.keys())):
            y = means_per_action[a]
            ax.plot(
                all_eps,
                y,
                label=f"{algo} — action {a}",
                linewidth=2.0,
                linestyle=linestyle,
                color=action_palette[idx % len(action_palette)],
            )

    ax.set_xlabel("Episode", fontsize=16)
    ax.set_ylabel("Number of agents choosing each action", fontsize=16)
    ax.set_title("Counts per action (averaged across replications with same name)", fontsize=14)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.tick_params(axis="both", labelsize=12)
    if show_legend:
        ax.legend(fontsize=10, loc=legend_location, ncol=2)



rcParams['font.family'] = 'Times New Roman'

# Create a single figure and axis
fig, ax = plt.subplots(figsize=(7, 4), dpi=300)  # Reduced width

colors = ["firebrick", "teal", "peru", "navy", "salmon", "slategray", "darkviolet"]

# Input directories and algorithm names
directories = [
    "../data/training_records_monetary_pricing_300_agents_long/episodes",
    "../data/training_records_monetary_pricing_300_agents_long_2/episodes",
    "../data/training_records_monetary_pricing_300_agents_long_3/episodes",
    "../data/training_records_monetary_pricing_300_agents_long_4/episodes",
    "../data/training_records_monetary_pricing_300_agents_long_5/episodes",
]

algorithm_names = [
   "MAPPO", "MAPPO", "MAPPO", "MAPPO", "MAPPO"
]

# Plot the data
"""plot_actions_with_error(
    ax,
    directories,
    algorithm_names,
    process_function=process_csv_files,
    colors=colors,
    smoothing_window=1,
    show_legend=True,
    extend_x=0
)"""
fig, ax = plt.subplots(figsize=(7, 4), dpi=300)

plot_action_counts(
    ax,
    directories=directories,
    algorithm_names=algorithm_names,
    process_function=process_action_counts,  # uses the new function above
    kind_filter="AV",        # or None to use all rows
    step=10,                 # sample every 10th episode file (set 1 for all)
    smoothing_window=1,      # set >1 to smooth the lines
    show_legend=True,
    legend_location="best"
)

plt.tight_layout()
plt.show()

