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
        means_by_ep = {}
        stds_by_ep = {}
        for episode in sorted(data.keys()):
            values = data[episode]
            mean = np.mean(values)
            above = [v for v in values if v > mean]
            below = [v for v in values if v < mean]
            std_above = np.std(above) if above else 0.0
            std_below = np.std(below) if below else 0.0
            means_by_ep[episode] = mean
            stds_by_ep[episode] = (std_below, std_above)

        # Choose your cutoff (make it a param if you like)
        min_episode = 0  # or 200 if you’re sure you have those episodes

       # Filter data to start from x=200
        filtered_episodes = sorted_episodes
        filtered_means     = [means_by_ep[ep] for ep in filtered_episodes]
        filtered_stds      = [stds_by_ep[ep] for ep in filtered_episodes]


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
    labels.append('Number of vehicles choosing route 0')  # Add label for the horizontal line

    # Optionally add the vertical line at episode 8200
    """if show_vertical_line:
        vertical_line = ax.axvline(
            x=400,
            color='black',
            linestyle='--',
            linewidth=2,
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
        ax.legend(fontsize=10, loc=legend_location, ncol=1)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(visible=True, linestyle='--', linewidth=0.5)




rcParams['font.family'] = 'Times New Roman'

# Create a single figure and axis
fig, ax = plt.subplots(figsize=(7, 4), dpi=300)  # Reduced width

colors = ["firebrick", "teal", "peru", "navy", "salmon", "slategray", "darkviolet"]

# Input directories and algorithm names
directories = [
    "../google_maps/training_records_google_maps_10_agents/episodes",
]

algorithm_names = [
   "MAPPO"
]

# Plot the data
plot_actions_with_error(
    ax,
    directories,
    algorithm_names,
    process_function=process_csv_files,
    colors=colors,
    smoothing_window=2,
    show_legend=True,
    extend_x=0
)

ax.set_ylim(0, 1)
ax.set_ylabel("Proportion of AVs \nchoosing route 0", fontsize=16)
ax.set_xlabel("Episode", fontsize=16)

plt.tight_layout()
plt.savefig("action_proportions.png", dpi=300, bbox_inches='tight')
