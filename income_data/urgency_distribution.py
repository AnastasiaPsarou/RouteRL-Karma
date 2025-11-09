import numpy as np
import matplotlib.pyplot as plt

# Parameters
p = 0.3  # geometric distribution parameter (adjust for desired skew)
n_samples = 300  # number of people or trips
max_urgency = 10   # as in the paper

# Generate urgency levels using a truncated geometric distribution
urgency_levels = np.random.geometric(p, n_samples)
urgency_levels = np.clip(urgency_levels, 1, max_urgency)  # limit to 1–10

# Plot the distribution
plt.hist(urgency_levels, bins=np.arange(1, max_urgency+2)-0.5, edgecolor='black')
plt.title('Simulated Urgency Level Distribution (1–10)')
plt.xlabel('Urgency Level')
plt.ylabel('Frequency')
plt.show()

# Example: Show first 20 generated urgency levels
print("Sample urgency levels:", urgency_levels[:20])
