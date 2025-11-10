import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

# ---- settings ----
csv_path = Path("agent_delays.csv")  # adjust path if needed
out_png = Path("agent_delay_income_scatter_routes3cmap.png")

# ---- load ----
df = pd.read_csv(csv_path)

# check columns
needed = {"id", "income", "avg_delay", "chosen_route_index"}
missing = needed.difference(df.columns)
if missing:
    raise ValueError(f"CSV missing columns: {sorted(missing)}")

# filter clean data
plot_df = df.dropna(subset=["income", "avg_delay", "chosen_route_index"]).copy()
plot_df["chosen_route_index"] = plot_df["chosen_route_index"].astype(int)

# quick stats
corr = plot_df[["income", "avg_delay"]].corr(method="pearson").iloc[0, 1]
print(f"Points plotted: {len(plot_df)}")
print(f"Pearson correlation (income vs. avg_delay): {corr:.4f}")

# ---- define a colormap with exactly 3 colors ----
cmap = ListedColormap(["#1f77b4", "#ff7f0e", "#2ca02c"])  # blue, orange, green

plt.figure(figsize=(8, 6))
sc = plt.scatter(
    plot_df["income"],
    plot_df["avg_delay"],
    c=plot_df["chosen_route_index"],
    cmap=cmap,
    vmin=0,
    vmax=2,
    alpha=0.85,
    edgecolors="k",
    linewidths=0.3,
)

# axes labels and title
plt.xlabel("Income")
plt.ylabel("Average Delay")
plt.title("Agent Average Delay vs. Income (3 Routes)")

# discrete colorbar ticks
cbar = plt.colorbar(sc, ticks=[0, 1, 2])
cbar.set_label("Chosen Route Index")

plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig(out_png, dpi=150)
plt.show()

print(f"Saved discrete 3-color scatter plot to: {out_png.resolve()}")
