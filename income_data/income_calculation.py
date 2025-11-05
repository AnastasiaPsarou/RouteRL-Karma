# ------------------------------------------------------------
# Global income distribution (monthly), stacked by region
# - Smooth & area-preserving on log-income axis (dincm)
# - Excel-style domain padding on both sides
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

CSV_PATH = "income_data.csv"

# -----------------------------
# Helpers
# -----------------------------
def pct_to_float(x):
    if pd.isna(x):
        return 0.0
    s = str(x).strip()
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except:
            return 0.0
    try:
        return float(s)
    except:
        return 0.0


def gaussian_kernel_units(sigma_units, step_units, radius_units=None):
    if sigma_units <= 0:
        return np.array([1.0])
    sigma_samples = max(1e-9, sigma_units / step_units)
    if radius_units is None:
        radius_samples = int(np.ceil(3 * sigma_samples))
    else:
        radius_samples = int(np.ceil(radius_units / step_units))
    t = np.arange(-radius_samples, radius_samples + 1, dtype=float)
    k = np.exp(-0.5 * (t / sigma_samples) ** 2)
    k /= k.sum()
    return k


def convolve_reflect(y, k):
    if len(k) == 1:
        return y.copy()
    r = (len(k) - 1) // 2
    y_ext = np.r_[y[r:0:-1], y, y[-2:-r-2:-1]]
    z = np.convolve(y_ext, k, mode="valid")
    return z


# -----------------------------
# 1) Load & clean
# -----------------------------
df = pd.read_csv(CSV_PATH, on_bad_lines="skip")
df.columns = df.columns.str.strip()

df = df[
    pd.to_numeric(df.get("dincm", np.nan), errors="coerce").notna()
    & pd.to_numeric(df.get("expdincm", np.nan), errors="coerce").notna()
].copy()
df["dincm"] = df["dincm"].astype(float)
df["expdincm"] = df["expdincm"].astype(float)

regions = [
    "North America", "Europe", "East Asia", "Russia & Central Asia",
    "MENA", "Latin America", "South & South-East Asia", "Sub-Saharan Africa"
]

for c in regions:
    df[c] = df[c].apply(pct_to_float)


# -----------------------------
# 2) Average duplicates on the log-income grid dincm
# -----------------------------
g = (
    df.groupby("dincm", as_index=False)[regions + ["expdincm"]]
      .mean()
      .sort_values("dincm")
      .reset_index(drop=True)
)

row_sum = g[regions].sum(axis=1).to_numpy()
if (row_sum > 0).any():
    first_nz = int(np.argmax(row_sum > 0))
    g = g.iloc[first_nz:].copy()

d = g["dincm"].to_numpy()
Y = g[regions].to_numpy()


# -----------------------------
# 3) Interpolate to fine log grid & smooth
# -----------------------------
# --- Build x_fine with exact endpoints and Excel-style padding ---
x_min_target = 1.0
x_max_target = 35000.0

# Excel-like small padding (~5% of domain on both sides)
pad_left  = 0.05 * x_min_target     # since lower bound is small, this adds ~0.05
pad_right = 0.05 * x_max_target     # ~1750 extra on right

x_min = x_min_target - pad_left
x_max = x_max_target + pad_right

# Build geometric grid so endpoints are exact
x_fine = np.geomspace(x_min, x_max, 1000)
d_fine = np.log(x_fine)

# Interpolate and smooth
Y_fine = np.empty((len(d_fine), Y.shape[1]))
for j in range(Y.shape[1]):
    Y_fine[:, j] = np.interp(d_fine, d, Y[:, j], left=0.0, right=0.0)

step_d = d_fine[1] - d_fine[0]
sigma_d = 0.10
k = gaussian_kernel_units(sigma_units=sigma_d, step_units=step_d)

Y_smooth = np.empty_like(Y_fine)
for j in range(Y_fine.shape[1]):
    y = np.maximum(Y_fine[:, j], 0.0)
    ys = convolve_reflect(y, k)
    Y_smooth[:, j] = np.clip(ys, 0.0, None)


# -----------------------------
# 4) Normalize (areas = population shares)
# -----------------------------
areas = np.trapz(Y_smooth, d_fine, axis=0)
total_area = areas.sum()
if total_area == 0:
    raise ValueError("All values are zero after smoothing; cannot normalize.")
Y_plot = Y_smooth / total_area

np.savez("income_distribution_arrays.npz",
         d_fine=d_fine, Y_plot=Y_plot, regions=regions)


# -----------------------------
# 5) Plot with Excel-style padding
# -----------------------------
ticks_high = [1, 5, 30, 50, 85, 150, 250, 450, 750, 1300,
               2500, 4000, 7000, 12000, 20000, 35000]
ticks = [t for t in ticks_high if x_min <= t <= x_max_target]

fig, ax = plt.subplots(figsize=(12, 6))
colors = ["#FFE800","#F4C300","#FF1F1F","#7E57C2",
          "#1F77B4","#2BC2F0","#7AC943","#9CD67C"]

ax.stackplot(x_fine, Y_plot.T, labels=regions, colors=colors)

ax.set_xscale("log")

# Excel-style padding (let plot breathe a little beyond data)
ax.set_xlim(x_min, x_max)
ax.margins(x=0.02)   # 2% extra visual padding, similar to Excel chart behavior

ax.xaxis.set_major_locator(FixedLocator(ticks))
ax.xaxis.set_minor_locator(FixedLocator([]))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:.0f}%"))

ax.set_title("Global income distribution (smooth; areas = regional population shares)", weight="bold")
ax.set_xlabel("Per adult monthly income (Euro PPP), log axis")
ax.set_ylabel("Density (∫ over log-income = 100%)")
ax.grid(True, axis="y", alpha=0.25)
ax.legend(ncol=2, frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig("global_income_distribution_smooth_excel_style.png", dpi=200)
plt.show()
