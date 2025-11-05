# sample_from_distribution.py
import numpy as np
import pandas as pd

NPZ_PATH = "income_distribution_arrays.npz"   # the file you saved earlier
N_SAMPLES = 800
OUT_CSV   = "income_samples.csv"

# ---------- Load arrays ----------
dat = np.load(NPZ_PATH, allow_pickle=True)
d_fine  = dat["d_fine"]               # log-income grid
Y_plot  = dat["Y_plot"]               # shape (n_points, n_regions); integrates to 1 over d when summed
regions = dat["regions"].tolist()     # list of region names

# ---------- Build total PDF over d and its CDF ----------
# Total density over d (sum across regions); clip + renormalize for safety
pdf_d = np.sum(np.clip(Y_plot, 0.0, None), axis=1)
# Renormalize to integrate to 1 over d
area = np.trapz(pdf_d, d_fine)
if area <= 0:
    raise ValueError("Total area is non-positive. Check inputs.")
pdf_d /= area

# CDF on the (uniform-in-d) grid using trapezoids
step = d_fine[1] - d_fine[0]
cdf = np.zeros_like(pdf_d)
cdf[1:] = np.cumsum(0.5 * (pdf_d[:-1] + pdf_d[1:]) * step)
# Normalize + enforce endpoints
cdf /= cdf[-1]
cdf[0] = 0.0
cdf[-1] = 1.0

# ---------- Sampling helpers ----------
rng = np.random.default_rng(42)  # set seed for reproducibility; change/remove as needed

def sample_income(n):
    """Sample incomes (Euro PPP per adult per month), ignoring region."""
    u = rng.random(n)
    d_samp = np.interp(u, cdf, d_fine)    # inverse CDF on log-income
    return np.exp(d_samp)                 # back to income

def sample_joint(n):
    """
    Sample (income, region) pairs:
      1) sample d from total density
      2) sample region ~ conditional shares at that d
    """
    u = rng.random(n)
    d_samp = np.interp(u, cdf, d_fine)
    x_samp = np.exp(d_samp)

    # Interpolate each region's density at sampled d
    # (Y_plot[:, j] is region j density over d; not normalized conditionally)
    weights = np.column_stack([
        np.interp(d_samp, d_fine, np.clip(Y_plot[:, j], 0.0, None), left=0.0, right=0.0)
        for j in range(Y_plot.shape[1])
    ])
    row_sums = weights.sum(axis=1)
    row_sums = np.where(row_sums <= 0, 1.0, row_sums)  # guard
    probs = weights / row_sums[:, None]

    # categorical draw from probs per row
    cum = np.cumsum(probs, axis=1)
    ru = rng.random(n)[:, None]
    idx = (ru > cum).sum(axis=1)

    reg = [regions[i] for i in idx]
    return x_samp, reg

# ---------- Do it ----------
# incomes only
incomes = sample_income(N_SAMPLES)

# or joint:
# incomes, regs = sample_joint(N_SAMPLES)

# Save to CSV
pd.DataFrame({"income": incomes}).to_csv(OUT_CSV, index=False)
print(f"Sampled {N_SAMPLES} incomes → {OUT_CSV}")
print("First 10 samples:", np.array2string(incomes[:10], precision=2))
