import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Episode loading helpers
# -----------------------
_EP_RE = re.compile(r"^ep(\d+)\.csv$")

def _episodes_dir(p: str) -> str:
    ep = os.path.join(p, "episodes")
    return ep if os.path.isdir(ep) else p

def list_episode_files(folder: str):
    folder = _episodes_dir(folder)
    if not os.path.isdir(folder):
        return []
    out = []
    for fn in os.listdir(folder):
        m = _EP_RE.match(fn)
        if m:
            out.append((int(m.group(1)), os.path.join(folder, fn)))
    out.sort(key=lambda t: t[0])
    return out

def load_episodes(folder: str, n_last: int = 30, episode_range=None, episodes=None):
    folder = _episodes_dir(folder)
    eps = list_episode_files(folder)
    if not eps:
        return None

    eps_map = {ep: path for ep, path in eps}

    if episodes is not None:
        wanted = sorted(set(int(e) for e in episodes))
        chosen = [(ep, eps_map[ep]) for ep in wanted if ep in eps_map]
    elif episode_range is not None:
        start, end = map(int, episode_range)
        chosen = [(ep, path) for ep, path in eps if start <= ep <= end]
    else:
        chosen = eps[-n_last:] if len(eps) >= n_last else eps

    if not chosen:
        return None

    dfs = []
    for ep, path in chosen:
        df = pd.read_csv(path, on_bad_lines="skip")
        df["episode"] = ep
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else None


def plot_income_vs_start_time_by_action_colored_urgency(
    algo_runs,
    scenario,
    episode_range=None,
    episodes=None,
    last_n=30,
    action_col="action",
    income_col="income",
    urgency_col="urgency",
    start_time_col="start_time",
    kind_col="kind",
    plot_kind="AV",
    actions=(0, 1, 2),
    urgency_threshold=0.5,   # NEW
    max_points=None,
    jitter_x=0.0,
    jitter_y=0.0,
    jitter_seed=0,
    alpha=0.5,
    s=20,
    figsize=(16, 5.2),
    sharex=True,
    sharey=True,
    cmap="viridis",
    save_path=None,
    show=True,
):
    """
    Create 3 subplots, one for each action.
    In each subplot:
      x = start_time
      y = income
      color = urgency
      points = individuals whose chosen action equals that subplot action

    Only rows with urgency > urgency_threshold are kept.
    """

    if scenario not in algo_runs:
        raise ValueError(f"Scenario '{scenario}' not found in algo_runs.")

    rng = np.random.default_rng(jitter_seed)
    run_dirs = algo_runs[scenario]

    parts = []
    for run_dir in run_dirs:
        df = load_episodes(run_dir, n_last=last_n, episode_range=episode_range, episodes=episodes)
        if df is None or df.empty:
            continue

        if plot_kind != "All" and kind_col in df.columns:
            df = df[df[kind_col] == plot_kind]
        if df.empty:
            continue

        needed = [action_col, income_col, urgency_col, start_time_col]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            print(f"[WARN] {run_dir}: missing {missing}. Skipping run.")
            continue

        tmp = df[[action_col, income_col, urgency_col, start_time_col]].copy()
        tmp[action_col] = pd.to_numeric(tmp[action_col], errors="coerce")
        tmp[income_col] = pd.to_numeric(tmp[income_col], errors="coerce")
        tmp[urgency_col] = pd.to_numeric(tmp[urgency_col], errors="coerce")
        tmp[start_time_col] = pd.to_numeric(tmp[start_time_col], errors="coerce")
        tmp = tmp.dropna(subset=[action_col, income_col, urgency_col, start_time_col])

        # NEW: keep only higher-urgency individuals
        tmp = tmp[tmp[urgency_col] > urgency_threshold]

        if not tmp.empty:
            parts.append(tmp)

    if not parts:
        raise RuntimeError(
            f"No valid data found for scenario '{scenario}' after filtering urgency > {urgency_threshold}."
        )

    data = pd.concat(parts, ignore_index=True)

    all_urg = data[urgency_col].to_numpy(dtype=float)
    all_urg = all_urg[np.isfinite(all_urg)]
    if len(all_urg) == 0:
        raise RuntimeError("No valid urgency values found.")

    vmin = np.nanmin(all_urg)
    vmax = np.nanmax(all_urg)

    fig, axes = plt.subplots(1, len(actions), figsize=figsize, sharex=sharex, sharey=sharey)
    if len(actions) == 1:
        axes = [axes]

    fig.subplots_adjust(right=0.88, wspace=0.18)
    all_limits = []
    scatter_artist = None

    for ax, act in zip(axes, actions):
        sub = data[data[action_col] == act].copy()

        if sub.empty:
            ax.set_title(f"Action {act} (no data)", fontsize=16)
            ax.grid(True, alpha=0.25)
            continue

        if max_points is not None and len(sub) > max_points:
            sub = sub.sample(n=max_points, random_state=jitter_seed).reset_index(drop=True)

        x = sub[start_time_col].to_numpy(dtype=float)
        y = sub[income_col].to_numpy(dtype=float)
        c = sub[urgency_col].to_numpy(dtype=float)

        if jitter_x > 0:
            x = x + rng.uniform(-jitter_x, jitter_x, size=len(x))
        if jitter_y > 0:
            y = y + rng.uniform(-jitter_y, jitter_y, size=len(y))

        sc = ax.scatter(
            x,
            y,
            c=c,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=s,
            alpha=alpha,
            marker="o",
            edgecolors="none",
        )
        scatter_artist = sc

        ax.set_title(f"Action {act}", fontsize=16)
        ax.grid(True, alpha=0.25)

        all_limits.append((
            np.nanmin(x), np.nanmax(x),
            np.nanmin(y), np.nanmax(y)
        ))

    axes[0].set_ylabel("Income", fontsize=15)
    for ax in axes:
        ax.set_xlabel("Start time", fontsize=15)
        ax.tick_params(axis="both", labelsize=12)

    if all_limits:
        xmin = min(t[0] for t in all_limits)
        xmax = max(t[1] for t in all_limits)
        ymin = min(t[2] for t in all_limits)
        ymax = max(t[3] for t in all_limits)
        for ax in axes:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

    cax = fig.add_axes([0.90, 0.15, 0.02, 0.72])
    cbar = fig.colorbar(scatter_artist, cax=cax)
    cbar.set_label("Urgency", fontsize=15)
    cbar.ax.tick_params(labelsize=12)

    fig.suptitle(
        f"{scenario}",
        fontsize=17
    )
    fig.tight_layout(rect=[0, 0, 0.88, 0.95])

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Episode loading helpers
# -----------------------
_EP_RE = re.compile(r"^ep(\d+)\.csv$")

def _episodes_dir(p: str) -> str:
    ep = os.path.join(p, "episodes")
    return ep if os.path.isdir(ep) else p

def list_episode_files(folder: str):
    folder = _episodes_dir(folder)
    if not os.path.isdir(folder):
        return []
    out = []
    for fn in os.listdir(folder):
        m = _EP_RE.match(fn)
        if m:
            out.append((int(m.group(1)), os.path.join(folder, fn)))
    out.sort(key=lambda t: t[0])
    return out

def load_episodes(folder: str, n_last: int = 30, episode_range=None, episodes=None):
    folder = _episodes_dir(folder)
    eps = list_episode_files(folder)
    if not eps:
        return None

    eps_map = {ep: path for ep, path in eps}

    if episodes is not None:
        wanted = sorted(set(int(e) for e in episodes))
        chosen = [(ep, eps_map[ep]) for ep in wanted if ep in eps_map]
    elif episode_range is not None:
        start, end = map(int, episode_range)
        chosen = [(ep, path) for ep, path in eps if start <= ep <= end]
    else:
        chosen = eps[-n_last:] if len(eps) >= n_last else eps

    if not chosen:
        return None

    dfs = []
    for ep, path in chosen:
        df = pd.read_csv(path, on_bad_lines="skip")
        df["episode"] = ep
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else None


def plot_income_vs_start_time_by_action_colored_urgency(
    algo_runs,
    scenario,
    episode_range=None,
    episodes=None,
    last_n=30,
    action_col="action",
    income_col="income",
    urgency_col="urgency",
    start_time_col="start_time",
    kind_col="kind",
    plot_kind="AV",
    actions=(0, 1, 2),
    urgency_threshold=0.5,   # NEW
    max_points=None,
    jitter_x=0.0,
    jitter_y=0.0,
    jitter_seed=0,
    alpha=0.5,
    s=20,
    figsize=(16, 5.2),
    sharex=True,
    sharey=True,
    cmap="viridis",
    save_path=None,
    show=True,
):
    """
    Create 3 subplots, one for each action.
    In each subplot:
      x = start_time
      y = income
      color = urgency
      points = individuals whose chosen action equals that subplot action

    Only rows with urgency > urgency_threshold are kept.
    """

    if scenario not in algo_runs:
        raise ValueError(f"Scenario '{scenario}' not found in algo_runs.")

    rng = np.random.default_rng(jitter_seed)
    run_dirs = algo_runs[scenario]

    parts = []
    for run_dir in run_dirs:
        df = load_episodes(run_dir, n_last=last_n, episode_range=episode_range, episodes=episodes)
        if df is None or df.empty:
            continue

        if plot_kind != "All" and kind_col in df.columns:
            df = df[df[kind_col] == plot_kind]
        if df.empty:
            continue

        needed = [action_col, income_col, urgency_col, start_time_col]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            print(f"[WARN] {run_dir}: missing {missing}. Skipping run.")
            continue

        tmp = df[[action_col, income_col, urgency_col, start_time_col]].copy()
        tmp[action_col] = pd.to_numeric(tmp[action_col], errors="coerce")
        tmp[income_col] = pd.to_numeric(tmp[income_col], errors="coerce")
        tmp[urgency_col] = pd.to_numeric(tmp[urgency_col], errors="coerce")
        tmp[start_time_col] = pd.to_numeric(tmp[start_time_col], errors="coerce")
        tmp = tmp.dropna(subset=[action_col, income_col, urgency_col, start_time_col])

        # NEW: keep only higher-urgency individuals
        tmp = tmp[tmp[urgency_col] > urgency_threshold]

        if not tmp.empty:
            parts.append(tmp)

    if not parts:
        raise RuntimeError(
            f"No valid data found for scenario '{scenario}' after filtering urgency > {urgency_threshold}."
        )

    data = pd.concat(parts, ignore_index=True)

    all_urg = data[urgency_col].to_numpy(dtype=float)
    all_urg = all_urg[np.isfinite(all_urg)]
    if len(all_urg) == 0:
        raise RuntimeError("No valid urgency values found.")

    vmin = np.nanmin(all_urg)
    vmax = np.nanmax(all_urg)

    fig, axes = plt.subplots(1, len(actions), figsize=figsize, sharex=sharex, sharey=sharey)
    if len(actions) == 1:
        axes = [axes]

    fig.subplots_adjust(right=0.88, wspace=0.18)
    all_limits = []
    scatter_artist = None

    for ax, act in zip(axes, actions):
        sub = data[data[action_col] == act].copy()

        if sub.empty:
            ax.set_title(f"Action {act} (no data)", fontsize=16)
            ax.grid(True, alpha=0.25)
            continue

        if max_points is not None and len(sub) > max_points:
            sub = sub.sample(n=max_points, random_state=jitter_seed).reset_index(drop=True)

        x = sub[start_time_col].to_numpy(dtype=float)
        y = sub[income_col].to_numpy(dtype=float)
        c = sub[urgency_col].to_numpy(dtype=float)

        if jitter_x > 0:
            x = x + rng.uniform(-jitter_x, jitter_x, size=len(x))
        if jitter_y > 0:
            y = y + rng.uniform(-jitter_y, jitter_y, size=len(y))

        sc = ax.scatter(
            x,
            y,
            c=c,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=s,
            alpha=alpha,
            marker="o",
            edgecolors="none",
        )
        scatter_artist = sc

        ax.set_title(f"Action {act}", fontsize=16)
        ax.grid(True, alpha=0.25)

        all_limits.append((
            np.nanmin(x), np.nanmax(x),
            np.nanmin(y), np.nanmax(y)
        ))

    axes[0].set_ylabel("Income", fontsize=15)
    for ax in axes:
        ax.set_xlabel("Start time", fontsize=15)
        ax.tick_params(axis="both", labelsize=12)

    if all_limits:
        xmin = min(t[0] for t in all_limits)
        xmax = max(t[1] for t in all_limits)
        ymin = min(t[2] for t in all_limits)
        ymax = max(t[3] for t in all_limits)
        for ax in axes:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

    cax = fig.add_axes([0.90, 0.15, 0.02, 0.72])
    cbar = fig.colorbar(scatter_artist, cax=cax)
    cbar.set_label("Urgency", fontsize=15)
    cbar.ax.tick_params(labelsize=12)

    fig.suptitle(
        f"{scenario}",
        fontsize=17
    )
    fig.tight_layout(rect=[0, 0, 0.88, 0.95])

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


def plot_income_vs_start_time_by_action_colored_urgency(
    algo_runs,
    scenario,
    episode_range=None,
    episodes=None,
    last_n=30,
    action_col="action",
    income_col="income",
    urgency_col="urgency",
    start_time_col="start_time",
    kind_col="kind",
    plot_kind="AV",
    actions=(0, 1, 2),
    urgency_threshold=None,    # e.g. 0.5 ; None means no filter
    income_threshold=None,     # NEW: e.g. 6000
    income_filter_mode="ge",   # NEW: "ge" or "le"
    max_points=None,
    jitter_x=0.0,
    jitter_y=0.0,
    jitter_seed=0,
    alpha=0.5,
    s=20,
    figsize=(16, 5.2),
    sharex=True,
    sharey=True,
    cmap="viridis",
    save_path=None,
    show=True,
):
    """
    Create 3 subplots, one for each action.
    In each subplot:
      x = start_time
      y = income
      color = urgency

    Optional filters:
      - urgency_threshold: keep only rows with urgency > urgency_threshold
      - income_threshold:
          if income_filter_mode == "ge": keep income >= income_threshold
          if income_filter_mode == "le": keep income <= income_threshold
    """

    if scenario not in algo_runs:
        raise ValueError(f"Scenario '{scenario}' not found in algo_runs.")

    if income_filter_mode not in {"ge", "le"}:
        raise ValueError("income_filter_mode must be 'ge' or 'le'.")

    rng = np.random.default_rng(jitter_seed)
    run_dirs = algo_runs[scenario]

    parts = []
    for run_dir in run_dirs:
        df = load_episodes(run_dir, n_last=last_n, episode_range=episode_range, episodes=episodes)
        if df is None or df.empty:
            continue

        if plot_kind != "All" and kind_col in df.columns:
            df = df[df[kind_col] == plot_kind]
        if df.empty:
            continue

        needed = [action_col, income_col, urgency_col, start_time_col]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            print(f"[WARN] {run_dir}: missing {missing}. Skipping run.")
            continue

        tmp = df[[action_col, income_col, urgency_col, start_time_col]].copy()
        tmp[action_col] = pd.to_numeric(tmp[action_col], errors="coerce")
        tmp[income_col] = pd.to_numeric(tmp[income_col], errors="coerce")
        tmp[urgency_col] = pd.to_numeric(tmp[urgency_col], errors="coerce")
        tmp[start_time_col] = pd.to_numeric(tmp[start_time_col], errors="coerce")
        tmp = tmp.dropna(subset=[action_col, income_col, urgency_col, start_time_col])

        # urgency filter
        if urgency_threshold is not None:
            tmp = tmp[tmp[urgency_col] > urgency_threshold]

        # income filter
        if income_threshold is not None:
            if income_filter_mode == "ge":
                tmp = tmp[tmp[income_col] >= income_threshold]
            else:
                tmp = tmp[tmp[income_col] <= income_threshold]

        if not tmp.empty:
            parts.append(tmp)

    if not parts:
        raise RuntimeError(
            f"No valid data found for scenario '{scenario}' after applying filters."
        )

    data = pd.concat(parts, ignore_index=True)

    all_urg = data[urgency_col].to_numpy(dtype=float)
    all_urg = all_urg[np.isfinite(all_urg)]
    if len(all_urg) == 0:
        raise RuntimeError("No valid urgency values found.")

    vmin = np.nanmin(all_urg)
    vmax = np.nanmax(all_urg)

    fig, axes = plt.subplots(1, len(actions), figsize=figsize, sharex=sharex, sharey=sharey)
    if len(actions) == 1:
        axes = [axes]

    fig.subplots_adjust(right=0.88, wspace=0.18)
    all_limits = []
    scatter_artist = None

    for ax, act in zip(axes, actions):
        sub = data[data[action_col] == act].copy()

        if sub.empty:
            ax.set_title(f"Action {act} (no data)", fontsize=16)
            ax.grid(True, alpha=0.25)
            continue

        if max_points is not None and len(sub) > max_points:
            sub = sub.sample(n=max_points, random_state=jitter_seed).reset_index(drop=True)

        x = sub[start_time_col].to_numpy(dtype=float)
        y = sub[income_col].to_numpy(dtype=float)
        c = sub[urgency_col].to_numpy(dtype=float)

        if jitter_x > 0:
            x = x + rng.uniform(-jitter_x, jitter_x, size=len(x))
        if jitter_y > 0:
            y = y + rng.uniform(-jitter_y, jitter_y, size=len(y))

        sc = ax.scatter(
            x,
            y,
            c=c,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=s,
            alpha=alpha,
            marker="o",
            edgecolors="none",
        )
        scatter_artist = sc

        ax.set_title(f"Action {act}", fontsize=16)
        ax.grid(True, alpha=0.25)

        all_limits.append((
            np.nanmin(x), np.nanmax(x),
            np.nanmin(y), np.nanmax(y)
        ))

    axes[0].set_ylabel("Income", fontsize=15)
    for ax in axes:
        ax.set_xlabel("Start time", fontsize=15)
        ax.tick_params(axis="both", labelsize=12)

    if all_limits:
        xmin = min(t[0] for t in all_limits)
        xmax = max(t[1] for t in all_limits)
        ymin = min(t[2] for t in all_limits)
        ymax = max(t[3] for t in all_limits)
        for ax in axes:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

    cax = fig.add_axes([0.90, 0.15, 0.02, 0.72])
    cbar = fig.colorbar(scatter_artist, cax=cax)
    cbar.set_label("Urgency", fontsize=15)
    cbar.ax.tick_params(labelsize=12)

    title = f"{scenario}"
    filters = []
    if urgency_threshold is not None:
        filters.append(f"urgency > {urgency_threshold}")
    if income_threshold is not None:
        op = ">=" if income_filter_mode == "ge" else "<="
        filters.append(f"income {op} {income_threshold}")
    if filters:
        title += " (" + ", ".join(filters) + ")"

    fig.suptitle(title, fontsize=17)
    fig.tight_layout(rect=[0, 0, 0.88, 0.95])

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig



import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Episode loading helpers
# -----------------------
_EP_RE = re.compile(r"^ep(\d+)\.csv$")

def _episodes_dir(p: str) -> str:
    ep = os.path.join(p, "episodes")
    return ep if os.path.isdir(ep) else p

def list_episode_files(folder: str):
    folder = _episodes_dir(folder)
    if not os.path.isdir(folder):
        return []
    out = []
    for fn in os.listdir(folder):
        m = _EP_RE.match(fn)
        if m:
            out.append((int(m.group(1)), os.path.join(folder, fn)))
    out.sort(key=lambda t: t[0])
    return out

def load_episodes(folder: str, n_last: int = 30, episode_range=None, episodes=None):
    folder = _episodes_dir(folder)
    eps = list_episode_files(folder)
    if not eps:
        return None

    eps_map = {ep: path for ep, path in eps}

    if episodes is not None:
        wanted = sorted(set(int(e) for e in episodes))
        chosen = [(ep, eps_map[ep]) for ep in wanted if ep in eps_map]
    elif episode_range is not None:
        start, end = map(int, episode_range)
        chosen = [(ep, path) for ep, path in eps if start <= ep <= end]
    else:
        chosen = eps[-n_last:] if len(eps) >= n_last else eps

    if not chosen:
        return None

    dfs = []
    for ep, path in chosen:
        df = pd.read_csv(path, on_bad_lines="skip")
        df["episode"] = ep
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else None


def get_filtered_data_for_scenario(
    algo_runs,
    scenario,
    episode_range=None,
    episodes=None,
    last_n=30,
    action_col="action",
    income_col="income",
    urgency_col="urgency",
    start_time_col="start_time",
    kind_col="kind",
    plot_kind="AV",
    urgency_threshold=None,
    income_threshold=None,
    income_filter_mode="ge",
):
    if scenario not in algo_runs:
        raise ValueError(f"Scenario '{scenario}' not found in algo_runs.")

    if income_filter_mode not in {"ge", "le"}:
        raise ValueError("income_filter_mode must be 'ge' or 'le'.")

    run_dirs = algo_runs[scenario]
    parts = []

    for run_dir in run_dirs:
        df = load_episodes(run_dir, n_last=last_n, episode_range=episode_range, episodes=episodes)
        if df is None or df.empty:
            continue

        if plot_kind != "All" and kind_col in df.columns:
            df = df[df[kind_col] == plot_kind]
        if df.empty:
            continue

        needed = [action_col, income_col, urgency_col, start_time_col]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            print(f"[WARN] {run_dir}: missing {missing}. Skipping run.")
            continue

        tmp = df[[action_col, income_col, urgency_col, start_time_col]].copy()
        tmp[action_col] = pd.to_numeric(tmp[action_col], errors="coerce")
        tmp[income_col] = pd.to_numeric(tmp[income_col], errors="coerce")
        tmp[urgency_col] = pd.to_numeric(tmp[urgency_col], errors="coerce")
        tmp[start_time_col] = pd.to_numeric(tmp[start_time_col], errors="coerce")
        tmp = tmp.dropna(subset=[action_col, income_col, urgency_col, start_time_col])

        if urgency_threshold is not None:
            tmp = tmp[tmp[urgency_col] > urgency_threshold]

        if income_threshold is not None:
            if income_filter_mode == "ge":
                tmp = tmp[tmp[income_col] >= income_threshold]
            else:
                tmp = tmp[tmp[income_col] <= income_threshold]

        if not tmp.empty:
            parts.append(tmp)

    if not parts:
        raise RuntimeError(f"No valid data found for scenario '{scenario}' after applying filters.")

    return pd.concat(parts, ignore_index=True)


def plot_scenario_on_axes(
    data,
    axes,
    scenario_title,
    actions=(0, 1, 2),
    action_col="action",
    income_col="income",
    urgency_col="urgency",
    start_time_col="start_time",
    max_points=None,
    jitter_x=0.0,
    jitter_y=0.0,
    jitter_seed=0,
    alpha=0.5,
    s=20,
    cmap="viridis",
    vmin=None,
    vmax=None,
):
    rng = np.random.default_rng(jitter_seed)
    scatter_artist = None
    all_limits = []

    for ax, act in zip(axes, actions):
        sub = data[data[action_col] == act].copy()

        if sub.empty:
            ax.set_title(f"Action {act} (no data)", fontsize=14)
            ax.grid(True, alpha=0.25)
            continue

        if max_points is not None and len(sub) > max_points:
            sub = sub.sample(n=max_points, random_state=jitter_seed).reset_index(drop=True)

        x = sub[start_time_col].to_numpy(dtype=float)
        y = sub[income_col].to_numpy(dtype=float)
        c = sub[urgency_col].to_numpy(dtype=float)

        if jitter_x > 0:
            x = x + rng.uniform(-jitter_x, jitter_x, size=len(x))
        if jitter_y > 0:
            y = y + rng.uniform(-jitter_y, jitter_y, size=len(y))

        sc = ax.scatter(
            x, y,
            c=c,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=s,
            alpha=alpha,
            marker="o",
            edgecolors="none",
        )
        scatter_artist = sc

        ax.set_title(f"{scenario_title} — Action {act}", fontsize=14)
        ax.grid(True, alpha=0.25)

        all_limits.append((np.nanmin(x), np.nanmax(x), np.nanmin(y), np.nanmax(y)))

    return scatter_artist, all_limits


def plot_two_scenarios_two_rows(
    algo_runs,
    scenarios=("monetary pricing", "karma pricing"),
    episode_range=None,
    episodes=None,
    last_n=30,
    action_col="action",
    income_col="income",
    urgency_col="urgency",
    start_time_col="start_time",
    kind_col="kind",
    plot_kind="AV",
    actions=(0, 1, 2),
    urgency_threshold=None,
    income_threshold=None,
    income_filter_mode="ge",
    max_points=None,
    jitter_x=0.0,
    jitter_y=0.0,
    jitter_seed=0,
    alpha=0.5,
    s=20,
    figsize=(16, 10),
    cmap="viridis",
    save_path=None,
    show=True,
):
    if len(scenarios) != 2:
        raise ValueError("Please provide exactly 2 scenarios for the 2-row layout.")

    # Load both datasets first
    scenario_data = {}
    all_urg = []

    for scenario in scenarios:
        data = get_filtered_data_for_scenario(
            algo_runs=algo_runs,
            scenario=scenario,
            episode_range=episode_range,
            episodes=episodes,
            last_n=last_n,
            action_col=action_col,
            income_col=income_col,
            urgency_col=urgency_col,
            start_time_col=start_time_col,
            kind_col=kind_col,
            plot_kind=plot_kind,
            urgency_threshold=urgency_threshold,
            income_threshold=income_threshold,
            income_filter_mode=income_filter_mode,
        )
        scenario_data[scenario] = data
        urg = data[urgency_col].to_numpy(dtype=float)
        urg = urg[np.isfinite(urg)]
        all_urg.append(urg)

    all_urg = np.concatenate(all_urg)
    vmin = np.nanmin(all_urg)
    vmax = np.nanmax(all_urg)

    fig, axes = plt.subplots(
        2, len(actions),
        figsize=figsize,
        sharex=True,
        sharey=True
    )

    # If len(actions)==1, make axes 2D-compatible
    if len(actions) == 1:
        axes = np.array(axes).reshape(2, 1)

    row_limits = []
    scatter_artist = None

    for row_idx, scenario in enumerate(scenarios):
        sc, limits = plot_scenario_on_axes(
            data=scenario_data[scenario],
            axes=axes[row_idx],
            scenario_title=scenario,
            actions=actions,
            action_col=action_col,
            income_col=income_col,
            urgency_col=urgency_col,
            start_time_col=start_time_col,
            max_points=max_points,
            jitter_x=jitter_x,
            jitter_y=jitter_y,
            jitter_seed=jitter_seed,
            alpha=alpha,
            s=s,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        if sc is not None:
            scatter_artist = sc
        row_limits.extend(limits)

    # Shared labels
    for r in range(2):
        axes[r, 0].set_ylabel("Income", fontsize=14)
        for c in range(len(actions)):
            axes[r, c].set_xlabel("Start time", fontsize=14)
            axes[r, c].tick_params(axis="both", labelsize=11)

    # Shared limits across all panels
    if row_limits:
        xmin = min(t[0] for t in row_limits)
        xmax = max(t[1] for t in row_limits)
        ymin = min(t[2] for t in row_limits)
        ymax = max(t[3] for t in row_limits)

        for r in range(2):
            for c in range(len(actions)):
                axes[r, c].set_xlim(xmin, xmax)
                axes[r, c].set_ylim(ymin, ymax)

    # Colorbar
    fig.subplots_adjust(right=0.88, hspace=0.35, wspace=0.18)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.72])
    cbar = fig.colorbar(scatter_artist, cax=cax)
    cbar.set_label("Urgency", fontsize=14)
    cbar.ax.tick_params(labelsize=11)

    title = "Income vs start time by action"
    filters = []
    if urgency_threshold is not None:
        filters.append(f"urgency > {urgency_threshold}")
    if income_threshold is not None:
        op = ">=" if income_filter_mode == "ge" else "<="
        filters.append(f"income {op} {income_threshold}")
    if filters:
        title += " (" + ", ".join(filters) + ")"

    fig.suptitle(title, fontsize=17)
    fig.tight_layout(rect=[0, 0, 0.88, 0.95])

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    return fig

ALGO_RUNS = {   


    "monetary pricing":[
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training/episodes",
    ],

    "karma pricing":[
        r"training_records_hyperippo_masked_300_agents_seed_18_karma_pricing_minimum_price_4_variable_start_times_smaller/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_19_karma_pricing_minimum_price_4_variable_start_times_smaller/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_20_karma_pricing_minimum_price_4_variable_start_times_smaller/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_21_karma_pricing_minimum_price_4_variable_start_times_smaller/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_22_karma_pricing_minimum_price_4_variable_start_times_smaller/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_23_karma_pricing_minimum_price_4_variable_start_times_smaller/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_24_karma_pricing_minimum_price_4_variable_start_times_smaller/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_25_karma_pricing_minimum_price_4_variable_start_times_smaller/episodes",
    ],


    }

"""plot_income_vs_start_time_by_action_colored_urgency(
    ALGO_RUNS,
    scenario="monetary pricing",
    episode_range=(1315, 1320),
    action_col="action",
    income_col="income",
    urgency_col="urgency",
    start_time_col="start_time",
    plot_kind="AV",
    actions=(0, 1, 2),
    urgency_threshold=0,
    jitter_x=0.02,
    jitter_y=20,
    alpha=0.55,
    s=40,
    save_path="imgs/karma_income_vs_start_time_by_action_urgency_gt_05.png",
    show=True,
)


plot_income_vs_start_time_by_action_colored_urgency(
    ALGO_RUNS,
    scenario="monetary pricing",
    episode_range=(1319, 1320),
    action_col="action",
    income_col="income",
    urgency_col="urgency",
    start_time_col="start_time",
    plot_kind="AV",
    actions=(0, 1, 2),
    urgency_threshold=0,
    income_threshold=6000,
    income_filter_mode="ge",
    jitter_x=0.02,
    jitter_y=20,
    alpha=0.55,
    s=40,
    save_path="imgs/karma_low_income_high_urgency.png",
    show=True,
)"""


plot_two_scenarios_two_rows(
    ALGO_RUNS,
    scenarios=("monetary pricing", "karma pricing"),
    episode_range=(1320, 1320),
    action_col="action",
    income_col="income",
    urgency_col="urgency",
    start_time_col="start_time",
    plot_kind="AV",
    actions=(0, 1, 2),
    urgency_threshold=0,
    income_threshold=0,
    income_filter_mode="ge",
    jitter_x=0.02,
    jitter_y=20,
    alpha=0.55,
    s=40,
    figsize=(16, 10),
    save_path="imgs/monetary_top_karma_bottom.png",
    show=True,
)