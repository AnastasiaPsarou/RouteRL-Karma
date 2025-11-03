import re
import glob
from pathlib import Path
import pandas as pd

def compute_agent_delays(
    episodes_dir: str,
    free_flow_csv: str,
    last_n: int = 10,
    output_csv: str | None = None,
    route_order: str = "appearance",  # or "alphabetical"
):
    """
    Compute per-agent average delay over the last N episodes and include the route
    chosen during testing (last N episodes).

    Output columns include:
      - id
      - episodes_used
      - records_used
      - avg_travel_time
      - free_flow_time
      - avg_delay
      - chosen_route_index
      - chosen_route_path
    """

    episodes_dir = Path(episodes_dir)

    # --- 1) Load free-flow times and build mappings
    df_ff = pd.read_csv(free_flow_csv)
    df_ff = df_ff.dropna(how="all")

    df_ff = df_ff.rename(columns={
        "origins": "origin",
        "destinations": "destination",
        "free_flow_time": "ff_time"
    })
    df_ff = df_ff.dropna(subset=["origin", "destination", "ff_time"])
    df_ff["origin"] = df_ff["origin"].astype(int)
    df_ff["destination"] = df_ff["destination"].astype(int)

    if route_order == "alphabetical":
        df_ff = df_ff.sort_values(["origin", "destination", "path"], kind="stable")
    elif route_order == "appearance":
        pass
    else:
        raise ValueError("route_order must be 'appearance' or 'alphabetical'")

    # Assign 0-based route indices per OD
    df_ff["route_index"] = df_ff.groupby(["origin", "destination"]).cumcount()

    # Build maps
    ff_time_map = df_ff.set_index(["origin", "destination", "route_index"])["ff_time"].to_dict()
    path_map    = df_ff.set_index(["origin", "destination", "route_index"])["path"].to_dict()

    # --- 2) Find and sort episode files
    episode_files = []
    for p in glob.glob(str(episodes_dir / "ep*.csv")):
        m = re.search(r"ep(\d+)\.csv$", Path(p).name)
        if m:
            ep = int(m.group(1))
            episode_files.append((ep, Path(p)))

    if not episode_files:
        raise FileNotFoundError(f"No episode CSVs matching 'ep{{number}}.csv' found in {episodes_dir}")

    episode_files.sort(key=lambda x: x[0])
    selected = episode_files[-last_n:]

    # --- 3) Load selected episodes
    dfs = []
    for ep_num, path in selected:
        df = pd.read_csv(path, on_bad_lines="skip")
        df["episode"] = ep_num
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.dropna(how="all")

    needed_cols = ["travel_time", "id", "action", "origin", "destination"]
    missing = [c for c in needed_cols if c not in df_all.columns]
    if missing:
        raise ValueError(f"Missing required columns in episode CSVs: {missing}")

    # Normalize dtypes
    for c in ["id", "action", "origin", "destination"]:
        df_all[c] = pd.to_numeric(df_all[c], errors="coerce").astype("Int64")
    df_all["travel_time"] = pd.to_numeric(df_all["travel_time"], errors="coerce")

    df_all = df_all.dropna(subset=["id", "action", "origin", "destination", "travel_time"])
    df_all["origin"] = df_all["origin"].astype(int)
    df_all["destination"] = df_all["destination"].astype(int)
    df_all["action"] = df_all["action"].astype(int)
    df_all["id"] = df_all["id"].astype(int)

    # --- 4) Attach free-flow time and route path
    df_all["ff_time"] = df_all.apply(
        lambda r: ff_time_map.get((r["origin"], r["destination"], r["action"])),
        axis=1
    )
    df_all["route_path"] = df_all.apply(
        lambda r: path_map.get((r["origin"], r["destination"], r["action"])),
        axis=1
    )

    unmatched = df_all["ff_time"].isna().sum()
    if unmatched > 0:
        print(f"Warning: {unmatched} rows had no free-flow match for (origin,destination,action). "
              f"Check route ordering or the free-flow file. Unmatched rows will be dropped.")

    df_all = df_all.dropna(subset=["ff_time"])

    # --- 5) Compute per-record delay
    df_all["delay"] = df_all["travel_time"] - df_all["ff_time"]

    # --- 6) Determine chosen route (mode of `action`)
    def mode_with_tiebreak(series):
        vc = series.value_counts()
        if vc.empty:
            return None
        max_count = vc.iloc[0]
        candidates = vc[vc == max_count].index.tolist()
        return min(candidates)

    chosen_idx = (
        df_all.groupby("id")["action"]
        .apply(mode_with_tiebreak)
        .rename("chosen_route_index")
        .reset_index()
    )

    def chosen_path_for_agent(sub):
        idx = sub["chosen_route_index"].iloc[0]
        paths = df_all.loc[(df_all["id"] == sub["id"].iloc[0]) & (df_all["action"] == idx), "route_path"]
        if paths.empty:
            return None
        vc = paths.value_counts()
        top_count = vc.iloc[0]
        candidates = vc[vc == top_count].index.tolist()
        return sorted(candidates)[0]

    chosen_paths = (
        chosen_idx
        .merge(df_all[["id"]].drop_duplicates(), on="id", how="left")
        .groupby("id", as_index=False)
        .apply(chosen_path_for_agent)
        .rename(columns={None: "chosen_route_path"})
    )

    if "id" not in chosen_paths.columns or "chosen_route_path" not in chosen_paths.columns:
        chosen_paths = chosen_paths.reset_index(names=["id"]).rename(columns={0: "chosen_route_path"})

    chosen = chosen_idx.merge(chosen_paths, on="id", how="left")

    # --- 7) Aggregate per agent
    agg = (
        df_all.groupby("id", as_index=False)
        .agg(
            episodes_used=("episode", "nunique"),
            records_used=("id", "size"),
            avg_travel_time=("travel_time", "mean"),
            free_flow_time=("ff_time", "mean"),   # <-- renamed here
            avg_delay=("delay", "mean"),
        )
        .sort_values("id")
    )

    # Merge chosen route info
    agg = agg.merge(chosen, on="id", how="left")

    # --- 8) Output
    for c in ["avg_travel_time", "free_flow_time", "avg_delay"]:
        agg[c] = agg[c].astype(float)

    pd.set_option("display.float_format", lambda x: f"{x:.6f}")
    print("\nPer-agent average delay over last {} episodes (with chosen route):".format(len(selected)))
    print(agg.to_string(index=False))

    # --- 9) Save
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        agg.to_csv(output_csv, index=False)
        print(f"\nSaved results to: {output_csv}")

    return agg

# ---------------------------
# Example usage:
# ---------------------------
results = compute_agent_delays(
    episodes_dir="../scenarios/monetary_pricing/training_records_monetary_pricing_10_agents/episodes",
    free_flow_csv="../scenarios/monetary_pricing/training_records_monetary_pricing_10_agents/paths.csv",
    last_n=10,
    output_csv="agent_delays.csv",
    route_order="appearance",  # or "alphabetical" if your route indexing uses sorted path names
)
