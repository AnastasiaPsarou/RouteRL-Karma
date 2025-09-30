import subprocess
import os
from pathlib import Path
import shutil
import time

CONFIG_FILE = "sumo_sim_2024_11_10_agents.sumocfg"
BASE_OUTPUT_DIR = Path("multi_seed_runs")
print(BASE_OUTPUT_DIR)
#SUMO_BINARY = r"C:/Users/mbertola/AppData/Local/sumo-1.22.0/bin/sumo.exe"
SUMO_BINARY = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"
#SUMO_BINARY = "sumo"
SEEDS = [111, 222, 333, 444, 555, 666, 777, 888]

# Detector output folders to capture after each run
DETECTOR_FOLDERS = ["A2_detectors", "loop_detector_outputs"]

# Store per-run durations
run_durations = {}

# Total timer
total_start = time.time()

def format_duration(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


for seed in SEEDS:
    print(f"\n▶️ Running simulation with seed {seed}")
    run_start = time.time()
    
    run_dir = BASE_OUTPUT_DIR / f"seed_{seed}"
    print('created folder ', run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Run SUMO
    tripinfo_path = run_dir / "tripinfo.xml"
    subprocess.run([
        SUMO_BINARY, "-c", CONFIG_FILE,
        "--seed", str(seed),
        "--tripinfo-output", str(tripinfo_path),
        "--no-warnings"
    ])

    # Copy loop detector outputs
    for folder_name in DETECTOR_FOLDERS:
        src_folder = Path(folder_name)
        if src_folder.exists():
            dest_folder = run_dir / folder_name
            dest_folder.mkdir(parents=True, exist_ok=True)
            for file in src_folder.glob("*"):
                shutil.copy(file, dest_folder)
                #print(f"  ✅ Copied {file.name} from {folder_name} to {dest_folder.name}")
        else:
            print(f"  ⚠️ Folder {folder_name} does not exist. Skipped.")

    run_duration = time.time() - run_start
    run_durations[seed] = run_duration
    print(f"⏱️  Finished seed {seed} in {format_duration(run_duration)} (hh:mm:ss)")

total_duration = time.time() - total_start

# Final report
print("\n📊 Run duration summary:")
for seed, duration in run_durations.items():
    print(f"⏱️  Finished seed {seed} in {format_duration(run_duration)} (hh:mm:ss)")

print(f"\n⏱️ Total runtime: {format_duration(total_duration)} (hh:mm:ss)")
