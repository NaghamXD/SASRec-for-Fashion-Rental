import os
import sys
import subprocess
import pandas as pd
import re
from datetime import datetime

# ==========================================
# 1. CONFIGURATION
# ==========================================
DELTAS = [0.3, 0.5, 0.8]

# Relative paths from the Project Root
DATASETS = [
    "data_70_30/clothing_items_train",
    "data_70_30/clothing_groups_train",
    "data_loo/clothing_items_train",
    "data_loo/clothing_groups_train"
]

def run_experiments():
    # 1. Determine Paths
    # Current script is in: .../Project/let_it_go/train_lg.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Project root is: .../Project/
    project_root = os.path.dirname(current_dir)

    for d in DELTAS:
        for ds in DATASETS:
            # Matches main_lg.py logic for folder naming
            train_dir_suffix = f"both_features_delta_{d}"
            
            print(f"\n>>> Running: {os.path.basename(ds)} | Delta: {d}")
            
            # 2. Construct Command
            # We run 'python let_it_go/main_lg.py' from the PROJECT ROOT
            # This ensures 'data/' paths in utils_lg.py work correctly.
            cmd = [
                sys.executable, "let_it_go/main_lg.py",
                f"--dataset={ds}",
                f"--train_dir={train_dir_suffix}",
                f"--delta_max={d}",
                "--use_visual=true",
                "--use_tags=true",
                "--device=mps", 
                "--num_epochs=200"
            ]
            
            # 3. Execute from Project Root
            subprocess.run(cmd, cwd=project_root)

def aggregate_results():
    results = []
    
    # We look for results in the Project Root (since we ran from there)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    print("\n>>> Aggregating Results from Project Root...")

    for d in DELTAS:
        for ds in DATASETS:
            dataset_basename = os.path.basename(ds)
            train_dir_suffix = f"both_features_delta_{d}"
            
            # The folder name created by main_lg.py is: {Basename}_{TrainDir}
            # e.g., clothing_items_train_both_features_delta_0.1
            actual_folder_name = f"{dataset_basename}_{train_dir_suffix}"
            
            # Look for this folder in the Project Root
            log_path = os.path.join(project_root, actual_folder_name, "log.txt")
            
            if os.path.exists(log_path):
                try:
                    with open(log_path, 'r') as f:
                        lines = f.readlines()
                        # Last line contains final metrics
                        if len(lines) > 1:
                            last_log = lines[-1].strip()
                            metrics = re.findall(r"(\d+\.\d+)", last_log)
                            
                            if len(metrics) >= 4:
                                results.append({
                                    "Dataset": dataset_basename,
                                    "Delta": d,
                                    "Val_NDCG": float(metrics[0]),
                                    "Val_HR": float(metrics[1]),
                                    "Test_NDCG": float(metrics[2]),
                                    "Test_HR": float(metrics[3])
                                })
                                print(f"   Loaded: {actual_folder_name}")
                except Exception as e:
                    print(f"   Error parsing {log_path}: {e}")
            else:
                print(f"   MISSING: {log_path}")

    # Save to CSV in let_it_go folder
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by=["Dataset", "Test_HR"], ascending=[True, False])
        output_path = os.path.join(current_dir, "delta_experiment_summary.csv")
        df.to_csv(output_path, index=False)
        print(f"\n=== SUMMARY SAVED TO: {output_path} ===")
        print(df.to_string(index=False))
    else:
        print("\nNo results found. Did the training run successfully?")

if __name__ == "__main__":
    run_experiments() 
    aggregate_results()