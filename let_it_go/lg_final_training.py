import os
import sys
import subprocess
import time

# ==========================================
#  CONFIGURATION
# ==========================================
# 1. Hyperparameters
HIDDEN_UNITS = 128
DEVICE = 'mps'         # 'mps' for Mac
DELTA_MAX = 0.3        # Your optimal delta
DROPOUT = 0.7          # Best dropout
LR = 0.001             # Best LR
EPOCHS = 1000           

# 2. Datasets
DATASETS = [
    "data_70_30/clothing_items_train",
    "data_70_30/clothing_groups_train",
    "data_loo/clothing_items_train",
    "data_loo/clothing_groups_train"
]

def run_training():
    # Setup Paths
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(current_script_path)
    
    # Locate main_lg.py
    main_script_path = os.path.join(project_root, "let_it_go", "main_lg.py")
    if not os.path.exists(main_script_path):
        main_script_path = os.path.join(project_root, "main_lg.py")
        
    if not os.path.exists(main_script_path):
        print(f"Error: Could not find main_lg.py at {main_script_path}")
        return

    print(f"--- Starting Training Job (Safe Mode) ---")
    print(f"Model: Let It Go (Content + Delta)")
    print(f"Hidden Units: {HIDDEN_UNITS} | Device: {DEVICE}")
    print(f"Delta Constraint: {DELTA_MAX}\n")

    for i, ds in enumerate(DATASETS):
        dataset_name = os.path.basename(ds)
        
        # --- FIX: GENERATE UNIQUE FOLDER NAMES ---
        # Detect the split type from the path
        if "70_30" in ds:
            split_suffix = "70_30"
        elif "loo" in ds:
            split_suffix = "loo"
        else:
            split_suffix = "generic"
            
        # Append split to directory name to prevent overwriting
        # Example result: both_features_delta_0.3_70_30
        train_dir_name = f"both_features_delta_{DELTA_MAX}_{split_suffix}"
        
        print(f"[{i+1}/{len(DATASETS)}] Training on: {dataset_name} ({split_suffix})")
        print(f"      Unique ID: {train_dir_name}")
        
        # Construct the command
        # Note: We use lowercase strings for booleans to be safe
        cmd = [
            sys.executable, main_script_path,
            f"--dataset={ds}",
            f"--train_dir={train_dir_name}",
            f"--batch_size=128",
            f"--lr={LR}",
            f"--maxlen=50",
            f"--hidden_units={HIDDEN_UNITS}",
            f"--num_blocks=2",
            f"--num_epochs={EPOCHS}",
            f"--num_heads=1",
            f"--dropout_rate={DROPOUT}",
            f"--l2_emb=0.0",
            f"--device={DEVICE}",
            f"--delta_max={DELTA_MAX}" 
        ]

        try:
            start_time = time.time()
            subprocess.run(cmd, check=True)
            duration = (time.time() - start_time) / 60
            print(f"      --> Done. Duration: {duration:.2f} min\n")

        except subprocess.CalledProcessError as e:
            print(f"      --> FAILED. Error code: {e.returncode}\n")
        except Exception as e:
            print(f"      --> ERROR: {e}\n")

if __name__ == "__main__":
    run_training()