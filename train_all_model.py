import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import copy
import time
current_dir = os.path.dirname(os.path.abspath(__file__))
python_dir = os.path.join(current_dir, 'python')
sys.path.append(python_dir)
from python.main import main 

# ==========================================
#  CONFIGURATION
# ==========================================

# 1. The "Winner" Hyperparameters from your tuning
BEST_PARAMS = {
    'batch_size': 128,
    'lr': 0.001,           # Best LR found
    'dropout_rate': 0.7,   # Best Dropout found
    'hidden_units': 128,
    'maxlen': 50,
    'num_blocks': 2,
    'num_heads': 1,
    'num_epochs': 200,     # Full training duration
    'device': 'mps',      # Change to 'mps' for Mac or 'cpu'
    'l2_emb': 0.0,
    'norm_first': False,
    'inference_only': False,
    'state_dict_path': None
}

# 2. Define the Datasets (Splits & Targets)
# Ensure these folders exist in your 'data/' directory
datasets = [
    # --- ITEMS ---
    {
        'desc': 'split_items', 
        'path': 'data_70_30/clothing_items_train'
    },
    {
        'desc': 'loo_items', 
        'path': 'data_loo/clothing_items_train'
    },
    
    # --- GROUPS (Uncomment if you generated group data) ---
    {
        'desc': 'split_groups', 
        'path': 'data_70_30/clothing_groups_train'
    },
    {
        'desc': 'loo_groups', 
        'path': 'data_loo/clothing_groups_train'
    },
]

# 3. Define the Feature Combinations
feature_sets = [
    {
        'name': 'no_features',
        'visual': False,
        'tags': False
    },
    {
        'name': 'tag_features',
        'visual': False,
        'tags': True
    },
    {
        'name': 'img_embed', # Naming match for your eval script
        'visual': True,
        'tags': False
    },
    {
        'name': 'both_features',
        'visual': True,
        'tags': True
    },
]

# ==========================================
#  ARGS CLASS (Dynamic)
# ==========================================
class Args:
    def __init__(self, dataset_path, train_dir_name, feat_config):
        # Paths
        self.dataset = dataset_path
        self.train_dir = train_dir_name
        
        # Features
        self.use_visual = feat_config['visual']
        self.use_tags = feat_config['tags']
        
        # Hyperparameters (Unpack global dict)
        for k, v in BEST_PARAMS.items():
            setattr(self, k, v)

# ==========================================
#  MAIN LOOP
# ==========================================
def run_all_training():
    total_runs = len(datasets) * len(feature_sets)
    current = 0
    
    print(f"--- Starting Batch Training: {total_runs} Models ---")
    print(f"Hyperparameters: LR={BEST_PARAMS['lr']}, Dropout={BEST_PARAMS['dropout_rate']}\n")

    for ds in datasets:
        for feat in feature_sets:
            current += 1
            
            # Construct standard folder name: e.g., "both_features_split_items"
            # This matches the pattern usually expected by evaluation scripts
            folder_name = f"{feat['name']}_{ds['desc']}"
            
            print(f"[{current}/{total_runs}] Training: {folder_name}")
            print(f"   Dataset: {ds['path']}")
            print(f"   Visual: {feat['visual']} | Tags: {feat['tags']}")
            
            # Create Args
            args = Args(ds['path'], folder_name, feat)
            
            try:
                start_time = time.time()
                
                # RUN TRAINING
                # We ignore the return values (valid scores) here as we rely on the saved .pth file
                main(args)
                
                duration = (time.time() - start_time) / 60
                print(f"   --> Done. Time: {duration:.2f} min\n")
                
            except Exception as e:
                print(f"   --> FAILED: {folder_name}")
                print(f"   Error: {e}\n")

if __name__ == "__main__":
    run_all_training()