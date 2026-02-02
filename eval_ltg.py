import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
import torch
import pickle
import numpy as np
import pandas as pd
import csv
import copy
import argparse
from datetime import datetime

# --- CHANGE 1: Import from the Let It Go submodule ---
# We add the submodule to path and import the specific SASRec class
sys.path.append(os.path.join(os.path.dirname(__file__), 'let_it_go'))
from let_it_go.model_lg import SASRec

# ==========================================
#  HELPER CLASSES & FUNCTIONS (KEEP THESE)
# ==========================================
def load_group_mapping(dataset_folder):
    """
    Loads mapping from Item_ID (int) -> Group_ID (str)
    to support Group-Based Discovery metrics.
    """
    # 1. Check for outfits.csv in typical locations
    csv_path = 'outfits.csv'
    if not os.path.exists(csv_path):
        # Try one level up if running from subfolder
        csv_path = '../outfits.csv'
    
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Group discovery metric will default to Item discovery.")
        return {}

    try:
        outfits = pd.read_csv(csv_path, sep=';')
        
        # 2. Load the Item Map (Int -> Str)
        map_path = os.path.join(dataset_folder, 'item_maps.pkl')
        if not os.path.exists(map_path):
            print(f"Warning: {map_path} not found.")
            return {}
            
        with open(map_path, 'rb') as f:
            user_map, item_map = pickle.load(f)

        # 3. Create Map: {Integer_ID: Group_ID_String}
        inv_item_map = {v: k for k, v in item_map.items()} # {1: 'ItemA_Str'}
        raw_item_to_group = dict(zip(outfits['id'], outfits['group'])) # {'ItemA_Str': 'GroupA'}
        
        item_int_to_group = {}
        for i_int, s_id in inv_item_map.items():
            if s_id in raw_item_to_group:
                item_int_to_group[i_int] = raw_item_to_group[s_id]
                
        print(f"--> [Info] Loaded Group Map with {len(item_int_to_group)} items.")
        return item_int_to_group
        
    except Exception as e:
        print(f"Warning: Failed to load group map: {e}")
        return {}
    
class AvailabilityMask:
    def __init__(self, orders_path, triplets_path, item_map, user_map):
        print("Building Availability Index...")
        self.item_map = item_map
        self.user_map = user_map
        # Assuming files are in current directory or specific path
        if not os.path.exists(orders_path):
            print(f"Warning: {orders_path} not found. Availability Mask might fail.")
            self.unavailable = {}
            return

        df1 = pd.read_csv(orders_path, sep=';')
        df2 = pd.read_csv(triplets_path, sep=';')
        cols = ['customer.id', 'outfit.id', 'rentalPeriod.start', 'rentalPeriod.end']
        full_df = pd.concat([df1[cols], df2[cols]])
        full_df['start'] = pd.to_datetime(full_df['rentalPeriod.start'])
        full_df['end'] = pd.to_datetime(full_df['rentalPeriod.end'])
        
        # Filter valid items
        full_df = full_df[full_df['outfit.id'].isin(self.item_map)]
        full_df['item_int'] = full_df['outfit.id'].map(self.item_map)
        full_df = full_df[full_df['customer.id'].isin(self.user_map)]
        full_df['user_int'] = full_df['customer.id'].map(self.user_map)
        
        self.date_index = {} 
        full_df = full_df.dropna(subset=['item_int', 'user_int', 'start', 'end'])
        for row in full_df.itertuples():
            current = row.start
            while current <= row.end:
                date_key = current.date() 
                if date_key not in self.date_index: self.date_index[date_key] = {}
                if int(row.item_int) not in self.date_index[date_key]: self.date_index[date_key][int(row.item_int)] = set()
                self.date_index[date_key][int(row.item_int)].add(int(row.user_int))
                current += pd.Timedelta(days=1)

    def get_unavailable_items(self, query_date, current_user_id):
        date_key = query_date.date()
        unavailable = []
        if date_key in self.date_index:
            for item_id, renters in self.date_index[date_key].items():
                if current_user_id not in renters:
                    unavailable.append(item_id)
        return unavailable

# --- YOUR EVALUATION LOGIC FUNCTIONS ---
def evaluate_static_logic(model, test_dict, train_seqs, history_dict, args, item_to_group_map={}):
    """Calculates metrics using fixed sequences (Set Recall)."""
    HR_10_all, HR_100_all = [], []
    HR_10_new, HR_100_new = [], []

    # Optimization: Move Embeddings to Device ONCE
    # (Assuming model.get_item_vector exists in 'let_it_go' model, or use embedding layer)
    # Since we imported from let_it_go, we likely need to handle embeddings carefully.
    # For safety with this specific script, we'll assume standard lookup unless specific method exists.
    
    for u, test_items in test_dict.items():
        if u not in train_seqs: continue
        if len(test_items) == 0: continue
        seq = train_seqs[u]
        seq = [0] * (args.maxlen - len(seq)) + seq[-args.maxlen:]
        
        # FIX: Ensure Input is Tensor on Device
        seq_input = torch.LongTensor([seq]).to(args.device)
        
        # 1. Convert History to GROUPS
        history_items = history_dict.get(u, set())
        history_groups = set()
        for item in history_items:
            if item in item_to_group_map:
                history_groups.add(item_to_group_map[item])

        # 2. Filter Test Items (Keep only those from New Groups)
        valid_new_items = []
        for x in test_items:
            t_group = item_to_group_map.get(x, None)
            if t_group:
                if t_group not in history_groups:
                    valid_new_items.append(x)
            else:
                # Fallback if map missing
                if x not in history_items:
                    valid_new_items.append(x)

        with torch.no_grad():
            log_feats = model.log2feats(seq_input)
            final_feat = log_feats[:, -1, :]
            
            # --- FEATURE COMPATIBILITY ---
            # In 'let_it_go', we often have a helper to get all item vectors
            if hasattr(model, 'get_item_vector'):
                all_indices = torch.arange(args.itemnum + 1).to(args.device)
                item_embs = model.get_item_vector(all_indices)
            else:
                item_embs = model.item_emb.weight
            
            logits = final_feat.matmul(item_embs.t())
            last_logits = logits[0].clone()
            last_logits[0] = -np.inf
            _, indices = torch.topk(last_logits, 100)
            recs = indices.cpu().numpy().tolist()

        # Hit logic
        hit_10 = any(x in recs[:10] for x in test_items)
        hit_100 = any(x in recs[:100] for x in test_items)
        HR_10_all.append(1 if hit_10 else 0)
        HR_100_all.append(1 if hit_100 else 0)

        if valid_new_items:
            hit_10_new = any(x in recs[:10] for x in valid_new_items)
            hit_100_new = any(x in recs[:100] for x in valid_new_items)
            HR_10_new.append(1 if hit_10_new else 0)
            HR_100_new.append(1 if hit_100_new else 0)

    return np.mean(HR_10_all), np.mean(HR_100_all), np.mean(HR_10_new), np.mean(HR_100_new)

# --- ROLLING EVALUATION LOGIC ---
def evaluate_rolling_logic(model, test_dict, train_seqs, date_dict, history_dict, args, masker=None, item_to_group_map={}):
    total_events = 0
    total_new_events = 0  # <--- NEW COUNTER
    hits_10_all, hits_100_all = 0, 0
    hits_10_new, hits_100_new = 0, 0
    
    # Precompute embeddings once if possible
    with torch.no_grad():
        if hasattr(model, 'get_item_vector'):
            all_indices = torch.arange(args.itemnum + 1).to(args.device)
            item_embs = model.get_item_vector(all_indices)
        else:
            item_embs = model.item_emb.weight

    for u, test_items in test_dict.items():
        if u not in train_seqs: continue
        curr_seq = train_seqs[u][:] 
        user_dates = date_dict.get(u, [])

        # 1. Build History Groups set ONCE for the user (to start)
        current_history_groups = set()
        for item in train_seqs[u]:
            if item in item_to_group_map:
                current_history_groups.add(item_to_group_map[item])

        for i, target_item in enumerate(test_items):
            current_date = user_dates[i] if i < len(user_dates) else None
            
            # 2. Input Setup (On GPU)
            seq_slice = curr_seq[-args.maxlen:]
            padded_seq = [0] * (args.maxlen - len(seq_slice)) + seq_slice
            seq_input = torch.LongTensor([padded_seq]).to(args.device)

            with torch.no_grad():
                log_feats = model.log2feats(seq_input)
                final_feat = log_feats[:, -1, :] 
                # Use precomputed embeddings
                logits = final_feat.matmul(item_embs.t()) 
                
                # Take first row, stay on GPU
                last_logits = logits[0]
                
                if masker and current_date is not None:
                    busy_ids = masker.get_unavailable_items(current_date, u)
                    if busy_ids:
                        valid_busy = [bid for bid in busy_ids if bid < last_logits.shape[0]]
                        if valid_busy:
                            # Create tensor directly on DEVICE to avoid crash
                            busy_indices = torch.tensor(valid_busy, dtype=torch.long, device=args.device)
                            last_logits.index_fill_(0, busy_indices, -float('inf'))
                
                last_logits[0] = -np.inf
                _, indices = torch.topk(last_logits, 100)
                recs = indices.cpu().numpy().tolist()
            
            target_group = item_to_group_map.get(target_item, None)
            
            # 2. Get Groups currently in History
            history_items = set(curr_seq) 
            history_groups = set()
            for h_item in history_items:
                if h_item in item_to_group_map:
                    history_groups.add(item_to_group_map[h_item])

            # --- METRIC CALCULATION ---
            is_hit_10 = target_item in recs[:10]
            is_hit_100 = target_item in recs[:100]
            
            # General Metrics
            hits_10_all += 1 if is_hit_10 else 0
            hits_100_all += 1 if is_hit_100 else 0
            total_events += 1
            
            # 3. Check Condition
            is_new = False
            if target_group:
                is_new = target_group not in history_groups
            else:
                is_new = target_item not in history_items

            if is_new:
                hits_10_new += 1 if is_hit_10 else 0
                hits_100_new += 1 if is_hit_100 else 0
                total_new_events += 1 

            # 4. UPDATE History Groups
            current_history_groups.add(target_group)
            
            # Update sequence
            curr_seq.append(target_item)
    
    if total_events == 0: return 0,0,0,0
    hr_10_new_avg = hits_10_new / total_new_events if total_new_events > 0 else 0
    hr_100_new_avg = hits_100_new / total_new_events if total_new_events > 0 else 0

    return hits_10_all/total_events, hits_100_all/total_events, hr_10_new_avg, hr_100_new_avg

# ==========================================
#  NEW AUTOMATION HELPERS
# ==========================================

def log_to_csv(filename, row_dict):
    """Appends a dictionary of results to a CSV file."""
    file_exists = os.path.isfile(filename)
    fieldnames = ['Features', 'Experiment_Name', 'Eval_Mode', 'HR@10', 'HR@100', 'HR@10_new', 'HR@100_new', 'Timestamp']
    
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)

# --- CHANGE 2: Define the 4 Let It Go Tasks ---
def get_eval_tasks():
    tasks = []
    
    # Task 1: 70-30 Items
    tasks.append({
        'name': f'LG_Delta_0.5_7030_Items',
        'model_dir': f'both_features_delta_0.5_70_30', # Suffix
        'dataset': 'data_70_30/clothing_items_train',
        'v': True, 't': True, 
        'label': f'Delta=0.5'
    })
    # Task 2: 70-30 Groups
    tasks.append({
        'name': f'LG_Delta_0.5_7030_Groups',
        'model_dir': f'both_features_delta_0.5_70_30',
        'dataset': 'data_70_30/clothing_groups_train',
        'v': True, 't': True, 
        'label': f'Delta=0.5'
    })
    '''    # Task 3: LOO Items
    tasks.append({
        'name': f'LG_Delta_0.5_LOO_Items',
        'model_dir': f'both_features_delta_0.5_loo',
        'dataset': 'data_loo/clothing_items_train',
        'v': True, 't': True, 
        'label': f'Delta=0.5'
    })
    # Task 4: LOO Groups
    tasks.append({
        'name': f'LG_Delta_0.5_LOO_Groups',
        'model_dir': f'both_features_delta_0.5_loo',
        'dataset': 'data_loo/clothing_groups_train',
        'v': True, 't': True, 
        'label': f'Delta=0.5'
    })'''
    return tasks

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

# ==========================================
#  MAIN EXECUTION LOOP
# ==========================================

if __name__ == '__main__':
    # 1. Parse Base Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='default')
    parser.add_argument('--train_dir', default='default')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--hidden_units', default=128, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=1000, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.7, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='mps', type=str) # Default to mps for Mac
    parser.add_argument('--inference_only', default=False, type=str2bool)
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', default=False, type=str2bool)
    parser.add_argument('--use_visual', default=True, type=str2bool)
    parser.add_argument('--use_tags', default=True, type=str2bool)
    # Added delta_max arg which is required for LG model
    parser.add_argument('--delta_max', default=0.5, type=float)

    base_args = parser.parse_args()
    
    output_csv = "evaluation_results_let_it_go_with_clip.csv"
    print(f"Results will be saved to: {output_csv}\n")

    tasks = get_eval_tasks()

    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] Processing: {task['name']} | Features: {task['label']}")
        
        # 2. Setup Args for this Task
        args = copy.deepcopy(base_args)
        args.dataset = task['dataset']
        # For LG model, train_dir usually isn't used directly in model init, but we set it
        args.train_dir = task['model_dir'] 
        args.use_visual = task['v']
        args.use_tags = task['t']

        # 3. Dynamic Data Loading Logic
        # ------------------------------------------------------------------
        # CORRECT PATH LOGIC
        # ------------------------------------------------------------------
        # Determine the raw folder name (e.g., 'data_70_30' or 'data_loo')
        raw_folder_name = os.path.dirname(task['dataset']) if 'dataset' in task else os.path.dirname(args.dataset)
        
        # Determine where the data folder is located
        if os.path.exists(os.path.join('data', raw_folder_name)):
            dataset_folder = os.path.join('data', raw_folder_name) # e.g. 'data/data_70_30'
        elif os.path.exists(raw_folder_name):
            dataset_folder = raw_folder_name
        else:
            # Fallback
            if 'loo' in task['name'].lower():
                dataset_folder = 'data/data_loo'
            else:
                dataset_folder = 'data/data_70_30'

        # Check for Group vs Item task
        is_group_task = 'group' in task['dataset'].lower() or 'group' in task['name'].lower()

        if is_group_task:
            print("  --> Group Task detected. Using direct ID checks (skipping group map).")
            group_map = {} 
        else:
            group_map = load_group_mapping(dataset_folder)
        
        data_root = dataset_folder 
        
        if is_group_task:
            pkl_filename = 'test_data_groups.pkl'
            map_filename = 'group_maps.pkl'
            train_txt_name = 'clothing_groups_train.txt'
        else:
            pkl_filename = 'test_data_items.pkl'
            map_filename = 'item_maps.pkl'
            train_txt_name = 'clothing_items_train.txt'
            
        pkl_path = os.path.join(data_root, pkl_filename)
        map_path = os.path.join(data_root, map_filename)
        train_file = os.path.join(data_root, train_txt_name)

        print(f"  Loading Data from: {data_root}")
        
        try:
            if not os.path.exists(pkl_path):
                print(f"  SKIPPING: Pickle file not found at {pkl_path}")
                continue

            # Load Maps
            with open(map_path, 'rb') as f:
                maps = pickle.load(f)
                user_map, item_map = maps[0], maps[1]
            
            # Load Test Data (Dicts)
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            test_dict = data['test']
            history_dict = data['history']
            date_dict = data.get('dates', {}) 
            
            # Load Training Sequences
            train_seqs = {}
            if os.path.exists(train_file):
                with open(train_file, 'r') as f:
                    for line in f:
                        u, i = map(int, line.split())
                        if u not in train_seqs: train_seqs[u] = []
                        train_seqs[u].append(i)
            else:
                 print(f"  Warning: Train file {train_file} not found.")

            # Set Model Dimensions from Maps
            args.usernum = len(user_map)
            args.itemnum = len(item_map)
            
            # --- Initialize Model ---
            # Now using the Let It Go SASRec
            model = SASRec(args.usernum, args.itemnum, args).to(args.device)
            
            # --- CHANGE 3: Correct Model Path Construction ---
            # 1. Get base dataset name (e.g., clothing_items_train)
            dataset_base = os.path.basename(task['dataset'])
            # 2. Combine with suffix (e.g., clothing_items_train_both_features_delta_0.3_70_30)
            model_folder_name = f"{dataset_base}_{task['model_dir']}"
            
            # 3. Look for the folder inside data_root (data/data_70_30/...)
            candidate_folder = os.path.join(raw_folder_name, model_folder_name)
            model_path = os.path.join(candidate_folder, 'SASRec_Best_Val.pth')
            
            # Check if it exists
            if not os.path.exists(model_path):
                print(f"  Skipping: Model not found at {model_path}")
                continue

            print(f"  Loading Weights: {os.path.basename(model_path)}")
            model.load_state_dict(torch.load(model_path, map_location=args.device))
            model.eval()
                
            # ==========================================
            #  RUN EVALUATION MODES
            # ==========================================
            if is_group_task:
                print(f"    --> Group Dataset detected. Skipping Rolling Eval (Running Static ONLY).")
            else:
                # 1. Rolling + No Mask
                print("    [1/3] Rolling (No Mask)...")
                try:
                    metrics = evaluate_rolling_logic(model, test_dict, train_seqs, date_dict, history_dict, args, masker=None, item_to_group_map=group_map)
                    log_to_csv(output_csv, {
                        'Features': task['label'],
                        'Experiment_Name': task['name'],
                        'Eval_Mode': 'Rolling (No Mask)',
                        'HR@10': metrics[0], 'HR@100': metrics[1],
                        'HR@10_new': metrics[2], 'HR@100_new': metrics[3],
                        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                except Exception as e:
                    print(f"    Error: {e}")

                # 2. Rolling + Mask
                print("    [2/3] Rolling (Availability Mask)...")
                try:
                    masker = AvailabilityMask('original_orders.csv', 'user_activity_triplets.csv', item_map, user_map)
                    metrics = evaluate_rolling_logic(model, test_dict, train_seqs, date_dict, history_dict, args, masker=masker, item_to_group_map=group_map)
                    log_to_csv(output_csv, {
                        'Features': task['label'],
                        'Experiment_Name': task['name'],
                        'Eval_Mode': 'Rolling (Availability Mask)',
                        'HR@10': metrics[0], 'HR@100': metrics[1],
                        'HR@10_new': metrics[2], 'HR@100_new': metrics[3],
                        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                except Exception as e:
                    print(f"    Error (Masking might not be set up): {e}")

            # 3. Static (Pure) - Run for EVERYONE
            print("    [3/3] Static Eval...")
            try:
                metrics = evaluate_static_logic(model, test_dict, train_seqs, history_dict, args, item_to_group_map=group_map)
                log_to_csv(output_csv, {
                    'Features': task['label'],
                    'Experiment_Name': task['name'],
                    'Eval_Mode': 'Static (Pure)',
                    'HR@10': metrics[0], 'HR@100': metrics[1],
                    'HR@10_new': metrics[2], 'HR@100_new': metrics[3],
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            except Exception as e:
                print(f"    Error: {e}")

        except Exception as e:
            print(f"CRITICAL FAILURE on task {task['name']}: {e}")
            import traceback
            traceback.print_exc()

    print("\nAll evaluations complete.")