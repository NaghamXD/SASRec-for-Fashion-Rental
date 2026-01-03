import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import pickle
import numpy as np
import pandas as pd
import csv
import copy
import argparse
from datetime import datetime
from python.model import SASRec

# ==========================================
#  HELPER CLASSES & FUNCTIONS (KEEP THESE)
# ==========================================

class AvailabilityMask:
    def __init__(self, orders_path, triplets_path, item_map, user_map):
        print("Building Availability Index...")
        self.item_map = item_map
        self.user_map = user_map
        # Assuming files are in current directory or specific path
        # You might need to adjust paths if they are in 'data/'
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
def evaluate_static_logic(model, test_dict, train_seqs, history_dict, args):
    """Calculates metrics using fixed sequences (Set Recall)."""
    HR_10_all, HR_100_all = [], []
    HR_10_new, HR_100_new = [], []

    for u, test_items in test_dict.items():
        if u not in train_seqs: continue
        seq = train_seqs[u]
        seq = [0] * (args.maxlen - len(seq)) + seq[-args.maxlen:]
        seq_input = np.array([seq])
        
        with torch.no_grad():
            log_feats = model.log2feats(seq_input)
            final_feat = log_feats[:, -1, :]
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

        history = history_dict.get(u, set())
        valid_test_items = [x for x in test_items if x not in history]
        if valid_test_items:
            hit_10_new = any(x in recs[:10] for x in valid_test_items)
            hit_100_new = any(x in recs[:100] for x in valid_test_items)
            HR_10_new.append(1 if hit_10_new else 0)
            HR_100_new.append(1 if hit_100_new else 0)

    return np.mean(HR_10_all), np.mean(HR_100_all), np.mean(HR_10_new), np.mean(HR_100_new)

# --- ROLLING EVALUATION LOGIC ---
def evaluate_rolling_logic(model, test_dict, train_seqs, date_dict, history_dict, args, masker=None):
    total_events = 0
    hits_10_all, hits_100_all = 0, 0
    hits_10_new, hits_100_new = 0, 0
    
    for u, test_items in test_dict.items():
        if u not in train_seqs: continue
        curr_seq = train_seqs[u][:] 
        user_dates = date_dict.get(u, [])
        for i, target_item in enumerate(test_items):
            current_date = user_dates[i] if i < len(user_dates) else None
            seq_input = np.array([curr_seq[-args.maxlen:] + [0] * max(0, args.maxlen - len(curr_seq))])
            with torch.no_grad():
                log_feats = model.log2feats(seq_input)
                final_feat = log_feats[:, -1, :] 
                item_embs = model.item_emb.weight 
                logits = final_feat.matmul(item_embs.t()) 
                last_logits = logits[0].clone()
                if masker and current_date is not None:
                    busy_ids = masker.get_unavailable_items(current_date, u)
                    if busy_ids:
                        busy_indices = torch.tensor(busy_ids).to(args.device)
                        last_logits.index_fill_(0, busy_indices, -float('inf'))
                last_logits[0] = -np.inf
                _, indices = torch.topk(last_logits, 100)
                recs = indices.cpu().numpy().tolist()
            
            is_hit_10 = target_item in recs[:10]
            is_hit_100 = target_item in recs[:100]
            hits_10_all += 1 if is_hit_10 else 0
            hits_100_all += 1 if is_hit_100 else 0
            if target_item not in history_dict.get(u, set()):
                hits_10_new += 1 if is_hit_10 else 0
                hits_100_new += 1 if is_hit_100 else 0
            total_events += 1
            curr_seq.append(target_item)
    
    if total_events == 0: return 0,0,0,0
    return hits_10_all/total_events, hits_100_all/total_events, hits_10_new/total_events, hits_100_new/total_events

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

def get_eval_tasks():
    tasks = []

    # --- SETTING 1: 70-30 SPLIT ---
    # Adjust 'model_dir' to match your actual folder names
    model_7030 = 'tag_features_split_items' 
    dataset_7030 = 'data_70_30/clothing_items_train'
    name_7030 = '70-30 Split (Items)'

    #tasks.append({'name': name_7030, 'model_dir': model_7030, 'dataset': dataset_7030, 'v': True, 't': True, 'label': 'Both features'})
    #tasks.append({'name': name_7030, 'model_dir': model_7030, 'dataset': dataset_7030, 'v': True, 't': False, 'label': 'Image features'})
    tasks.append({'name': name_7030, 'model_dir': model_7030, 'dataset': dataset_7030, 'v': False, 't': True, 'label': 'Tag features'})
    #tasks.append({'name': name_7030, 'model_dir': model_7030, 'dataset': dataset_7030, 'v': False, 't': False, 'label': 'No features'})

    # --- SETTING 2: LEAVE-ONE-OUT SPLIT ---
    model_loo = 'tag_features_loo_items' 
    dataset_loo = 'data_loo/clothing_items_train'
    name_loo = 'Leave-One-Out (Items)'

    #tasks.append({'name': name_loo, 'model_dir': model_loo, 'dataset': dataset_loo, 'v': True, 't': True, 'label': 'Both features'})
    #tasks.append({'name': name_loo, 'model_dir': model_loo, 'dataset': dataset_loo, 'v': True, 't': False, 'label': 'Image features'})
    tasks.append({'name': name_loo, 'model_dir': model_loo, 'dataset': dataset_loo, 'v': False, 't': True, 'label': 'Tag features'})
    #tasks.append({'name': name_loo, 'model_dir': model_loo, 'dataset': dataset_loo, 'v': False, 't': False, 'label': 'No features'})

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
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=1000, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='mps', type=str) # Default to mps for Mac
    parser.add_argument('--inference_only', default=False, type=str2bool)
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', default=False, type=str2bool)
    parser.add_argument('--use_visual', default=True, type=str2bool)
    parser.add_argument('--use_tags', default=True, type=str2bool)

    base_args = parser.parse_args()
    
    output_csv = "evaluation_results.csv"
    print(f"Results will be saved to: {output_csv}\n")

    tasks = get_eval_tasks()

    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] Processing: {task['name']} | Features: {task['label']}")
        
        # 2. Setup Args for this Task
        args = copy.deepcopy(base_args)
        args.dataset = task['dataset']
        args.train_dir = task['model_dir']
        args.use_visual = task['v']
        args.use_tags = task['t']

        # 3. Dynamic Data Loading
        # We need to construct paths similar to how main.py or your old evaluate_model did
        # Assuming args.dataset is like 'data_70_30/clothing_items_train'
        
        # Extract folder: 'data_70_30'
        dataset_subdir = os.path.dirname(args.dataset)
        # Construct full root: 'data/data_70_30'
        data_root = os.path.join('data', dataset_subdir)
        
        # Determine Pickle File Name
        if 'group' in args.dataset:
            pkl_filename = 'test_data_groups.pkl'
            map_filename = 'group_maps.pkl'
        else:
            pkl_filename = 'test_data_items.pkl'
            map_filename = 'item_maps.pkl'
            
        pkl_path = os.path.join(data_root, pkl_filename)
        map_path = os.path.join(data_root, map_filename)
        train_file = os.path.join(data_root, os.path.basename(args.dataset) + '.txt')

        print(f"  Loading Data from: {data_root}")
        
        try:
            # Load Maps
            with open(map_path, 'rb') as f:
                maps = pickle.load(f)
                user_map, item_map = maps[0], maps[1]
            
            # Load Test Data (Dicts)
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            test_dict = data['test']
            history_dict = data['history']
            date_dict = data.get('dates', {}) # Safe get in case missing
            
            # Load Training Sequences (for filtering)
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
            model = SASRec(args.usernum, args.itemnum, args).to(args.device)
            
            # --- Load Weights ---
            # Construct path: {dataset}_{model_dir}/SASRec.epoch=40.pth
            # Note: args.dataset here is relative path, we might need just the folder name 
            # If your main.py saves as "data_70_30/clothing_items_train_both_features...", check that.
            # Usually main.py does: args.dataset + '_' + args.train_dir
            
            # 1. Define the folder (ensure this variable matches your previous code)
            folder = args.dataset + '_' + args.train_dir
            model_path = None  # Safe initialization

            if os.path.exists(folder):
                # 2. Filter: Only get .pth files that explicitly have "epoch=" in the name
                pth_files = [f for f in os.listdir(folder) 
                             if f.endswith('.pth') and 'epoch=' in f]
                
                if pth_files:
                    # 3. Find the one with the highest epoch number
                    latest = max(pth_files, key=lambda x: int(x.split('epoch=')[1].split('.')[0]))
                    model_path = os.path.join(folder, latest)

            # 4. Load only if a valid path was found
            if model_path:
                print(f"  Loading Model: {model_path}")
                model.load_state_dict(torch.load(model_path, map_location=args.device), strict=False)
                model.eval()
            else:
                print(f"  !! SKIPPING: No 'epoch=' model found in {folder}")
                continue
                
            # ==========================================
            #  RUN EVALUATION MODES
            # ==========================================

            # Check if this is a "Group" task
            is_group_task = 'group' in args.dataset.lower() or 'group' in task['name'].lower()

            if is_group_task:
                print(f"    --> Group Dataset detected. Skipping Rolling Eval (Running Static ONLY).")
            else:
                # 1. Rolling + No Mask (Only for Items)
                print("    [1/3] Rolling (No Mask)...")
                try:
                    metrics = evaluate_rolling_logic(model, test_dict, train_seqs, date_dict, history_dict, args, masker=None)
                    log_to_csv(output_csv, {
                        'Features': task['label'],
                        'Experiment_Name': task['name'],
                        'Eval_Mode': 'Rolling (No Mask)',
                        'HR@10': metrics[0],
                        'HR@100': metrics[1],
                        'HR@10_new': metrics[2],
                        'HR@100_new': metrics[3],
                        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                except Exception as e:
                    print(f"    Error: {e}")

                # 2. Rolling + Mask (Only for Items)
                print("    [2/3] Rolling (Availability Mask)...")
                try:
                    # Initialize Masker (Global files assumed, adjust if dataset specific)
                    masker = AvailabilityMask('original_orders.csv', 'user_activity_triplets.csv', item_map, user_map)
                    
                    metrics = evaluate_rolling_logic(model, test_dict, train_seqs, date_dict, history_dict, args, masker=masker)
                    log_to_csv(output_csv, {
                        'Features': task['label'],
                        'Experiment_Name': task['name'],
                        'Eval_Mode': 'Rolling (Availability Mask)',
                        'HR@10': metrics[0],
                        'HR@100': metrics[1],
                        'HR@10_new': metrics[2],
                        'HR@100_new': metrics[3],
                        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                except Exception as e:
                    print(f"    Error (Masking might not be set up): {e}")

            # 3. Static (Pure) - Run for EVERYONE
            print("    [3/3] Static Eval...")
            try:
                metrics = evaluate_static_logic(model, test_dict, train_seqs, history_dict, args)
                log_to_csv(output_csv, {
                    'Features': task['label'],
                    'Experiment_Name': task['name'],
                    'Eval_Mode': 'Static (Pure)',
                    'HR@10': metrics[0],
                    'HR@100': metrics[1],
                    'HR@10_new': metrics[2],
                    'HR@100_new': metrics[3],
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            except Exception as e:
                print(f"    Error: {e}")

        except Exception as e:
            print(f"CRITICAL FAILURE on task {task['name']}: {e}")

    print("\nAll evaluations complete.")