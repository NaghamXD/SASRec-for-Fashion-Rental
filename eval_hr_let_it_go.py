import os
import sys

# --- FIX 1: Allow importing from the subdirectory ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'let_it_go'))

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import pickle
import numpy as np
import pandas as pd
import csv
import copy
import argparse
from datetime import datetime

# Import from the subdirectory module
from let_it_go.model_lg import SASRec

# ==========================================
#  HELPER CLASSES & FUNCTIONS
# ==========================================
def load_group_mapping(data_root):
    """
    Loads mapping from Item_ID (int) -> Group_ID (str)
    """
    # Check for outfits.csv in the data root or root dir
    csv_path = os.path.join(data_root, 'outfits.csv')
    if not os.path.exists(csv_path):
        csv_path = 'outfits.csv' # Try current dir
    
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Group discovery metric will default to Item discovery.")
        return {}

    try:
        outfits = pd.read_csv(csv_path, sep=';')
        
        # Load the Item Map
        map_path = os.path.join(data_root, 'item_maps.pkl')
        if not os.path.exists(map_path):
            print(f"Warning: {map_path} not found.")
            return {}
            
        with open(map_path, 'rb') as f:
            maps = pickle.load(f)
            # maps is usually [user_map, item_map]
            item_map = maps[1]

        # Create Map: {Integer_ID: Group_ID_String}
        inv_item_map = {v: k for k, v in item_map.items()} 
        raw_item_to_group = dict(zip(outfits['id'], outfits['group'])) 
        
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
        
        if not os.path.exists(orders_path):
            print(f"Warning: {orders_path} not found. Availability Mask might fail.")
            self.date_index = {}
            return

        df1 = pd.read_csv(orders_path, sep=';')
        df2 = pd.read_csv(triplets_path, sep=';')
        cols = ['customer.id', 'outfit.id', 'rentalPeriod.start', 'rentalPeriod.end']
        
        # Ensure cols exist
        if not all(c in df1.columns for c in cols):
            print("Warning: CSV columns mismatch for Availability Mask.")
            self.date_index = {}
            return

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
                item_id = int(row.item_int)
                if item_id not in self.date_index[date_key]: self.date_index[date_key][item_id] = set()
                self.date_index[date_key][item_id].add(int(row.user_int))
                current += pd.Timedelta(days=1)

    def get_unavailable_items(self, query_date, current_user_id):
        date_key = query_date.date()
        unavailable = []
        if date_key in self.date_index:
            for item_id, renters in self.date_index[date_key].items():
                if current_user_id not in renters:
                    unavailable.append(item_id)
        return unavailable

def evaluate_static_logic(model, test_dict, train_seqs, history_dict, args, item_to_group_map={}):
    HR_10_all, HR_100_all = [], []
    HR_10_new, HR_100_new = [], []

    # Placeholder for Global Counts (Assuming existing file)
    if os.path.exists('original_orders.csv'):
        all_orders = pd.read_csv('original_orders.csv', sep=';')
        global_group_counts = all_orders['group'].value_counts().to_dict()
    else:
        global_group_counts = {}

    for u, test_items in test_dict.items():
        if u not in train_seqs: continue
        if len(test_items) == 0: continue
        
        seq = train_seqs[u]
        seq = [0] * (args.maxlen - len(seq)) + seq[-args.maxlen:]
        seq_input = np.array([seq])
        
        # Identify "New" items (Cold-Start)
        valid_new_items = []
        for x in test_items:
            t_group = item_to_group_map.get(x, None)
            if t_group and global_group_counts.get(t_group, 0) == 1:
                valid_new_items.append(x)
            elif not t_group: # Fallback
                valid_new_items.append(x)

        with torch.no_grad():
            log_feats = model.log2feats(seq_input)
            final_feat = log_feats[:, -1, :]
            
            # Representation Adjustment (Section 3.3)
            all_item_indices = torch.arange(model.item_num + 1).to(args.device)
            item_embs = model.get_item_vector(all_item_indices)
            
            logits = final_feat.matmul(item_embs.t())
            last_logits = logits[0].clone()
            last_logits[0] = -np.inf 
            
            _, indices = torch.topk(last_logits, 100)
            recs = indices.cpu().numpy().tolist()

        hit_10 = any(x in recs[:10] for x in test_items)
        hit_100 = any(x in recs[:100] for x in test_items)
        HR_10_all.append(1 if hit_10 else 0)
        HR_100_all.append(1 if hit_100 else 0)

        if valid_new_items:
            hit_10_new = any(x in recs[:10] for x in valid_new_items)
            hit_100_new = any(x in recs[:100] for x in valid_new_items)
            HR_10_new.append(1 if hit_10_new else 0)
            HR_100_new.append(1 if hit_100_new else 0)

    # Handle empty lists
    m_h10 = np.mean(HR_10_all) if HR_10_all else 0
    m_h100 = np.mean(HR_100_all) if HR_100_all else 0
    m_h10n = np.mean(HR_10_new) if HR_10_new else 0
    m_h100n = np.mean(HR_100_new) if HR_100_new else 0

    return m_h10, m_h100, m_h10n, m_h100n

def evaluate_rolling_logic(model, test_dict, train_seqs, date_dict, history_dict, args, masker=None, item_to_group_map={}):
    total_events = 0
    total_new_events = 0 
    hits_10_all, hits_100_all = 0, 0
    hits_10_new, hits_100_new = 0, 0
    
    if os.path.exists('original_orders.csv'):
        all_orders = pd.read_csv('original_orders.csv', sep=';')
        global_group_counts = all_orders['group'].value_counts().to_dict()
    else:
        global_group_counts = {}

    with torch.no_grad():
        all_item_indices = torch.arange(model.item_num + 1).to(args.device)
        item_embs = model.get_item_vector(all_item_indices) 

    for u, test_items in test_dict.items():
        if u not in train_seqs: continue
        curr_seq = train_seqs[u][:] 
        user_dates = date_dict.get(u, [])

        for i, target_item in enumerate(test_items):
            current_date = user_dates[i] if i < len(user_dates) else None

            seq_slice = curr_seq[-args.maxlen:]
            padded_seq = [0] * (args.maxlen - len(seq_slice)) + seq_slice
            seq_input = torch.LongTensor([padded_seq]).to(args.device)
            
            with torch.no_grad():
                log_feats = model.log2feats(seq_input)
                final_feat = log_feats[:, -1, :] 
                
                logits = final_feat.matmul(item_embs.t())
                last_logits = logits[0].clone()
                
                if masker and current_date is not None:
                    busy_ids = masker.get_unavailable_items(current_date, u)
                    if busy_ids:
                        # Ensure busy_ids are within valid range
                        valid_busy = [bid for bid in busy_ids if bid < model.item_num + 1]
                        if valid_busy:
                            busy_indices = torch.tensor(valid_busy).to(args.device)
                            last_logits.index_fill_(0, busy_indices, -float('inf'))
                
                last_logits[0] = -np.inf
                _, indices = torch.topk(last_logits, 100)
                recs = indices.cpu().numpy().tolist()
            
            is_hit_10 = target_item in recs[:10]
            is_hit_100 = target_item in recs[:100]
            
            hits_10_all += 1 if is_hit_10 else 0
            hits_100_all += 1 if is_hit_100 else 0
            total_events += 1
            
            target_group = item_to_group_map.get(target_item, None)
            is_globally_rare = global_group_counts.get(target_group, 0) == 1

            if is_globally_rare:
                hits_10_new += 1 if is_hit_10 else 0
                hits_100_new += 1 if is_hit_100 else 0
                total_new_events += 1 

            curr_seq.append(target_item)
    
    if total_events == 0: return 0,0,0,0
    
    hr_10_new_avg = hits_10_new / total_new_events if total_new_events > 0 else 0
    hr_100_new_avg = hits_100_new / total_new_events if total_new_events > 0 else 0

    return hits_10_all/total_events, hits_100_all/total_events, hr_10_new_avg, hr_100_new_avg

# ==========================================
#  AUTOMATION HELPERS
# ==========================================
def log_to_csv(filename, row_dict):
    file_exists = os.path.isfile(filename)
    fieldnames = ['Features', 'Experiment_Name', 'Eval_Mode', 'HR@10', 'HR@100', 'HR@10_new', 'HR@100_new', 'Timestamp']
    
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)

def get_eval_tasks():
    tasks = []
    delta_values = [0.05, 0.1, 0.2]
    
    for d in delta_values:
        # Task 1: 70-30 Items
        tasks.append({
            'name': f'70-30 Split Items (Delta {d})',
            'model_dir_suffix': f'both_features_delta_{d}', # The suffix used in train_lg.py
            'dataset': 'data_70_30/clothing_items_train',
            'v': True, 't': True, 
            'label': f'Split Items (Delta={d})'
        })

        # Task 2: 70-30 Groups
        tasks.append({
            'name': f'70-30 Split Groups (Delta {d})',
            'model_dir_suffix': f'both_features_delta_{d}',
            'dataset': 'data_70_30/clothing_groups_train',
            'v': True, 't': True, 
            'label': f'Split Groups (Delta={d})'
        })

        # Task 3: LOO Items
        tasks.append({
            'name': f'LOO Items (Delta {d})',
            'model_dir_suffix': f'both_features_delta_{d}',
            'dataset': 'data_loo/clothing_items_train',
            'v': True, 't': True, 
            'label': f'LOO Items Delta={d}'
        })
        
        # Task 4: LOO Groups
        tasks.append({
            'name': f'LOO Groups (Delta {d})',
            'model_dir_suffix': f'both_features_delta_{d}',
            'dataset': 'data_loo/clothing_groups_train',
            'v': True, 't': True, 
            'label': f'LOO Groups Delta={d}'
        })
    
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
    parser = argparse.ArgumentParser()
    # Dummy args to satisfy SASRec constructor if needed
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
    parser.add_argument('--device', default='mps', type=str) 
    parser.add_argument('--inference_only', default=False, type=str2bool)
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', default=False, type=str2bool)
    parser.add_argument('--use_visual', default=True, type=str2bool)
    parser.add_argument('--use_tags', default=True, type=str2bool)
    parser.add_argument('--delta_max', default=0.1, type=float)

    base_args = parser.parse_args()
    
    output_csv = "evaluation_results_final.csv"
    print(f"Results will be saved to: {output_csv}\n")

    tasks = get_eval_tasks()

    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] Processing: {task['name']}")
        
        args = copy.deepcopy(base_args)
        args.use_visual = task['v']
        args.use_tags = task['t']
        
        # --- PATH LOGIC ---
        # 1. Resolve Data Root
        # task['dataset'] is relative (e.g. 'data_70_30/clothing_items_train')
        # We expect data to be in 'data/data_70_30/...'
        dataset_rel_path = task['dataset']
        dataset_subdir = os.path.dirname(dataset_rel_path) # e.g. 'data_70_30'
        
        # Check if 'data/' prefix is needed
        if os.path.exists(os.path.join('data', dataset_subdir)):
            data_root = os.path.join('data', dataset_subdir)
        else:
            # Fallback if running inside data folder or similar
            data_root = dataset_subdir
            
        print(f"  --> Data Root: {data_root}")

        # 2. Determine File Names
        is_group_task = 'group' in task['dataset'].lower()
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

        # 3. Load Data
        if not os.path.exists(pkl_path) or not os.path.exists(map_path):
            print(f"  !! SKIPPING: Data files missing at {data_root}")
            continue

        try:
            with open(map_path, 'rb') as f:
                maps = pickle.load(f)
                user_map, item_map = maps[0], maps[1]
            
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            test_dict = data['test']
            history_dict = data['history']
            date_dict = data.get('dates', {})
            
            train_seqs = {}
            if os.path.exists(train_file):
                with open(train_file, 'r') as f:
                    for line in f:
                        u, i = map(int, line.split())
                        if u not in train_seqs: train_seqs[u] = []
                        train_seqs[u].append(i)
            
            # 4. Load Model
            # This is the critical fix. 
            # main_lg.py names folder as: "{basename(dataset)}_{train_dir}"
            # Models are inside 'let_it_go/' directory
            
            dataset_basename = os.path.basename(task['dataset']) # e.g., 'clothing_items_train'
            model_folder_name = f"{dataset_basename}_{task['model_dir_suffix']}"
            
            # Look inside 'let_it_go' folder
            model_dir_path = os.path.join('let_it_go', model_folder_name)
            
            if not os.path.exists(model_dir_path):
                print(f"  !! SKIPPING: Model folder not found: {model_dir_path}")
                continue
                
            # Find best epoch model
            pth_files = [f for f in os.listdir(model_dir_path) if f.endswith('.pth') and 'epoch=' in f]
            if not pth_files:
                print(f"  !! SKIPPING: No checkpoints in {model_dir_path}")
                continue
                
            latest = max(pth_files, key=lambda x: int(x.split('epoch=')[1].split('.')[0]))
            model_path = os.path.join(model_dir_path, latest)
            print(f"  --> Loading Model: {model_path}")

            # Initialize and Load
            args.usernum = len(user_map)
            args.itemnum = len(item_map)
            
            # Pass correct args to model (specifically for data path in model init if needed)
            # Temporarily overwrite args.dataset to allow model to find embeddings if it tries to load them
            # The model_lg.py uses args.dataset to find embeddings path
            # It expects something like 'data_70_30/clothing_items_train'
            args.dataset = task['dataset'] 
            
            model = SASRec(args.usernum, args.itemnum, args).to(args.device)
            model.load_state_dict(torch.load(model_path, map_location=args.device), strict=False)
            model.eval()

            # 5. Run Evaluations
            group_map = {}
            if not is_group_task:
                group_map = load_group_mapping(data_root)

            # Static Eval
            print("    [1/2] Static Eval...")
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
                print(f"    Error in Static Eval: {e}")

            # Rolling Eval (Only for Items)
            if not is_group_task:
                print("    [2/2] Rolling Eval (Availability Mask)...")
                try:
                    masker = AvailabilityMask('original_orders.csv', 'user_activity_triplets.csv', item_map, user_map)
                    metrics = evaluate_rolling_logic(model, test_dict, train_seqs, date_dict, history_dict, args, masker=masker, item_to_group_map=group_map)
                    log_to_csv(output_csv, {
                        'Features': task['label'],
                        'Experiment_Name': task['name'],
                        'Eval_Mode': 'Rolling (Masked)',
                        'HR@10': metrics[0], 'HR@100': metrics[1],
                        'HR@10_new': metrics[2], 'HR@100_new': metrics[3],
                        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                except Exception as e:
                    print(f"    Error in Rolling Eval: {e}")

        except Exception as e:
            print(f"CRITICAL ERROR on task {task['name']}: {e}")