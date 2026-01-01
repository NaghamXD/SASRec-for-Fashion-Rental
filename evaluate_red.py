import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import pickle
import numpy as np
import pandas as pd
import csv
from datetime import datetime
from python.model import SASRec

# --- OPTIMIZED AVAILABILITY MASK ---
class AvailabilityMask:
    def __init__(self, orders_path, triplets_path, item_map, user_map):
        print("Building Availability Index...")
        self.item_map = item_map
        self.user_map = user_map
        df1 = pd.read_csv(orders_path, sep=';')
        df2 = pd.read_csv(triplets_path, sep=';')
        cols = ['customer.id', 'outfit.id', 'rentalPeriod.start', 'rentalPeriod.end']
        full_df = pd.concat([df1[cols], df2[cols]])
        full_df['start'] = pd.to_datetime(full_df['rentalPeriod.start'])
        full_df['end'] = pd.to_datetime(full_df['rentalPeriod.end'])
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

# --- STATIC EVALUATION LOGIC (From evaluate_paper_metrics_wo_availability.py) ---
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

def save_to_csv(model_path, dataset, eval_mode, hr10, hr100, hr10_n, hr100_n):
    csv_file = 'benchmark_results_red.csv'
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'Model_Name', 'Dataset', 'Eval_Mode', 'HR@10_All', 'HR@100_All', 'HR@10_New', 'HR@100_New'])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), os.path.basename(model_path), dataset, eval_mode, f"{hr10:.4f}", f"{hr100:.4f}", f"{hr10_n:.4f}", f"{hr100_n:.4f}"])

def evaluate_model(model_path, dataset_name, test_pkl_path):
    print(f"\nEvaluating {dataset_name}...")
    # Extract the split folder (e.g., 'data/data_70_30') from the test_pkl_path
    split_folder = os.path.dirname(test_pkl_path)
    
    # 1. Update Map Path to look in the correct split subfolder
    map_path = os.path.join(split_folder, 'group_maps.pkl' if 'group' in dataset_name else 'item_maps.pkl')
    with open(map_path, 'rb') as f:
        maps = pickle.load(f)
        user_map, item_map = maps[0], maps[1]
    with open(test_pkl_path, 'rb') as f:
        data = pickle.load(f)
    test_dict, history_dict, date_dict = data['test'], data['history'], data.get('dates', {})
    
    train_seqs = {}
    train_file_path = os.path.join(split_folder, f'{dataset_name}.txt')
    with open(train_file_path, 'r') as f:
        for line in f:
            u, i = map(int, line.split())
            if u not in train_seqs: train_seqs[u] = []
            train_seqs[u].append(i)

    class Args:
        def __init__(self):
            self.usernum, self.itemnum = len(user_map), len(item_map)
            self.maxlen, self.hidden_units = 200, 50
            self.num_blocks, self.num_heads = 2, 1
            self.dropout_rate, self.device = 0.2, 'mps'
            self.norm_first, self.dataset = False, dataset_name

    args = Args()
    model = SASRec(args.usernum, args.itemnum, args).to(args.device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    model.eval()

    if 'group' in dataset_name:
        print("--> Using STATIC Evaluation (Standard for Groups)")
        r10, r100, r10_n, r100_n = evaluate_static_logic(model, test_dict, train_seqs, history_dict, args)
        save_to_csv(model_path, dataset_name, "Standard_Static", r10, r100, r10_n, r100_n)
    else:
        # RUN THREE PASSES FOR ITEMS
        masker = AvailabilityMask('original_orders.csv', 'user_activity_triplets.csv', item_map, user_map)
        
        # 1. Rolling + Mask
        print("\n--- Pass 1: Rolling + Availability Mask ---")
        r10, r100, r10_n, r100_n = evaluate_rolling_logic(model, test_dict, train_seqs, date_dict, history_dict, args, masker=masker)
        save_to_csv(model_path, dataset_name, "Rolling_Masked", r10, r100, r10_n, r100_n)

        # 2. Rolling + No Mask
        print("--- Pass 2: Rolling + Pure Preference (No Mask) ---")
        r10, r100, r10_n, r100_n = evaluate_rolling_logic(model, test_dict, train_seqs, date_dict, history_dict, args, masker=None)
        save_to_csv(model_path, dataset_name, "Rolling_NoMask", r10, r100, r10_n, r100_n)

        # 3. Static + No Mask
        print("--- Pass 3: Static Evaluation (Set Recall) ---")
        r10, r100, r10_n, r100_n = evaluate_static_logic(model, test_dict, train_seqs, history_dict, args)
        save_to_csv(model_path, dataset_name, "Static_NoMask", r10, r100, r10_n, r100_n)

if __name__ == "__main__":
    for folder, d_name, pkl in [('data_loo/clothing_items_train_item_model', 'clothing_items_train', 'data/data_loo/test_data_items.pkl'),
                                ('data_loo/clothing_groups_train_group_model', 'clothing_groups_train', 'data/data_loo/test_data_groups.pkl')]:
        if os.path.exists(folder):
            pth_files = [f for f in os.listdir(folder) if f.endswith('.pth')]
            if pth_files:
                latest = max(pth_files, key=lambda x: int(x.split('epoch=')[1].split('.')[0]))
                evaluate_model(f'{folder}/{latest}', d_name, pkl)