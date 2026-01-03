import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import pickle
import numpy as np
import pandas as pd
import csv
from datetime import datetime
from python.model import SASRec

# --- AVAILABILITY MASK CLASS ---
class AvailabilityMask:
    def __init__(self, orders_path, triplets_path, item_map, user_map):
        print("Building Availability Index (with User ID)...")
        self.item_map = item_map
        self.user_map = user_map
        
        # 1. Load Data
        df1 = pd.read_csv(orders_path, sep=';')
        df2 = pd.read_csv(triplets_path, sep=';')
        cols = ['customer.id', 'outfit.id', 'rentalPeriod.start', 'rentalPeriod.end']
        full_df = pd.concat([df1[cols], df2[cols]])
        
        # 2. Convert Dates
        full_df['start'] = pd.to_datetime(full_df['rentalPeriod.start'])
        full_df['end'] = pd.to_datetime(full_df['rentalPeriod.end'])
        
        # 3. Filter and Map
        full_df = full_df[full_df['outfit.id'].isin(self.item_map)]
        full_df['item_int'] = full_df['outfit.id'].map(self.item_map)
        full_df = full_df[full_df['customer.id'].isin(self.user_map)]
        full_df['user_int'] = full_df['customer.id'].map(self.user_map)
        
        # 4. Create Lookup
        self.reservations = {}
        for row in full_df.itertuples():
            if row.item_int not in self.reservations: 
                self.reservations[row.item_int] = []
            self.reservations[row.item_int].append((row.start, row.end, row.user_int))
            
    def get_unavailable_items(self, query_date, current_user_id):
        unavailable = []
        for iid, intervals in self.reservations.items():
            for start, end, renter_id in intervals:
                if start <= query_date <= end:
                    if renter_id != current_user_id: # Don't block self
                        unavailable.append(iid)
                        break 
        return unavailable

# --- MAIN EVALUATION ---
def evaluate_model(model_path, dataset_name, test_pkl_path):
    print(f"\nEvaluating {dataset_name}...")
    
    # 1. Load Maps & Data
    with open('data/item_maps.pkl', 'rb') as f:
        # Depending on how you saved them, this might load user/item or user/group maps
        # For groups, we might need 'data/group_maps.pkl' if the IDs differ.
        # But usually user_map is consistent.
        maps = pickle.load(f)
        user_map = maps[0]
        item_map = maps[1]
        
    with open(test_pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    test_dict = data['test']
    history_dict = data['history']
    date_dict = data.get('dates', {}) 

    # Load Training Sequences
    train_seqs = {}
    user_num, item_num = 0, 0
    with open(f'data/{dataset_name}.txt', 'r') as f:
        for line in f:
            u, i = map(int, line.split())
            user_num = max(user_num, u)
            item_num = max(item_num, i)
            if u not in train_seqs: train_seqs[u] = []
            train_seqs[u].append(i)
            
    # 2. Logic Switch: ITEMS vs GROUPS
    # If "groups" is in the name, we use the OLD (Static) method.
    # If "items" is in the name, we use the NEW (Rolling+Mask) method.
    is_group_model = 'groups' in dataset_name
    use_mask = 'items' in dataset_name and not is_group_model
    
    masker = None
    if use_mask:
        masker = AvailabilityMask('original_orders.csv', 'user_activity_triplets.csv', item_map, user_map)

    # 3. Load Model
    class Args:
        def __init__(self):
            self.usernum = user_num
            self.itemnum = item_num
            self.maxlen = 50
            self.hidden_units = 50 
            self.num_blocks = 2
            self.num_heads = 1
            self.dropout_rate = 0.2
            self.device = 'mps'
            self.norm_first = False

    args = Args()
    model = SASRec(args.usernum, args.itemnum, args).to(args.device)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    # 4. Metrics Storage
    HR_10_all, HR_100_all = [], []
    HR_10_new, HR_100_new = [], []

    print(f"Testing on {len(test_dict)} users...")
    
    # ==========================================
    # MODE 1: STATIC EVALUATION (For GROUPS)
    # ==========================================
    if is_group_model:
        print("--> Using STATIC Evaluation (Standard for Groups)")
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
                last_logits = logits[0]
                last_logits[0] = -np.inf
                
                probs, indices = torch.topk(last_logits, 100)
                recs = indices.cpu().numpy().tolist()

            # "Set Recall" Logic (Easier, matches previous results)
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

    # ==========================================
    # MODE 2: ROLLING EVALUATION (For ITEMS)
    # ==========================================
    else:
        print("--> Using ROLLING Evaluation + AVAILABILITY MASK (For Items)")
        total_events = 0
        hits_10_all, hits_100_all = 0, 0
        hits_10_new, hits_100_new = 0, 0
        
        for u, test_items in test_dict.items():
            if u not in train_seqs: continue
            
            curr_seq = train_seqs[u]
            user_dates = date_dict.get(u, [])
            
            for i, target_item in enumerate(test_items):
                current_date = user_dates[i] if i < len(user_dates) else None
                
                # Input: Last 50 items
                seq_input = np.array([curr_seq[-args.maxlen:] + [0] * max(0, args.maxlen - len(curr_seq))])
                
                with torch.no_grad():
                    log_feats = model.log2feats(seq_input)
                    final_feat = log_feats[:, -1, :] 
                    item_embs = model.item_emb.weight 
                    logits = final_feat.matmul(item_embs.t()) 
                    last_logits = logits[0]
                    
                    # Apply Mask
                    if use_mask and current_date is not None:
                        busy_ids = masker.get_unavailable_items(current_date, u)
                        if busy_ids:
                            busy_indices = torch.tensor(busy_ids).to(args.device)
                            last_logits.index_fill_(0, busy_indices, -float('inf'))

                    last_logits[0] = -np.inf
                    probs, indices = torch.topk(last_logits, 100)
                    recs = indices.cpu().numpy().tolist()
                
                # "Next Item" Logic (Stricter, but required for Masking)
                is_hit_10 = target_item in recs[:10]
                is_hit_100 = target_item in recs[:100]
                
                hits_10_all += 1 if is_hit_10 else 0
                hits_100_all += 1 if is_hit_100 else 0
                
                if target_item not in history_dict.get(u, set()):
                    hits_10_new += 1 if is_hit_10 else 0
                    hits_100_new += 1 if is_hit_100 else 0
                
                total_events += 1
                curr_seq.append(target_item)
        
        # Normalize for print
        m_hr10_all = hits_10_all / total_events if total_events > 0 else 0
        m_hr100_all = hits_100_all / total_events if total_events > 0 else 0
        m_hr10_new = hits_10_new / total_events if total_events > 0 else 0
        m_hr100_new = hits_100_new / total_events if total_events > 0 else 0

        # Override lists so printing below works
        HR_10_all = [m_hr10_all] # Dummy list to make mean() work below
        HR_100_all = [m_hr100_all]
        HR_10_new = [m_hr10_new]
        HR_100_new = [m_hr100_new]

    # Calculate Final Averages
    m_hr10_all = np.mean(HR_10_all)
    m_hr100_all = np.mean(HR_100_all)
    m_hr10_new = np.mean(HR_10_new)
    m_hr100_new = np.mean(HR_100_new)

    print(f"RESULTS for {dataset_name}:")
    print(f"HR@10 (All):  {m_hr10_all:.4f}")
    print(f"HR@100 (All): {m_hr100_all:.4f}")
    print(f"HR@10 (New):  {m_hr10_new:.4f}")
    print(f"HR@100 (New): {m_hr100_new:.4f}")

    # CSV Saving
    csv_file = 'benchmark_results_with_availability_mask.csv'
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'Model_Name', 'Dataset', 'HR@10_All', 'HR@100_All', 'HR@10_New', 'HR@100_New'])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            os.path.basename(model_path),
            dataset_name,
            f"{m_hr10_all:.4f}", f"{m_hr100_all:.4f}", f"{m_hr10_new:.4f}", f"{m_hr100_new:.4f}"
        ])
    print(f"✅ Saved results to {csv_file}")

if __name__ == "__main__":
    # Items
    if os.path.exists('clothing_items_train_item_model'): 
        files = os.listdir('clothing_items_train_item_model')
        pth_files = [f for f in files if f.endswith('.pth')]
        if pth_files:
            latest = sorted(pth_files)[-1]
            evaluate_model(f'clothing_items_train_item_model/{latest}', 'clothing_items_train', 'data/test_data_items.pkl')
            
    # Groups
    if os.path.exists('clothing_groups_train_group_model'):
        files = os.listdir('clothing_groups_train_group_model')
        pth_files = [f for f in files if f.endswith('.pth')]
        if pth_files:
            latest = sorted(pth_files)[-1]
            evaluate_model(f'clothing_groups_train_group_model/{latest}', 'clothing_groups_train', 'data/test_data_groups.pkl')