import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import pickle
import numpy as np
from python.model import SASRec

def evaluate_model(model_path, dataset_name, test_pkl_path):
    print(f"\nEvaluating {dataset_name}...")
    
    # 1. Load Data
    # We need the args used during training to rebuild the model structure
    # Usually saved in the checkpoint or we assume standard args
    dataset_path = f'data/{dataset_name}.txt'
    
    # Load the test/history data we prepared
    with open(test_pkl_path, 'rb') as f:
        data = pickle.load(f)
    test_dict = data['test']       # {uid: [test_item_1, test_item_2...]}
    history_dict = data['history'] # {uid: {train_item_1, train_item_2...}}

    # Load User/Item counts from the text file to initialize model
    user_num = 0
    item_num = 0
    with open(dataset_path, 'r') as f:
        for line in f:
            u, i = line.rstrip().split(' ')
            user_num = max(user_num, int(u))
            item_num = max(item_num, int(i))
            
    # 2. Load Model
    # Note: Adjust args to match your training command
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
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    # 3. Metrics
    HR_10_all = []
    HR_100_all = []
    HR_10_new = []
    HR_100_new = []

    # 4. Evaluation Loop
    # We need to feed the Training Sequence to get the Next Prediction
    # Since SASRec is sequential, we just need the LAST state after the full training sequence.
    
    # Re-read training sequences
    train_seqs = {}
    with open(dataset_path, 'r') as f:
        for line in f:
            u, i = map(int, line.split())
            if u not in train_seqs: train_seqs[u] = []
            train_seqs[u].append(i)

    print(f"Testing on {len(test_dict)} users...")
    
    for u, test_items in test_dict.items():
            if u not in train_seqs: continue
            
            seq = train_seqs[u]
            
            # Pad sequence
            seq = [0] * (args.maxlen - len(seq)) + seq[-args.maxlen:]
            
            # FIX: Convert to a NumPy array (CPU) for the model input
            seq_input = np.array([seq])
            
            with torch.no_grad():
                # 1. Get the sequence embedding (hidden state)
                log_feats = model.log2feats(seq_input) 
                
                # 2. Extract the state at the last time step
                final_feat = log_feats[:, -1, :] # Shape: [1, hidden_units]

                # 3. Calculate scores for ALL items
                item_embs = model.item_emb.weight # Shape: [itemnum+1, hidden_units]
                logits = final_feat.matmul(item_embs.t()) # Shape: [1, itemnum+1]

                # 4. Use these logits for ranking
                last_logits = logits[0]
                
                # --- THIS WAS MISSING ---
                # Set padding (item 0) to negative infinity so it's never recommended
                last_logits[0] = -np.inf
                
                # Get Top 100 candidates
                probs, indices = torch.topk(last_logits, 100)
                recs = indices.cpu().numpy().tolist()
                # ------------------------

            # ------------------------------------------------
            # METRIC 1: Standard (All items)
            # ------------------------------------------------
            hit_10 = any(x in recs[:10] for x in test_items)
            hit_100 = any(x in recs[:100] for x in test_items)
            
            HR_10_all.append(1 if hit_10 else 0)
            HR_100_all.append(1 if hit_100 else 0)

            # ------------------------------------------------
            # METRIC 2: New / Unique (Exclude items seen in history)
            # ------------------------------------------------
            history = history_dict.get(u, set())
            valid_test_items = [x for x in test_items if x not in history]
            
            if valid_test_items:
                hit_10_new = any(x in recs[:10] for x in valid_test_items)
                hit_100_new = any(x in recs[:100] for x in valid_test_items)
                
                HR_10_new.append(1 if hit_10_new else 0)
                HR_100_new.append(1 if hit_100_new else 0)

# Calculate Averages
    m_hr10_all = np.mean(HR_10_all)
    m_hr100_all = np.mean(HR_100_all)
    m_hr10_new = np.mean(HR_10_new)
    m_hr100_new = np.mean(HR_100_new)

    print(f"RESULTS for {dataset_name}:")
    print(f"HR@10 (All):  {m_hr10_all:.4f}")
    print(f"HR@100 (All): {m_hr100_all:.4f}")
    print(f"HR@10 (New):  {m_hr10_new:.4f}")
    print(f"HR@100 (New): {m_hr100_new:.4f}")

    # --- SAVE TO CSV ---
    import csv
    from datetime import datetime
    
    csv_file = 'benchmark_results.csv'
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        # Write header if file is new
        if not file_exists:
            writer.writerow(['Timestamp', 'Model_Name', 'Dataset', 'HR@10_All', 'HR@100_All', 'HR@10_New', 'HR@100_New'])
        
        # Get model filename for reference
        model_filename = os.path.basename(model_path)
        
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_filename,
            dataset_name,
            f"{m_hr10_all:.4f}",
            f"{m_hr100_all:.4f}",
            f"{m_hr10_new:.4f}",
            f"{m_hr100_new:.4f}"
        ])
    print(f"✅ Saved results to {csv_file}")

if __name__ == "__main__":
    # Evaluate ITEMS
    # Check if the FOLDER exists (not a specific file)
    if os.path.exists('clothing_items_train_item_model'): 
        # Find all files in that folder
        files = os.listdir('clothing_items_train_item_model')
        # Filter for .pth files (model checkpoints)
        pth_files = [f for f in files if f.endswith('.pth')]
        
        if pth_files:
            # Sort them and pick the last one (usually the highest epoch)
            latest = sorted(pth_files)[-1]
            print(f"Found latest model: {latest}")
            
            # Run evaluation using this latest file
            evaluate_model(f'clothing_items_train_item_model/{latest}', 'clothing_items_train', 'data/test_data_items.pkl')
        else:
            print("Folder exists, but no .pth model files found inside.")
    else:
        print("Could not find the 'item_model' folder.")
        
    # Evaluate GROUPS
    if os.path.exists('clothing_groups_train_group_model'): # Check your specific epoch file
        files = os.listdir('clothing_groups_train_group_model')
        pth_files = [f for f in files if f.endswith('.pth')]
        if pth_files:
            latest = sorted(pth_files)[-1]
            evaluate_model(f'clothing_groups_train_group_model/{latest}', 'clothing_groups_train', 'data/test_data_groups.pkl')