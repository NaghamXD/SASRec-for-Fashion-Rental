import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pandas as pd
import numpy as np
import pickle
import math

# ---------------------------------------------------------
# 1. HELPER FUNCTIONS
# ---------------------------------------------------------
def remove_consecutive_duplicates(df, date_col="rentalPeriod.start"):
    df[date_col] = pd.to_datetime(df[date_col])
    item_col = 'outfit.id' if 'outfit.id' in df.columns else 'group.id'
    
    drop_indexes = []
    df = df.sort_values(by=['customer.id', date_col])

    for i, (customer_id, group) in enumerate(df.groupby('customer.id')):
        counts = group[item_col].value_counts()
        repeated_ids = counts[counts > 1].index
        
        for rep_id in repeated_ids:
            instances = group[group[item_col] == rep_id]
            prev_idx = 0
            for k in range(1, len(instances)):
                curr_time = instances.iloc[k][date_col]
                prev_time = instances.iloc[prev_idx][date_col]
                
                if (curr_time - prev_time).days < 30:
                    drop_indexes.append(instances.index[k])
                else:
                    prev_idx = k
    print(f"Removing {len(drop_indexes)} duplicates based on 30-day rule.")
    return df.drop(drop_indexes)

def leave_percentage_out_split_with_dates(sequence, dates, percentage=0.3):
    """Splits both items and their dates."""
    seq_len = len(sequence)
    if seq_len < 2:
        return sequence, [], dates, []
        
    num_test = max(math.floor(seq_len * percentage), 1)
    num_train = seq_len - num_test
    
    return sequence[:num_train], sequence[-num_test:], dates[:num_train], dates[-num_test:]

# ---------------------------------------------------------
# 2. MAIN PROCESSING
# ---------------------------------------------------------
def process_data():
    print("Loading data...")
    df_orders = pd.read_csv('original_orders.csv', sep=';')
    df_triplets = pd.read_csv('user_activity_triplets.csv', sep=';')
    
    cols = ['customer.id', 'outfit.id', 'rentalPeriod.start']
    full_df = pd.concat([df_orders[cols], df_triplets[cols]])
    full_df['rentalPeriod.start'] = pd.to_datetime(full_df['rentalPeriod.start'])
    
    print("Loading outfits map...")
    df_outfits = pd.read_csv('outfits.csv', sep=';')
    item_to_group = dict(zip(df_outfits['id'], df_outfits['group']))

    # =====================================================
    # MODE A: INDIVIDUAL ITEMS
    # =====================================================
    print("\n--- Processing ITEMS ---")
    
    # --- CHANGE: Build Map from MASTER LIST (outfits.csv), not history ---
    unique_items = df_outfits['id'].unique()
    # Note: Users still come from history because we don't have a 'users.csv' master list usually
    unique_users = full_df['customer.id'].unique()
    
    user_map = {u: i+1 for i, u in enumerate(unique_users)}
    item_map = {item: i+1 for i, item in enumerate(unique_items)}
    
    print(f"Total Unique Items in Catalogue: {len(item_map)}")
    print(f"Total Unique Users in History:   {len(user_map)}")
    
    with open('data/item_maps.pkl', 'wb') as f:
        pickle.dump((user_map, item_map), f)
    # ---------------------------------------------------------------------

    df_items = full_df.copy()
    df_items = remove_consecutive_duplicates(df_items)
    df_items = df_items.sort_values(by=['customer.id', 'rentalPeriod.start'])

    train_file_lines = []
    test_data = {} 
    test_dates = {} 
    history_data = {}

    print("Generating 70/30 Splits for Items...")
    # Group both ID and Date
    grouped = df_items.groupby('customer.id')[['outfit.id', 'rentalPeriod.start']].apply(
        lambda x: list(zip(x['outfit.id'], x['rentalPeriod.start']))
    )
    
    for user_str, pairs in grouped.items():
        if user_str not in user_map: continue
        u_id = user_map[user_str]
        
        # Unzip pairs
        raw_items, raw_dates = zip(*pairs)
        
        # Map IDs
        seq_ints = [item_map[i] for i in raw_items if i in item_map]
        valid_dates = [d for i, d in zip(raw_items, raw_dates) if i in item_map]

        if len(seq_ints) != len(valid_dates): continue 

        # Split with Dates
        train_seq, test_seq, train_d, test_d = leave_percentage_out_split_with_dates(seq_ints, valid_dates, percentage=0.3)
        
        if len(train_seq) > 0:
            for item in train_seq:
                train_file_lines.append(f"{u_id} {item}")
            
            if len(test_seq) > 0:
                test_data[u_id] = test_seq
                test_dates[u_id] = test_d 
            
            history_data[u_id] = set(train_seq)

    if not os.path.exists('data'): os.makedirs('data')
    with open('data/clothing_items_train.txt', 'w') as f:
        f.write('\n'.join(train_file_lines))
        
    with open('data/test_data_items.pkl', 'wb') as f:
        pickle.dump({'test': test_data, 'dates': test_dates, 'history': history_data}, f)
        
    print(f"Saved Items Train: {len(train_file_lines)} lines")

    # =====================================================
    # MODE B: GROUPS
    # =====================================================
    print("\n--- Processing GROUPS ---")
    df_groups = full_df.copy()
    
    # 1. Map Outfits to Groups
    df_groups['group.id'] = df_groups['outfit.id'].map(item_to_group)
    df_groups = df_groups.dropna(subset=['group.id'])
    
    # 2. Clean Duplicates
    df_groups = remove_consecutive_duplicates(df_groups)
    df_groups = df_groups.sort_values(by=['customer.id', 'rentalPeriod.start'])
    
    # --- CHANGE: Build Map from MASTER LIST (outfits.csv) ---
    # This ensures EVERY group in the catalogue gets an ID, even if never rented
    unique_groups = df_outfits['group'].unique()
    group_map = {g: i+1 for i, g in enumerate(unique_groups)}
    
    # User map can stay history-based
    group_user_map = {u: i+1 for i, u in enumerate(df_groups['customer.id'].unique())}
    
    print(f"Total Unique Groups in Catalogue: {len(group_map)}") 
    
    with open('data/group_maps.pkl', 'wb') as f:
        pickle.dump((group_user_map, group_map), f)
    # --------------------------------------------------------

    # 4. Create Splits
    train_file_lines = []
    test_data = {}
    test_dates = {}
    history_data = {}

    print("Generating 70/30 Splits for Groups...")
    # Group both ID and Date (We keep dates for consistency, even if not strictly used for masking yet)
    grouped = df_groups.groupby('customer.id')[['group.id', 'rentalPeriod.start']].apply(
        lambda x: list(zip(x['group.id'], x['rentalPeriod.start']))
    )
    
    for user_str, pairs in grouped.items():
        if user_str not in group_user_map: continue
        u_id = group_user_map[user_str]
        
        raw_items, raw_dates = zip(*pairs)
        
        seq_ints = [group_map[g] for g in raw_items if g in group_map]
        valid_dates = [d for i, d in zip(raw_items, raw_dates) if i in group_map]
        
        # Split
        train_seq, test_seq, train_d, test_d = leave_percentage_out_split_with_dates(seq_ints, valid_dates, percentage=0.3)
        
        if len(train_seq) > 0:
            for item in train_seq:
                train_file_lines.append(f"{u_id} {item}")
            
            if len(test_seq) > 0:
                test_data[u_id] = test_seq
                test_dates[u_id] = test_d
            
            history_data[u_id] = set(train_seq)

    # 5. Save Group Files
    with open('data/clothing_groups_train.txt', 'w') as f:
        f.write('\n'.join(train_file_lines))
        
    with open('data/test_data_groups.pkl', 'wb') as f:
        pickle.dump({'test': test_data, 'dates': test_dates, 'history': history_data}, f)
        
    print(f"Saved Groups Train: {len(train_file_lines)} lines")
    print("Done!")

if __name__ == "__main__":
    process_data()