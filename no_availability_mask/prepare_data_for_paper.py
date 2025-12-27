import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import pandas as pd
import numpy as np
import pickle
import math

# ---------------------------------------------------------
# 1. HELPER FUNCTIONS (From Paper)
# ---------------------------------------------------------

def remove_consecutive_duplicates(df, date_col="rentalPeriod.start"):
    """
    Paper's logic: Removes repeats of the same item/group if they 
    occur within 30 days of the previous rental.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    # Identify item column (could be 'outfit.id' or 'group.id')
    item_col = 'outfit.id' if 'outfit.id' in df.columns else 'group.id'
    
    drop_indexes = []
    
    # Sort just in case
    df = df.sort_values(by=['customer.id', date_col])

    for i, (customer_id, group) in enumerate(df.groupby('customer.id')):
        # Find ids that appear more than once for this user
        counts = group[item_col].value_counts()
        repeated_ids = counts[counts > 1].index
        
        for rep_id in repeated_ids:
            # Get all instances of this item for this user
            instances = group[group[item_col] == rep_id]
            
            # Check time gaps
            prev_idx = 0
            for k in range(1, len(instances)):
                curr_time = instances.iloc[k][date_col]
                prev_time = instances.iloc[prev_idx][date_col]
                
                if (curr_time - prev_time).days < 30:
                    # Too close, mark for deletion
                    drop_indexes.append(instances.index[k])
                else:
                    # Valid gap, this becomes the new baseline
                    prev_idx = k
                    
    print(f"Removing {len(drop_indexes)} duplicates based on 30-day rule.")
    return df.drop(drop_indexes)

def leave_percentage_out_split(sequence, percentage=0.3):
    """
    Splits a list into Training (first 70%) and Test (last 30%).
    """
    seq_len = len(sequence)
    if seq_len < 2:
        return sequence, [] # Not enough data to split
        
    num_test = max(math.floor(seq_len * percentage), 1)
    num_train = seq_len - num_test
    
    return sequence[:num_train], sequence[-num_test:]

# ---------------------------------------------------------
# 2. MAIN PROCESSING
# ---------------------------------------------------------

def process_data():
    print("Loading data...")
    # Load Interactions
    df_orders = pd.read_csv('original_orders.csv', sep=';')
    df_triplets = pd.read_csv('user_activity_triplets.csv', sep=';')
    
    cols = ['customer.id', 'outfit.id', 'rentalPeriod.start']
    full_df = pd.concat([df_orders[cols], df_triplets[cols]])
    full_df['rentalPeriod.start'] = pd.to_datetime(full_df['rentalPeriod.start'])
    
    # Load Outfits (for Group mapping)
    print("Loading outfits map...")
    df_outfits = pd.read_csv('outfits.csv', sep=';')
    # Create Dictionary: outfit.id -> group.id
    item_to_group = dict(zip(df_outfits['id'], df_outfits['group']))

    # -----------------------------------------------------
    # MODE A: INDIVIDUAL ITEMS
    # -----------------------------------------------------
    print("\n--- Processing ITEMS ---")
    df_items = full_df.copy()
    
    # 1. Clean Duplicates (Paper Logic)
    df_items = remove_consecutive_duplicates(df_items)
    df_items = df_items.sort_values(by=['customer.id', 'rentalPeriod.start'])
    
    # 2. Convert to Integers
    unique_users = df_items['customer.id'].unique()
    unique_items = df_items['outfit.id'].unique()
    
    user_map = {u: i+1 for i, u in enumerate(unique_users)}
    item_map = {item: i+1 for i, item in enumerate(unique_items)}
    
    # Save maps for later analysis
    with open('data/item_maps.pkl', 'wb') as f:
        pickle.dump((user_map, item_map), f)

    # 3. Create Splits
    train_file_lines = []
    test_data = {} # user_id -> [list of test items]
    history_data = {} # user_id -> [list of train items] (for 'Unique' check)

    print("Generating 70/30 Splits for Items...")
    grouped = df_items.groupby('customer.id')['outfit.id'].apply(list)
    
    for user_str, item_list in grouped.items():
        if user_str not in user_map: continue
        u_id = user_map[user_str]
        
        # Map string items to integers
        seq_ints = [item_map[i] for i in item_list if i in item_map]
        
        # Split
        train_seq, test_seq = leave_percentage_out_split(seq_ints, percentage=0.3)
        
        if len(train_seq) > 0:
            # Add to Training File
            for item in train_seq:
                train_file_lines.append(f"{u_id} {item}")
            
            # Store Test Data
            if len(test_seq) > 0:
                test_data[u_id] = test_seq
            
            history_data[u_id] = set(train_seq)

    # 4. Save Item Files
    if not os.path.exists('data'): os.makedirs('data')
    
    with open('data/clothing_items_train.txt', 'w') as f:
        f.write('\n'.join(train_file_lines))
        
    with open('data/test_data_items.pkl', 'wb') as f:
        pickle.dump({'test': test_data, 'history': history_data}, f)

    print(f"Saved Items Train: {len(train_file_lines)} lines")
    print(f"Saved Items Test Users: {len(test_data)}")

    # -----------------------------------------------------
    # MODE B: GROUPS
    # -----------------------------------------------------
    print("\n--- Processing GROUPS ---")
    df_groups = full_df.copy()
    
    # 1. Map Outfits to Groups
    # If an outfit is not in outfits.csv, we drop it or keep raw ID? 
    # Usually drop or assume singleton. Let's map and drop NaNs.
    df_groups['group.id'] = df_groups['outfit.id'].map(item_to_group)
    df_groups = df_groups.dropna(subset=['group.id'])
    
    # 2. Clean Duplicates (Paper Logic - applied to GROUPS now)
    # Important: Re-run cleaning because Item A and Item B might be in same Group G.
    # We don't want Sequence: G, G.
    df_groups = remove_consecutive_duplicates(df_groups)
    df_groups = df_groups.sort_values(by=['customer.id', 'rentalPeriod.start'])
    
    # 3. Convert to Integers
    unique_groups = df_groups['group.id'].unique()
    group_map = {g: i+1 for i, g in enumerate(unique_groups)}
    
    # Reuse User Map from Items to keep User IDs consistent? 
    # Yes, usually better. But some users might disappear if they only rented invalid groups.
    # Let's strictly stick to the users present here.
    group_user_map = {u: i+1 for i, u in enumerate(df_groups['customer.id'].unique())}

    with open('data/group_maps.pkl', 'wb') as f:
        pickle.dump((group_user_map, group_map), f)

    # 4. Create Splits
    train_file_lines = []
    test_data = {}
    history_data = {}

    print("Generating 70/30 Splits for Groups...")
    grouped = df_groups.groupby('customer.id')['group.id'].apply(list)
    
    for user_str, group_list in grouped.items():
        if user_str not in group_user_map: continue
        u_id = group_user_map[user_str]
        
        # Map string groups to integers
        seq_ints = [group_map[g] for g in group_list if g in group_map]
        
        # Split
        train_seq, test_seq = leave_percentage_out_split(seq_ints, percentage=0.3)
        
        if len(train_seq) > 0:
            for item in train_seq:
                train_file_lines.append(f"{u_id} {item}")
            
            if len(test_seq) > 0:
                test_data[u_id] = test_seq
            
            history_data[u_id] = set(train_seq)

    # 5. Save Group Files
    with open('data/clothing_groups_train.txt', 'w') as f:
        f.write('\n'.join(train_file_lines))
        
    with open('data/test_data_groups.pkl', 'wb') as f:
        pickle.dump({'test': test_data, 'history': history_data}, f)
        
    print(f"Saved Groups Train: {len(train_file_lines)} lines")
    print(f"Saved Groups Test Users: {len(test_data)}")
    print("Done!")

if __name__ == "__main__":
    process_data()