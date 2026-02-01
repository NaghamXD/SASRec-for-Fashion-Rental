import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pandas as pd
import numpy as np
import pickle
import math

# ---------------------------------------------------------
# 1. HELPER FUNCTIONS
# ---------------------------------------------------------
'''def leave_percentage_out_split_with_dates(sequence, dates, percentage=0.3):
    """
    If percentage is None: Leave-One-Out (last item for test).
    If percentage is float: Percentage split (e.g. 0.3).
    """
    seq_len = len(sequence)
    if seq_len < 2:
        return sequence, [], dates, []
        
    if percentage is None:
        num_test = 1 #
    else:
        num_test = max(math.floor(seq_len * percentage), 1) #
    
    num_train = seq_len - num_test
    return sequence[:num_train], sequence[-num_test:], dates[:num_train], dates[-num_test:]
'''
#new version to include validation set
def leave_percentage_out_split_with_dates(sequence, dates, test_percentage=0.3, val_percentage=0.1):
    """
    splits sequence into Train -> Valid -> Test chronologically.
    
    Args:
        test_percentage: 
            - If None: treated as Leave-One-Out (1 item for test).
            - If float: percentage of sequence for test.
        val_percentage:
            - If None: treated as Leave-One-Out (1 item for validation).
            - If float: percentage of sequence for validation.
    """
    seq_len = len(sequence)
    
    # 1. Determine Test Size
    if test_percentage is None:
        num_test = 1
    else:
        num_test = max(math.floor(seq_len * test_percentage), 1)
        
    # 2. Determine Validation Size
    if val_percentage is None:
        num_valid = 1
    else:
        # If doing % split, take valid from the remaining part, or just a fixed % of total
        num_valid = max(math.floor(seq_len * val_percentage), 1)

    # Safety check: ensure we have enough data for all splits
    # We need at least 1 item for Train + 1 for Valid + 1 for Test
    if seq_len < 3: 
        # Not enough data for a proper 3-way split
        # Fallback: Just return what we can (e.g., empty train/valid)
        # This depends on how aggressive you want to be filtering short sequences
        return [], [], [], [], [], [] 

    # 3. Calculate Indices
    # [ --- TRAIN --- | --- VALID --- | --- TEST --- ]
    #                 ^               ^
    #             idx_val          idx_test
    
    idx_test_start = seq_len - num_test
    idx_val_start = idx_test_start - num_valid
    
    # Ensure validation doesn't eat into index 0 (leave at least 1 item for train)
    if idx_val_start < 1:
        idx_val_start = 1
        
    # Slicing
    train_seq = sequence[:idx_val_start]
    valid_seq = sequence[idx_val_start:idx_test_start]
    test_seq  = sequence[idx_test_start:]
    
    train_dates = dates[:idx_val_start]
    valid_dates = dates[idx_val_start:idx_test_start]
    test_dates  = dates[idx_test_start:]
    
    return train_seq, valid_seq, test_seq, train_dates, valid_dates, test_dates

def remove_consecutive_duplicates(df, date_col="rentalPeriod.start"):
    df[date_col] = pd.to_datetime(df[date_col])
    item_col = 'outfit.id' if 'outfit.id' in df.columns else 'group.id'
    df = df.sort_values(by=['customer.id', date_col])
    
    drop_indexes = []
    for i, (customer_id, group) in enumerate(df.groupby('customer.id')):
        counts = group[item_col].value_counts()
        repeated_ids = counts[counts > 1].index
        for rep_id in repeated_ids:
            instances = group[group[item_col] == rep_id]
            prev_idx = 0
            for k in range(1, len(instances)):
                if (instances.iloc[k][date_col] - instances.iloc[prev_idx][date_col]).days < 30:
                    drop_indexes.append(instances.index[k])
                else:
                    prev_idx = k
    return df.drop(drop_indexes)

# ---------------------------------------------------------
# 2. MAIN PROCESSING
# ---------------------------------------------------------
'''def process_data(percentage=0.3, folder_name='data_70_30'):
    # Create the specific sub-folder
    target_dir = os.path.join('data', folder_name)
    if not os.path.exists(target_dir): 
        os.makedirs(target_dir)
        print(f"Created directory: {target_dir}")
        
    df_orders = pd.read_csv('original_orders.csv', sep=';')
    df_triplets = pd.read_csv('user_activity_triplets.csv', sep=';')
    df_outfits = pd.read_csv('outfits.csv', sep=';')
    item_to_group = dict(zip(df_outfits['id'], df_outfits['group']))
    
    full_df = pd.concat([df_orders, df_triplets])
    full_df['rentalPeriod.start'] = pd.to_datetime(full_df['rentalPeriod.start'])

    # Map Generation (Uses master catalog)
    user_map = {u: i+1 for i, u in enumerate(full_df['customer.id'].unique())}
    item_map = {item: i+1 for i, item in enumerate(df_outfits['id'].unique())}
    group_map = {g: i+1 for i, g in enumerate(df_outfits['group'].unique())}

    # Save maps in the specific folder
    with open(os.path.join(target_dir, 'item_maps.pkl'), 'wb') as f:
        pickle.dump((user_map, item_map), f)
    with open(os.path.join(target_dir, 'group_maps.pkl'), 'wb') as f:
        pickle.dump((user_map, group_map), f)

    # Process both Items and Groups
    for mode in ['items', 'groups']:
        print(f"\n--- Processing {mode.upper()} ({folder_name}) ---")
        df = full_df.copy()
        
        if mode == 'groups':
            df['target_id'] = df['outfit.id'].map(item_to_group)
            df = df.dropna(subset=['target_id'])
            current_map = group_map
        else:
            df['target_id'] = df['outfit.id']
            current_map = item_map

        df = remove_consecutive_duplicates(df)
        
        train_lines, test_data, test_dates, history_data = [], {}, {}, {}
        grouped = df.groupby('customer.id')[['target_id', 'rentalPeriod.start']].apply(
            lambda x: list(zip(x['target_id'], x['rentalPeriod.start']))
        )

        for user_str, pairs in grouped.items():
            u_id = user_map[user_str]
            seq_ints = [current_map[i] for i, d in pairs if i in current_map]
            dates = [d for i, d in pairs if i in current_map]

            train_seq, test_seq, train_d, test_d = leave_percentage_out_split_with_dates(
                seq_ints, dates, percentage=percentage
            )

            if len(train_seq) > 0:
                for item in train_seq:
                    train_lines.append(f"{u_id} {item}")
                test_data[u_id] = test_seq
                test_dates[u_id] = test_d
                history_data[u_id] = set(train_seq)

        # Save files to sub-folder
        with open(os.path.join(target_dir, f'clothing_{mode}_train.txt'), 'w') as f:
            f.write('\n'.join(train_lines))
        with open(os.path.join(target_dir, f'test_data_{mode}.pkl'), 'wb') as f:
            pickle.dump({'test': test_data, 'dates': test_dates, 'history': history_data}, f)'''

def process_data(percentage=0.3, folder_name='data_70_30'):
    # Create the specific sub-folder
    target_dir = os.path.join('data', folder_name)
    if not os.path.exists(target_dir): 
        os.makedirs(target_dir)
        print(f"Created directory: {target_dir}")
        
    df_orders = pd.read_csv('original_orders.csv', sep=';')
    df_triplets = pd.read_csv('user_activity_triplets.csv', sep=';')
    df_outfits = pd.read_csv('outfits.csv', sep=';')
    item_to_group = dict(zip(df_outfits['id'], df_outfits['group']))
    
    full_df = pd.concat([df_orders, df_triplets])
    full_df['rentalPeriod.start'] = pd.to_datetime(full_df['rentalPeriod.start'])

    # Map Generation (Uses master catalog)
    user_map = {u: i+1 for i, u in enumerate(full_df['customer.id'].unique())}
    item_map = {item: i+1 for i, item in enumerate(df_outfits['id'].unique())}
    group_map = {g: i+1 for i, g in enumerate(df_outfits['group'].unique())}

    # Save maps in the specific folder
    with open(os.path.join(target_dir, 'item_maps.pkl'), 'wb') as f:
        pickle.dump((user_map, item_map), f)
    with open(os.path.join(target_dir, 'group_maps.pkl'), 'wb') as f:
        pickle.dump((user_map, group_map), f)

    # Process both Items and Groups
    for mode in ['items', 'groups']:
        print(f"\n--- Processing {mode.upper()} ({folder_name}) ---")
        df = full_df.copy()
        
        if mode == 'groups':
            df['target_id'] = df['outfit.id'].map(item_to_group)
            df = df.dropna(subset=['target_id'])
            current_map = group_map
        else:
            df['target_id'] = df['outfit.id']
            current_map = item_map

        df = remove_consecutive_duplicates(df)
        
        grouped = df.groupby('customer.id')[['target_id', 'rentalPeriod.start']].apply(
            lambda x: list(zip(x['target_id'], x['rentalPeriod.start']))
        )

        # Initialize containers including validation
        train_lines = []
        test_data, test_dates = {}, {}
        valid_data, valid_dates = {}, {} # New dictionaries
        history_data = {} 

        for user_str, pairs in grouped.items():
            u_id = user_map[user_str]
            seq_ints = [current_map[i] for i, d in pairs if i in current_map]
            dates = [d for i, d in pairs if i in current_map]

            # DETERMINE SPLIT CONFIG BASED ON MODE
            if percentage is None:
                # LOO Mode: 1 for valid, 1 for test
                p_test = None
                p_val = None 
            else:
                # 70/30 Mode: 30% test, 10% valid (resulting in 60% train)
                p_test = percentage
                p_val = 0.1 

            tr_s, va_s, te_s, tr_d, va_d, te_d = leave_percentage_out_split_with_dates(
                seq_ints, dates, test_percentage=p_test, val_percentage=p_val
            )

            # Only proceed if we actually got a valid split
            if len(tr_s) > 0:
                # Write Train to TXT
                for item in tr_s:
                    train_lines.append(f"{u_id} {item}")
                
                # Save Test Sequences
                test_data[u_id] = te_s
                test_dates[u_id] = te_d
                
                # Save Valid Sequences
                valid_data[u_id] = va_s
                valid_dates[u_id] = va_d
                
                # History now includes Train (AND optionally Valid, depending on evaluation logic)
                # Usually history for testing is Train + Valid
                history_data[u_id] = set(tr_s) 

        # Save files to sub-folder
        with open(os.path.join(target_dir, f'clothing_{mode}_train.txt'), 'w') as f:
            f.write('\n'.join(train_lines))
            
        # Updated Pickle Dump
        with open(os.path.join(target_dir, f'test_data_{mode}.pkl'), 'wb') as f:
            pickle.dump({
                'test': test_data, 
                'test_dates': test_dates, 
                'valid': valid_data,     # Added validation
                'valid_dates': valid_dates, 
                'history': history_data
            }, f)

if __name__ == "__main__":
    # Run 70-30 Split
    process_data(percentage=0.3, folder_name='data_70_30')
    
    # Run Leave-One-Out
    process_data(percentage=None, folder_name='data_loo')
    print("\nAll data prepared in separate folders.")