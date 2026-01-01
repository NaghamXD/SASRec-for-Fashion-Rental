import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pandas as pd
import numpy as np
import pickle
import math

# ---------------------------------------------------------
# 1. HELPER FUNCTIONS
# ---------------------------------------------------------
def leave_percentage_out_split_with_dates(sequence, dates, percentage=0.3):
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
            pickle.dump({'test': test_data, 'dates': test_dates, 'history': history_data}, f)

if __name__ == "__main__":
    # Run 70-30 Split
    process_data(percentage=0.3, folder_name='data_70_30')
    
    # Run Leave-One-Out
    process_data(percentage=None, folder_name='data_loo')
    print("\nAll data prepared in separate folders.")