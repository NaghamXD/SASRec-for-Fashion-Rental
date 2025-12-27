import numpy as np
import pandas as pd
import pickle
import os

def process_group_features():
    print("--- Generaring Group Embeddings ---")
    
    # 1. Load Maps
    # We need both maps to link Group_ID -> [Item_IDs]
    try:
        with open('data/item_maps.pkl', 'rb') as f:
            item_maps = pickle.load(f)
            item_map = item_maps[1] # Item String -> Item Int
            
        with open('data/group_maps.pkl', 'rb') as f:
            group_maps = pickle.load(f)
            group_map = group_maps[1] # Group String -> Group Int
    except FileNotFoundError:
        print("Error: Map files not found.")
        return

    # 2. Load Outfits to link Items to Groups
    df = pd.read_csv('outfits.csv', sep=';')
    # Create dictionary: Group_String -> List of Item_Strings
    group_to_items_str = df.groupby('group')['id'].apply(list).to_dict()

    # 3. Load Existing ITEM Matrices
    try:
        item_img_matrix = np.load('data/pretrained_item_emb_1280.npy')
        print(f"Loaded Item Images: {item_img_matrix.shape}")
    except:
        item_img_matrix = None
        
    try:
        item_tag_matrix = np.load('data/pretrained_tag_emb.npy')
        print(f"Loaded Item Tags: {item_tag_matrix.shape}")
    except:
        item_tag_matrix = None

    # 4. Initialize GROUP Matrices
    num_groups = len(group_map)
    
    # Images (1280 dim)
    if item_img_matrix is not None:
        group_img_matrix = np.zeros((num_groups + 1, 1280), dtype=np.float32)
    
    # Tags (whatever dim the item tags are)
    if item_tag_matrix is not None:
        tag_dim = item_tag_matrix.shape[1]
        group_tag_matrix = np.zeros((num_groups + 1, tag_dim), dtype=np.float32)

    print(f"Processing {num_groups} groups...")
    
    # 5. Aggregate
    for grp_str, grp_int in group_map.items():
        # Find items belonging to this group
        items_in_group = group_to_items_str.get(grp_str, [])
        
        # Convert to Integer IDs (and filter out ones not in our map)
        item_ints = [item_map[i] for i in items_in_group if i in item_map]
        
        if not item_ints:
            continue
            
        # -- Process Images (Average) --
        if item_img_matrix is not None:
            # Get all vectors for these items
            vectors = item_img_matrix[item_ints]
            # Average them (Mean Pooling)
            # Axis 0 is the number of items
            avg_vector = np.mean(vectors, axis=0)
            group_img_matrix[grp_int] = avg_vector
            
        # -- Process Tags (Max/Union) --
        if item_tag_matrix is not None:
            # Get all tag vectors
            tag_vectors = item_tag_matrix[item_ints]
            # Take MAX (if any item has tag 'Red', the Group is 'Red')
            union_vector = np.max(tag_vectors, axis=0)
            group_tag_matrix[grp_int] = union_vector

    # 6. Save
    if item_img_matrix is not None:
        np.save('data/pretrained_group_emb_1280.npy', group_img_matrix)
        print("Saved data/pretrained_group_emb_1280.npy")
        
    if item_tag_matrix is not None:
        np.save('data/pretrained_group_tag_emb.npy', group_tag_matrix)
        print("Saved data/pretrained_group_tag_emb.npy")

if __name__ == "__main__":
    process_group_features()