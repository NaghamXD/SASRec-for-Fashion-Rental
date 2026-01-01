import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import pandas as pd
import pickle
import sys

def process_images_from_folder():
    # ---------------------------------------------------------
    # CONFIGURATION
    # ---------------------------------------------------------
    # The CSV file linking outfits to pictures
    CSV_PATH = 'picture_triplets.csv' 
    
    # The folder containing the 50.3k .npy files
    EMBEDDINGS_DIR = 'data/embeddings/EfficientNet_V2_L_final/' 
    
    # Change these lines in process_tags.py
    SPLIT = "data_70_30" # or "data_loo"
    MAP_PATH = f'data/{SPLIT}/item_maps.pkl'
    OUTPUT_PATH = f'data/{SPLIT}/pretrained_tag_emb.npy'
    # ---------------------------------------------------------

    print("--- Step 1: Loading Mappings ---")
    if not os.path.exists(MAP_PATH):
        print(f"Error: {MAP_PATH} not found. Run prepare_data.py first.")
        return

    with open(MAP_PATH, 'rb') as f:
        maps = pickle.load(f)
        item_map = maps[1] # item_map is the second element (User, Item)
    
    num_items = len(item_map)
    print(f"Total items in model to find images for: {num_items}")

    print("\n--- Step 2: Processing CSV ---")
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    # Try reading with different separators
    try:
        df = pd.read_csv(CSV_PATH, sep=';')
    except:
        print("Warning: Failed to read with ';' separator, trying ','...")
        df = pd.read_csv(CSV_PATH, sep=',')

    # Clean column names (remove spaces)
    df.columns = [c.strip() for c in df.columns]
    
    # Verify headers exist
    required_cols = ['outfit.id', 'picture.id', 'displayOrder']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV is missing one of these columns: {required_cols}")
        print(f"Found columns: {df.columns}")
        return

    # Sort by outfit and displayOrder so the best image (0) is first
    df = df.sort_values(by=['outfit.id', 'displayOrder'])
    
    # Keep only the FIRST image for each outfit
    df_best = df.drop_duplicates(subset=['outfit.id'], keep='first')
    
    # Create a dictionary for fast lookup: Outfit ID -> Picture ID
    outfit_to_pic = dict(zip(df_best['outfit.id'], df_best['picture.id']))
    print(f"Found image mappings for {len(outfit_to_pic)} unique outfits.")

    print("\n--- Step 3: detecting Dimension ---")
    # Check one file to detect if it is 1280 or something else
    first_pic_id = df_best.iloc[0]['picture.id']
    sample_path = os.path.join(EMBEDDINGS_DIR, f"{first_pic_id}.npy")
    
    if not os.path.exists(sample_path):
        # If first one missing, try to find ANY valid file
        found = False
        for pid in df_best['picture.id']:
            p = os.path.join(EMBEDDINGS_DIR, f"{pid}.npy")
            if os.path.exists(p):
                sample_path = p
                found = True
                break
        if not found:
            print(f"Error: Could not find any .npy files in '{EMBEDDINGS_DIR}'. Check folder name.")
            return

    try:
        sample_emb = np.load(sample_path)
        embed_dim = sample_emb.shape[0]
        print(f"Detected embedding dimension: {embed_dim}")
    except Exception as e:
        print(f"Error loading sample file: {e}")
        return

    print("\n--- Step 4: Building Matrix ---")
    # Initialize Matrix with Zeros
    # Shape: [num_items + 1, embed_dim] (Index 0 is padding)
    emb_matrix = np.zeros((num_items + 1, embed_dim), dtype=np.float32)
    
    success_count = 0
    missing_count = 0
    
    # Iterate through the ITEM MAP (the model's items)
    for item_str, item_int in item_map.items():
        # 1. Do we have a picture ID for this item?
        if item_str in outfit_to_pic:
            pic_id = outfit_to_pic[item_str]
            file_name = f"{pic_id}.npy"
            file_path = os.path.join(EMBEDDINGS_DIR, file_name)
            
            # 2. Does the .npy file exist?
            if os.path.exists(file_path):
                try:
                    emb = np.load(file_path)
                    if emb.shape[0] == embed_dim:
                        emb_matrix[item_int] = emb
                        success_count += 1
                    else:
                        # Handle size mismatch if any
                        pass
                except:
                    pass
            else:
                missing_count += 1
        else:
            missing_count += 1
            
        # Progress Bar
        if (success_count + missing_count) % 1000 == 0:
            sys.stdout.write(f"\rProcessed: {success_count + missing_count}/{num_items}")
            sys.stdout.flush()

    print(f"\n\nDone!")
    print(f"Successfully loaded images: {success_count}")
    print(f"Missing images (zeros): {missing_count}")
    
    # I added also filling missing embeddings with the mean of existing ones
    final_matrix = emb_matrix.copy()
    # Calculate the mean of only the rows we successfully filled (skip index 0 and zeros)
    mask = np.any(final_matrix != 0, axis=1)
    mean_vec = final_matrix[mask].mean(axis=0)

    # Fill the missing items (where the row is all zeros) with the mean
    missing_mask = ~mask
    # Don't fill index 0 (padding)
    missing_mask[0] = False 
    final_matrix[missing_mask] = mean_vec

    np.save(OUTPUT_PATH, final_matrix)
    # Save
    #np.save(OUTPUT_PATH, emb_matrix)
    print(f"Saved matrix to: {OUTPUT_PATH}")
    print(f"Matrix Shape: {final_matrix.shape}")

if __name__ == "__main__":
    process_images_from_folder()