import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MultiLabelBinarizer

def process_tags():
    # 1. Config
    OUTFITS_CSV = 'outfits.csv'
    # Change these lines in process_tags.py
    SPLIT = "data_70_30" # or "data_loo"
    MAP_PATH = f'data/{SPLIT}/item_maps.pkl'
    OUTPUT_PATH = f'data/{SPLIT}/pretrained_tag_emb.npy'

    print("Loading data...")
    if not os.path.exists(MAP_PATH):
        print("Error: Map file not found. Run prepare_data.py first.")
        return

    # Load Item Map
    with open(MAP_PATH, 'rb') as f:
        maps = pickle.load(f)
        item_map = maps[1] # item_map
        
    # Load Outfits
    try:
        df = pd.read_csv(OUTFITS_CSV, sep=';')
    except:
        print("Warning: Could not read with ';', trying ','...")
        df = pd.read_csv(OUTFITS_CSV, sep=',')

    # 2. Parse Helper Function
    def parse_list_string(x):
        if pd.isna(x): return []
        # Remove brackets and quotes to get clean strings
        x = str(x).replace('[', '').replace(']', '').replace("'", "").replace('"', "")
        # Split by comma and strip whitespace
        tokens = [t.strip() for t in x.split(',')]
        # Filter out empty strings just in case
        return [t for t in tokens if t]

    print("Parsing Tags and Categories...")
    # Apply parsing to both columns
    # Make sure these column names match your CSV exactly
    df['parsed_tags'] = df['outfit_tags'].apply(parse_list_string)
    df['parsed_cats'] = df['tag_categories'].apply(parse_list_string)
    
    # 3. Fuse Tags with Categories (e.g., "Color_Black")
    def fuse_tags(row):
        tags = row['parsed_tags']
        cats = row['parsed_cats']
        
        fused_list = []
        # Zip them together. If lengths mismatch, it stops at the shorter one.
        for tag, cat in zip(tags, cats):
            # Create a combined token
            token = f"{cat}_{tag}"
            fused_list.append(token)
            
        return fused_list

    df['fused_tokens'] = df.apply(fuse_tags, axis=1)
    
    # Debug: Show an example
    print("\nSample Fused Tags:")
    print(df[['outfit_tags', 'fused_tokens']].head(1))

    # 4. Align with Model IDs
    num_items = len(item_map)
    aligned_tags = [[] for _ in range(num_items + 1)] # Index 0 is padding
    
    # specific logic: Outfit ID -> List of Fused Tags
    id_to_tags = dict(zip(df['id'], df['fused_tokens']))
    
    found_count = 0
    for item_str, item_int in item_map.items():
        if item_str in id_to_tags:
            aligned_tags[item_int] = id_to_tags[item_str]
            found_count += 1
            
    print(f"\nFound tags for {found_count} out of {num_items} items.")

    # 5. Multi-Hot Encoding
    print("Binarizing tags...")
    mlb = MultiLabelBinarizer()
    
    # This creates a matrix of shape (num_items+1, num_unique_fused_tokens)
    tag_matrix = mlb.fit_transform(aligned_tags)
    
    print(f"Detected {len(mlb.classes_)} unique Category_Tag combinations.")
    print(f"Matrix Shape: {tag_matrix.shape}")
    
    # 6. Save
    np.save(OUTPUT_PATH, tag_matrix.astype(np.float32))
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    process_tags()