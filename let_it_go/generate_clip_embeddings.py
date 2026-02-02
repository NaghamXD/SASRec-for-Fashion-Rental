import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import pickle
import ast
# ==========================================
# CONFIGURATION
# ==========================================
# 1. Path to your images folder
IMAGE_FOLDER = "data/images" 

# 2. Path to your metadata file (CSV or TXT)
# It must have columns that link Item ID -> Image Filename -> Tags
# 2. Path to your metadata files
# Ensure these point to the correct files you uploaded
PICTURES_PATH = 'picture_triplets.csv'
OUTFITS_PATH = "outfits.csv" 
MAP_PATH = 'data/data_70_30/item_maps.pkl'

# 3. Model Choice
MODEL_ID = "patrickjohncyh/fashion-clip" #Use this for fashion-specific

# ==========================================
# 1. SETUP MODEL
# ==========================================
print(f"Loading CLIP model: {MODEL_ID}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = "mps"

model = CLIPModel.from_pretrained(MODEL_ID).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_ID)
print(f"Model loaded on {device}")

# ==========================================
# 2. LOAD DATA
# ==========================================
def load_data():
    """
    Loads mapping from Item_ID (SASRec Int) -> Image Path & Description.
    """
    print("Reading CSVs and Mapping files...")

    # A. Check if files exist
    if not os.path.exists(MAP_PATH):
        raise FileNotFoundError(f"Could not find {MAP_PATH}. You need this to map UUIDs to Integers.")
    
    # B. Load the SASRec Integer Map
    with open(MAP_PATH, 'rb') as f:
        # Usually returns (user_map, item_map)
        _, item_map = pickle.load(f) 
        # item_map format: {'outfit.fff...': 1, 'outfit.abc...': 2}

    # C. Load Dataframes
    df_outfits = pd.read_csv(OUTFITS_PATH, sep=';')
    df_pics = pd.read_csv(PICTURES_PATH, sep=';')

    # D. Merge Pictures with Outfits
    # Strip whitespace just in case
    df_pics['outfit.id'] = df_pics['outfit.id'].str.strip()
    df_outfits['id'] = df_outfits['id'].str.strip()

    # Drop duplicates to ensure one image per outfit (picking the first one)
    df_pics = df_pics.sort_values('displayOrder').drop_duplicates(subset=['outfit.id'])
    full_data = pd.merge(df_outfits, df_pics, left_on='id', right_on='outfit.id', how='left')

    items_to_process = []
    
    # E. Iterate through the SASRec Map (This ensures perfect alignment)
    print(f"Processing {len(item_map)} items...")
    
    missing_count = 0
    
    for uuid, sasrec_id in item_map.items():
        # Find the row in the CSV matching this UUID
        row = full_data[full_data['id'] == uuid]
        
        if row.empty:
            continue
        
        row = row.iloc[0] # Get the series

        # --- Prepare Text (Description + Tags) ---
        desc = str(row['description'])
        
        # Clean tags: "['Tag1', 'Tag2']" -> "Tag1, Tag2"
        try:
            raw_tags = row['outfit_tags']
            if pd.notna(raw_tags):
                # parse string list to actual list
                tags_list = ast.literal_eval(raw_tags) 
                tags_str = ", ".join(tags_list)
            else:
                tags_str = ""
        except:
            tags_str = str(raw_tags)

        # Final Text for CLIP
        full_text = f"Category: {row['group']}. Tags: {tags_str}. Description: {desc}"
        full_text = full_text[:77] # Truncate to avoid CLIP errors

        # --- Prepare Image Path ---
        # RAW format from CSV: "['picture', '0a0a9183...'].jpg"
        # TARGET format: "0a0a9183....jpg"
        raw_filename = str(row['file_name'])
        
        # CLEANING LOGIC:
        # 1. Remove "['picture', '"
        # 2. Remove "'].jpg" (temporarily, to get clean hash)
        # 3. Remove "']" (if .jpg wasn't there)
        clean_hash = raw_filename.replace("['picture', '", "").replace("'].jpg", "").replace("']", "")
        
        # 4. Add .jpg back
        clean_filename = clean_hash + ".jpg"

        image_path = os.path.join(IMAGE_FOLDER, clean_filename)
        
        # Verify file exists (Optional, helpful for debugging)
        if not os.path.exists(image_path):
             # print(f"Warning: File not found {image_path}") # Uncomment to debug
             missing_count += 1
             continue

        items_to_process.append({
            'id': sasrec_id,     # CRITICAL: This is the integer index (1, 2, 3...)
            'path': image_path,
            'text': full_text
        })

    if missing_count > 0:
        print(f"Warning: Could not find images for {missing_count} items.")

    return items_to_process

items = load_data()
print(f"Found {len(items)} items with valid images/text.")

# ==========================================
# 3. GENERATE EMBEDDINGS
# ==========================================
# CLIP Base-32 outputs 512-dim vectors
output_dim = 512
max_id = max([x['id'] for x in items]) if items else 0

# Create empty matrices (+1 for padding index 0)
image_matrix = np.zeros((max_id + 1, output_dim), dtype=np.float32)
text_matrix = np.zeros((max_id + 1, output_dim), dtype=np.float32)

print("Starting inference...")
model.eval() 

with torch.no_grad():
    for item in tqdm(items):
        idx = item['id']
        path = item['path']
        text = item['text']
        
        try:
            # A. Load and Process Image
            image = Image.open(path).convert("RGB")
            
            # B. Prepare Inputs
            inputs = processor(
                text=[text],
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)
            
            # C. Forward Pass
            outputs = model(**inputs)
            
            # D. Get Embeddings & Normalize
            img_embed = outputs.image_embeds
            img_embed = img_embed / img_embed.norm(p=2, dim=-1, keepdim=True)
            
            txt_embed = outputs.text_embeds
            txt_embed = txt_embed / txt_embed.norm(p=2, dim=-1, keepdim=True)
            
            # E. Save to Matrix
            image_matrix[idx] = img_embed.cpu().numpy()[0]
            text_matrix[idx] = txt_embed.cpu().numpy()[0]
            
        except Exception as e:
            print(f"Skipping Item {idx}: {e}")

# ==========================================
# 4. SAVE FILES
# ==========================================
print("Saving .npy files...")
# We save to the same folder as MAP_PATH so SASRec can find them easily
output_folder = os.path.dirname(MAP_PATH)
np.save(os.path.join(output_folder, "image_feat.npy"), image_matrix)
np.save(os.path.join(output_folder, "text_feat.npy"), text_matrix)

print(f"Success! Files saved to {output_folder}")