import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import pandas as pd
import os
import pickle
import ast
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
IMAGE_FOLDER = "data/images" 
OUTFITS_PATH = 'outfits.csv'
PICTURES_PATH = 'picture_triplets.csv'

# SELECT YOUR TARGET MAP HERE
# ---------------------------------------------------------
# OPTION A: ITEMS (Standard - 1 image per item)
# MAP_PATH = 'data_70_30/clothing_items_train/item_maps.pkl'

# OPTION B: GROUPS (Averaging Mode - Mean of all images in group)
MAP_PATH = 'data/data_70_30/group_maps.pkl'
# ---------------------------------------------------------

MODEL_ID = "patrickjohncyh/fashion-clip"

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
def load_data_for_averaging():
    print(f"Processing Map: {MAP_PATH}")

    if not os.path.exists(MAP_PATH):
        raise FileNotFoundError(f"Missing map file: {MAP_PATH}")
    
    with open(MAP_PATH, 'rb') as f:
        _, item_map = pickle.load(f)

    # Load CSVs
    df_outfits = pd.read_csv(OUTFITS_PATH, sep=';')
    df_pics = pd.read_csv(PICTURES_PATH, sep=';')

    # Cleanup IDs
    df_pics['outfit.id'] = df_pics['outfit.id'].str.strip()
    df_outfits['id'] = df_outfits['id'].str.strip()
    df_outfits['group'] = df_outfits['group'].str.strip()

    # Merge Pictures
    # Note: We do NOT drop duplicates here because we want ALL images for a group
    df_pics = df_pics.sort_values('displayOrder')
    full_data = pd.merge(df_outfits, df_pics, left_on='id', right_on='outfit.id', how='left')

    # Detect Mode
    first_uuid = list(item_map.keys())[0]
    is_group_mode = 'group' in str(first_uuid)
    print(f"--> Mode Detected: {'GROUPS (Averaging)' if is_group_mode else 'ITEMS'}")

    groups_to_process = []
    
    for uuid, sasrec_id in item_map.items():
        # A. Find all rows for this entity
        if is_group_mode:
            rows = full_data[full_data['group'] == uuid]
        else:
            rows = full_data[full_data['id'] == uuid]

        if rows.empty:
            continue

        # B. Collect all valid image paths and texts
        valid_paths = []
        valid_texts = []
        
        # Iterate over all variants in this group
        # (Dropna ensures we only look at rows with images)
        for _, row in rows.dropna(subset=['file_name']).iterrows():
            # Image Path
            raw_filename = str(row['file_name'])
            clean_hash = raw_filename.replace("['picture', '", "").replace("'].jpg", "").replace("']", "")
            clean_filename = clean_hash + ".jpg"
            full_path = os.path.join(IMAGE_FOLDER, clean_filename)
            
            if os.path.exists(full_path):
                valid_paths.append(full_path)
                
                # Text (We collect text too, to average it or pick unique tags)
                desc = str(row['description'])
                try:
                    raw_tags = row['outfit_tags']
                    tags_str = ", ".join(ast.literal_eval(raw_tags)) if pd.notna(raw_tags) else ""
                except:
                    tags_str = str(raw_tags)
                
                full_text = f"Category: {row['group']}. Tags: {tags_str}. Description: {desc}"
                valid_texts.append(full_text[:77])

        # Remove duplicates to save computation (e.g. if 5 sizes share the same photo)
        valid_paths = list(set(valid_paths))
        valid_texts = list(set(valid_texts))

        if valid_paths:
            groups_to_process.append({
                'id': sasrec_id,
                'paths': valid_paths, # LIST of images
                'texts': valid_texts  # LIST of texts
            })

    return groups_to_process

items = load_data_for_averaging()
print(f"Found {len(items)} entities. Starting Multi-Image Averaging...")

# ==========================================
# 3. GENERATE (AVERAGE) & SAVE
# ==========================================
output_dim = 512
max_id = max([x['id'] for x in items]) if items else 0
image_matrix = np.zeros((max_id + 1, output_dim), dtype=np.float32)
text_matrix = np.zeros((max_id + 1, output_dim), dtype=np.float32)

model.eval()

with torch.no_grad():
    for item in tqdm(items):
        idx = item['id']
        paths = item['paths']
        texts = item['texts']
        
        # Containers for this group's vectors
        group_img_embeds = []
        group_txt_embeds = []

        # Process all images in this group
        for i, path in enumerate(paths):
            try:
                # We pair image[i] with text[i] (or text[0] if fewer texts)
                curr_text = texts[i] if i < len(texts) else texts[0]
                
                image = Image.open(path).convert("RGB")
                inputs = processor(text=[curr_text], images=image, return_tensors="pt", padding=True, truncation=True).to(device)
                outputs = model(**inputs)

                # Normalize individual vectors first
                img_e = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
                txt_e = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)

                group_img_embeds.append(img_e.cpu().numpy()[0])
                group_txt_embeds.append(txt_e.cpu().numpy()[0])
            except Exception as e:
                # print(f"Error on image {path}: {e}")
                pass

        # === THE AVERAGING STEP ===
        if group_img_embeds:
            # 1. Stack all vectors: Shape (N_images, 512)
            img_stack = np.stack(group_img_embeds)
            txt_stack = np.stack(group_txt_embeds)

            # 2. Compute Mean: Shape (512,)
            mean_img = np.mean(img_stack, axis=0)
            mean_txt = np.mean(txt_stack, axis=0)

            # 3. Re-Normalize the result (Crucial! The mean of unit vectors is not a unit vector)
            # This pushes the vector back onto the hypersphere surface
            mean_img = mean_img / (np.linalg.norm(mean_img) + 1e-9)
            mean_txt = mean_txt / (np.linalg.norm(mean_txt) + 1e-9)

            image_matrix[idx] = mean_img
            text_matrix[idx] = mean_txt

# Save
output_folder = os.path.dirname(MAP_PATH)
print(f"Saving Averaged Embeddings to: {output_folder}")
np.save(os.path.join(output_folder, "group_image_feat.npy"), image_matrix)
np.save(os.path.join(output_folder, "group_text_feat.npy"), text_matrix)
print("Done.")