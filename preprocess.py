import pandas as pd
import os

# 1. Define file paths
file_orders = 'original_orders.csv'
file_activity = 'user_activity_triplets.csv'
output_file = 'data/clothing.txt'

# 2. Load the data (specifying the ; separator)
print("Loading data...")
df1 = pd.read_csv(file_orders, sep=';')
df2 = pd.read_csv(file_activity, sep=';')

# 3. Select only relevant columns and merge
# We need User, Item, and Time
cols = ['customer.id', 'outfit.id', 'rentalPeriod.start']
df1 = df1[cols]
df2 = df2[cols]

full_df = pd.concat([df1, df2])

# 4. Convert time to datetime objects for correct sorting
full_df['rentalPeriod.start'] = pd.to_datetime(full_df['rentalPeriod.start'])

# 5. Sort by User then by Date
# This ensures the sequence is correct: Item A -> Item B -> Item C
full_df = full_df.sort_values(by=['customer.id', 'rentalPeriod.start'])

# 6. Convert String IDs to Integers
# SASRec requires IDs to be 1, 2, 3... (0 is reserved for padding)
print("Converting IDs to integers...")

# Create Mappings
unique_users = full_df['customer.id'].unique()
user_to_id = {user: i+1 for i, user in enumerate(unique_users)}

unique_items = full_df['outfit.id'].unique()
item_to_id = {item: i+1 for i, item in enumerate(unique_items)}

# Apply Mappings
full_df['user_int'] = full_df['customer.id'].map(user_to_id)
full_df['item_int'] = full_df['outfit.id'].map(item_to_id)

# 7. Save to the format SASRec expects
# Format: user_id item_id
# (The model splits the sequence itself based on user_id)
if not os.path.exists('data'):
    os.makedirs('data')

print(f"Saving to {output_file}...")
full_df[['user_int', 'item_int']].to_csv(output_file, sep=' ', index=False, header=False)

print("Done!")
print(f"Total Interactions: {len(full_df)}")
print(f"Total Users: {len(unique_users)}")
print(f"Total Items: {len(unique_items)}")