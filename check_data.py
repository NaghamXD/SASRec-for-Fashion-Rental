import pickle
import os
import sys

# Change this to match the file you are using
FILE_PATH = 'data/data_70_30/test_data_items.pkl' 

def inspect_pickle():
    if not os.path.exists(FILE_PATH):
        print(f"ERROR: File not found at {FILE_PATH}")
        return

    print(f"--- Inspecting {FILE_PATH} ---")
    
    with open(FILE_PATH, 'rb') as f:
        data = pickle.load(f)
    
    # 1. Check Top-Level Keys
    print(f"Keys found: {list(data.keys())}")
    
    # 2. Check Validation Data
    if 'valid' in data:
        valid_data = data['valid']
        num_users = len(valid_data)
        print(f"\n[VALIDATION SET]")
        print(f"Number of users with validation data: {num_users}")
        
        if num_users > 0:
            # Show a sample
            first_user = list(valid_data.keys())[0]
            print(f"Sample User {first_user}: {valid_data[first_user]}")
        else:
            print("WARNING: 'valid' key exists but contains 0 users.")
    else:
        print("\n[CRITICAL WARNING]")
        print("The key 'valid' is MISSING from the pickle file.")
        print("This is why your results are 0.0. The model has nothing to evaluate.")

    # 3. Check Test Data (Comparison)
    if 'test' in data:
        print(f"\n[TEST SET]")
        print(f"Number of users with test data: {len(data['test'])}")

if __name__ == "__main__":
    inspect_pickle()