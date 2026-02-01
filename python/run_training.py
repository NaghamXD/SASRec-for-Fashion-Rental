import pandas as pd
import argparse
import time
from main import main # Import your main function

# 1. Define the Grid
dropout_rates = [0.5, 0.7]
embed_learning_rates = [0.01, 0.5]
learning_rates = [0.001]
# Placeholder class to mimic argparse if your main expects an object
class Args:
    def __init__(self):
        # Set your default fixed parameters here
        self.dataset = 'data_loo/clothing_items_train'
        self.train_dir = 'tuning_logs'
        self.batch_size = 128
        self.maxlen = 50
        self.hidden_units = 128
        self.num_blocks = 2
        self.num_epochs = 200
        self.num_heads = 1
        self.device = 'mps'  # or 'cuda' or 'cpu'
        self.inference_only = False
        self.state_dict_path = None
        self.norm_first = False
        
        # --- NEW PARAMS (Required by your updated main.py) ---
        self.use_visual = True
        self.use_tags = True
              
        
        # Variables we will change in the loop
        self.dropout_rate = 0.0
        self.lr = 0.0
        self.l2_emb = 0.0   

def run_grid_search():
    results = []
    
    total_runs = len(dropout_rates) * len(embed_learning_rates)
    current_run = 0

    print(f"Starting Grid Search: {total_runs} combinations...")
    print(f"Dataset: {Args().dataset}")
    
    for drop in dropout_rates:
        for embed_lr in embed_learning_rates:
            for lr in learning_rates:    
                current_run += 1
                print(f"\n--- Run {current_run}/{total_runs} | Dropout: {drop} | LR: {lr} | embed_LR: {embed_lr} ---")
                
                # 2. Setup Arguments
                args = Args()
                args.dropout_rate = drop
                args.lr_emb = embed_lr
                args.lr = lr
                # 3. Run Training (and catch the return values from Step 1)
                # We assume main(args) now returns (best_ndcg, best_hit)
                try:
                    start_time = time.time()
                    best_ndcg, best_hit = main(args)
                    duration = time.time() - start_time
                    
                    # 4. Log Data
                    results.append({
                        'dropout_rate': drop,
                        'emb_learning_rate': embed_lr,
                        'learning_rate': lr,
                        'valid_ndcg': best_ndcg,
                        'valid_hit_rate': best_hit,
                        'duration_sec': round(duration, 2)
                    })
                    
                    print(f"Result: NDCG={best_ndcg:.4f}, Hit={best_hit:.4f}")

                except Exception as e:
                    print(f"FAILED run with Drop {drop}, LR {lr}, embed_LR {embed_lr}")
                    print(e)

    # Save Results
    if results:
        df = pd.DataFrame(results)
        df.to_csv('tuning_results_validation.csv', index=False)
        print("\nGrid Search Complete. Results saved to 'tuning_results_validation.csv'.")
    else:
        print("\nGrid Search failed to produce any results.")

if __name__ == "__main__":
    run_grid_search()