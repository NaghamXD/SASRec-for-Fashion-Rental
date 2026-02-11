import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# Part 1: Plotting the Delta Hyperparameter Sweep
# ---------------------------------------------------------

# 1. Load the summary data
df_summary = pd.read_csv('delta_experiment_summary_drop_out_0.7.csv')

# 2. Sort by Delta for a proper line plot
df_summary = df_summary.sort_values(by='Delta')

# 3. Setup the plot style
sns.set(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot NDCG Scores
axes[0].plot(df_summary['Delta'], df_summary['Val_NDCG'], marker='o', label='Val NDCG', linestyle='-', color='royalblue')
axes[0].plot(df_summary['Delta'], df_summary['Test_NDCG'], marker='s', label='Test NDCG', linestyle='--', color='navy')
axes[0].set_title('Impact of Delta on NDCG@10 (Dropout=0.7)', fontsize=14)
axes[0].set_xlabel('Delta', fontsize=12)
axes[0].set_ylabel('NDCG@10', fontsize=12)
axes[0].legend()
axes[0].grid(True)

# Plot Hit Rate Scores
axes[1].plot(df_summary['Delta'], df_summary['Val_HR'], marker='o', label='Val HR', linestyle='-', color='mediumseagreen')
axes[1].plot(df_summary['Delta'], df_summary['Test_HR'], marker='s', label='Test HR', linestyle='--', color='darkgreen')
axes[1].set_title('Impact of Delta on Hit Rate@10 (Dropout=0.7)', fontsize=14)
axes[1].set_xlabel('Delta', fontsize=12)
axes[1].set_ylabel('HR@10', fontsize=12)
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('delta_tuning_results.png')
plt.show()

# ---------------------------------------------------------
# Part 2: Plotting the Detailed Evaluation for Best Model (Delta=0.2)
# ---------------------------------------------------------

# 1. Load the data
df_eval = pd.read_csv('evaluation_results_let_it_go_with_clip_0.2_delta.csv')

# 2. Create a 'Task' column to distinguish between Items and Groups
# We look at the 'Experiment_Name' column to find the keyword
df_eval['Task'] = df_eval['Experiment_Name'].apply(lambda x: 'Groups' if 'Groups' in x else 'Items')

# 3. Create the Plot with 'hue' to separate Items and Groups
plt.figure(figsize=(10, 6))

# 'hue' is the key change here: it splits the bars based on the Task
sns.barplot(x='Eval_Mode', y='HR@10', hue='Task', data=df_eval, palette='viridis')

plt.title('Performance of Optimal Model (Delta=0.2): Items vs. Groups', fontsize=14)
plt.xlabel('Evaluation Mode', fontsize=12)
plt.ylabel('Hit Rate @ 10', fontsize=12)
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Task Type')

plt.tight_layout()
plt.savefig('best_model_items_vs_groups.png')
plt.show()

print("Plot generated: 'best_model_items_vs_groups.png'")