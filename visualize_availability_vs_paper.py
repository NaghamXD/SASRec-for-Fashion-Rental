import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

# 1. Load User Data
csv_data = """Features,Experiment_Name,Eval_Mode,HR@10,HR@100,HR@10_new,HR@100_new,Timestamp
Both features,Leave-One-Out (Items),Rolling (Availability Mask),0.0239,0.1076,0.0116,0.0712,2026-01-03 01:26:10
Image features,Leave-One-Out (Items),Rolling (Availability Mask),0.0185,0.1128,0.0108,0.0819,2026-01-03 12:53:19
Tag features,Leave-One-Out (Items),Rolling (Availability Mask),0.0211,0.1105,0.0108,0.0747,2026-01-03 15:01:09
No features,Leave-One-Out (Items),Rolling (Availability Mask),0.0164,0.0981,0.0095,0.0696,2026-01-03 14:51:13"""

df_user = pd.read_csv(io.StringIO(csv_data))
# Clean up feature names for the plot
df_user['Method'] = df_user['Features'].str.replace(' features', '', regex=False).str.title() + ' (Yours)'

# 2. Load Paper Data
# Creating a DataFrame manually for the paper benchmarks
paper_data = {
    'HR@10': 0.024214,
    'HR@100': 0.100206,
    'HR@10_new': 0.026629,
    'HR@100_new': 0.102637
}
# Convert to same format as user data
df_paper = pd.DataFrame([paper_data])
df_paper['Method'] = 'Paper Baseline'

# 3. Combine and Reshape (Melt)
df_full = pd.concat([df_paper, df_user], ignore_index=True)

df_melted = df_full.melt(
    id_vars=['Method'],
    value_vars=['HR@10', 'HR@100', 'HR@10_new', 'HR@100_new'],
    var_name='Metric',
    value_name='Hit Rate'
)

# 4. Visualization
sns.set_theme(style="whitegrid")

# Define a specific order so the Paper Baseline appears first (as a reference)
method_order = ['Paper Baseline', 'No (Yours)', 'Tag (Yours)', 'Image (Yours)', 'Both (Yours)']

g = sns.catplot(
    data=df_melted,
    x='Method',
    y='Hit Rate',
    hue='Method',      # <--- FIX: Assign 'x' variable to 'hue'
    legend=False,
    col='Metric',
    col_wrap=2,        # 2x2 grid
    kind='bar',
    height=4,
    aspect=1.5,
    palette='Set2',    # distinct colors
    sharey=False       # Important: Allows each metric to use its own scale
)

# 5. Polish the Chart
g.fig.suptitle('Benchmarking: Your Rolling Evaluation vs Paper Baseline', y=1.02, fontsize=16, fontweight='bold')

for ax in g.axes.flat:
    # Rotate x-axis labels
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha('right')
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)

plt.tight_layout()
plt.show()