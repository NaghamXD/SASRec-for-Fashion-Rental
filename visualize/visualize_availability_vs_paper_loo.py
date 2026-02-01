import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io


# 1. Load User Data
csv_data = """Features,Experiment_Name,Eval_Mode,HR@10,HR@100,HR@10_new,HR@100_new,Timestamp
Both features,Leave-One-Out (Items),Rolling (Availability Mask),0.03390701258669407,0.11816080143847932,0.013562136728480487,0.08137282037088292,2026-01-04 01:12:53
Image features,Leave-One-Out (Items),Rolling (Availability Mask),0.026714615977395325,0.12175699974312869,0.011624688624411846,0.0916136174923886,2026-01-04 01:15:37
Tag features,Leave-One-Out (Items),Rolling (Availability Mask),0.03056768558951965,0.12715129720010274,0.013838915029061722,0.09189039579296983,2026-01-04 01:18:20
No features,Leave-One-Out (Items),Rolling (Availability Mask),0.02902645774466992,0.11071153352170562,0.011347910323830611,0.08247993357320786,2026-01-04 01:20:55
"""

df_user = pd.read_csv(io.StringIO(csv_data))
# Clean up feature names for the plot
df_user['Method'] = df_user['Features'].str.replace(' features', '', regex=False).str.title() + ' (SASRec)'

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
method_order = ['Paper Baseline', 'No (SASRec)', 'Tag (SASRec)', 'Image (SASRec)', 'Both (SASRec)']

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
g.fig.suptitle('Benchmarking: SASRec Rolling Evaluation vs Paper Baseline', y=1.02, fontsize=16, fontweight='bold')

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