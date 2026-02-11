import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io


# 1. Load User Data
csv_data = """Features,Experiment_Name,Eval_Mode,HR@10,HR@100,HR@10_new,HR@100_new,Timestamp
Both features,70-30 Split (Groups),Static (Pure),0.11572052401746726,0.3373362445414847,0.05009208103130755,0.27771639042357277,2026-02-10 17:51:39
Image features,70-30 Split (Groups),Static (Pure),0.10844250363901019,0.33770014556040756,0.03941068139963168,0.26850828729281767,2026-02-10 17:54:06
Tag features,70-30 Split (Groups),Static (Pure),0.10589519650655022,0.35662299854439594,0.04604051565377532,0.2869244935543278,2026-02-10 17:56:30
No features,70-30 Split (Groups),Static (Pure),0.11280931586608442,0.33879184861717615,0.04014732965009208,0.26519337016574585,2026-02-10 17:58:46
"""

df_user = pd.read_csv(io.StringIO(csv_data))
# Clean up feature names for the plot
df_user['Method'] = df_user['Features'].str.replace(' features', '', regex=False).str.title() + ' (SASRec)'

# 2. Load Paper Data
# Creating a DataFrame manually for the paper benchmarks
paper_data = {
    'HR@10': 0.0776,
    'HR@100': 0.2633,
    'HR@10_new': 0.0625,
    'HR@100_new': 0.2429
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
g.fig.suptitle(
    'Benchmarking: SASRec Static Evaluation vs Paper Baseline - Groups',
    fontsize=16,
    fontweight='bold'
)

g.fig.subplots_adjust(top=0.90)

for ax in g.axes.flat:
    # Rotate x-axis labels
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha('right')
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)

plt.tight_layout()
plt.savefig('static_group_vs_paper.png')
plt.show()