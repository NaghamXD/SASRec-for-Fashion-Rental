import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io


# 1. Load User Data
csv_data = """Features,Experiment_Name,Eval_Mode,HR@10,HR@100,HR@10_new,HR@100_new,Timestamp
Both features,70-30 Split (Items),Static (Pure),0.08806404657933042,0.27729257641921395,0.03167587476979742,0.20589318600368325,2026-02-10 17:51:05
Image features,70-30 Split (Items),Static (Pure),0.09024745269286755,0.29002911208151383,0.026519337016574586,0.2151012891344383,2026-02-10 17:53:34
Tag features,70-30 Split (Items),Static (Pure),0.10007278020378457,0.2929403202328967,0.02725598526703499,0.19705340699815838,2026-02-10 17:55:57
No features,70-30 Split (Items),Static (Pure),0.11754002911208151,0.30458515283842796,0.022099447513812154,0.19410681399631677,2026-02-10 17:58:17
"""

df_user = pd.read_csv(io.StringIO(csv_data))
# Clean up feature names for the plot
df_user['Method'] = df_user['Features'].str.replace(' features', '', regex=False).str.title() + ' (SASRec)'

# 2. Load Paper Data
# Creating a DataFrame manually for the paper benchmarks
paper_data = {
    'HR@10': 0.0607,
    'HR@100': 0.1957,
    'HR@10_new': 0.0447,
    'HR@100_new': 0.2193
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
    'Benchmarking: SASRec Static Evaluation vs Paper Baseline - Items',
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
plt.savefig('static_items_vs_paper.png')
plt.show()