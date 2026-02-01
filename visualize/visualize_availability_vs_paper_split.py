import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io


# 1. Load User Data
csv_data = """Features,Experiment_Name,Eval_Mode,HR@10,HR@100,HR@10_new,HR@100_new,Timestamp
Both features,70-30 Split (Items),Static (Pure),0.0868225019265348,0.25404572309273055,0.02468487394957983,0.1722689075630252,2026-01-04 01:12:28
Image features,70-30 Split (Items),Static (Pure),0.09118931415360904,0.26303621885435396,0.025735294117647058,0.1796218487394958,2026-01-04 01:15:12
Tag features,70-30 Split (Items),Static (Pure),0.08425378885178525,0.26868738761880295,0.027836134453781514,0.19170168067226892,2026-01-04 01:17:55
No features,70-30 Split (Items),Static (Pure),0.07629077832006165,0.2504495247880812,0.020483193277310924,0.17358193277310924,2026-01-04 01:20:32
"""

df_user = pd.read_csv(io.StringIO(csv_data))
# Clean up feature names for the plot
df_user['Method'] = df_user['Features'].str.replace(' features', '', regex=False).str.title() + ' (SASRec)'

# 2. Load Paper Data
# Creating a DataFrame manually for the paper benchmarks
paper_data = {
    'HR@10': 0.0607,
    'HR@100': 0.1957,
    'HR@10_new': 0.0625,
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
g.fig.suptitle('Benchmarking: SASRec Static Evaluation vs Paper Baseline - items', y=1.02, fontsize=16, fontweight='bold')

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