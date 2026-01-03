import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

# 1. Load the Data
csv_data = """Features,Experiment_Name,Eval_Mode,HR@10,HR@100,HR@10_new,HR@100_new,Timestamp
Both features,70-30 Split (Items),Rolling (No Mask),0.0151,0.0870,0.0051,0.0416,2026-01-03 01:24:22
Both features,70-30 Split (Items),Rolling (Availability Mask),0.0187,0.1057,0.0068,0.0528,2026-01-03 01:25:28
Both features,70-30 Split (Items),Static (Pure),0.0457,0.1337,0.0249,0.1728,2026-01-03 01:25:44
Both features,Leave-One-Out (Items),Rolling (No Mask),0.0200,0.0950,0.0095,0.0616,2026-01-03 01:25:53
Both features,Leave-One-Out (Items),Rolling (Availability Mask),0.0239,0.1076,0.0116,0.0712,2026-01-03 01:26:10
Both features,Leave-One-Out (Items),Static (Pure),0.0160,0.0529,0.0117,0.0653,2026-01-03 01:26:27
Both features,70-30 Split (Groups),Static (Pure),0.0507,0.1517,0.0331,0.2271,2026-01-03 12:12:17
Both features,Leave-One-Out (Groups),Static (Pure),0.0201,0.0749,0.0191,0.1074,2026-01-03 12:12:33
Image features,70-30 Split (Items),Rolling (No Mask),0.0157,0.0911,0.0043,0.0420,2026-01-03 12:51:38
Image features,70-30 Split (Items),Rolling (Availability Mask),0.0206,0.1115,0.0062,0.0555,2026-01-03 12:52:41
Image features,70-30 Split (Items),Static (Pure),0.0480,0.1385,0.0257,0.1812,2026-01-03 12:52:55
Image features,Leave-One-Out (Items),Rolling (No Mask),0.0154,0.0953,0.0092,0.0676,2026-01-03 12:53:04
Image features,Leave-One-Out (Items),Rolling (Availability Mask),0.0185,0.1128,0.0108,0.0819,2026-01-03 12:53:19
Image features,Leave-One-Out (Items),Static (Pure),0.0123,0.0548,0.0096,0.0757,2026-01-03 12:53:34
Image features,70-30 Split (Groups),Static (Pure),0.0545,0.1589,0.0305,0.2352,2026-01-03 13:05:52
Image features,Leave-One-Out (Groups),Static (Pure),0.0199,0.0730,0.0191,0.1038,2026-01-03 13:06:07
Tag features,70-30 Split (Items),Rolling (No Mask),0.0137,0.0828,0.0047,0.0405,2026-01-03 14:59:25
Tag features,70-30 Split (Items),Rolling (Availability Mask),0.0179,0.1020,0.0065,0.0530,2026-01-03 15:00:29
Tag features,70-30 Split (Items),Static (Pure),0.0443,0.1414,0.0286,0.1943,2026-01-03 15:00:45
Tag features,Leave-One-Out (Items),Rolling (No Mask),0.0182,0.0948,0.0092,0.0637,2026-01-03 15:00:53
Tag features,Leave-One-Out (Items),Rolling (Availability Mask),0.0211,0.1105,0.0108,0.0747,2026-01-03 15:01:09
Tag features,Leave-One-Out (Items),Static (Pure),0.0147,0.0581,0.0117,0.0768,2026-01-03 15:01:24
Tag features,70-30 Split (Groups),Static (Pure),0.0521,0.1551,0.0326,0.2247,2026-01-03 13:13:30
Tag features,Leave-One-Out (Groups),Static (Pure),0.0191,0.0713,0.0158,0.0968,2026-01-03 13:13:46
No features,70-30 Split (Items),Rolling (No Mask),0.0133,0.0801,0.0056,0.0405,2026-01-03 14:49:38
No features,70-30 Split (Items),Rolling (Availability Mask),0.0178,0.1003,0.0073,0.0528,2026-01-03 14:50:37
No features,70-30 Split (Items),Static (Pure),0.0402,0.1318,0.0205,0.1747,2026-01-03 14:50:51
No features,Leave-One-Out (Items),Rolling (No Mask),0.0134,0.0827,0.0072,0.0583,2026-01-03 14:50:58
No features,Leave-One-Out (Items),Rolling (Availability Mask),0.0164,0.0981,0.0095,0.0696,2026-01-03 14:51:13
No features,Leave-One-Out (Items),Static (Pure),0.0135,0.0510,0.0087,0.0702,2026-01-03 14:51:26
No features,70-30 Split (Groups),Static (Pure),0.0571,0.1537,0.0305,0.2169,2026-01-03 14:56:00
No features,Leave-One-Out (Groups),Static (Pure),0.0192,0.0721,0.0160,0.0977,2026-01-03 14:56:13
"""

df = pd.read_csv(io.StringIO(csv_data))

# 2. Setup the Plotting
# We can loop through metrics, but let's just plot the most important one: HR@100
# (You can change y_metric to 'HR@10', 'HR@100', 'HR@10_new', or 'HR@100_new')
y_metric = 'HR@10'

sns.set_theme(style="whitegrid")

# Create a FacetGrid: Rows = Feature Sets
g = sns.catplot(
    data=df,
    x='Experiment_Name',
    y=y_metric,
    hue='Eval_Mode',
    col='Features',       # Create a separate chart for each Feature type
    col_wrap=2,           # 2 charts per row (2x2 grid)
    kind='bar',
    height=4,
    aspect=1.6,
    palette='magma'       # 'magma', 'viridis', or 'deep' look good
)

# 3. Polish the Layout
g.fig.suptitle(f'Comparison of {y_metric} Across Feature Sets', y=1.02, fontsize=16, fontweight='bold')

# Loop through axes to clean up labels
for ax in g.axes.flat:
    # Rotate x-axis labels
    for label in ax.get_xticklabels():
        label.set_rotation(25)
        label.set_ha('right')
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', padding=2, fontsize=8, rotation=90)

# Move Legend
sns.move_legend(
    g, "upper left",
    bbox_to_anchor=(0, 1),
    title='Evaluation Mode',
    frameon=True
)

plt.tight_layout()
plt.show()