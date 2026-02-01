import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

# 1. Load Data
csv_data = """Features,Experiment_Name,Eval_Mode,HR@10,HR@100,HR@10_new,HR@100_new,Timestamp
Both features,70-30 Split (Items),Rolling (No Mask),0.010869565217391304,0.0660377358490566,0.005234667777050741,0.046041282493605375,2026-01-30 12:18:12
Both features,70-30 Split (Items),Rolling (Availability Mask),0.010869565217391304,0.0660377358490566,0.005234667777050741,0.046041282493605375,2026-01-30 12:20:10
Both features,70-30 Split (Items),Static (Pure),0.08806404657933042,0.27729257641921395,0.03167587476979742,0.20589318600368325,2026-01-30 12:20:28
Both features,Leave-One-Out (Items),Rolling (No Mask),0.018922852983988356,0.09497816593886463,0.007120253164556962,0.06487341772151899,2026-01-30 12:20:47
Both features,Leave-One-Out (Items),Rolling (Availability Mask),0.018922852983988356,0.09497816593886463,0.007120253164556962,0.06487341772151899,2026-01-30 12:21:12
Both features,Leave-One-Out (Items),Static (Pure),0.018922852983988356,0.09497816593886463,0.007120253164556962,0.06487341772151899,2026-01-30 12:21:28
Both features,70-30 Split (Groups),Static (Pure),0.11572052401746726,0.3373362445414847,0.05009208103130755,0.27771639042357277,2026-01-30 12:21:44
Both features,Leave-One-Out (Groups),Static (Pure),0.020014556040756915,0.11754002911208151,0.009098101265822785,0.09572784810126582,2026-01-30 12:22:01
Image features,70-30 Split (Items),Rolling (No Mask),0.012561525840853158,0.07060090237899919,0.0060079709713877815,0.04931294985426209,2026-01-30 12:23:36
Image features,70-30 Split (Items),Rolling (Availability Mask),0.012561525840853158,0.07060090237899919,0.0060079709713877815,0.04931294985426209,2026-01-30 12:25:16
Image features,70-30 Split (Items),Static (Pure),0.09024745269286755,0.29002911208151383,0.026519337016574586,0.2151012891344383,2026-01-30 12:25:29
Image features,Leave-One-Out (Items),Rolling (No Mask),0.017831149927219795,0.0975254730713246,0.006724683544303798,0.07041139240506329,2026-01-30 12:25:43
Image features,Leave-One-Out (Items),Rolling (Availability Mask),0.017831149927219795,0.0975254730713246,0.006724683544303798,0.07041139240506329,2026-01-30 12:26:05
Image features,Leave-One-Out (Items),Static (Pure),0.017831149927219795,0.0975254730713246,0.006724683544303798,0.07041139240506329,2026-01-30 12:26:19
Image features,70-30 Split (Groups),Static (Pure),0.10844250363901019,0.33770014556040756,0.03941068139963168,0.26850828729281767,2026-01-30 12:26:32
Image features,Leave-One-Out (Groups),Static (Pure),0.023289665211062592,0.11426491994177583,0.01305379746835443,0.09295886075949367,2026-01-30 12:26:45
Tag features,70-30 Split (Items),Rolling (No Mask),0.014150943396226415,0.07101107465135356,0.006483849860210576,0.04592231277139968,2026-01-30 12:28:22
Tag features,70-30 Split (Items),Rolling (Availability Mask),0.014150943396226415,0.07101107465135356,0.006483849860210576,0.04592231277139968,2026-01-30 12:30:04
Tag features,70-30 Split (Items),Static (Pure),0.10007278020378457,0.2929403202328967,0.02725598526703499,0.19705340699815838,2026-01-30 12:30:16
Tag features,Leave-One-Out (Items),Rolling (No Mask),0.017831149927219795,0.08660844250363901,0.007515822784810127,0.06091772151898734,2026-01-30 12:30:30
Tag features,Leave-One-Out (Items),Rolling (Availability Mask),0.017831149927219795,0.08660844250363901,0.007515822784810127,0.06091772151898734,2026-01-30 12:30:51
Tag features,Leave-One-Out (Items),Static (Pure),0.017831149927219795,0.08660844250363901,0.007515822784810127,0.06091772151898734,2026-01-30 12:31:04
Tag features,70-30 Split (Groups),Static (Pure),0.10589519650655022,0.35662299854439594,0.04604051565377532,0.2869244935543278,2026-01-30 12:31:17
Tag features,Leave-One-Out (Groups),Static (Pure),0.020014556040756915,0.11972343522561864,0.00949367088607595,0.09612341772151899,2026-01-30 12:31:29
No features,70-30 Split (Items),Rolling (No Mask),0.017124692370795735,0.08593109105824447,0.00529415263815359,0.04675510082683957,2026-01-30 12:32:55
No features,70-30 Split (Items),Rolling (Availability Mask),0.017124692370795735,0.08593109105824447,0.00529415263815359,0.04675510082683957,2026-01-30 12:34:31
No features,70-30 Split (Items),Static (Pure),0.11754002911208151,0.30458515283842796,0.022099447513812154,0.19410681399631677,2026-01-30 12:34:43
No features,Leave-One-Out (Items),Rolling (No Mask),0.01710334788937409,0.0906113537117904,0.007120253164556962,0.07041139240506329,2026-01-30 12:34:55
No features,Leave-One-Out (Items),Rolling (Availability Mask),0.01710334788937409,0.0906113537117904,0.007120253164556962,0.07041139240506329,2026-01-30 12:35:15
No features,Leave-One-Out (Items),Static (Pure),0.01710334788937409,0.0906113537117904,0.007120253164556962,0.07041139240506329,2026-01-30 12:35:27
No features,70-30 Split (Groups),Static (Pure),0.11280931586608442,0.33879184861717615,0.04014732965009208,0.26519337016574585,2026-01-30 12:35:38
No features,Leave-One-Out (Groups),Static (Pure),0.020014556040756915,0.11098981077147016,0.009098101265822785,0.08900316455696203,2026-01-30 12:35:50
"""

df = pd.read_csv(io.StringIO(csv_data))

# 2. Data Preprocessing
# Extract clean 'Split Type' and 'Target' from the 'Experiment_Name'
df['Split_Type'] = df['Experiment_Name'].apply(lambda x: 'Leave-One-Out' if 'Leave-One-Out' in x else '70-30 Split')
df['Target'] = df['Experiment_Name'].apply(lambda x: 'Groups' if 'Groups' in x else 'Items')

# IMPORTANT: Filter for only 'Static (Pure)' Evaluation
# This ensures we are comparing apples-to-apples, as Groups only have Static evaluation.
df_static = df[df['Eval_Mode'] == 'Static (Pure)'].copy()

# Melt the 4 metrics into rows so we can plot them all at once
df_melted = df_static.melt(
    id_vars=['Features', 'Split_Type', 'Target'],
    value_vars=['HR@10', 'HR@100', 'HR@10_new', 'HR@100_new'],
    var_name='Metric',
    value_name='Hit Rate'
)

# Define a logical order for features on the X-axis
feature_order = ['No features', 'Image features', 'Tag features', 'Both features']

# 3. Create the Visualization
sns.set_theme(style="whitegrid")

# Create a Faceted Grid
# Rows = Metric (HR@10, HR@100...)
# Cols = Split Type (70-30 vs Leave-One-Out)
g = sns.catplot(
    data=df_melted,
    x='Features',
    y='Hit Rate',
    hue='Target',       # Compare Items vs Groups side-by-side
    row='Metric',       # One row per metric
    col='Split_Type',   # One column per split type
    kind='bar',
    palette='coolwarm', # Good contrast for comparisons
    height=3, 
    aspect=2,
    sharey='row',       # Share Y-axis within the same metric row, but not across metrics
    order=feature_order # Enforce the logical order
)

# 4. Polish the Layout
g.fig.suptitle('Impact of Feature Sets: Items vs Groups (Static Evaluation)', y=1.01, fontsize=16, fontweight='bold')

# Loop through axes to clean up labels
for ax in g.axes.flat:
    # Rotate x-axis labels slightly
    for label in ax.get_xticklabels():
        label.set_rotation(15)
        label.set_ha('center')
    
    # Add value labels
    for container in ax.containers:
        # Use simple formatting to avoid clutter
        ax.bar_label(container, fmt='%.4f', padding=2, fontsize=7)

# Move Legend to a clean spot (upper left of the whole figure isn't great for grids, 
# so we let Seaborn place it outside right, or force it top-left of the first plot)
sns.move_legend(g, "upper right", bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()