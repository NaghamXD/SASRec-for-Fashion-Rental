import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

# 1. Load Data
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

# 2. Data Cleaning & Feature Extraction
df['Split_Type'] = df['Experiment_Name'].apply(lambda x: '70-30 Split' if '70-30' in x else 'Leave-One-Out')
df['Target'] = df['Experiment_Name'].apply(lambda x: 'Groups' if 'Groups' in x else 'Items')

# Create a clean label for the "Winner" (Features + Mode)
# Example: "Image / Static" or "Both / Rolling(Avail)"
df['Config_Label'] = df['Features'].str.replace(' features', '') + "\n" + df['Eval_Mode'].str.replace(' (Pure)', '', regex=False).str.replace('Availability Mask', 'Avail.', regex=False)

# 3. Find the Winner for each Metric in each Subgroup
metrics = ['HR@10', 'HR@100', 'HR@10_new', 'HR@100_new']
results = []

for split in ['70-30 Split', 'Leave-One-Out']:
    for target in ['Items', 'Groups']:
        subset = df[(df['Split_Type'] == split) & (df['Target'] == target)]
        
        for metric in metrics:
            if not subset.empty:
                # Find the row with the max value for this metric
                best_row = subset.loc[subset[metric].idxmax()]
                results.append({
                    'Split': split,
                    'Target': target,
                    'Metric': metric,
                    'Max_Score': best_row[metric],
                    'Winning_Config': best_row['Config_Label']
                })

df_winners = pd.DataFrame(results)

# 4. Visualization
sns.set_theme(style="whitegrid")

# Create a 2x2 FacetGrid (Rows=Split, Cols=Target)
g = sns.catplot(
    data=df_winners,
    x='Metric',
    y='Max_Score',
    col='Target',
    row='Split',
    kind='bar',
    height=4.5,
    aspect=1.3,
    color='cornflowerblue', # Uniform color since the labels explain the differences
    sharey=False # Critical: Allows each subplot to scale to its own max (Groups scores are higher)
)

g.fig.suptitle('Best Performing Configurations by Metric', y=1.02, fontsize=16, fontweight='bold')

# 5. Add Labels to Bars (This is the most important part)
for ax in g.axes.flat:
    # Get the data for this specific subplot
    for container in ax.containers:
        # Loop through each bar to add custom text (The Winning Config Name)
        labels = []
        for val, metric_name in zip(container.datavalues, ax.get_xticklabels()):
            # Find the matching config name from our dataframe
            # (We have to match the current Facet's Split/Target + the Bar's Metric)
            # A simpler way with seaborn barplot is to iterate the patches directly, 
            # but getting the specific label requires mapping back to the data.
            pass 
    
    # Custom annotation loop
    # We iterate over the dataframe rows that belong to this ax
    # Extract Split/Target from title or ax attributes
    # Note: Seaborn titles are like "Target = Groups | Split = 70-30 Split"
    title = ax.get_title() 
    current_target = 'Groups' if 'Groups' in title else 'Items'
    current_split = '70-30 Split' if '70-30' in title else 'Leave-One-Out'
    
    subset = df_winners[(df_winners['Split'] == current_split) & (df_winners['Target'] == current_target)]
    
    # We must ensure the order matches the X-axis (metrics list)
    subset = subset.set_index('Metric').reindex(metrics).reset_index()
    
    # Add the text annotations
    rects = ax.patches
    for rect, label, score in zip(rects, subset['Winning_Config'], subset['Max_Score']):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, 
            height/2, # Place text in middle of bar
            label, 
            ha='center', va='center', 
            rotation=90, color='white', fontweight='bold', fontsize=10
        )
        # Add score on top
        ax.text(
            rect.get_x() + rect.get_width() / 2, 
            height, 
            f'{score:.3f}', 
            ha='center', va='bottom', 
            fontsize=9, color='black'
        )

plt.tight_layout()
plt.show()