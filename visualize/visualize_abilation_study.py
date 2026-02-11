import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns

# The CSV data provided
csv_data = """Features,Experiment_Name,Eval_Mode,HR@10,HR@100,HR@10_new,HR@100_new,Timestamp
Both features,70-30 Split (Items),Rolling (No Mask),0.010869565217391304,0.0660377358490566,0.005234667777050741,0.046041282493605375,2026-02-10 17:49:52
Both features,70-30 Split (Items),Rolling (Availability Mask),0.015227645611156686,0.08490566037735849,0.00791148652667896,0.0617452858247576,2026-02-10 17:50:59
Both features,70-30 Split (Items),Static (Pure),0.08806404657933042,0.27729257641921395,0.03167587476979742,0.20589318600368325,2026-02-10 17:51:05
Both features,70-30 Split (Groups),Static (Pure),0.11572052401746726,0.3373362445414847,0.05009208103130755,0.27771639042357277,2026-02-10 17:51:39
Image features,70-30 Split (Items),Rolling (No Mask),0.012561525840853158,0.07060090237899919,0.0060079709713877815,0.04931294985426209,2026-02-10 17:52:28
Image features,70-30 Split (Items),Rolling (Availability Mask),0.015535274815422478,0.09146841673502872,0.00791148652667896,0.06525489262982571,2026-02-10 17:53:28
Image features,70-30 Split (Items),Static (Pure),0.09024745269286755,0.29002911208151383,0.026519337016574586,0.2151012891344383,2026-02-10 17:53:34
Image features,70-30 Split (Groups),Static (Pure),0.10844250363901019,0.33770014556040756,0.03941068139963168,0.26850828729281767,2026-02-10 17:54:06
Tag features,70-30 Split (Items),Rolling (No Mask),0.014150943396226415,0.07101107465135356,0.006483849860210576,0.04592231277139968,2026-02-10 17:54:54
Tag features,70-30 Split (Items),Rolling (Availability Mask),0.017945036915504513,0.09126333059885151,0.00886324430432455,0.061983225269168996,2026-02-10 17:55:51
Tag features,70-30 Split (Items),Static (Pure),0.10007278020378457,0.2929403202328967,0.02725598526703499,0.19705340699815838,2026-02-10 17:55:57
Tag features,70-30 Split (Groups),Static (Pure),0.10589519650655022,0.35662299854439594,0.04604051565377532,0.2869244935543278,2026-02-10 17:56:30
No features,70-30 Split (Items),Rolling (No Mask),0.017124692370795735,0.08593109105824447,0.00529415263815359,0.04675510082683957,2026-02-10 17:57:16
No features,70-30 Split (Items),Rolling (Availability Mask),0.021739130434782608,0.106285890073831,0.006900243887930522,0.06269704360240319,2026-02-10 17:58:12
No features,70-30 Split (Items),Static (Pure),0.11754002911208151,0.30458515283842796,0.022099447513812154,0.19410681399631677,2026-02-10 17:58:17
No features,70-30 Split (Groups),Static (Pure),0.11280931586608442,0.33879184861717615,0.04014732965009208,0.26519337016574585,2026-02-10 17:58:46"""

# Load data into a pandas DataFrame
df = pd.read_csv(io.StringIO(csv_data))

# Filter for rows where Eval_Mode is 'Static (Pure)'
df_static = df[df['Eval_Mode'] == 'Static (Pure)'].copy()

# Create a new column 'Split' to distinguish between Items and Groups
# It checks if 'Groups' is in the Experiment_Name, otherwise assigns 'Items'
df_static['Split'] = df_static['Experiment_Name'].apply(lambda x: 'Groups' if 'Groups' in x else 'Items')

# Define the 4 metrics to compare
metrics = ['HR@10', 'HR@100', 'HR@10_new', 'HR@100_new']

# Create subplots (2x2 grid)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Generate a bar plot for each metric
for i, metric in enumerate(metrics):
    sns.barplot(
        data=df_static,
        x='Features',
        y=metric,
        hue='Split',
        ax=axes[i],
        palette='viridis' # Color scheme
    )
    axes[i].set_title(f'Comparison of {metric}')
    axes[i].set_ylabel(metric)
    axes[i].set_xlabel('')
    axes[i].legend(title='Split Type')
    # Rotate x-axis labels for better readability
    axes[i].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('static_comparison_plots.png')
plt.show()