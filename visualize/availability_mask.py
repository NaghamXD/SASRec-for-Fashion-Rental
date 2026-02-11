import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Define the data
data = {
    'Model': [
        'Residual Quantization "Let It Go"', 'Residual Quantization "Let It Go"', 
        'Feature-Enhanced SASRec', 'Feature-Enhanced SASRec'
    ],
    'Mask': [
        'No Mask', 'Availability Mask',
        'No Mask', 'Availability Mask'
    ],
    # HR@10 Data
    'HR@10': [0.0141, 0.0180, 0.0109, 0.0152],
    # HR@100 Data
    'HR@100': [0.0817, 0.1009, 0.0660, 0.0849],
    # HR@10-new Data
    'HR@10-new': [0.0061, 0.0079, 0.0052, 0.0079],
    # HR@100-new Data
    'HR@100-new': [0.0477, 0.0621, 0.0460, 0.0617]
}

df = pd.DataFrame(data)

# Melt the DataFrame to long format for easier plotting with seaborn
df_melted = df.melt(
    id_vars=['Model', 'Mask'],
    var_name='Metric',
    value_name='Score'
)

# 2. Create the Faceted Plot
# We use catplot to create a grid of subplots, one for each metric
g = sns.catplot(
    data=df_melted,
    x='Model',
    y='Score',
    hue='Mask',
    col='Metric',
    kind='bar',
    col_wrap=2,        # 2 columns in the grid
    height=4,          # Height of each subplot
    aspect=1.5,        # Width/Height ratio
    palette='viridis', # Color scheme
    sharey=False       # Important: Let each metric have its own y-axis scale
)

# 3. Styling
g.set_titles("{col_name}")  # Set title for each subplot
g.set_axis_labels("", "Hit Ratio")
g.legend.set_title("Evaluation Mode")

# Add numeric labels on top of bars
for ax in g.axes.flat:
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=10)

plt.subplots_adjust(top=0.9) # Make room for the main title
g.fig.suptitle('Impact of Availability Mask on Model Performance', fontsize=16)
plt.savefig('impact_of_availability_mask.png')
plt.show()