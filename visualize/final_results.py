import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Prepare the Data
data = [
    # Residual Quantization "Let It Go"
    {"Model": "Residual Quantization - Let It Go", "Split": "Items", "HR@10": 0.0855, "HR@100": 0.3006, "HR@10-new": 0.0320, "HR@100-new": 0.2011},
    {"Model": "Residual Quantization - Let It Go", "Split": "Groups", "HR@10": 0.1146, "HR@100": 0.3661, "HR@10-new": 0.0431, "HR@100-new": 0.2781},
    
    # Feature-Enhanced SASRec
    {"Model": "Feature-Enhanced SASRec", "Split": "Items", "HR@10": 0.0881, "HR@100": 0.2773, "HR@10-new": 0.0317, "HR@100-new": 0.2059},
    {"Model": "Feature-Enhanced SASRec", "Split": "Groups", "HR@10": 0.1157, "HR@100": 0.3373, "HR@10-new": 0.0501, "HR@100-new": 0.2777},
    
    # Paper BaseLine
    {"Model": "Paper Baseline", "Split": "Items", "HR@10": 0.0607, "HR@100": 0.1957, "HR@10-new": 0.0447, "HR@100-new": 0.2193},
    {"Model": "Paper Baseline", "Split": "Groups", "HR@10": 0.0776, "HR@100": 0.2633, "HR@10-new": 0.0625, "HR@100-new": 0.2429}
]

df = pd.DataFrame(data)

# 2. Reshape for Plotting
df_melted = df.melt(
    id_vars=["Model", "Split"], 
    value_vars=["HR@10", "HR@100", "HR@10-new", "HR@100-new"],
    var_name="Metric", 
    value_name="Score"
)

# 3. Create the Visualization
# We define a custom palette to make "Let It Go" stand out
custom_palette = {
    "Residual Quantization - Let It Go": "#2ecc71",       # Bright Green
    "Feature-Enhanced SASRec": "#3498db",        # Blue
    "Paper Baseline": "#95a5a6"          # Grey
}

g = sns.catplot(
    data=df_melted,
    x="Split", 
    y="Score", 
    hue="Model", 
    col="Metric", 
    kind="bar",
    col_wrap=2, 
    height=4, 
    aspect=1.5,
    palette=custom_palette,
    sharey=False # Independent y-axis for each metric
)

# 4. Styling
g.set_titles("{col_name}")
g.set_axis_labels("", "Hit Ratio")
g.legend.set_title("Model Architecture")

# Add numeric labels
for ax in g.axes.flat:
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)

plt.subplots_adjust(top=0.9)
g.fig.suptitle('Benchmarking Our Two Enhanced SASRec Models vs. State-of-the-Art Baselines', fontsize=16)
plt.savefig('Final_results_mask.png')

plt.show()