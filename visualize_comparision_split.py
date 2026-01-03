import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Paper Data ---
data_paper = {
    'Method': [
        'Pop', 'Pop', 'Rep', 'Rep', 'Rep + Pop', 'Rep + Pop', 
        'BPR', 'BPR', 'ALS', 'ALS', 
        'Img Embed (Paper)', 'Img Embed (Paper)', 'Tag Embed (Paper)', 'Tag Embed (Paper)'
    ],
    'User Type': [
        'Items', 'Groups', 'Items', 'Groups', 'Items', 'Groups', 
        'Items', 'Groups', 'Items', 'Groups', 
        'Items', 'Groups', 'Items', 'Groups'
    ],
    # Note: I renamed 'Ind' to 'Items' to match your terminology
    'HR@10': [0.0220, 0.0451, 0.0546, 0.0756, 0.0607, 0.0776, 0.0371, 0.0479, 0.0379, 0.0551, 0.0355, 0.0386, 0.0541, 0.0634],
    'HR@100': [0.1158, 0.2088, 0.1255, 0.1532, 0.1957, 0.2633, 0.1949, 0.2223, 0.1755, 0.2398, 0.1334, 0.1443, 0.2324, 0.2378],
    'Source': ['Paper'] * 14

    
}
df_paper = pd.DataFrame(data_paper)

# --- 2. Your Results (70-30 Split, Static Only) ---
# I extracted these values manually from your previous CSV content
data_mine = {
    'Method': [
        'No Features (Yours)', 'No Features (Yours)',
        'Tag Features (Yours)', 'Tag Features (Yours)',
        'Img Features (Yours)', 'Img Features (Yours)',
        'Both Features (Yours)', 'Both Features (Yours)'
    ],
    'User Type': [
        'Items', 'Groups', 'Items', 'Groups', 'Items', 'Groups', 'Items', 'Groups'
    ],
    'HR@10': [
        0.0402, 0.0571,  # No Features
        0.0443, 0.0521,  # Tag Features
        0.0480, 0.0545,  # Img Features
        0.0457, 0.0507   # Both Features
    ],
    'HR@100': [
        0.1318, 0.1537,  # No Features
        0.1414, 0.1551,  # Tag Features
        0.1385, 0.1589,  # Img Features
        0.1337, 0.1517   # Both Features
    ],
    'Source': ['Yours'] * 8
}
df_mine = pd.DataFrame(data_mine)

# Combine them
df_full = pd.concat([df_paper, df_mine], ignore_index=True)

# Define a metric to plot (Change to HR@10 if needed)
metric = 'HR@10'


#'''
plt.figure(figsize=(14, 6))
sns.set_theme(style="whitegrid")

# Create barplot
ax = sns.barplot(
    data=df_full, 
    x='Method', 
    y=metric, 
    hue='User Type', 
    palette='Paired' # Paired colors help distinguish Items vs Groups clearly
)

# Customizing to separate Paper vs Yours visually
plt.axvline(x=6.5, color='black', linestyle='--', linewidth=1) # Line separating Paper vs Yours
plt.text(3, df_full[metric].max(), 'Paper Baselines', ha='center', fontsize=12, fontweight='bold')
plt.text(9, df_full[metric].max(), 'Your Models', ha='center', fontsize=12, fontweight='bold', color='darkblue')

plt.title(f'Global Comparison: Paper Baselines vs Your Approaches ({metric})', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Hit Rate')
plt.legend(title='Target')

# Add values
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', padding=3, fontsize=8, rotation=90)

plt.tight_layout()
plt.show()
'''

# Filter: Keep only Embed/Feature methods
target_methods = [
    'Img Embed (Paper)', 'Tag Embed (Paper)', 
    'Img Features (Yours)', 'Tag Features (Yours)', 'Both Features (Yours)'
]
df_focus = df_full[df_full['Method'].isin(target_methods)].copy()

plt.figure(figsize=(10, 6))

# Plot
ax = sns.barplot(
    data=df_focus, 
    x='Method', 
    y=metric, 
    hue='User Type', 
    palette='viridis'
)

plt.title(f'Direct Comparison: Feature-Based Approaches ({metric})', fontsize=15, fontweight='bold')
plt.xticks(rotation=30, ha='right')
plt.ylabel('Hit Rate')

# Add values
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)

plt.tight_layout()
plt.show()
'''