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
    'HR@10': [0.010502, 0.008965, 0.015881, 0.021004, 0.021260, 0.022797, 0.011783, 0.017674, 0.009734, 0.017418, 0.024987, 0.025760, 0.024214, 0.027563],
    'HR@100': [0.046875, 0.064037, 0.034580, 0.046875, 0.075051, 0.098617, 0.084016, 0.083760, 0.060195, 0.089395, 0.073673, 0.077280, 0.100206, 0.100979],
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
        0.0135, 0.0192,  # No Features
        0.0147, 0.0191,  # Tag Features
        0.0123, 0.0199,  # Img Features
        0.0160, 0.0201   # Both Features
    ],
    'HR@100': [
        0.0510, 0.0721,  # No Features
        0.0581, 0.0713,  # Tag Features
        0.0548, 0.0730,  # Img Features
        0.0529, 0.0749   # Both Features
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