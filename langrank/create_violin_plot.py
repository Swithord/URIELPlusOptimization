import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the data
mi_data = pd.read_csv('results/mi_results.csv')
phi_data = pd.read_csv('results/phi_results.csv')
baseline_data = pd.read_csv('results/baseline_results_imputed.csv')

# Create a figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

# Metrics to plot (excluding num_features column)
metrics = ['mt', 'dep', 'el', 'pos']
names = ["Machine Translation", "Dependency Parsing", "Entity Linking", "Part-of-Speech Tagging"]

# Create violin plots for each metric
for idx, metric in enumerate(metrics):
    # Prepare data for plotting
    plot_data = pd.DataFrame({
        'Method': ['MI'] * len(mi_data) + ['PHI'] * len(phi_data),
        'Value': pd.concat([mi_data[metric], phi_data[metric]]),
    })
    
    # Create violin plot with split using hue
    sns.violinplot(data=plot_data, x='Method', y='Value', hue='Method', ax=axes[idx], split=True)
    
    # Add horizontal line for baseline
    baseline_value = baseline_data[metric].iloc[0]
    axes[idx].axhline(y=baseline_value, color='red', linestyle='--', label='Baseline')
    
    axes[idx].set_title(f'{names[idx]}')
    axes[idx].set_ylabel('NDCG@3')
    axes[idx].legend()
    
# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('violin_plots.png', dpi=300, bbox_inches='tight')
plt.close() 