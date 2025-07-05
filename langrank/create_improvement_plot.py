import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the data
mi_data = pd.read_csv('results/mi_results.csv')
phi_data = pd.read_csv('results/phi_results.csv')
baseline_data = pd.read_csv('results/baseline_results_imputed.csv')

# Metrics to average
metrics = ['mt', 'dep', 'el', 'pos']

# Calculate average improvements
mi_improvements = []
phi_improvements = []

for metric in metrics:
    baseline_value = baseline_data[metric].iloc[0]
    
    # Calculate percentage improvements for this metric
    mi_improvement = [(val - baseline_value) / baseline_value * 100 for val in mi_data[metric]]
    phi_improvement = [(val - baseline_value) / baseline_value * 100 for val in phi_data[metric]]
    
    mi_improvements.append(mi_improvement)
    phi_improvements.append(phi_improvement)

# Convert to numpy arrays and calculate mean and std
mi_improvements = np.array(mi_improvements)
phi_improvements = np.array(phi_improvements)

mi_mean = np.mean(mi_improvements, axis=0)
phi_mean = np.mean(phi_improvements, axis=0)

mi_std = np.std(mi_improvements, axis=0)
phi_std = np.std(phi_improvements, axis=0)

# Create the plot
plt.figure(figsize=(10, 6))

# Plot lines with error bands
plt.plot(mi_data['num_features'], mi_mean, marker='o', color='#2ecc71', 
         label='MI', linewidth=2)
plt.fill_between(mi_data['num_features'], 
                 mi_mean - mi_std, mi_mean + mi_std,
                 color='#2ecc71', alpha=0.2)

plt.plot(phi_data['num_features'], phi_mean, marker='s', color='#e74c3c', 
         label='PHI', linewidth=2)
plt.fill_between(phi_data['num_features'], 
                 phi_mean - phi_std, phi_mean + phi_std,
                 color='#e74c3c', alpha=0.2)

# Add horizontal line at 0% (baseline)
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# Customize plot
plt.title('Average Improvement over Baseline Across All Tasks')
plt.xlabel('Number of Features')
plt.ylabel('Average Improvement over Baseline (%)')
plt.grid(True, alpha=0.3)
plt.legend()

# Add value labels on points
for x, y in zip(mi_data['num_features'], mi_mean):
    plt.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=10)
for x, y in zip(phi_data['num_features'], phi_mean):
    plt.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                xytext=(0,-15), ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('average_improvement_plot.png', dpi=300, bbox_inches='tight')
plt.close() 