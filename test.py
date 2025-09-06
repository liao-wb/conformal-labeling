import matplotlib.pyplot as plt
import numpy as np

# Data
targets = [80, 85, 90, 95]  # Target levels
models = ['Mistral-7B', 'GPT-3.5-turbo', 'GPT-4-turbo']
data = {
    80: [45.9, 12.8, 19.2],
    85: [26.0, 18.5, 23.1],
    90: [15.7, 24.2, 55.7],
    95: [14.5, 7.0, 28.9]
}

# Prepare data for plotting
x = np.arange(len(targets))
width = 0.2  # Width of bars
spacing_factor = 1  # Spacing between bars

fig, ax = plt.subplots(figsize=(16, 8))  # Figure size

# Define colors for each model
colors = ['#FF6B6B', '#4ECDC4', '#FFD166']

# Plot the "Three Together" stacked bar
bottoms = np.zeros(len(targets))
for i, model in enumerate(models):
    values = [data[t][i] for t in targets]
    ax.bar(x - width, values, width, bottom=bottoms,
           color=colors[i], alpha=0.8)
    bottoms += values
# Add outline for "Three Together"
total_bar = ax.bar(x - width, [sum(data[t]) for t in targets], width,
                   label="Three Together", color='none', edgecolor='black', linewidth=1)

# Plot individual bars for each model in reverse order
bars = [total_bar]
for i, model in enumerate(reversed(models)):  # Reverse the order: GPT-4-turbo, GPT-3.5-turbo, Mistral-7B
    model_idx = len(models) - 1 - i  # Map to original data indices
    values = [data[t][model_idx] for t in targets]
    offset = width * i  # GPT-4-turbo at x (i=0), others follow
    bar = ax.bar(x + offset, values, width, label=model, color=colors[model_idx], alpha=0.8)
    bars.append(bar)

# Customize the plot
ax.set_xlabel('Target Human Agreement (1 - α; %)', fontsize=12, fontweight='bold')
ax.set_ylabel('Evaluator Composition (%)', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Across Agreement Levels', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(targets)

# Create legend with reversed order for "Three Together"
handles, labels = ax.get_legend_handles_labels()
# Move "Three Together" to the end
three_together_idx = labels.index("Three Together")
handles.append(handles.pop(three_together_idx))
labels.append(labels.pop(three_together_idx))
ax.legend(handles, labels, title='Models', bbox_to_anchor=(1.05, 1), loc='upper right')

# Add value labels on top of bars
for i, target in enumerate(targets):
    # Three Together
    total_height = 0
    for j, value in enumerate([data[target][k] for k in range(len(models))]):
        ax.text(i - width, total_height + value / 2, f'{value:.1f}%',
                ha='center', va='center', fontsize=9, color='black')
        total_height += value
    ax.text(i - width, total_height + 1, f'{sum(data[target]):.1f}%',
            ha='center', va='bottom', fontsize=9, color='black')

    # Individual models in reverse order
    for j, model in enumerate(reversed(models)):
        model_idx = len(models) - 1 - j
        value = data[target][model_idx]
        offset = width * j
        ax.text(i + offset, value / 2, f'{value:.1f}%',
                ha='center', va='center', fontsize=9, color='black')

# Add grid for better readability
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Adjust layout and display
plt.tight_layout()
plt.show()