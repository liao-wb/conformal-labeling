import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

def plot_results(models, target_fdr_list, fdr_list, power_list, fdr_std_list=None, power_std_list=None, column_name="FDR and Power",
                     figsize=(18, 7), max_cols=3):
    """
       Plot FDR (False Discovery Rate) and Power results for multiple models with optional error bands.
       Parameters:
       -----------
       models : list of str, len(models) = n_models
           Names of the models to plot (will appear as subplot titles)
           Every model will be one subplot. For example, if len(models) = 6, it will plot 6 different figures
       target_fdr_list : list of arrays, shape: [ numpy array of shape [k, ] for _ in range(n_models)], k means the number of point in a figure
           List of target FDR levels for each model (x-axis values)
       fdr_list : list of arrays, shape: the same as target_fdr_list
           List of achieved FDR values for each model (y-axis values for FDR curve)
       power_list : list of arrays, shape: the same as target_fdr_list
           List of power values for each model (y-axis values for power curve)
       fdr_std_list : list of arrays, optional
           List of standard deviations for FDR values (for error bands)
       power_std_list : list of arrays, optional
           List of standard deviations for power values (for error bands)
       column_name : str, optional
           Label for the y-axis (default: "FDR and Power")
       figsize : tuple, optional
           Figure size in inches (default: (12, 6))
       max_cols : int, optional
           Maximum number of columns in the subplot grid (default: 3)
       """
    SMALL_SIZE = 18
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 16

    plt.rc('font', size=MEDIUM_SIZE)
    plt.rc('axes', titlesize=MEDIUM_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=MEDIUM_SIZE)
    plt.rc('ytick', labelsize=MEDIUM_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)

    # Calculate grid dimensions
    n_models = len(models)
    n_rows = (n_models + max_cols - 1) // max_cols
    n_cols = min(n_models, max_cols)

    # Create figure
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False, constrained_layout=True)
    axs = axs.flatten()

    # Plot each model's results
    for i, (model, target_fdrs, fdrs, powers) in enumerate(zip(models, target_fdr_list, fdr_list, power_list)):
        ax = axs[i]

        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_facecolor('#f0f4f8')

        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.spines['bottom'].set_color('#D3D3D3')
        ax.spines['left'].set_color('#D3D3D3')
        ax.spines['right'].set_color('#D3D3D3')
        ax.spines['top'].set_color('#D3D3D3')

        ax.plot(target_fdrs, fdrs, 'bo--', label='FDR', markersize=6)
        if fdr_std_list is not None:
            ax.fill_between(target_fdrs, np.clip(fdrs - fdr_std_list[i], 0, 1),
                          np.clip(fdrs + fdr_std_list[i], 0, 1), alpha=1,
                          edgecolor='lightblue', facecolor='lightblue')

        ax.plot(target_fdrs, powers, 'rs--', label='Power', markersize=6)
        if power_std_list is not None:
            ax.fill_between(target_fdrs, np.clip(powers - power_std_list[i], 0, 1),
                          np.clip(powers + power_std_list[i], 0, 1), alpha=1,
                          edgecolor='lightpink', facecolor='lightpink')


        # Reference line
        ax.plot([0, 1], [0, 1], 'g--', alpha=0.75, zorder=0)

        # Customize subplot
        ax.set_title(f'{model}')
        if i // n_cols >= n_rows - 1:
            ax.set_xlabel("Target level of FDR")
        if i % n_cols == 0:
            ax.set_ylabel(column_name)

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.5, color='#E0E0E0')
        ax.legend(loc='best', prop={'size': 16})

    # Hide unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')
    plt.savefig("calibration_variance.pdf")
    plt.show()


def plot_results_with_budget_save(models, target_fdr_list, fdr_list, power_list, budget_save_list=None, fdr_std_list=None, power_std_list=None, budget_save_std_list=None ,column_name="FDR and Power",
                     figsize=(12, 6), max_cols=3):
    SMALL_SIZE = 18
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 16

    plt.rc('font', size=MEDIUM_SIZE)
    plt.rc('axes', titlesize=MEDIUM_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=MEDIUM_SIZE)
    plt.rc('ytick', labelsize=MEDIUM_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)

    # Calculate grid dimensions
    n_models = len(models)
    n_rows = (n_models + max_cols - 1) // max_cols
    n_cols = min(n_models, max_cols)

    # Create figure
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False, constrained_layout=True)
    axs = axs.flatten()

    # Plot each model's results
    for i, (model, target_fdrs, fdrs, powers, budget_save) in enumerate(zip(models, target_fdr_list, fdr_list, power_list, budget_save_list)):
        ax = axs[i]

        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_facecolor('#f0f4f8')

        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.spines['bottom'].set_color('#D3D3D3')
        ax.spines['left'].set_color('#D3D3D3')
        ax.spines['right'].set_color('#D3D3D3')
        ax.spines['top'].set_color('#D3D3D3')

        ax.plot(target_fdrs, fdrs, 'bo--', label='FDR', markersize=6)
        if fdr_std_list is not None:
            ax.fill_between(target_fdrs, np.clip(fdrs - fdr_std_list[i], 0, 1),
                          np.clip(fdrs + fdr_std_list[i], 0, 1), alpha=0.5,
                          edgecolor='lightblue', facecolor='lightblue')

        """ax.plot(target_fdrs, powers, 'rs--', label='Power', markersize=6)
        if power_std_list is not None:
            ax.fill_between(target_fdrs, np.clip(powers - power_std_list[i], 0, 1),
                          np.clip(powers + power_std_list[i], 0, 1), alpha=0.5,
                          edgecolor='lightpink', facecolor='lightpink')"""

        ax.plot(target_fdrs, budget_save, 'rs--', label='Budget save', markersize=6)  # 'C1' = orange
        if budget_save_std_list is not None:
            ax.fill_between(target_fdrs, np.clip(budget_save - budget_save_std_list[i], 0, 1),
                            np.clip(budget_save + budget_save_std_list[i], 0, 1), alpha=0.5,
                            edgecolor='moccasin', facecolor='moccasin')

        # Reference line
        ax.plot([0, 1], [0, 1], 'g--', alpha=0.75, zorder=0)

        # Customize subplot
        ax.set_title(f'{model}')
        if i // n_cols >= n_rows - 1:
            ax.set_xlabel("Target level of FDR")
        if i % n_cols == 0:
            ax.set_ylabel(column_name)

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.5, color='#E0E0E0')
        ax.legend(loc='best', prop={'size': 16})

    # Hide unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.show()


def plot_selective_evaluation():
    large_font_size = 20
    small_font_size = 16
    target_fdr_list = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90]
    all_fdp_list = [4.56, 9.84, 12.24, 15.50, 18.04, 19.74, 20.88, 21.02, 21.63, 21.62, 22.12]
    mistral_fdr_list = np.array([0.04302288, 0.09887138, 0.14896379, 0.1980011,  0.26550419, 0.26537725, 0.26519281,
  0.26479521, 0.26535569, 0.26542994, 0.26536287]) * 100
    gpt3_fdr_list = np.array([0.04838759, 0.09824116, 0.15003452, 0.19994998, 0.23733174, 0.23706347, 0.23709701,
                     0.23713533, 0.23681916, 0.23687425, 0.23698683]) * 100
    gpt4_fdr_list = np.array([0.0437033,  0.10314587, 0.14944392, 0.20050551, 0.2141485,  0.21397844, 0.21423952,
  0.21392096, 0.21370299, 0.21447186, 0.21368623]) * 100
    plt.rcParams['axes.edgecolor'] = '#CCCCCC'  # Lighter axis spine color

    # Create figure with gridspec for unequal subplot widths
    fig = plt.figure(figsize=(25, 8))  # Increased figure width slightly
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])  # Right plot wider than left
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Left plot (ax1)

    ax1.plot([0, 100], [0, 100], 'g--', alpha=0.75, zorder=0, label="Target FDR level")
    ax1.plot(
        target_fdr_list, all_fdp_list,
        marker='o',
        label="Multi-Model Labeling",
        markersize=8, linestyle='-', linewidth=3, color="#FF8C00", alpha=1.0
    )

    ax1.plot(
        target_fdr_list, mistral_fdr_list,
        marker='o',
        label="Mistral-7B",
        markersize=8, linestyle='-', linewidth=3, color='#4682B4', alpha=1.0
    )
    ax1.plot(
        target_fdr_list, gpt3_fdr_list,
        marker='o',
        label="GPT-3.5",
        markersize=8, linestyle='-', linewidth=3, color="#FF6B6B", alpha=1.0
    )
    ax1.plot(
        target_fdr_list, gpt4_fdr_list,
        marker='o',
        label="GPT-4",
        markersize=8, linestyle='-', linewidth=3, color='#6495ED', alpha=1.0
    )

    ax1.set_xlabel("Target FDR (α; %)", fontsize=large_font_size)
    ax1.set_ylabel("FDR", fontsize=large_font_size)
    ax1.tick_params(axis='both', labelsize=small_font_size)
    ax1.legend(loc='upper left', fontsize=small_font_size, framealpha=1, shadow=True)

    # Right plot (ax2) - Bar chart
    targets = [10, 15, 20]  # Target levels
    models = ['Mistral-7B', 'GPT-3.5-turbo', 'GPT-4-turbo']
    data0 = {
        10: [30.6, 12.9, 3.85],
        15: [45.60, 8.2, 7.87],
        20: [60.30, 5.48, 7.50]
    }
#5: [1.27, 21.1, 0.5],
#20: [45.63, 9.02, 6.44]
    data = {
        10: [31.32, 36.33, 45.86],
        15: [45.26, 50.26, 60.80],
        20: [60.15, 68.81, 76.66]
    }
#5: [1.24, 15.83, 9.5],
#20: [60.15, 68.81, 76.66]

    # Prepare data for plotting
    x = np.arange(len(targets))
    width = 0.22  # Width of bars

    # Define colors for each model
    colors = ['#4682B4', "#FF6B6B", '#6495ED']
    #['#4682B4', "#6495ED", '#FF6B6B']
    #['#FF6B6B', '#4ECDC4', '#FFD166']
    # Plot the "Three Together" stacked bar
    bottoms = np.zeros(len(targets))
    for i, model in enumerate(models):
        values = [data0[t][i] for t in targets]
        ax2.bar(x - width, values, width, bottom=bottoms,
                color=colors[i], alpha=0.8)
        bottoms += values
    # Add outline for "Three Together"
    total_bar = ax2.bar(x - width, [sum(data0[t]) for t in targets], width,
                        label="Three Together", color='none', linewidth=1)

    # Plot individual bars for each model in reverse order
    bars = [total_bar]
    for i, model in enumerate(reversed(models)):  # Reverse the order: GPT-4-turbo, GPT-3.5-turbo, Mistral-7B
        model_idx = len(models) - 1 - i  # Map to original data indices
        values = [data[t][model_idx] for t in targets]
        offset = width * i  # GPT-4-turbo at x (i=0), others follow
        bar = ax2.bar(x + offset, values, width, label=model, color=colors[model_idx], alpha=0.8)
        bars.append(bar)

    # Customize the plot
    ax2.set_xlabel('Target FDR (α; %)', fontsize=large_font_size)
    ax2.set_ylabel('Budget Save (%)', fontsize=large_font_size)
    #ax2.set_title('Model Performance Across Agreement Levels', fontsize=large_font_size)
    ax2.set_xticks(x)
    ax2.set_xticklabels(targets, fontsize=small_font_size)
    ax2.tick_params(axis='y', labelsize=small_font_size)

    # Create legend with reversed order for "Three Together"
    handles, labels = ax2.get_legend_handles_labels()

    #ax2.legend(handles, labels, bbox_to_anchor=(1, 1), loc='upper left', fontsize=small_font_size)

    number_font_size = 12
    # Add value labels on top of bars
    for i, target in enumerate(targets):
        # Three Together
        total_height = 0
        for j, value in enumerate([data0[target][k] for k in range(len(models))]):
            ax2.text(i - width, total_height + value / 2, f'{value:.1f}%',
                     ha='center', va='center', fontsize=number_font_size, color='white')
            total_height += value
        ax2.text(i - width, total_height + 1, f'{sum(data0[target]):.1f}%',
                 ha='center', va='bottom', fontsize=small_font_size, color='black')

        # Individual models in reverse order
        for j, model in enumerate(reversed(models)):
            model_idx = len(models) - 1 - j
            value = data[target][model_idx]
            offset = width * j
            ax2.text(i + offset, value+2, f'{value:.1f}%',
                     ha='center', va='center', fontsize=small_font_size, color='black')

    # Add grid for better readability
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    # Adjust layout
    plt.tight_layout()
    plt.savefig("multimodel.pdf")
    plt.show()

plot_selective_evaluation()
