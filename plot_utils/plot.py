import matplotlib.pyplot as plt
import numpy as np


def plot_results(models, target_fdr_list, fdr_list, power_list, fdr_std_list=None, power_std_list=None, column_name="FDR and Power",
                     figsize=(12, 6), max_cols=3):
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
                          np.clip(fdrs + fdr_std_list[i], 0, 1), alpha=0.5,
                          edgecolor='lightblue', facecolor='lightblue')

        ax.plot(target_fdrs, powers, 'rs--', label='Power', markersize=6)
        if power_std_list is not None:
            ax.fill_between(target_fdrs, np.clip(powers - power_std_list[i], 0, 1),
                          np.clip(powers + power_std_list[i], 0, 1), alpha=0.5,
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


# Example usage with synthetic dataset
if __name__ == "__main__":
    # Generate example dataset for 4 models
    models = ['GPT-3.5', 'LLaMA-2-7B', 'LLaMA-2-13B', 'OPT-6.7B']

    # Create target FDRs (same for all models in this example)
    base_targets = np.linspace(0.05, 0.95, 10)
    target_fdr_list = [base_targets.copy() for _ in models]

    # Generate synthetic FDR and Power dataset for each model
    fdr_list = []
    power_list = []

    for i in range(len(models)):
        # FDR that roughly tracks target but with some deviation
        fdrs = base_targets * (0.9 + 0.1 * np.random.randn(len(base_targets)))
        fdrs = np.clip(fdrs, 0, 1)

        # Power that increases with target FDR
        powers = 0.3 + 0.6 * base_targets + 0.1 * np.random.randn(len(base_targets))
        powers = np.clip(powers, 0, 1)

        fdr_list.append(fdrs)
        power_list.append(powers)

    std_list = np.random.uniform(low=0.1, high=0.5, size=(4, 10)) * 0.5
    # Plot the results
    plot_results(models, target_fdr_list, fdr_list, power_list, fdr_std_list=std_list, power_std_list=std_list)