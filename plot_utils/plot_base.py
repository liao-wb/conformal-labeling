"""
Base Plotting Utilities.
Contains functions for standard FDR/Power plots and bar charts.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Style Constants
COLORS = {
    'fdr': 'blue', 
    'power': 'red', 
    'budget': 'orange',
    'mistral': '#4682B4',
    'gpt3': '#FF6B6B',
    'gpt4': '#6495ED'
}
STYLES = {
    'linewidth': 3,
    'markersize': 10,
    'alpha_line': 0.8,
    'alpha_fill': 0.2
}

def set_style(medium_size=20, big_size=24):
    """Sets matplotlib style parameters."""
    plt.rc('font', size=medium_size)
    plt.rc('axes', titlesize=big_size)
    plt.rc('axes', labelsize=medium_size)
    plt.rc('xtick', labelsize=medium_size)
    plt.rc('ytick', labelsize=medium_size)
    plt.rc('legend', fontsize=medium_size - 4)
    plt.rc('figure', titlesize=big_size)
    plt.rcParams['axes.edgecolor'] = '#CCCCCC'

def plot_fdr_power_curves(
    models, target_fdr_list, fdr_list, power_list, 
    fdr_std_list=None, power_std_list=None, 
    budget_save_list=None, budget_save_std_list=None,
    figsize=(18, 6), max_cols=3, output_file=None
):
    """
    Plots FDR and Power (and optionally Budget Save) curves for multiple models.
    """
    set_style()
    n_models = len(models)
    n_rows = (n_models + max_cols - 1) // max_cols
    n_cols = min(n_models, max_cols)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False, constrained_layout=True)
    axs = axs.flatten()

    for i, (model, target_fdrs) in enumerate(zip(models, target_fdr_list)):
        ax = axs[i]
        
        # Extract data for current model
        curr_fdrs = fdr_list[i]
        curr_powers = power_list[i] if power_list else None
        curr_budget = budget_save_list[i] if budget_save_list else None
        
        # Grid and Spines
        ax.set_facecolor('#f9f9f9')
        ax.grid(True, alpha=0.3, color='#E0E0E0')
        for spine in ax.spines.values():
            spine.set_color('#D3D3D3')

        # 1. FDR
        ax.plot(target_fdrs, curr_fdrs, 'o--', label='FDR', color=COLORS['fdr'], **STYLES)
        if fdr_std_list is not None:
            std = fdr_std_list[i]
            ax.fill_between(target_fdrs, np.clip(curr_fdrs - std, 0, 1), 
                            np.clip(curr_fdrs + std, 0, 1), color=COLORS['fdr'], alpha=STYLES['alpha_fill'])

        # 2. Power (Optional)
        if curr_powers is not None:
            ax.plot(target_fdrs, curr_powers, 's--', label='Power', color=COLORS['power'], **STYLES)
            if power_std_list is not None:
                std = power_std_list[i]
                ax.fill_between(target_fdrs, np.clip(curr_powers - std, 0, 1), 
                                np.clip(curr_powers + std, 0, 1), color=COLORS['power'], alpha=STYLES['alpha_fill'])

        # 3. Budget Save (Optional)
        if curr_budget is not None:
            ax.plot(target_fdrs, curr_budget, '^--', label='Budget Save', color=COLORS['budget'], **STYLES)
            if budget_save_std_list is not None:
                std = budget_save_std_list[i]
                ax.fill_between(target_fdrs, np.clip(curr_budget - std, 0, 1), 
                                np.clip(curr_budget + std, 0, 1), color=COLORS['budget'], alpha=STYLES['alpha_fill'])

        # Reference Line (y=x)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, lw=1)

        ax.set_title(model)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1.02)
        
        if i // n_cols == n_rows - 1:
            ax.set_xlabel("Target FDR Level")
        if i % n_cols == 0:
            ax.set_ylabel("Metric Value")
            
        if i == 0: # Legend only on first plot to reduce clutter
            ax.legend(loc='best')

    # Hide empty subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    plt.show()

# Note: The 'plot_selective_evaluation' function in original code was highly specific 
# to a set of hardcoded numbers (Mistral/GPT results). I have omitted it here as it 
# seems to be a one-off paper figure generation script rather than a reusable utility. 
# If needed, it can be kept as a standalone script in `experiments/paper_figures/`.
