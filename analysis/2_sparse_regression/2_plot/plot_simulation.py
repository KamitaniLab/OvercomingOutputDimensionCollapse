# %%
"""
Simulation plotting utilities for sparse regression analysis.

This notebook provides interactive plotting functions for analyzing
sparse regression simulation results, including data scale vs risk relationships
for different data types and settings.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
# Global plot settings
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 28
plt.rcParams["lines.linewidth"] = 2.0

# %%
# Before running the following, please move to the root directory (~/overcoming_output_dimension_collapse)
# cd ../../..


# %%
def convert_to_array(value):
    """
    Convert string representation of array into numpy array.

    Parameters:
    - value: string representation of array (e.g., '[1.0 2.0 3.0]')

    Returns:
    - numpy array of floats
    """
    # Remove the brackets, split by space, and convert to float
    cleaned_value = value.replace("[", "").replace("]", "").strip()
    return np.array([float(num) for num in cleaned_value.split()])


# %%
def plot_data_scale_vs_risk(
    df,
    datatype,
    setting,
    a_list=None,
    line_width=3,
    figsize=(10, 7),
    ylim=(0, 1.1),
    show_grid=True,
    log_scale=True,
):
    """
    Plot data scale vs risk relationship for different sparsity parameters.

    Parameters:
    - df: DataFrame containing simulation results
    - datatype: data type to filter data (e.g., 'baseline', 'gaussian-weight')
    - setting: setting to filter data (e.g., 'select-optimal')
    - a_list: list of sparsity parameters to plot
    - line_width: line width for curves
    - figsize: figure size
    - ylim: tuple of (min, max) for y-axis limits
    - show_grid: whether to show grid
    - log_scale: whether to use log scale for x-axis
    """
    # Create a copy to avoid SettingWithCopyWarning
    if a_list is None:
        a_list = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
    df_filtered = df[(df["data_type"] == datatype) & (df["setting"] == setting)].copy()
    df_filtered["mse_mean"] = df_filtered["mse"].apply(np.mean)
    df_filtered["data_scale"] = df_filtered["n_train"] / df_filtered["d_in"]
    df_pivot = df_filtered.pivot(
        index="data_scale", columns="nonzero_ratio", values="mse_mean"
    )

    plt.figure(figsize=figsize)
    colors = plt.cm.viridis(np.linspace(0.95, 0.1, len(a_list)))

    for nonzero_ratio, color in zip(a_list, colors, strict=False):
        plt.plot(
            df_pivot.index,
            df_pivot[nonzero_ratio],
            label=f"a={nonzero_ratio:.2f}",
            color=color,
            linewidth=line_width,
        )

    # Set plot properties
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(line_width)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if log_scale:
        plt.xscale("log")
    plt.ylim(ylim[0], ylim[1])
    if show_grid:
        plt.grid(True)
    plt.show()


# %%
# Load and prepare simulation data
df = pd.read_csv("assets/2_sparse_regression/simulation/results.csv")
df["mse"] = df["mse"].apply(convert_to_array)
df

# %%
# Plot results for different data types
plot_data_scale_vs_risk(df, "baseline", "select-optimal")

# %%
plot_data_scale_vs_risk(df, "gaussian-weight", "select-optimal")

# %%
plot_data_scale_vs_risk(df, "correlated-signal", "select-optimal")

# %%
plot_data_scale_vs_risk(df, "input-noise", "select-optimal")

# %%
