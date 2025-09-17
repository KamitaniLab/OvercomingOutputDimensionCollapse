# %%
"""
Plotting utilities for ODC (Output Dimension Collapse) analysis.

This notebook provides plotting functions for analyzing MSE results
between true features and projected/decoded features for different
sample sizes and data types (natural vs artificial images).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
# Before running the following, please move to the root directory (~/overcoming_output_dimension_collapse)
# cd ../../..

# %%
# Global plot settings
plt.rcParams["figure.dpi"] = 300
plt.rcParams["axes.linewidth"] = 3
plt.rcParams["font.size"] = 28
plt.rcParams["lines.linewidth"] = 2.0

# %%
# Configuration for file paths
BASE_PATH = "./assets/1_ODC_on_real_data/features"
SAMPLE_SIZES = [1200, 600, 300, 150]
DATASETS = {"natural": "ImageNetTest", "artificial": "ArtificialShapes"}
FEATURE_TYPES = {
    "projected": "best-prediction-feature",
    "decoded": "decoded-feature-ridge-alpha1000.0",
}


# %%
def generate_file_paths():
    """Generate file paths for all combinations of datasets, feature types, and sample sizes."""
    file_paths = {}

    for dataset_name, dataset_dir in DATASETS.items():
        file_paths[dataset_name] = {}
        for feature_name, feature_dir in FEATURE_TYPES.items():
            file_paths[dataset_name][feature_name] = [
                f"{BASE_PATH}/{dataset_dir}/{feature_dir}/{size}/mse_concatenated.pkl"
                for size in SAMPLE_SIZES
            ]

    return file_paths


# %%
# Generate file paths
file_paths = generate_file_paths()

# Extract specific file name lists for backward compatibility
projected_feature_file_names = file_paths["natural"]["projected"]
decoded_feature_file_names = file_paths["natural"]["decoded"]
projected_feature_artificial_file_names = file_paths["artificial"]["projected"]
decoded_feature_artificial_file_names = file_paths["artificial"]["decoded"]


# %%
def load_data():
    """Load MSE data from pickle files."""
    projected_feature_files = [
        pd.read_pickle(file_name) for file_name in projected_feature_file_names
    ]
    decoded_feature_files = [
        pd.read_pickle(file_name) for file_name in decoded_feature_file_names
    ]
    projected_feature_artificial_files = [
        pd.read_pickle(file_name)
        for file_name in projected_feature_artificial_file_names
    ]
    decoded_feature_artificial_files = [
        pd.read_pickle(file_name) for file_name in decoded_feature_artificial_file_names
    ]

    return (
        projected_feature_files,
        decoded_feature_files,
        projected_feature_artificial_files,
        decoded_feature_artificial_files,
    )


# %%
def plot_mse_line_chart(
    projected_feature_files,
    projected_feature_artificial_files,
    sample_names=None,
    figsize=(12, 6),
    xlim=(0, 90),
):
    """
    Plot line chart showing MSE values for different sample sizes.

    Parameters:
    - projected_feature_files: list of DataFrames containing projected feature MSE data
    - projected_feature_artificial_files: list of DataFrames containing artificial projected feature MSE data
    - sample_names: list of sample size labels
    - figsize: figure size tuple
    - xlim: x-axis limits tuple
    """
    if sample_names is None:
        sample_names = ["1200 samples", "600 samples", "300 samples", "150 samples"]

    # Get MSE values for each setting
    projected_feature_mses = [
        projected_feature_file["mse"]
        for projected_feature_file in projected_feature_files
    ]
    projected_feature_mses_artificial = [
        projected_feature_artificial_file["mse"]
        for projected_feature_artificial_file in projected_feature_artificial_files
    ]
    projected_feature_mses_concatenated = [
        np.concatenate(
            [projected_feature_mses[i], projected_feature_mses_artificial[i]]
        )
        for i in range(len(projected_feature_mses))
    ]

    # Sort by 150 samples (smallest sample size) MSE values
    sorted_indices = np.argsort(projected_feature_mses_concatenated[-1])
    projected_feature_mses_concatenated_sorted = [
        mse_values[sorted_indices] for mse_values in projected_feature_mses_concatenated
    ]

    # Create line plot
    plt.figure(figsize=figsize, dpi=300)
    indices = np.arange(len(projected_feature_mses_concatenated_sorted[0]))

    for i, mse_values in reversed(
        list(enumerate(projected_feature_mses_concatenated_sorted))
    ):
        plt.plot(indices, mse_values, label=sample_names[i], alpha=0.7)

    # Format plot
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.ticklabel_format(style="sci", scilimits=[-1, 3])
    plt.grid(axis="y")
    plt.ylim(0, None)
    plt.xlim(xlim[0], xlim[1])
    plt.xticks([])
    plt.show()


# %%
def plot_error_percentage_comparison(
    projected_feature_file,
    decoded_feature_file,
    projected_feature_artificial_file,
    decoded_feature_artificial_file,
    figsize=(6, 12),
    ylim=(0, 100),
):
    """
    Plot bar chart comparing error percentages between natural and artificial images.

    Parameters:
    - projected_feature_file: DataFrame containing projected feature MSE data
    - decoded_feature_file: DataFrame containing decoded feature MSE data
    - projected_feature_artificial_file: DataFrame containing artificial projected feature MSE data
    - decoded_feature_artificial_file: DataFrame containing artificial decoded feature MSE data
    - figsize: figure size tuple
    - ylim: y-axis limits tuple
    """
    # Calculate error percentages for natural images
    decoded_error = decoded_feature_file["mse"]
    projected_error = projected_feature_file["mse"]
    projected_error_percentage = (projected_error / decoded_error) * 100

    # Calculate error percentages for artificial images
    decoded_error_artificial = decoded_feature_artificial_file["mse"]
    projected_error_artificial = projected_feature_artificial_file["mse"]
    projected_error_percentage_artificial = (
        projected_error_artificial / decoded_error_artificial
    ) * 100

    # Create bar plot
    plt.figure(figsize=figsize, dpi=300)
    plt.bar(
        "Natural images",
        projected_error_percentage.mean(),
        yerr=projected_error_percentage.std(),
        width=0.8,
        capsize=4,
        color="#5A79A5",
    )
    plt.bar(
        "Artificial images",
        projected_error_percentage_artificial.mean(),
        yerr=projected_error_percentage_artificial.std(),
        width=0.8,
        capsize=4,
        color="#C69756",
    )

    # Format plot
    plt.ylim(ylim[0], ylim[1])
    for spine in plt.gca().spines.values():
        spine.set_linewidth(3)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.grid(axis="y")
    plt.tick_params(labelbottom=False, labelleft=True, labelright=False, labeltop=False)
    plt.show()


# %%
def plot_mse_scatter(
    projected_feature_file,
    decoded_feature_file,
    projected_feature_artificial_file,
    decoded_feature_artificial_file,
    figsize=(10, 10),
    natural_color="#5A79A5",
    artificial_color="#C69756",
):
    """
    Plot scatter plot comparing projected vs decoded feature MSE for natural and artificial images.

    Parameters:
    - projected_feature_file: DataFrame containing projected feature MSE data
    - decoded_feature_file: DataFrame containing decoded feature MSE data
    - projected_feature_artificial_file: DataFrame containing artificial projected feature MSE data
    - decoded_feature_artificial_file: DataFrame containing artificial decoded feature MSE data
    - figsize: figure size tuple
    - natural_color: color for natural images
    - artificial_color: color for artificial images
    """
    plt.figure(figsize=figsize, dpi=300)

    # Plot natural images
    plt.scatter(
        projected_feature_file["mse"],
        decoded_feature_file["mse"],
        s=25,
        color=natural_color,
        label="Natural images",
    )

    # Plot artificial images
    plt.scatter(
        projected_feature_artificial_file["mse"],
        decoded_feature_artificial_file["mse"],
        s=25,
        color=artificial_color,
        marker="s",
        label="Artificial images",
    )

    # Add diagonal line
    max_val = (
        max(
            projected_feature_file["mse"].max(),
            decoded_feature_file["mse"].max(),
            projected_feature_artificial_file["mse"].max(),
            decoded_feature_artificial_file["mse"].max(),
        )
        * 1.05
    )
    plt.xlim(0, max_val)
    plt.xticks([2 * 1e5, 4 * 1e5, 6 * 1e5])
    plt.ylim(0, max_val)
    plt.yticks([0, 2 * 1e5, 4 * 1e5, 6 * 1e5])
    plt.plot([0, max_val], [0, max_val], "--", color="black", linewidth=1)

    # Format plot
    plt.ticklabel_format(style="sci", scilimits=[-1, 3])
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.grid(True)
    plt.show()


# %%
# Load data and create plots
(
    projected_feature_files,
    decoded_feature_files,
    projected_feature_artificial_files,
    decoded_feature_artificial_files,
) = load_data()

# %%
# Plot MSE line chart for different sample sizes
plot_mse_line_chart(projected_feature_files, projected_feature_artificial_files)

# %%
# Plot error percentage comparison for first sample size (8x150)
plot_error_percentage_comparison(
    projected_feature_files[0],
    decoded_feature_files[0],
    projected_feature_artificial_files[0],
    decoded_feature_artificial_files[0],
)

# %%
# Plot scatter plots for all sample sizes
for (
    i,
    (
        projected_feature_file,
        decoded_feature_file,
        projected_feature_artificial_file,
        decoded_feature_artificial_file,
    ),
) in enumerate(
    zip(
        projected_feature_files,
        decoded_feature_files,
        projected_feature_artificial_files,
        decoded_feature_artificial_files,
        strict=False,
    )
):
    sample_names = ["1200 samples", "600 samples", "300 samples", "150 samples"]
    print(f"Plotting {sample_names[i]}")
    plot_mse_scatter(
        projected_feature_file,
        decoded_feature_file,
        projected_feature_artificial_file,
        decoded_feature_artificial_file,
    )

# %%
