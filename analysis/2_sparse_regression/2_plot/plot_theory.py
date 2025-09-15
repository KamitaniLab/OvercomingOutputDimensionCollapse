# %%
"""
Theory plotting utilities for sparse regression analysis.

This notebook provides interactive plotting functions for analyzing
sparse regression theory, including risk landscapes and optimal selection curves.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from matplotlib.colors import LinearSegmentedColormap
from overcoming_output_dimension_collapse.sparse_regression.solve_theory import SparseRegressionTheory

# Global plot settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 28
plt.rcParams['lines.linewidth'] = 2.0

# %%
# Initialize solver
solver = SparseRegressionTheory()

# %%
# Create custom colormap for risk landscapes
# Redsの一部だけを抽出して新しいカラーマップを作成
reds_cmap = plt.cm.Reds
colors = reds_cmap(np.linspace(0.0, 0.7, 256))  # 0.0から0.7の範囲を使用
custom_reds = LinearSegmentedColormap.from_list('custom_reds', colors)

# %%
def plot_risk_landscape(sigma, data_scale, a, 
                       p_sel_range=(0.01, 1.0), pi_range=(0.01, 1.0), 
                       n_points=100, figsize=(10, 10), vmax=1.2):
    """
    Plot risk landscape with constraints.
    
    Parameters:
    - sigma: noise level
    - data_scale: data scaling parameter
    - a: sparsity parameter
    - p_sel_range: tuple of (min, max) for p_sel values
    - pi_range: tuple of (min, max) for pi values
    - n_points: number of points for each dimension
    - figsize: figure size
    - vmax: maximum value for colorbar
    """
    # Parameters
    p_sel_values = np.linspace(p_sel_range[0], p_sel_range[1], n_points)
    pi_values = np.linspace(pi_range[0], pi_range[1], n_points)

    # Compute risk landscape
    risk_values = np.zeros((len(p_sel_values), len(pi_values)))
    for i, p_sel in enumerate(p_sel_values):
        for j, pi in enumerate(pi_values):
            risk_values[i, j] = solver.risk_from_p_sel_and_pi(p_sel, pi, sigma, data_scale)

    # Create plot
    plt.figure(figsize=figsize)

    # Plot contour
    cp = plt.contourf(p_sel_values, pi_values, risk_values.T, 
                     levels=10, cmap=custom_reds, vmin=0, vmax=vmax)

    # Add constraint lines
    X, Y = np.meshgrid(np.linspace(0, 1, 101), np.linspace(0, 1, 101))
    mask = Y < X * a
    plt.contourf(X, Y, mask.T, levels=[0.5, 1], colors='black', hatches=['//'], alpha=0)

    # Add boundary lines
    plt.plot([0.0, a], [0.0, 1.0], color='black', linewidth=2)
    plt.plot([0.0, 1.0], [0.0, 1.0], color='black', linestyle='--', linewidth=2)

    # Set plot properties
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlim(0, 1.01)
    plt.ylim(0, 1.01)
    plt.xticks(np.arange(0.2, 1.1, 0.2))
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.gca().set_aspect('equal', adjustable='box')

    plt.show()

# %%
# Plot 1: Risk landscape with constraints
plot_risk_landscape(sigma=0.1, data_scale=0.2, a=0.08)

# %%
plot_risk_landscape(sigma=0.1, data_scale=0.8, a=0.08)
# %%
def plot_colorbar_only(sigma, data_scale, a,
                      p_sel_range=(0.01, 1.0), pi_range=(0.01, 1.0),
                      n_points=100, figsize=(10, 4), vmax=1.2):
    """
    Plot colorbar only for risk landscape.
    
    Parameters:
    - sigma: noise level
    - data_scale: data scaling parameter
    - a: sparsity parameter (not used in computation, kept for consistency)
    - p_sel_range: tuple of (min, max) for p_sel values
    - pi_range: tuple of (min, max) for pi values
    - n_points: number of points for each dimension
    - figsize: figure size
    - vmax: maximum value for colorbar
    """
    # Parameters
    p_sel_values = np.linspace(p_sel_range[0], p_sel_range[1], n_points)
    pi_values = np.linspace(pi_range[0], pi_range[1], n_points)

    # Compute risk landscape (dummy data for colorbar)
    risk_values = np.zeros((len(p_sel_values), len(pi_values)))
    for i, p_sel in enumerate(p_sel_values):
        for j, pi in enumerate(pi_values):
            risk_values[i, j] = solver.risk_from_p_sel_and_pi(p_sel, pi, sigma, data_scale)

    # Create colorbar plot
    plt.figure(figsize=figsize)
    cp = plt.contourf(p_sel_values, pi_values, risk_values.T, 
                     levels=10, cmap=custom_reds, vmin=0, vmax=vmax)
    cbar = plt.colorbar(cp, orientation='horizontal')
    cbar.outline.set_visible(False)
    plt.gca().set_visible(False)  # Hide main plot area
    plt.show()

# %%
# Plot 2: Colorbar only
plot_colorbar_only(sigma=0.1, data_scale=0.8, a=0.04)

# %%
def plot_pi_curves(sigma, data_scale, a_list,
                  p_sel_range=(0.01, 1.0), n_points=100, figsize=(10, 10), 
                  linewidth=2, show_grid=True):
    """
    Plot pi curves for different sparsity parameters.
    
    Parameters:
    - sigma: noise level
    - data_scale: data scaling parameter
    - a_list: list of sparsity parameters to plot
    - p_sel_range: tuple of (min, max) for p_sel values
    - n_points: number of points for the curve
    - figsize: figure size
    - linewidth: line width for curves
    - show_grid: whether to show grid
    """
    # Parameters
    p_sel_values_line = np.linspace(p_sel_range[0], p_sel_range[1], n_points)

    # Create plot
    plt.figure(figsize=figsize)
    colors = plt.cm.viridis(np.linspace(0.95, 0.1, len(a_list)))

    # Plot curves for each a value
    for idx, a in enumerate(a_list):
        pi_line = [solver.solve_pi_given_a_p_select_and_data_scale(a, p_sel, data_scale, sigma) 
                   for p_sel in p_sel_values_line]
        plt.plot(p_sel_values_line, pi_line, color=colors[idx], label=f'a={a}', linewidth=linewidth)

    # Add diagonal line
    plt.plot([0.0, 1.0], [0.0, 1.0], color='black', linestyle='--', linewidth=linewidth)

    # Set plot properties
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlim(0, 1.01)
    plt.ylim(0, 1.01)
    plt.xticks(np.arange(0.2, 1.1, 0.2))
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.gca().set_aspect('equal')
    if show_grid:
        plt.grid(True)
    plt.show()

# %%
# Plot 3: Pi curves for different sparsity parameters
plot_pi_curves(sigma=0.1, data_scale=0.2, a_list=[0.01, 0.02, 0.04, 0.08, 0.16, 0.32])
# %%
plot_pi_curves(sigma=0.1, data_scale=0.8, a_list=[0.01, 0.02, 0.04, 0.08, 0.16, 0.32])
# %%
def compute_optimal_results(sigma, a_vals, data_scale_vals):
    """
    Compute optimal results for different parameters.
    
    Parameters:
    - sigma: noise level
    - a_vals: list of sparsity parameters
    - data_scale_vals: list of data scale values
    
    Returns:
    - DataFrame with results
    """
    # Compute optimal results
    results = [
        (a, data_scale, *solver.solve_optimal_p_select_and_risk(a, sigma, data_scale))
        for a, data_scale in itertools.product(a_vals, data_scale_vals)
    ]

    # Compute naive results (p_select = 1)
    naive_results = [
        (1, data_scale, 1, *solver.solve_risk_and_tau2_for_p_select(1, sigma, data_scale, 1))
        for data_scale in data_scale_vals
    ]

    results.extend(naive_results)

    # Convert to DataFrame
    df_results = pd.DataFrame(results, columns=['a', 'data_scale', 'p_select', 'risk', 'tau2'])
    return df_results

# %%
def plot_risk_vs_data_scale(sigma, a_vals, data_scale_vals=None, 
                           figsize=(10, 7), line_width=3, ylim=(0, 1.1),
                           show_grid=True, log_scale=True):
    """
    Plot risk vs data scale for different sparsity parameters.
    
    Parameters:
    - sigma: noise level
    - a_vals: list of sparsity parameters
    - data_scale_vals: list of data scale values (if None, auto-generate)
    - figsize: figure size
    - line_width: line width for curves
    - ylim: tuple of (min, max) for y-axis limits
    - show_grid: whether to show grid
    - log_scale: whether to use log scale for x-axis
    """
    if data_scale_vals is None:
        data_scale_vals = [(2 ** k) / 100 for k in np.arange(0, 10, 0.1)]
    
    # Compute results
    df_results = compute_optimal_results(sigma, a_vals, data_scale_vals)

    # Create plot
    plt.figure(figsize=figsize)
    colors = plt.cm.viridis(np.linspace(0.95, 0.1, len(a_vals)))

    # Plot curves for each a value
    for a, color in zip(a_vals, colors):
        subset = df_results[df_results['a'] == a]
        plt.plot(subset['data_scale'], subset['risk'], 
                 label=f'a={a:.2f}', color=color, linewidth=line_width)

    # Plot naive solution
    naive_subset = df_results[df_results['a'] == 1]
    plt.plot(naive_subset['data_scale'], naive_subset['risk'], 
             label='naive', color='black', linewidth=line_width)

    # Set plot properties
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(line_width)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if log_scale:
        plt.xscale('log')
    plt.ylim(ylim[0], ylim[1])
    if show_grid:
        plt.grid(True)
    plt.show()

# %%
# Compute optimal results for different parameters
sigma = 0.1
a_vals = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
data_scale_vals = [(2 ** k) / 100 for k in np.arange(0, 10, 0.1)]

df_results = compute_optimal_results(sigma, a_vals, data_scale_vals)

# %%
# Plot 4: Risk vs data scale
plot_risk_vs_data_scale(sigma=0.1, a_vals=a_vals)

# %%