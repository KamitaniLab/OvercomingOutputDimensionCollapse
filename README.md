# Overcoming Output Dimension Collapse

This repository contains the code for the paper:
“Overcoming Output Dimension Collapse: How Sparsity Enables Zero-shot Brain-to-Image Reconstruction at Small Data Scales” by Kenya Otsuka, Yoshihiro Nagano, and Yukiyasu Kamitani.

## Overview

The repository consists of two main analysis pipelines:

1. **ODC on Real Data** (`analysis/1_ODC_on_real_data/`) - Analysis of ODC using real brain imaging data
2. **Sparse Regression** (`analysis/2_sparse_regression/`) - Theoretical and simulation-based analysis of sparse regression

## Installation

This project uses `uv` for dependency management. Install dependencies with:

```bash
uv sync
```

## Quick Start

### Dataset Download

Download the required datasets using the provided script:

```bash
uv run python scripts/download.py
```

This will automatically download and organize the following data:

- True features of each dataset in `data/features`
- fMRI data (.h5) of each dataset in `data/fmri`
- Model parameters in `data/models_shared`

The script downloads data from:

- [Deep Image Reconstruction](https://figshare.com/articles/dataset/Deep_Image_Reconstruction/7033577)
- [brain-decoding-cookbook](https://figshare.com/articles/dataset/brain-decoding-cookbook/21564384)
- [Spurious reconstruction from brain activity](https://figshare.com/articles/dataset/Spurious_reconstruction_from_brain_activity/27013342)

### ODC on Real Data Analysis

```bash
# Step 1: Calculate best prediction
bash analysis/1_ODC_on_real_data/1_calculate_best_prediction/calculate_best_prediction.sh

# Step 2: Calculate brain prediction
bash analysis/1_ODC_on_real_data/2_calculate_brain_prediction/calculate_brain_prediction.sh

# Step 3: Reconstruction
bash analysis/1_ODC_on_real_data/3_reconstruction/iCNN.sh

# Step 4: Plot results
bash analysis/1_ODC_on_real_data/4_plot/calculate-mse.sh
```

**Note**: After running the shell script, you can plot the results:

- `analysis/1_ODC_on_real_data/4_plot/plot.py` - Plot results

### Sparse Regression Analysis

```bash
# Step 1: Run simulations
bash analysis/2_sparse_regression/1_simulation/simulation.sh
```

**Note**: After running the shell script, you can plot the results:

- `analysis/2_sparse_regression/2_plot/plot_simulation.py` - Plot simulation results
- `analysis/2_sparse_regression/2_plot/plot_theory.py` - Plot theoretical results

## Project Structure

```
├── src/overcoming_output_dimension_collapse/   # Main package
│   ├── icnn_replication/                       # iCNN replication code
│   └── sparse_regression/                      # Sparse regression code
├── analysis/                                   # Analysis pipelines
│   ├── 1_ODC_on_real_data/                     # Real data ODC analysis
│   └── 2_sparse_regression/                    # Sparse regression analysis
├── assets/                                     # Generated results and data
├── data/                                       # Downloaded data
│   ├── features/                               # True features
│   ├── fmri/                                   # fMRI data
│   └── model_shared/                           # Model parameters
└── README.md                                   # This file
```

## Usage Notes

- All scripts should be run from the project root directory
- The analysis pipelines take significant time to complete
- Use VS Code's Interactive Window for plotting scripts (execute `# %%` cells)
