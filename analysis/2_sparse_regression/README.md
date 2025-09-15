# Sparse Regression Analysis Pipeline

This directory contains the analysis pipeline for studying sparse regression with different data types and sparsity levels. The analysis consists of two main steps that should be executed in order.

## Pipeline Overview

The analysis pipeline consists of the following steps:

1. **Simulation** (`1_simulation/`)
2. **Plot Results** (`2_plot/`)

## Execution Order

Execute the following shell scripts in order from the **project root directory**:

### Step 1: Simulation

```bash
bash analysis/2_sparse_regression/1_simulation/simulation.sh
```

### Step 2: Plot Results

After running the shell script, you can plot the results:

Run with VS Code's Interactive Window. Open the plotting scripts and execute each `# %%` cell (Shift+Enter):

- `analysis/2_sparse_regression/2_plot/plot_simulation.py` - Plot simulation results
- `analysis/2_sparse_regression/2_plot/plot_theory.py` - Plot theoretical results

Details: [VS Code Interactive Window Documentation](https://code.visualstudio.com/docs/python/jupyter-support-py)

## Simulation Parameters

The simulation runs with the following data types and sparsity levels:

**Data Types:**

- `baseline`
- `gaussian-weight`
- `correlated-signal`
- `input-noise`

**Nonzero Ratios:**

- 0.01, 0.02, 0.04, 0.08, 0.16, 0.32

## Output

The analysis will generate results in the `assets/2_sparse_regression/` directory with the following structure:

```
assets/2_sparse_regression/
├── simulation/                    # Simulation results
│   ├── baseline/
│   ├── gaussian-weight/
│   ├── correlated-signal/
│   └── input-noise/
└── plots/                        # Generated plots
    ├── simulation/
    └── theory/
```

## Notes

- Make sure to run the scripts from the project root directory
- The plotting step depends on the output of the simulation step
- Check the individual `.py` files for specific parameters and requirements
- **The simulation takes a significant amount of time to complete** due to multiple parameter combinations
- Consider running steps individually due to the long execution time
