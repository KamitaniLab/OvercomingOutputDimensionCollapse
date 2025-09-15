# ODC (Output Dimension Collapse) Analysis Pipeline

This directory contains the analysis pipeline for studying Output Dimension Collapse on real data. The analysis consists of four main steps that should be executed in order.

## Pipeline Overview

The analysis pipeline consists of the following steps:

1. **Calculate Best Prediction** (`1_calculate_best_prediction/`)
2. **Calculate Brain Prediction** (`2_calculate_brain_prediction/`)
3. **Reconstruction** (`3_reconstruction/`)
4. **Plot Results** (`4_plot/`)

## Execution Order

Execute the following shell scripts in order from the **project root directory**:

### Step 1: Calculate Best Prediction

```bash
bash analysis/1_ODC_on_real_data/1_calculate_best_prediction/calculate_best_prediction.sh
```

### Step 2: Calculate Brain Prediction

```bash
bash analysis/1_ODC_on_real_data/2_calculate_brain_prediction/calculate_brain_prediction.sh
```

### Step 3: Reconstruction

```bash
bash analysis/1_ODC_on_real_data/3_reconstruction/iCNN.sh
```

### Step 4: Plot Results

```bash
bash analysis/1_ODC_on_real_data/4_plot/calculate-mse.sh
```

**Note**: After running the shell script, you can plot the results:

Run with VS Code's Interactive Window. Open `plot.py` and execute each `# %%` cell (Shift+Enter). Details: [VS Code Interactive Window Documentation](https://code.visualstudio.com/docs/python/jupyter-support-py)

## Output

The analysis will generate results in the `assets/1_ODC_on_real_data/` directory with the following structure:

```
assets/1_ODC_on_real_data/
├── calculate_best_prediction/     # Best prediction results
├── features/                      # Feature data
│   ├── ImageNetTest/
│   └── ArtificialShapes/
└── reconstruction/                # Reconstruction results
    ├── ImageNetTest/
    └── ArtificialShapes/
```

## Notes

- Make sure to run the scripts from the project root directory
- Each step depends on the output of the previous step
- Check the individual `.sh` files for specific parameters and requirements
- **The analysis takes a significant amount of time to complete**
- Consider running steps individually due to the long execution time
