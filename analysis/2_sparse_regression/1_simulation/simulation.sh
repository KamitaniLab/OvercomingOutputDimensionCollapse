#! /bin/bash

data_types=(
  "baseline"
  "gaussian-weight"
  "correlated-signal"
  "input-noise"
)
nonzero_ratios=(
  "0.01"
  "0.02"
  "0.04"
  "0.08"
  "0.16"
  "0.32"
)

for data_type in ${data_types[@]}; do
  for nonzero_ratio in ${nonzero_ratios[@]}; do
    echo "Processing $data_type $nonzero_ratio"
    uv run python analysis/2_sparse_regression/1_simulation/simulation.py --data_type $data_type --nonzero_ratio $nonzero_ratio
  done
done