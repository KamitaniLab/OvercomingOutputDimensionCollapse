#! /bin/bash

settings=(
  "1200"
  "600"
  "300"
  "150"
)
test_dataset_names=(
  "ImageNetTest"
  "ArtificialShapes"
)
for setting in ${settings[@]}; do
  for test_dataset_name in ${test_dataset_names[@]}; do
    echo "Processing $setting $test_dataset_name best prediction feature"
    uv run python analysis/1_ODC_on_real_data/4_plot/calculate-mse.py --setting $setting --feature_type "best-prediction-feature" --test_dataset_name $test_dataset_name
    echo "Processing $setting $test_dataset_name brain prediction feature"
    uv run python analysis/1_ODC_on_real_data/4_plot/calculate-mse.py --setting $setting --feature_type "decoded-feature-ridge-alpha1000.0" --test_dataset_name $test_dataset_name
  done
done