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
  echo "Processing setting: $setting"
  uv run python analysis/1_ODC_on_real_data/1_calculate_best_prediction/calculate_psudo_inv.py --setting $setting 
  for test_dataset_name in ${test_dataset_names[@]}; do
    echo "Processing $test_dataset_name"
    uv run python analysis/1_ODC_on_real_data/1_calculate_best_prediction/calculate_best_prediction.py --setting $setting --test_dataset_name $test_dataset_name
  done
done