#! /bin/bash

settings=(
  "1200"
  "600"
  "300"
  # "150"
)
test_dataset_names=(
  "ImageNetTest"
  "ArtificialShapes"
)
echo "Processing true features"
for test_dataset_name in ${test_dataset_names[@]}; do
  echo "Processing $test_dataset_name"
  uv run python analysis/1_ODC_on_real_data/3_reconstruction/iCNN.py --test_dataset_name $test_dataset_name
done
for setting in ${settings[@]}; do
  for test_dataset_name in ${test_dataset_names[@]}; do
    echo "Processing $setting best prediction feature"
    uv run python analysis/1_ODC_on_real_data/3_reconstruction/iCNN_best_prediction.py --setting $setting --test_dataset_name $test_dataset_name
    echo "Processing $setting brain prediction feature"
    uv run python analysis/1_ODC_on_real_data/3_reconstruction/iCNN_brain_prediction.py --setting $setting --test_dataset_name $test_dataset_name
  done
done