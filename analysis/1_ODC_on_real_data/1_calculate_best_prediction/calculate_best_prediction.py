from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from bdpy.dataform import Features, save_array
from bdpy.dl.torch.models import layer_map
from torch.utils.data import DataLoader

from overcoming_output_dimension_collapse.icnn_replication.dataset import (
    FeaturesDataset,
    RenameFeatureKeys,
)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Make projected feature")
    parser.add_argument(
        "--setting",
        type=str,
        required=True,
        choices=["1200", "600", "300", "150"],
        help="Setting for # of training samples",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./assets/1_ODC_on_real_data/features",
        help="Output directory for results",
    )
    parser.add_argument(
        "--test_dataset_name",
        type=str,
        default="ImageNetTest",
        help="Test dataset name",
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        default=None,
        help="Directory for features (default: based on test_dataset_name)",
    )
    parser.add_argument(
        "--proc_dir",
        type=str,
        default="./assets/1_ODC_on_real_data/calculate_best_prediction",
        help="Directory for processing",
    )

    args = parser.parse_args()

    # Set default feature_dir based on test_dataset_name
    if args.feature_dir is None:
        if args.test_dataset_name == "ImageNetTest":
            args.feature_dir = "./data/features/ImageNetTest/caffe/VGG19/"
        elif args.test_dataset_name == "ArtificialShapes":
            args.feature_dir = "./data/features/ArtificialShapes/caffe/VGG19/"
        else:
            raise ValueError(f"Unknown test_dataset_name: {args.test_dataset_name}")

    return args


def get_config(args):
    """Get configuration settings"""
    configs = {
        "feature_root_path": Path(args.feature_dir),
        "results_dir_projection": Path(
            f"{args.output_dir}/{args.test_dataset_name}/best-prediction-feature/{args.setting}"
        ),
        "pseudo_inverse_dir": Path(args.proc_dir) / args.setting,
        "feature_map_path": Path(args.proc_dir) / "feature_map.pkl",
    }

    return configs


def get_layer_names():
    """Get layer names to process"""
    to_layer_name = layer_map("vgg19")
    to_path_name = dict(zip(to_layer_name.values(), to_layer_name.keys(), strict=False))
    layer_names = list(to_layer_name.values())

    # Remove "relu" layers
    layer_names = [
        layer_name
        for layer_name in layer_names
        if "relu" not in to_path_name[layer_name]
    ]

    to_layer_name = {k: v for k, v in to_layer_name.items() if v in layer_names}

    return layer_names, to_path_name, to_layer_name


def get_distortion_model(root_path: Path):
    """Get distortion model"""
    print("Loading standard scaler...")
    with (root_path / "standard_scaler.pkl").open("rb") as f:
        standard_scaler = pickle.load(f)
    mean_vector = standard_scaler.mean_
    std_vector = standard_scaler.scale_

    print("Loading projection matrix...")
    with (root_path / "pseudo_inverse.pkl").open("rb") as f:
        pseudo_inverse = pickle.load(f)  # (N, N)
    with (root_path / "selected_features.pkl").open("rb") as f:
        selected_features = pickle.load(f)  # (N, D)

    def distortion_model(h):
        # h.shape = (D,)
        h = (h - mean_vector) / std_vector
        distorted_h = selected_features.T @ (pseudo_inverse @ (selected_features @ h))
        return distorted_h * std_vector + mean_vector

    return distortion_model


def get_feature_selector(feature_map: dict):
    """Get feature selector"""
    feature_map_slice_idx = {}
    slice_idx = 0
    for layer_name, feature_shape in feature_map.items():
        feature_map_slice_idx[layer_name] = slice(
            slice_idx, slice_idx + np.prod(feature_shape)
        )
        slice_idx += np.prod(feature_shape)

    def feature_selector(feature, layer_name: str):
        selected_feature = feature[feature_map_slice_idx[layer_name]].reshape(
            feature_map[layer_name]
        )
        return np.expand_dims(selected_feature, axis=0)

    return feature_selector


def load_feature_map(feature_map_path: Path):
    """Load feature map"""
    print("Loading feature map...")
    with feature_map_path.open("rb") as f:
        feature_map = pickle.load(f)

    # Layer name mapping
    layer_name_dict = {
        "conv1_1": "features[0]",
        "conv1_2": "features[2]",
        "conv2_1": "features[5]",
        "conv2_2": "features[7]",
        "conv3_1": "features[10]",
        "conv3_2": "features[12]",
        "conv3_3": "features[14]",
        "conv3_4": "features[16]",
        "conv4_1": "features[19]",
        "conv4_2": "features[21]",
        "conv4_3": "features[23]",
        "conv4_4": "features[25]",
        "conv5_1": "features[28]",
        "conv5_2": "features[30]",
        "conv5_3": "features[32]",
        "conv5_4": "features[34]",
        "fc6": "classifier[0]",
        "fc7": "classifier[3]",
        "fc8": "classifier[6]",
    }

    feature_map_renamed = {layer_name_dict[k]: v for k, v in feature_map.items()}
    return feature_map_renamed


def setup_data_loader(feature_root_path: Path, to_layer_name: dict):
    """Setup data loader"""
    print("Setting up data loader...")

    # Get stimulus names from the features store
    features_store = Features(feature_root_path.as_posix())
    stimulus_names = features_store.labels
    print(f"Found {len(stimulus_names)} stimuli in features")

    features_dataset = FeaturesDataset(
        root_path=feature_root_path,
        layer_path_names=list(to_layer_name.keys()),
        stimulus_names=stimulus_names,
        transform=RenameFeatureKeys(to_layer_name),
    )

    data_loader = DataLoader(
        features_dataset,
        batch_size=1,
        num_workers=1,
    )

    return data_loader, stimulus_names


def process_stimulus(
    idx: int,
    features,
    stimulus_name,
    layer_names,
    feature_selector,
    distortion_model,
    results_dir_projection,
    to_path_name,
):
    """Process individual stimulus"""
    print(f"Stimulus [{idx+1}]: {stimulus_name}")

    print("Making best prediction features...")

    # Check if all required layers are present
    missing_layers = [layer for layer in layer_names if layer not in features]
    if missing_layers:
        raise ValueError(f"Missing layers in features: {missing_layers}")

    # Concatenate features
    concated_feature = np.concatenate(
        [
            features[layer_name].detach().cpu().numpy().flatten()
            for layer_name in layer_names
        ]
    )

    # Apply distortion model
    distorted_concated_feature = distortion_model(concated_feature)

    # Process each layer
    for layer_name in layer_names:
        feature = feature_selector(distorted_concated_feature, layer_name)

        # Save best prediction features
        save_dir = os.path.join(results_dir_projection, to_path_name[layer_name])
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, f"{stimulus_name}.mat")
        save_array(save_file, feature, key="feat", dtype=np.float32, sparse=False)

        print(f"Saved {layer_name}: {feature.shape}")

    print("Done.")


if __name__ == "__main__":
    """Main processing"""
    # Parse arguments
    args = parse_arguments()

    # Get configuration
    configs = get_config(args)

    # Get layer names
    layer_names, to_path_name, to_layer_name = get_layer_names()
    print(f"Processing layers: {layer_names}")

    # Setup data loader
    data_loader, stimulus_names = setup_data_loader(
        configs["feature_root_path"], to_layer_name
    )

    # Load distortion model
    print("Loading distortion model...")
    if not configs["pseudo_inverse_dir"].exists():
        raise FileNotFoundError(
            f"Pseudo inverse directory not found: {configs['pseudo_inverse_dir']}"
        )

    distortion_model = get_distortion_model(configs["pseudo_inverse_dir"])

    # Load feature map and create selector
    print("Setting up feature selector...")
    if not configs["feature_map_path"].exists():
        raise FileNotFoundError(f"Feature map not found: {configs['feature_map_path']}")

    feature_map_renamed = load_feature_map(configs["feature_map_path"])
    feature_selector = get_feature_selector(feature_map_renamed)

    # Process all stimuli
    print("Starting processing...")
    print(f"Processing {len(data_loader)} stimuli")
    for idx, data in enumerate(data_loader):
        try:
            stimulus_name = stimulus_names[idx]
            process_stimulus(
                idx,
                data,
                stimulus_name,
                layer_names,
                feature_selector,
                distortion_model,
                configs["results_dir_projection"],
                to_path_name,
            )
        except Exception as e:
            print(f"Error processing stimulus {idx}: {e}")
            continue

    print("Processing completed!")
