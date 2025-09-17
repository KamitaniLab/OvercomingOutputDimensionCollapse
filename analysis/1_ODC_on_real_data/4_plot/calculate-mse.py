from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from bdpy.dataform import Features
from bdpy.dl.torch.models import layer_map
from overcoming_output_dimension_collapse.icnn_replication.dataset import (
    DecodedFeaturesDataset,
    FeaturesDataset,
    RenameFeatureKeys,
)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Evaluate MSE between features and decoded features"
    )
    parser.add_argument(
        "--setting",
        type=str,
        required=True,
        choices=["1200", "600", "300", "150"],
        help="Setting for feature extraction",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        required=True,
        choices=["best-prediction-feature", "decoded-feature-ridge-alpha1000.0"],
        help="Feature type for feature extraction",
    )
    parser.add_argument(
        "--test_dataset_name",
        type=str,
        default="ImageNetTest",
        help="Test dataset name",
    )
    parser.add_argument(
        "--feature_dir", type=str, default=None, help="Directory for features"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./assets/1_ODC_on_real_data/features",
        help="Output directory for results",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="sub-01",
        help="Subject to use for brain data (default: sub-01)",
    )
    parser.add_argument(
        "--roi", type=str, default="VC", help="ROI to use for brain data (default: VC)"
    )

    args = parser.parse_args()

    if args.feature_dir is None:
        if args.test_dataset_name == "ImageNetTest":
            args.feature_dir = "./data/features/ImageNetTest/caffe/VGG19/"
        elif args.test_dataset_name == "ArtificialShapes":
            args.feature_dir = "./data/features/ArtificialShapes/caffe/VGG19/"
        else:
            raise ValueError(f"Unknown test_dataset_name: {args.test_dataset_name}")
    return args


def get_configuration(args) -> dict[str, Any]:
    """Get configuration parameters for the MSE calculation experiment."""
    return {
        "feature_type": args.feature_type,
        "subject": args.subject,
        "roi": args.roi,
        "paths": {
            "feature_root": Path(args.feature_dir),
            "decoded_feature_root": Path(
                f"{args.output_dir}/{args.test_dataset_name}/{args.feature_type}/{args.setting}"
            ),
        },
    }


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


def setup_data_loader(
    config: dict[str, Any], to_layer_name: dict
) -> tuple[FeaturesDataset, FeaturesDataset, list]:
    """Setup data loader"""
    print("Setting up data loader...")

    paths = config["paths"]
    feature_type = config["feature_type"]
    subject = config["subject"]
    roi = config["roi"]

    # Get stimulus names from the features store
    features_store = Features(paths["feature_root"].as_posix())
    stimulus_names = features_store.labels
    print(f"Found {len(stimulus_names)} stimuli in features")

    # Features dataset (always use original features)
    features_dataset = FeaturesDataset(
        root_path=paths["feature_root"],
        layer_path_names=list(to_layer_name.keys()),
        stimulus_names=stimulus_names,
        transform=RenameFeatureKeys(to_layer_name),
    )

    # Decoded/processed features dataset
    if feature_type == "best-prediction-feature":
        # For best-prediction-feature, use FeaturesDataset
        decoded_features_dataset = FeaturesDataset(
            root_path=paths["decoded_feature_root"],
            layer_path_names=list(to_layer_name.keys()),
            stimulus_names=stimulus_names,
            transform=RenameFeatureKeys(to_layer_name),
        )
    else:
        # For decoded features, use DecodedFeaturesDataset
        decoded_features_dataset = DecodedFeaturesDataset(
            root_path=paths["decoded_feature_root"],
            layer_path_names=list(to_layer_name.keys()),
            subject_id=subject,
            roi=roi,
            stimulus_names=stimulus_names,
            transform=RenameFeatureKeys(to_layer_name),
        )

    return features_dataset, decoded_features_dataset, stimulus_names


def calculate_mse(features, decoded_features, layer_names):
    """Calculate MSE between features and decoded features"""
    # Concatenate features
    concated_features = np.concatenate(
        [features[layer_name].flatten() for layer_name in layer_names]
    )
    concated_decoded_features = np.concatenate(
        [decoded_features[layer_name].flatten() for layer_name in layer_names]
    )

    # Calculate MSE
    mse = np.mean((concated_features - concated_decoded_features) ** 2)

    return mse


def process_stimulus(idx: int, features, decoded_features, stimulus_name, layer_names):
    """Process individual stimulus"""
    print(f"Stimulus [{idx+1}]: {stimulus_name}")

    # Check if all required layers are present
    missing_layers = [layer for layer in layer_names if layer not in features]
    if missing_layers:
        raise ValueError(f"Missing layers in features: {missing_layers}")

    missing_decoded_layers = [
        layer for layer in layer_names if layer not in decoded_features
    ]
    if missing_decoded_layers:
        raise ValueError(
            f"Missing layers in decoded features: {missing_decoded_layers}"
        )

    # Calculate MSE
    mse = calculate_mse(features, decoded_features, layer_names)

    print(f"MSE: {mse}")

    return mse


def save_results(mse_list, config: dict[str, Any]):
    """Save results"""
    print("Saving results...")

    # Create results DataFrame
    perf_df = pd.DataFrame(columns=["mse"])
    perf_df["mse"] = mse_list

    # Save results
    save_path = config["paths"]["decoded_feature_root"] / "mse_concatenated.pkl"
    perf_df.to_pickle(save_path)
    print(f"MSE results saved to {save_path}")

    return save_path


if __name__ == "__main__":
    """Main processing"""
    # Parse arguments
    args = parse_arguments()

    # Get configuration
    config = get_configuration(args)

    print("Configuration:")
    print(f"  Feature type: {config['feature_type']}")
    print(f"  Subject: {config['subject']}")
    print(f"  ROI: {config['roi']}")
    print(f"  Feature root: {config['paths']['feature_root']}")
    print(f"  Decoded feature root: {config['paths']['decoded_feature_root']}")

    # Get layer names
    layer_names, to_path_name, to_layer_name = get_layer_names()
    print(f"Processing layers: {layer_names}")

    # Setup datasets
    features_dataset, decoded_features_dataset, stimulus_names = setup_data_loader(
        config, to_layer_name
    )

    # Process all stimuli
    print("Starting MSE evaluation...")
    print(f"Processing {len(stimulus_names)} stimuli")

    mse_list = []

    for idx in range(len(stimulus_names)):
        try:
            stimulus_name = stimulus_names[idx]
            features = features_dataset[idx]
            decoded_features = decoded_features_dataset[idx]

            mse = process_stimulus(
                idx, features, decoded_features, stimulus_name, layer_names
            )
            mse_list.append(mse)
        except Exception as e:
            print(f"Error processing stimulus {idx}: {e}")
            continue

    # Save results
    save_path = save_results(mse_list, config)

    print("MSE evaluation completed!")
    print(f"Results saved to: {save_path}")
