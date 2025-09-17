from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from bdpy.dataform import save_array
from bdpy.dl.torch.models import layer_map
from torch.utils.data import DataLoader


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Make decoded feature")
    parser.add_argument(
        "--setting",
        type=str,
        required=True,
        choices=["1200", "600", "300", "150"],
        help="Setting for feature extraction",
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
        "--proc_dir",
        type=str,
        default="./assets/1_ODC_on_real_data/calculate_best_prediction",
        help="Directory for processing",
    )
    parser.add_argument(
        "--ridge_alpha",
        type=float,
        default=1000.0,
        help="Ridge regression alpha value (default: 1000.0)",
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

    return parser.parse_args()


def get_config(args):
    """Get configuration settings"""
    configs = {
        "results_dir_projection": Path(
            f"{args.output_dir}/{args.test_dataset_name}/decoded-feature-ridge-alpha{args.ridge_alpha}/{args.setting}"
        ),
        "concatenated_feature_dir": Path(args.proc_dir) / args.setting,
        "feature_map_path": Path(args.proc_dir) / "feature_map.pkl",
        "subject": args.subject,
        "roi": args.roi,
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


def get_projector(root_path: Path):
    """Get projector"""
    print("Loading standard scaler...")
    with (root_path / "standard_scaler.pkl").open("rb") as f:
        standard_scaler = pickle.load(f)
    mean_vector = standard_scaler.mean_
    std_vector = standard_scaler.scale_

    print("Loading selected features...")
    with (root_path / "selected_features.pkl").open("rb") as f:
        Y_train = pickle.load(f)

    # Y_train.shape = (N, D)

    def projector(m):
        # m.shape = (N,)
        y_pred = Y_train.T @ m  # (D,)
        return y_pred * std_vector + mean_vector

    return projector


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


def load_decoded_data(
    proc_dir: Path, setting: str, test_dataset_name: str, ridge_alpha: float
):
    """Load decoded data"""
    print("Loading decoded data...")

    # Load decoded I dataset
    input_path = (
        proc_dir / setting / test_dataset_name / f"ridge-alpha{ridge_alpha}/I_pred.pkl"
    )
    with input_path.open("rb") as f:
        decoded_I_dataset = pickle.load(f)
    print(f"decoded_I_dataset.shape: {decoded_I_dataset.shape}")

    # Load labels
    input_path = (
        proc_dir / setting / test_dataset_name / f"ridge-alpha{ridge_alpha}/labels.pkl"
    )
    with input_path.open("rb") as f:
        labels = pickle.load(f)
    print(f"labels.shape: {labels.shape}")

    return decoded_I_dataset, labels


def setup_data_loader(decoded_I_dataset, labels):
    """Setup data loader"""
    print("Setting up data loader...")

    # Create a simple dataset that returns decoded data and labels
    class DecodedDataset:
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            return self.data[index], self.labels[index]

    dataset = DecodedDataset(decoded_I_dataset, labels)

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
    )

    return data_loader


def process_stimulus(
    idx: int,
    decoded_I,
    label,
    layer_names,
    feature_selector,
    projector,
    results_dir_projection,
    to_path_name,
    subject: str,
    roi: str,
):
    """Process individual stimulus"""
    stimulus_name = label[0]
    print(f"Stimulus [{idx+1}]: {stimulus_name}")

    print("Making decoded features...")

    # Apply projector
    decoded_feature = projector(decoded_I.detach().cpu().numpy().flatten())

    # Process each layer
    for layer_name in layer_names:
        feature = feature_selector(decoded_feature, layer_name)

        # Save decoded features
        save_dir = os.path.join(
            results_dir_projection, to_path_name[layer_name], subject, roi
        )
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

    # Load decoded data
    decoded_I_dataset, labels = load_decoded_data(
        Path(args.proc_dir), args.setting, args.test_dataset_name, args.ridge_alpha
    )

    # Setup data loader
    data_loader = setup_data_loader(decoded_I_dataset, labels)

    # Load projector
    print("Loading projector...")
    if not configs["concatenated_feature_dir"].exists():
        raise FileNotFoundError(
            f"Concatenated feature directory not found: {configs['concatenated_feature_dir']}"
        )

    projector = get_projector(configs["concatenated_feature_dir"])

    # Load feature map and create selector
    print("Setting up feature selector...")
    if not configs["feature_map_path"].exists():
        raise FileNotFoundError(f"Feature map not found: {configs['feature_map_path']}")

    feature_map_renamed = load_feature_map(configs["feature_map_path"])
    feature_selector = get_feature_selector(feature_map_renamed)

    # Process all stimuli
    print("Starting processing...")
    print(f"Processing {len(data_loader)} stimuli")
    for idx, (decoded_I, label) in enumerate(data_loader):
        try:
            process_stimulus(
                idx,
                decoded_I,
                label,
                layer_names,
                feature_selector,
                projector,
                configs["results_dir_projection"],
                to_path_name,
                configs["subject"],
                configs["roi"],
            )
        except Exception as e:
            print(f"Error processing stimulus {idx}: {e}")
            continue

    print("Processing completed!")
