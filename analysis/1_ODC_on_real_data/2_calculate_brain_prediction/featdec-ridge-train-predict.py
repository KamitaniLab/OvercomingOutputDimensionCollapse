import argparse
import pickle
from pathlib import Path

import bdpy
import numpy as np
from bdpy.dataform import Features
from sklearn.linear_model import Ridge


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train and predict feature decoding")
    parser.add_argument(
        "--setting",
        type=str,
        required=True,
        choices=["1200", "600", "300", "150"],
        help="Setting for feature extraction",
    )
    parser.add_argument(
        "--alpha",
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
    parser.add_argument(
        "--test_dataset_name",
        type=str,
        default="ImageNetTest",
        help="Test dataset name",
    )
    parser.add_argument(
        "--brain_train_dir",
        type=str,
        default="./data/fmri/",
        help="Directory to training brain data",
    )
    parser.add_argument(
        "--brain_test_dir",
        type=str,
        default="./data/fmri/",
        help="Directory to test brain data",
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        default="./data/features/ImageNetTraining/caffe/VGG19/",
        help="Directory to feature directory (default: ./data/features/ImageNetTraining/caffe/VGG19/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./assets/1_ODC_on_real_data/calculate_best_prediction",
        help="Output directory for results (default: ./assets/1_ODC_on_real_data/calculate_best_prediction)",
    )

    args = parser.parse_args()

    return args


def get_config(args):
    """Get configuration settings"""
    configs = {
        "alpha": args.alpha,
        "subject": args.subject,
        "data_brain_train_path": Path(args.brain_train_dir)
        / f"{args.subject}_ImageNetTraining_fmriprep_volume_native.h5",
        "data_brain_test_path": Path(args.brain_test_dir)
        / f"{args.subject}_{args.test_dataset_name}_fmriprep_volume_native.h5",
        "roi": f"ROI_{args.roi}",
        "feature_label_dir": Path(args.feature_dir),
        "assets_root_path": Path(
            f"{args.output_dir}/{args.setting}/{args.test_dataset_name}/ridge-alpha{args.alpha}"
        ),
    }

    return configs


def load_brain_data(data_brain_train_path: Path, data_brain_test_path: Path, roi: str):
    """Load brain data from training and test datasets"""
    print("Loading brain data...")

    # Load training data
    data_brain_train = bdpy.BData(data_brain_train_path)
    x_train = data_brain_train.select(roi)
    x_train_labels = data_brain_train.get_label("stimulus_name")

    # Load test data
    data_brain_test = bdpy.BData(data_brain_test_path)
    x_test = data_brain_test.select(roi)
    x_test_labels = data_brain_test.get_label("stimulus_name")

    # Average across runs for test data
    x_test_labels_unique = np.unique(x_test_labels)
    x_test = np.vstack(
        [
            np.mean(x_test[(np.array(x_test_labels) == xl).flatten(), :], axis=0)
            for xl in x_test_labels_unique
        ]
    )

    print("Done.")
    return x_train, x_train_labels, x_test, x_test_labels_unique


def load_features(feature_label_dir: Path):
    """Load feature labels and create identity matrix"""
    print("Loading features...")

    data_features = Features(feature_label_dir)
    y_labels = data_features.labels

    return y_labels


def select_training_samples(y_labels: list, setting: str):
    """Select training samples based on the setting"""
    print("Selecting training samples...")

    if setting == "1200":
        idxs = np.arange(0, 1200)
        idxs.sort()
    elif setting == "600":
        idxs = np.concatenate([np.arange(i, 1200, 8) for i in range(4)])
        idxs.sort()
    elif setting == "300":
        idxs = np.concatenate([np.arange(i, 1200, 8) for i in range(2)])
        idxs.sort()
    elif setting == "150":
        idxs = np.arange(0, 1200, 8)
    else:
        raise ValueError(f"Invalid setting: {setting}")

    selected_labels = [y_labels[i] for i in idxs]

    print("Done.")
    return idxs, selected_labels


def prepare_training_data(
    x_train: np.ndarray, x_train_labels: list, idxs: list, y_labels: list
):
    """Prepare training data by matching brain data with feature labels"""
    # Select x_train and y_train based on matching labels
    x_index = np.array([idx for idx, xl in enumerate(x_train_labels) if xl in y_labels])
    x_train_labels = [x_train_labels[idx] for idx in x_index]
    y_index = np.array(
        [np.where(np.array(y_labels) == xl) for xl in x_train_labels]
    ).flatten()

    x_train = x_train[x_index, :]
    y_train = np.eye(len(idxs))[y_index, :]
    return x_train, y_train


def normalize_data(x_train: np.ndarray, x_test: np.ndarray):
    """Normalize brain data using training statistics"""
    x_train_mean = np.mean(x_train, axis=0)
    x_train_std = np.std(x_train, axis=0)

    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    return x_train, x_test, x_train_mean, x_train_std


def train_and_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float
):
    """Train Ridge regression model and make predictions"""
    print("Training...")
    model = Ridge(alpha=alpha)
    model.fit(x_train, y_train)

    print("Predicting...")
    y_pred = model.predict(x_test)

    return model, y_pred


def save_results(
    assets_root_path: Path,
    x_train_mean: np.ndarray,
    x_train_std: np.ndarray,
    model: Ridge,
    y_pred: np.ndarray,
    x_test_labels_unique: np.ndarray,
):
    """Save all results to files"""
    print("Saving...")

    # Save normalization parameters
    output_path = assets_root_path / "x_mean.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(x_train_mean, f)

    output_path = assets_root_path / "x_std.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(x_train_std, f)

    # Save model
    output_path = assets_root_path / "model.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    # Save predicted features
    output_path = assets_root_path / "I_pred.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(y_pred, f)

    # Save labels
    output_path = assets_root_path / "labels.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(x_test_labels_unique, f)

    print("All done.")


if __name__ == "__main__":
    """Main processing"""
    # Parse arguments
    args = parse_arguments()

    # Get configuration
    configs = get_config(args)

    # Create output directory
    configs["assets_root_path"].mkdir(parents=True, exist_ok=True)

    # Load brain data
    x_train, x_train_labels, x_test, x_test_labels_unique = load_brain_data(
        configs["data_brain_train_path"],
        configs["data_brain_test_path"],
        configs["roi"],
    )

    # Load features
    y_labels = load_features(configs["feature_label_dir"])

    # Select training samples
    idxs, selected_labels = select_training_samples(y_labels, args.setting)

    # Prepare training data
    x_train, y_train = prepare_training_data(
        x_train, x_train_labels, idxs, selected_labels
    )

    # Normalize data
    x_train, x_test, x_train_mean, x_train_std = normalize_data(x_train, x_test)

    # Train and predict
    model, y_pred = train_and_predict(x_train, y_train, x_test, configs["alpha"])

    # Save results
    save_results(
        configs["assets_root_path"],
        x_train_mean,
        x_train_std,
        model,
        y_pred,
        x_test_labels_unique,
    )
