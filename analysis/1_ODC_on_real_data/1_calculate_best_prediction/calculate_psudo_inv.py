"""
Calculate pseudo inverse for different training sample settings.

This script loads training features, selects samples based on different settings,
standardizes the data, and calculates the pseudo inverse for analysis.
"""

import os
import pickle
import argparse
from typing import Dict, List, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler
from bdpy.dataform import Features
from tqdm import tqdm




def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Calculate pseudo inverse for different settings')
    parser.add_argument('--setting', type=str, required=True, 
                       choices=['1200', '600', '300', '150'],
                       help='Setting for # of training samples')
    parser.add_argument('--output_dir', type=str, default='./assets/1_ODC_on_real_data/calculate_best_prediction',
                       help='Output directory for results')
    parser.add_argument('--training_data_path', type=str, 
                       default='./data/features/ImageNetTraining/caffe/VGG19/',
                       help='Path to the training features data')
    parser.add_argument('--layers', nargs='+', 
                       default=['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4', 'fc6', 'fc7', 'fc8'],
                       help='Specific layers to process (space-separated)')
    return parser.parse_args()


def load_features_and_map(data_path: str, layers: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, tuple]]:
    """
    Load features and create feature map.
    
    Args:
        data_path: Path to the training features data
        layers: List of layer names to process
        
    Returns:
        Tuple of (features_dict, feature_map_dict)
    """
    print('Loading data and features...')
    data_features = Features(data_path)
    features = {}
    feature_map = {}

    for layer in tqdm(layers, desc="Processing layers"):
        feature_layer = data_features.get(layer)
        features[layer] = feature_layer.reshape(feature_layer.shape[0], -1)  # (N, D)
        feature_map[layer] = feature_layer.shape[1:]  # Get shape during loading
    
    print('Done.')
    return features, feature_map


def save_feature_map(feature_map: Dict[str, tuple], output_dir: str) -> None:
    """
    Save feature map if it doesn't exist.
    
    Args:
        feature_map: Dictionary containing feature shapes
        output_dir: Output directory path
    """
    feature_map_path = os.path.join(output_dir, 'feature_map.pkl')
    if not os.path.exists(feature_map_path):
        print('Saving feature map...')
        os.makedirs(output_dir, exist_ok=True)
        with open(feature_map_path, 'wb') as f:
            pickle.dump(feature_map, f)
        print(f"Feature map is saved at {feature_map_path}")


def concatenate_features(features: Dict[str, np.ndarray], layers: List[str]) -> np.ndarray:
    """
    Concatenate features from all layers.
    
    Args:
        features: Dictionary containing features for each layer
        layers: List of layer names
        
    Returns:
        Concatenated features array
    """
    print('Concatenating features...')
    concat_features = np.concatenate([features[layer] for layer in layers], axis=1)
    print('Done.')
    return concat_features


def select_training_samples(concat_features: np.ndarray, setting: str) -> np.ndarray:
    """
    Select training samples based on the setting.
    
    Args:
        concat_features: Concatenated features array
        setting: Setting name (1200, 600, 300, or 150)
        
    Returns:
        Selected features array
    """
    print(f'Selecting {setting} training samples...')
    
    # Select indices based on setting
    if setting == '1200':
        idxs = np.arange(0, 1200)
    elif setting == '600':
        idxs = np.concatenate([np.arange(i, 1200, 8) for i in range(4)])
        idxs.sort()
    elif setting == '300':
        idxs = np.concatenate([np.arange(i, 1200, 8) for i in range(2)])
        idxs.sort()
    elif setting == '150':
        idxs = np.arange(0, 1200, 8)
    else:
        raise ValueError(f'Invalid setting: {setting}')
    
    selected_features = concat_features[idxs, :]
    print('Done.')
    return selected_features


def standardize_features(features: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """
    Standardize features using StandardScaler.
    
    Args:
        features: Input features array
        
    Returns:
        Tuple of (standardized_features, scaler)
    """
    print('Standardizing...')
    scaler = StandardScaler()
    scaler.fit(features)
    standardized_features = scaler.transform(features)
    print('Done.')
    return standardized_features, scaler


def calculate_pseudo_inverse(features: np.ndarray) -> np.ndarray:
    """
    Calculate pseudo inverse of the feature matrix.
    
    Args:
        features: Standardized features array
        
    Returns:
        Pseudo inverse matrix
    """
    print('Calculating pseudo inverse...')
    pseudo_inverse = np.linalg.pinv(features @ features.T)
    print('Done.')
    return pseudo_inverse


def save_results(setting: str, output_dir: str, scaler: StandardScaler, 
                selected_features: np.ndarray, pseudo_inverse: np.ndarray) -> None:
    """
    Save all results to files.
    
    Args:
        setting: Setting name
        output_dir: Output directory path
        scaler: Fitted StandardScaler
        selected_features: Selected and standardized features
        pseudo_inverse: Calculated pseudo inverse matrix
    """
    print('Saving...')
    setting_output_dir = os.path.join(output_dir, setting)
    os.makedirs(setting_output_dir, exist_ok=True)

    # Save StandardScaler
    scaler_path = os.path.join(setting_output_dir, 'standard_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"StandardScaler is saved at {scaler_path}")

    # Save selected features
    selected_features_path = os.path.join(setting_output_dir, 'selected_features.pkl')
    with open(selected_features_path, 'wb') as f:
        pickle.dump(selected_features, f)
    print(f"Selected features are saved at {selected_features_path}")

    # Save pseudo inverse
    pseudo_inverse_path = os.path.join(setting_output_dir, 'pseudo_inverse.pkl')
    with open(pseudo_inverse_path, 'wb') as f:
        pickle.dump(pseudo_inverse, f)
    print(f"Pseudo inverse is saved at {pseudo_inverse_path}")

    print('Done.')


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Load data and features, create feature map
    features, feature_map = load_features_and_map(args.training_data_path, args.layers)
    
    # Save feature map if it doesn't exist
    save_feature_map(feature_map, args.output_dir)
    
    # Concatenate features
    concat_features = concatenate_features(features, args.layers)
    
    # Select training samples
    selected_features = select_training_samples(concat_features, args.setting)
    
    # Standardize features
    selected_features_standardized, scaler = standardize_features(selected_features)
    
    # Calculate pseudo inverse
    pseudo_inverse = calculate_pseudo_inverse(selected_features_standardized)
    
    # Save results
    save_results(args.setting, args.output_dir, scaler, 
                selected_features_standardized, pseudo_inverse)
    
    print('All done.')
