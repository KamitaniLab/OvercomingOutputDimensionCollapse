from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple
import argparse

import numpy as np
import torch
import torch.optim as optim
from bdpy.dl.torch.models import VGG19, layer_map
from overcoming_output_dimension_collapse.icnn_replication import image_domain
from overcoming_output_dimension_collapse.icnn_replication.critic import TargetNormalizedMSE
from overcoming_output_dimension_collapse.icnn_replication.dataset import (
    FeaturesDataset,
    RenameFeatureKeys,
)
from overcoming_output_dimension_collapse.icnn_replication.encoder import Encoder
from overcoming_output_dimension_collapse.icnn_replication.generator import DeepImagePriorGenerator
from overcoming_output_dimension_collapse.icnn_replication.pipeline import FeatureInversionPipeline
from PIL import Image
from torch.utils.data import DataLoader
from bdpy.dataform import Features
import wandb


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='iCNN reconstruction experiment')

    parser.add_argument('--setting', type=str, required=True, 
                       choices=['1200', '600', '300', '150'],
                       help='Setting for feature extraction')
    parser.add_argument('--output_dir', type=str, 
                       default='./assets/1_ODC_on_real_data/reconstruction',
                       help='Output directory for results')
    parser.add_argument('--test_dataset_name', type=str, default='ImageNetTest',
                       help='Test dataset name')
    parser.add_argument('--feature_dir', type=str, 
                       default='./assets/1_ODC_on_real_data/features',
                       help='Directory for features')
    parser.add_argument('--model_path', type=str,
                       default='./data/model_shared/pytorch/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.pt',
                       help='Path to the model file')

    parser.add_argument('--num_iterations', type=int, default=800,
                       help='Number of iterations for reconstruction')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for optimizer')
    parser.add_argument('--max_trials', type=int, default=3,
                       help='Maximum number of trials for reconstruction')
    parser.add_argument('--loss_threshold', type=float, default=0.9,
                       help='Loss threshold for reconstruction')
    return parser.parse_args()


def get_configuration(args) -> Dict[str, Any]:
    """Get configuration parameters for the iCNN experiment."""
    return {
        "project_name": "overcoming_output_dimension_collapse",
        "experiment_name": "iCNN-reconstruction",
        "feature_network_param_path": args.model_path,
        "paths": {
            "output_root": Path(args.output_dir) / args.test_dataset_name / "best-prediction-feature"/ args.setting,
            "feature_root": Path(args.feature_dir) / args.test_dataset_name / "best-prediction-feature"/ args.setting,
        },
        "optimization": {
            "num_iterations": args.num_iterations,
            "log_interval": 100,
            "learning_rate": args.learning_rate,
            "max_trials": args.max_trials,
            "loss_threshold": args.loss_threshold,
        }
    }


def get_layer_names():
    """Get layer names to process"""
    to_layer_name = layer_map("vgg19")
    to_path_name = {
        layer_name: layer_path_name
        for layer_name, layer_path_name in zip(
            to_layer_name.values(), to_layer_name.keys()
        )
    }
    layer_names = list(to_layer_name.values())
    
    # Remove "relu" layers
    layer_names = [
        layer_name
        for layer_name in layer_names
        if "relu" not in to_path_name[layer_name]
    ]

    to_layer_name = {k: v for k, v in to_layer_name.items() if v in layer_names}
    
    return layer_names, to_path_name, to_layer_name


def setup_data_loader(config: Dict[str, Any]) -> Tuple[DataLoader, list, list]:
    """Setup data loader and layer mapping for the experiment."""
    print("Setting up data loader...")
    
    paths = config["paths"]
    
    # Use feature_root directly from config instead of constructing path
    feature_root_path = paths["feature_root"]
    
    # Setup layer mapping (excluding relu layers)
    layer_names, to_path_name, to_layer_name = get_layer_names()

    # Get stimulus names from the features store (similar to calculate_best_prediction.py)
    features_store = Features(feature_root_path.as_posix())
    stimulus_names = features_store.labels
    print(f"Found {len(stimulus_names)} stimuli in features")
    
    # Create features dataset only
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
    
    return data_loader, layer_names, stimulus_names


def setup_models_and_pipeline(config: Dict[str, Any], layer_names: list) -> FeatureInversionPipeline:
    """Setup models and pipeline for feature inversion."""
    print("Setting up models and pipeline...")
    
    optimization_config = config["optimization"]
    
    # Setup device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print(f"Using device: {device}")
    
    # Load feature network
    feature_network = VGG19()
    feature_network.load_state_dict(torch.load(config["feature_network_param_path"]))
    feature_network.to(device)
    
    # Setup encoder
    encoder = Encoder(
        feature_network=feature_network,
        layer_names=layer_names,
        domain=image_domain.BdPyVGGDomain(device=device, dtype=dtype),
        device=device,
    )
    
    # Setup generator
    generator = DeepImagePriorGenerator(
        image_shape=(224, 224),
        batch_size=1,
        device=device,
    )
    
    # Setup critic and optimizer
    critic = TargetNormalizedMSE()
    optimizer = optim.AdamW(generator.parameters(), lr=optimization_config["learning_rate"])
    scheduler = None
    
    # Setup pipeline
    pipeline = FeatureInversionPipeline(
        generator=generator,
        encoder=encoder,
        critic=critic,
        optimizer=optimizer,
        scheduler=scheduler,
        num_iterations=optimization_config["num_iterations"],
        log_interval=optimization_config["log_interval"],
        with_wandb=True,
    )
    
    return pipeline


def run_reconstruction(config: Dict[str, Any], data_loader: DataLoader, pipeline: FeatureInversionPipeline, stimulus_names: list) -> None:
    """Run the reconstruction process for all stimuli."""
    print("Starting reconstruction process...")
    
    optimization_config = config["optimization"]
    paths = config["paths"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    for idx, features in enumerate(data_loader):
        stimulus_name = stimulus_names[idx]
        print(f"Processing stimulus [{idx+1}/{len(data_loader)}]: {stimulus_name}")
        
        try:
            # Initialize wandb for this stimulus
            wandb.init(
                project=config["project_name"],
                group=config["experiment_name"],
                config={
                    "stimulus_name": stimulus_name,
                    "target_feature_type": "features",
                    "feature_network_param_path": config["feature_network_param_path"],
                },
            )
            
            # Prepare target features
            target_features = {
                k: v.to(device=device, dtype=dtype) for k, v in features.items()
            }
            
            # Run reconstruction
            print("Reconstructing image from features...")
            generated_images = image_domain.finalize(
                pipeline(
                    target_features, 
                    max_trials=optimization_config["max_trials"], 
                    loss_threshold=optimization_config["loss_threshold"]
                )
            )
            
            # Save reconstructed image
            image = Image.fromarray(
                generated_images[0].detach().cpu().numpy().astype(np.uint8)
            )
            savedir = paths["output_root"]
            savedir.mkdir(parents=True, exist_ok=True)
            image.save(savedir / f"{stimulus_name}.jpg")
            print(f"Saved reconstructed image to: {savedir / f'{stimulus_name}.jpg'}")
            
        except Exception as e:
            print(f"Error processing stimulus {stimulus_name}: {str(e)}")
            continue
        finally:
            wandb.finish()


if __name__ == "__main__":
    # Main execution flow
    print("Starting iCNN reconstruction experiment...")
    
    # Parse command line arguments
    args = parse_arguments()
    print(f"Output directory: {args.output_dir}")
    print(f"Feature directory: {args.feature_dir}")
    
    # Get configuration
    config = get_configuration(args)
    print(f"Experiment: {config['experiment_name']}")
    print(f"Project: {config['project_name']}")
    
    # Setup data loader
    data_loader, layer_names, stimulus_names = setup_data_loader(config)
    print(f"Loaded {len(data_loader)} stimuli for reconstruction")
    
    # Setup models and pipeline
    pipeline = setup_models_and_pipeline(config, layer_names)
    
    # Run reconstruction
    run_reconstruction(config, data_loader, pipeline, stimulus_names)
    
    print("Reconstruction experiment completed!")
