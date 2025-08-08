#!/usr/bin/env python3
"""
Speech Emotion Recognition (SER) Project
Main training script for baseline models

This script supports training three baseline models:
- MLP (Multi-Layer Perceptron) for MFCC features
- Shallow CNN for Mel Spectrogram features  
- ResNet-18 for Mel Spectrogram features

Usage:
    python main.py --config config.yaml --model mlp
    python main.py --config config.yaml --model shallow_cnn
    python main.py --config config.yaml --model resnet18
"""

import argparse
import logging
import os
import sys
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Import our modules
from data_processing.data_loader import load_config
from models import create_model
from data_processing.feature_extractor import create_feature_data_loaders
from trainer import Trainer


def setup_logging(config: dict):
    """Setup logging configuration"""
    log_dir = config['logging']['log_dir']
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, 'main.log')

    # Clear any existing handlers first
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),  # Overwrite log file
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Force reconfiguration in Python 3.8+
    )

    logging.info("Logging system initialized")
    logging.info(f"Log file: {log_file}")


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Random seeds set to {seed}")


def validate_config(config: dict, model_name: str):
    """Validate configuration"""
    # Check if model exists in config
    if model_name not in config['models']:
        available_models = list(config['models'].keys())
        raise ValueError(f"Model '{model_name}' not found in config. Available models: {available_models}")

    # Check if data directory exists
    data_path = Path(config['dataset']['data_path'])
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    # Dataset-specific checks
    ds_name = config['dataset']['name'].lower()
    if ds_name == 'ravdess':
        if config['dataset'].get('use_speech', False):
            speech_path = data_path / config['dataset']['speech_path']
            if not speech_path.exists():
                raise FileNotFoundError(f"Speech data directory not found: {speech_path}")
        if config['dataset'].get('use_song', False):
            song_path = data_path / config['dataset']['song_path']
            if not song_path.exists():
                raise FileNotFoundError(f"Song data directory not found: {song_path}")
    elif ds_name == 'iemocap':
        # For IEMOCAP, ensure Session folders exist (basic check)
        has_session = any((data_path / f'Session{i}').exists() for i in range(1, 6))
        if not has_session:
            raise FileNotFoundError(f"IEMOCAP sessions not found under: {data_path}")

    logging.info("Configuration validation passed")


def print_system_info():
    """Print system information"""
    logging.info("=== System Information ===")
    logging.info(f"Python version: {sys.version}")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logging.info(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
    logging.info("=" * 30)


def print_model_info(model: torch.nn.Module, config: dict):
    """Print model information"""
    model_name = config['training']['model_name']
    model_config = config['models'][model_name]

    logging.info("=== Model Information ===")
    logging.info(f"Model type: {model_config['type']}")
    logging.info(f"Feature type: {model_config['feature_type']}")
    logging.info(f"Number of classes: {model_config['num_classes']}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")

    logging.info("=" * 30)


def print_training_info(config: dict):
    """Print training configuration"""
    training_config = config['training']

    logging.info("=== Training Configuration ===")
    logging.info(f"Batch size: {training_config['batch_size']}")
    logging.info(f"Number of epochs: {training_config['num_epochs']}")
    logging.info(f"Learning rate: {training_config['learning_rate']}")
    logging.info(f"Optimizer: {training_config['optimizer']}")
    logging.info(f"Weight decay: {training_config['weight_decay']}")
    logging.info(f"Early stopping patience: {training_config['early_stopping']['patience']}")
    logging.info("=" * 30)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Speech Emotion Recognition Training')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--model', type=str, choices=['mlp',  'shallow_cnn', 'resnet18', 'ast_tiny', 'ast_small', 'ast_base', 'ast_audioset'],
                       help='Model to train (overrides config file)')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'auto'],
                       help='Device to use for training (overrides config file)')
    parser.add_argument('--batch-size', type=int,
                       help='Batch size (overrides config file)')
    parser.add_argument('--epochs', type=int,
                       help='Number of epochs (overrides config file)')
    parser.add_argument('--lr', type=float,
                       help='Learning rate (overrides config file)')

    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config(args.config)
        logging.info(f"Loaded configuration from {args.config}")

        # Override config with command line arguments
        if args.model:
            config['training']['model_name'] = args.model
        if args.device:
            config['device'] = args.device
        if args.batch_size:
            config['training']['batch_size'] = args.batch_size
        if args.epochs:
            config['training']['num_epochs'] = args.epochs
        if args.lr:
            config['training']['learning_rate'] = args.lr

        # Prepare experiment directory: experiments/<timestamp>_<model>
        model_name = config['training']['model_name']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_name = config['dataset']['name'].lower()
        exp_name = f"{timestamp}_{dataset_name}_{model_name}"
        exp_root = os.path.join('experiments', exp_name)
        os.makedirs(exp_root, exist_ok=True)

        # Override paths to point into experiment directory
        config['training']['checkpoint_dir'] = os.path.join(exp_root, 'checkpoints')
        os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
        config['logging']['log_dir'] = exp_root
        os.makedirs(config['logging']['log_dir'], exist_ok=True)

        # Save resolved config snapshot
        try:
            import yaml  # local import to avoid global dependency if unused
            resolved_cfg_path = os.path.join(exp_root, 'config_resolved.yaml')
            with open(resolved_cfg_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
        except Exception as e:
            print(f"Warning: failed to save resolved config: {e}")

        # Setup logging to experiment directory
        setup_logging(config)

        # Print system information
        print_system_info()

        # Set random seeds for reproducibility
        set_random_seeds(config['dataset']['random_seed'])

        # Validate configuration
        model_name = config['training']['model_name']
        validate_config(config, model_name)

        logging.info(f"Starting training for model: {model_name}")

        # Create model
        logging.info("Creating model...")
        model = create_model(config)
        print_model_info(model, config)

        # Create data loaders
        logging.info("Creating data loaders...")
        train_loader, val_loader = create_feature_data_loaders(config)
        logging.info(f"Training batches: {len(train_loader)}")
        logging.info(f"Validation batches: {len(val_loader)}")

        # Print training information
        print_training_info(config)

        # Create trainer
        trainer = Trainer(config)

        # Start training
        logging.info("=" * 50)
        logging.info("Starting training process...")
        logging.info("=" * 50)

        best_metrics = trainer.train(model, train_loader, val_loader)

        # Print final results
        logging.info("=" * 50)
        logging.info("Training completed successfully!")
        logging.info("=" * 50)
        logging.info("Final Results:")
        for metric, value in best_metrics.items():
            logging.info(f"  {metric.capitalize()}: {value:.4f}")

        # Save final summary
        summary_path = os.path.join(config['logging']['log_dir'], f'{config["training"]["model_name"]}_training_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Speech Emotion Recognition Training Summary\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Model: {config['training']['model_name']}\n")
            f.write(f"Feature type: {config['models'][model_name]['feature_type']}\n")
            f.write(f"Dataset: {config['dataset']['name']}\n")
            f.write(f"Training samples: {len(train_loader.dataset)}\n")
            f.write(f"Validation samples: {len(val_loader.dataset)}\n")
            f.write(f"\nBest Validation Results:\n")
            for metric, value in best_metrics.items():
                f.write(f"  {metric.capitalize()}: {value:.4f}\n")

        logging.info(f"Training summary saved to: {summary_path}")

    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}")
        logging.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()