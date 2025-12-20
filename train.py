"""
Main training script for deepfake detection model
Run this script to train the model on your dataset

Usage:
    python train.py
"""

import os
import sys
import yaml
import torch
import random
import numpy as np
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.dirname(__file__))

from src.data.dataset import DeepfakeDataset
from src.data.transforms import get_transforms
from src.models.model import get_model
from src.training.trainer import Trainer
from src.utils.logger import setup_logger


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main training function"""
    
    # Load configuration
    config_path = 'config/config.yaml'
    print(f"Loading configuration from {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    set_seed(config['seed'])
    
    # Setup logger
    logger = setup_logger(
        name='deepfake_training',
        log_dir=config['logging']['log_dir'],
        log_file='training.log'
    )
    
    logger.info("="*80)
    logger.info("DEEPFAKE DETECTION - TRAINING")
    logger.info("="*80)
    logger.info(f"Project: {config['project']['name']}")
    logger.info(f"Version: {config['project']['version']}")
    logger.info(f"Description: {config['project']['description']}")
    
    # Setup device
    device = config['hardware']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available. Using CPU instead.")
        device = 'cpu'
    
    logger.info(f"\nDevice: {device}")
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Data paths
    data_root = config['data']['dataset_path']
    train_dir = os.path.join(data_root, config['data']['train_dir'])
    val_dir = os.path.join(data_root, config['data']['val_dir'])
    
    logger.info(f"\nData directories:")
    logger.info(f"  Train: {train_dir}")
    logger.info(f"  Validation: {val_dir}")
    
    # Check if directories exist
    if not os.path.exists(train_dir):
        logger.error(f"Training directory not found: {train_dir}")
        sys.exit(1)
    if not os.path.exists(val_dir):
        logger.error(f"Validation directory not found: {val_dir}")
        sys.exit(1)
    
    # Get transforms
    logger.info("\nSetting up data transformations...")
    train_transform = get_transforms(config, mode='train')
    val_transform = get_transforms(config, mode='val')
    
    # Create datasets
    logger.info("\nLoading datasets...")
    train_dataset = DeepfakeDataset(
        data_dir=train_dir,
        transform=train_transform,
        class_names=config['data']['classes']
    )
    
    val_dataset = DeepfakeDataset(
        data_dir=val_dir,
        transform=val_transform,
        class_names=config['data']['classes']
    )
    
    # Create data loaders
    logger.info("\nCreating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    logger.info(f"Batch size: {config['training']['batch_size']}")
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    
    # Create model
    logger.info("\nInitializing model...")
    model = get_model(config, device)
    
    # Create trainer
    logger.info("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        logger=logger
    )
    
    # Start training
    logger.info("\n" + "="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80 + "\n")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"\nTraining failed with error: {e}", exc_info=True)
        raise
    
    logger.info("\nTraining script finished successfully!")
    logger.info(f"Best model saved at: {os.path.join(config['checkpoint']['save_dir'], 'best_model.pth')}")
    logger.info(f"Training logs saved at: {config['logging']['log_dir']}")
    logger.info(f"Metrics saved at: {config['logging']['metrics_dir']}")


if __name__ == '__main__':
    main()
