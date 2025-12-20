"""
Training pipeline for deepfake detection model
Handles training loop, validation, metrics tracking, and checkpointing
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional

from ..utils.metrics import MetricsCalculator, calculate_batch_accuracy
from ..utils.logger import MetricsLogger, TrainingProgress
from ..utils.visualize import (
    plot_training_history, plot_confusion_matrix, 
    plot_roc_curve, plot_class_wise_metrics
)


class Trainer:
    """
    Trainer class for deepfake detection model
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary
        device: Device to train on
        logger: Python logger instance
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str,
        logger
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.logger = logger
        
        # Training configuration
        self.epochs = config['training']['epochs']
        self.learning_rate = config['training']['learning_rate']
        
        # Setup criterion (loss function)
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer
        optimizer_name = config['training']['optimizer'].lower()
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=config['training']['weight_decay']
            )
        elif optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=config['training']['weight_decay']
            )
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=config['training']['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Setup learning rate scheduler
        scheduler_config = config['training']['scheduler']
        scheduler_type = scheduler_config['type'].lower()
        
        if scheduler_type == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=scheduler_config['patience'],
                factor=scheduler_config['factor'],
                min_lr=scheduler_config['min_lr']
            )
            self.logger.info("Learning rate scheduler: ReduceLROnPlateau")
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 10),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=scheduler_config['min_lr']
            )
        else:
            self.scheduler = None
        
        # Setup metrics calculator
        class_names = config['data'].get('classes', ['Fake', 'Real'])
        self.metrics_calculator = MetricsCalculator(
            num_classes=config['model']['num_classes'],
            class_names=class_names
        )
        
        # Setup metrics logger
        log_dir = config['logging']['metrics_dir']
        os.makedirs(log_dir, exist_ok=True)
        
        metrics_to_track = [
            'train_loss', 'val_loss', 'train_acc', 'val_acc',
            'precision', 'recall', 'f1_score', 'roc_auc',
            'tpr', 'tnr', 'fpr', 'fnr'
        ]
        
        self.metrics_logger = MetricsLogger(
            filepath=os.path.join(log_dir, 'training_metrics.csv'),
            metrics_names=metrics_to_track
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Early stopping
        self.early_stopping_enabled = config['training']['early_stopping']['enabled']
        self.early_stopping_patience = config['training']['early_stopping']['patience']
        self.early_stopping_min_delta = config['training']['early_stopping']['min_delta']
        
        # Checkpointing
        self.checkpoint_dir = config['checkpoint']['save_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Progress tracker
        self.progress = TrainingProgress(self.epochs)
        
        self.logger.info(f"Trainer initialized with {optimizer_name} optimizer")
        self.logger.info(f"Learning rate: {self.learning_rate}")
        self.logger.info(f"Scheduler: {scheduler_type}")
    
    def train_epoch(self, epoch: int) -> tuple:
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.epochs} [Train]')
        
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            batch_acc = calculate_batch_accuracy(outputs, labels)
            running_corrects += batch_acc * inputs.size(0)
            total_samples += inputs.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / total_samples,
                'acc': running_corrects / total_samples
            })
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, epoch: int) -> Dict:
        """
        Validate for one epoch
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        running_loss = 0.0
        total_samples = 0
        
        all_labels = []
        all_predictions = []
        all_probabilities = []
        
        # Progress bar
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch}/{self.epochs} [Val]  ')
        
        with torch.no_grad():
            for inputs, labels in pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
                
                # Get predictions and probabilities
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)
                
                # Store for metrics calculation
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({'loss': running_loss / total_samples})
        
        # Calculate metrics
        val_loss = running_loss / total_samples
        
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        metrics = self.metrics_calculator.calculate_metrics(
            all_labels, all_predictions, all_probabilities
        )
        
        metrics['val_loss'] = val_loss
        
        return metrics
    
    def train(self):
        """
        Main training loop
        """
        self.logger.info("Starting training...")
        self.logger.info(f"Total epochs: {self.epochs}")
        self.logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(1, self.epochs + 1):
            self.progress.start_epoch(epoch)
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate_epoch(epoch)
            val_loss = val_metrics['val_loss']
            val_acc = val_metrics['accuracy']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_metrics['f1_score'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            
            # Print comprehensive metrics
            print(f"\n{'='*80}")
            print(f"EPOCH {epoch}/{self.epochs} SUMMARY")
            print(f"{'='*80}")
            print(f"Training   - Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")
            print(f"Validation - Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")
            print(f"{'='*80}")
            
            # Print detailed validation metrics
            self.metrics_calculator.print_metrics(val_metrics, prefix="Validation")
            
            # Log metrics to CSV
            metrics_to_log = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'precision': val_metrics['precision'],
                'recall': val_metrics['recall'],
                'f1_score': val_metrics['f1_score'],
                'roc_auc': val_metrics.get('roc_auc', 0),
                'tpr': val_metrics.get('tpr', 0),
                'tnr': val_metrics.get('tnr', 0),
                'fpr': val_metrics.get('fpr', 0),
                'fnr': val_metrics.get('fnr', 0)
            }
            self.metrics_logger.log_epoch(epoch, metrics_to_log)
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Print progress
            self.progress.print_progress(train_loss, val_loss, val_acc)
            
            # Save best model
            if val_acc > self.best_val_acc + self.early_stopping_min_delta:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.patience_counter = 0
                
                if self.config['checkpoint']['save_best']:
                    checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
                    self.save_checkpoint(epoch, val_acc, checkpoint_path)
                    self.logger.info(f"✓ New best model saved! Accuracy: {val_acc:.4f}")
            else:
                self.patience_counter += 1
            
            # Save periodic checkpoints
            if epoch % self.config['checkpoint']['save_frequency'] == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
                self.save_checkpoint(epoch, val_acc, checkpoint_path)
            
            # Save last model
            if self.config['checkpoint']['save_last']:
                checkpoint_path = os.path.join(self.checkpoint_dir, 'last_model.pth')
                self.save_checkpoint(epoch, val_acc, checkpoint_path)
            
            # Early stopping check
            if self.early_stopping_enabled and self.patience_counter >= self.early_stopping_patience:
                self.logger.info(f"\nEarly stopping triggered after {epoch} epochs")
                self.logger.info(f"Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
                break
        
        # Training complete
        self.logger.info("\n" + "="*80)
        self.logger.info("TRAINING COMPLETE!")
        self.logger.info("="*80)
        self.logger.info(f"Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
        self.logger.info(f"Total training time: {sum(self.progress.epoch_times):.2f} seconds")
        self.logger.info(f"Average time per epoch: {self.progress.get_avg_epoch_time():.2f} seconds")
        
        # Save training history plots
        if self.config['logging']['save_plots']:
            self.save_training_plots()
    
    def save_checkpoint(self, epoch: int, accuracy: float, filepath: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': accuracy,
            'history': self.history
        }
        torch.save(checkpoint, filepath)
    
    def save_training_plots(self):
        """Save training history visualizations"""
        plot_dir = self.config['logging']['metrics_dir']
        
        # Training history plot
        plot_training_history(
            self.history,
            save_path=os.path.join(plot_dir, 'training_history.png'),
            show=False
        )
        
        self.logger.info(f"Training plots saved to {plot_dir}")
