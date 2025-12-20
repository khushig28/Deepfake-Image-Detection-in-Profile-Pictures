"""
Logging utilities for training and evaluation
Provides structured logging to console and files
"""

import os
import logging
import sys
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = 'deepfake_detection',
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger with console and file handlers
    
    Args:
        name: Logger name
        log_dir: Directory to save log files
        log_file: Log file name (optional)
        level: Logging level
    
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_dir provided)
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'training_{timestamp}.log'
        
        file_path = os.path.join(log_dir, log_file)
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {file_path}")
    
    return logger


class MetricsLogger:
    """
    Logger for tracking metrics across epochs
    Saves to CSV for easy analysis
    """
    
    def __init__(self, filepath: str, metrics_names: list):
        """
        Initialize metrics logger
        
        Args:
            filepath: Path to CSV file
            metrics_names: List of metric names to track
        """
        self.filepath = filepath
        self.metrics_names = ['epoch'] + metrics_names
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Initialize CSV file with headers
        with open(filepath, 'w') as f:
            f.write(','.join(self.metrics_names) + '\n')
    
    def log_epoch(self, epoch: int, metrics: dict):
        """
        Log metrics for an epoch
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
        """
        values = [str(epoch)]
        
        for metric_name in self.metrics_names[1:]:  # Skip 'epoch'
            value = metrics.get(metric_name, 'N/A')
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        
        # Append to CSV
        with open(self.filepath, 'a') as f:
            f.write(','.join(values) + '\n')
    
    def __str__(self):
        return f"MetricsLogger(filepath={self.filepath})"


class TrainingProgress:
    """
    Track and display training progress
    """
    
    def __init__(self, total_epochs: int):
        """
        Initialize progress tracker
        
        Args:
            total_epochs: Total number of epochs
        """
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.start_time = None
        self.epoch_times = []
    
    def start_epoch(self, epoch: int):
        """Start timing an epoch"""
        self.current_epoch = epoch
        self.start_time = datetime.now()
    
    def end_epoch(self):
        """End timing an epoch and record duration"""
        if self.start_time is not None:
            duration = (datetime.now() - self.start_time).total_seconds()
            self.epoch_times.append(duration)
            return duration
        return 0
    
    def get_avg_epoch_time(self) -> float:
        """Get average epoch time"""
        if len(self.epoch_times) == 0:
            return 0
        return sum(self.epoch_times) / len(self.epoch_times)
    
    def get_eta(self) -> str:
        """
        Get estimated time to completion
        
        Returns:
            ETA as formatted string
        """
        if len(self.epoch_times) == 0:
            return "Unknown"
        
        avg_time = self.get_avg_epoch_time()
        remaining_epochs = self.total_epochs - self.current_epoch
        eta_seconds = avg_time * remaining_epochs
        
        hours = int(eta_seconds // 3600)
        minutes = int((eta_seconds % 3600) // 60)
        seconds = int(eta_seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def print_progress(self, train_loss: float, val_loss: float, val_acc: float):
        """
        Print epoch progress
        
        Args:
            train_loss: Training loss
            val_loss: Validation loss
            val_acc: Validation accuracy
        """
        epoch_time = self.end_epoch()
        avg_time = self.get_avg_epoch_time()
        eta = self.get_eta()
        
        print(f"\n{'='*80}")
        print(f"Epoch [{self.current_epoch}/{self.total_epochs}] Complete")
        print(f"{'='*80}")
        print(f"Time: {epoch_time:.2f}s | Avg: {avg_time:.2f}s/epoch | ETA: {eta}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"{'='*80}\n")
