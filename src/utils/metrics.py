"""
Comprehensive metrics calculation for model evaluation
Includes accuracy, precision, recall, F1-score, confusion matrix, ROC-AUC, etc.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
from typing import Dict, Tuple


class MetricsCalculator:
    """
    Calculate comprehensive metrics for binary classification
    """
    
    def __init__(self, num_classes: int = 2, class_names: list = None):
        """
        Initialize metrics calculator
        
        Args:
            num_classes: Number of classes
            class_names: Names of classes
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
    
    def calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_prob: np.ndarray = None
    ) -> Dict:
        """
        Calculate comprehensive metrics
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional, for ROC-AUC)
        
        Returns:
            Dictionary containing all metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = precision_per_class[i] if i < len(precision_per_class) else 0
            metrics[f'recall_{class_name}'] = recall_per_class[i] if i < len(recall_per_class) else 0
            metrics[f'f1_{class_name}'] = f1_per_class[i] if i < len(f1_per_class) else 0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # For binary classification, calculate additional metrics
        if self.num_classes == 2 and len(cm) == 2:
            tn, fp, fn, tp = cm.ravel()
            
            # True Positive Rate (Sensitivity/Recall)
            metrics['tpr'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['sensitivity'] = metrics['tpr']
            
            # True Negative Rate (Specificity)
            metrics['tnr'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['specificity'] = metrics['tnr']
            
            # False Positive Rate
            metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            # False Negative Rate
            metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            # Positive Predictive Value (Precision)
            metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            # Negative Predictive Value
            metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # ROC-AUC (if probabilities provided)
        if y_prob is not None:
            try:
                if self.num_classes == 2:
                    # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    # Multi-class
                    metrics['roc_auc'] = roc_auc_score(
                        y_true, y_prob, multi_class='ovr', average='weighted'
                    )
            except Exception as e:
                print(f"Warning: Could not calculate ROC-AUC: {e}")
                metrics['roc_auc'] = 0.0
        
        return metrics
    
    def print_metrics(self, metrics: Dict, prefix: str = ""):
        """
        Print metrics in a formatted way
        
        Args:
            metrics: Dictionary of metrics
            prefix: Prefix for printing (e.g., 'Train', 'Val')
        """
        print(f"\n{'='*60}")
        print(f"{prefix} Metrics")
        print(f"{'='*60}")
        
        # Main metrics
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        # Per-class metrics
        print(f"\nPer-Class Metrics:")
        for class_name in self.class_names:
            print(f"  {class_name}:")
            if f'precision_{class_name}' in metrics:
                print(f"    Precision: {metrics[f'precision_{class_name}']:.4f}")
            if f'recall_{class_name}' in metrics:
                print(f"    Recall:    {metrics[f'recall_{class_name}']:.4f}")
            if f'f1_{class_name}' in metrics:
                print(f"    F1-Score:  {metrics[f'f1_{class_name}']:.4f}")
        
        # Binary classification specific metrics
        if 'tpr' in metrics:
            print(f"\nBinary Classification Metrics:")
            print(f"  TPR (Sensitivity): {metrics['tpr']:.4f}")
            print(f"  TNR (Specificity): {metrics['tnr']:.4f}")
            print(f"  FPR:               {metrics['fpr']:.4f}")
            print(f"  FNR:               {metrics['fnr']:.4f}")
            print(f"  PPV (Precision):   {metrics['ppv']:.4f}")
            print(f"  NPV:               {metrics['npv']:.4f}")
        
        # Confusion matrix
        if 'confusion_matrix' in metrics:
            print(f"\nConfusion Matrix:")
            cm = metrics['confusion_matrix']
            
            # Print header
            print(f"{'':>12}", end='')
            for class_name in self.class_names:
                print(f"{class_name:>12}", end='')
            print()
            
            # Print matrix
            for i, class_name in enumerate(self.class_names):
                print(f"{class_name:>12}", end='')
                for j in range(len(self.class_names)):
                    if i < len(cm) and j < len(cm[i]):
                        print(f"{cm[i][j]:>12}", end='')
                    else:
                        print(f"{'0':>12}", end='')
                print()
        
        print(f"{'='*60}\n")
    
    def get_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """
        Get sklearn classification report
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
        
        Returns:
            Classification report string
        """
        return classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            zero_division=0
        )


def calculate_batch_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculate accuracy for a batch
    
    Args:
        outputs: Model outputs (logits)
        labels: Ground truth labels
    
    Returns:
        Accuracy as float
    """
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct / total if total > 0 else 0
