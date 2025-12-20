"""
Visualization utilities for training metrics and results
Creates plots for loss, accuracy, confusion matrix, ROC curve, etc.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from typing import List, Dict, Optional


# Set style for better-looking plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def plot_training_history(
    history: Dict[str, List],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot training history (loss and accuracy curves)
    
    Args:
        history: Dictionary containing training history
                 Expected keys: 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    show: bool = True,
    normalize: bool = False
):
    """
    Plot confusion matrix as a heatmap
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
        normalize: Whether to normalize the confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt=fmt, 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot ROC curve
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities for positive class
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, 
        color='darkorange', 
        lw=2, 
        label=f'ROC curve (AUC = {roc_auc:.3f})'
    )
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_metrics_progression(
    metrics_dict: Dict[str, List],
    metric_name: str,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot progression of a specific metric over epochs
    
    Args:
        metrics_dict: Dictionary containing metric values per epoch
        metric_name: Name of the metric to plot
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    if metric_name not in metrics_dict:
        print(f"Warning: Metric '{metric_name}' not found in metrics dictionary")
        return
    
    values = metrics_dict[metric_name]
    epochs = range(1, len(values) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, values, 'b-', marker='o', linewidth=2, markersize=6)
    plt.title(f'{metric_name.replace("_", " ").title()} Over Epochs', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"{metric_name} progression plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_class_wise_metrics(
    metrics: Dict,
    class_names: List[str],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot class-wise precision, recall, and F1-score
    
    Args:
        metrics: Dictionary containing per-class metrics
        class_names: List of class names
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    metric_types = ['precision', 'recall', 'f1']
    metric_values = {m: [] for m in metric_types}
    
    # Extract per-class metrics
    for class_name in class_names:
        for metric_type in metric_types:
            key = f"{metric_type}_{class_name}"
            if key in metrics:
                metric_values[metric_type].append(metrics[key])
            else:
                metric_values[metric_type].append(0)
    
    # Create grouped bar chart
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width, metric_values['precision'], width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, metric_values['recall'], width, label='Recall', color='#2ecc71')
    bars3 = ax.bar(x + width, metric_values['f1'], width, label='F1-Score', color='#e74c3c')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Class-wise Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9)
    
    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class-wise metrics plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
