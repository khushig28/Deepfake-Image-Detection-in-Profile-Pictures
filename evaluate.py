"""
Evaluation script for deepfake detection model
Evaluates trained model on test dataset and generates comprehensive reports

Usage:
    python evaluate.py --checkpoint outputs/checkpoints/best_model.pth
"""

import os
import sys
import yaml
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(__file__))

from src.data.dataset import DeepfakeDataset
from src.data.transforms import get_transforms
from src.models.model import DeepfakeDetector, load_checkpoint
from src.utils.metrics import MetricsCalculator
from src.utils.visualize import plot_confusion_matrix, plot_roc_curve, plot_class_wise_metrics
from src.utils.logger import setup_logger


def evaluate_model(model, test_loader, device, config):
    """
    Evaluate model on test set
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to use
        config: Configuration dictionary
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    print("\nEvaluating model on test set...")
    pbar = tqdm(test_loader, desc='Testing')
    
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            # Store results
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    class_names = config['data'].get('classes', ['Fake', 'Real'])
    metrics_calculator = MetricsCalculator(
        num_classes=config['model']['num_classes'],
        class_names=class_names
    )
    
    metrics = metrics_calculator.calculate_metrics(
        all_labels, all_predictions, all_probabilities
    )
    
    return metrics, all_labels, all_predictions, all_probabilities


def main():
    """Main evaluation function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate deepfake detection model')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='outputs/checkpoints/best_model.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/evaluation',
        help='Directory to save evaluation results'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(
        name='deepfake_evaluation',
        log_dir=args.output_dir,
        log_file='evaluation.log'
    )
    
    logger.info("="*80)
    logger.info("DEEPFAKE DETECTION - MODEL EVALUATION")
    logger.info("="*80)
    
    # Setup device
    device = config['hardware']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available. Using CPU instead.")
        device = 'cpu'
    
    logger.info(f"Device: {device}")
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    
    # Load model
    logger.info("\nInitializing model...")
    model = DeepfakeDetector(
        architecture=config['model']['architecture'],
        num_classes=config['model']['num_classes'],
        pretrained=False,  # We're loading weights
        dropout=config['model']['dropout']
    )
    
    model, epoch, best_acc = load_checkpoint(model, args.checkpoint, device)
    model = model.to(device)
    
    logger.info(f"Model loaded from epoch {epoch} with validation accuracy {best_acc:.4f}")
    
    # Load test data
    data_root = config['data']['dataset_path']
    test_dir = os.path.join(data_root, config['data']['test_dir'])
    
    if not os.path.exists(test_dir):
        logger.error(f"Test directory not found: {test_dir}")
        sys.exit(1)
    
    logger.info(f"\nTest directory: {test_dir}")
    
    # Get transforms
    test_transform = get_transforms(config, mode='test')
    
    # Create dataset
    logger.info("\nLoading test dataset...")
    test_dataset = DeepfakeDataset(
        data_dir=test_dir,
        transform=test_transform,
        class_names=config['data']['classes']
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    logger.info(f"Test samples: {len(test_dataset)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Evaluate
    metrics, y_true, y_pred, y_prob = evaluate_model(model, test_loader, device, config)
    
    # Print metrics
    class_names = config['data'].get('classes', ['Fake', 'Real'])
    metrics_calculator = MetricsCalculator(
        num_classes=config['model']['num_classes'],
        class_names=class_names
    )
    
    print("\n" + "="*80)
    print("TEST SET EVALUATION RESULTS")
    print("="*80)
    metrics_calculator.print_metrics(metrics, prefix="Test Set")
    
    # Print classification report
    print("\nDetailed Classification Report:")
    print(metrics_calculator.get_classification_report(y_true, y_pred))
    
    # Save visualizations
    logger.info("\nGenerating visualizations...")
    
    # Confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names=class_names,
        save_path=os.path.join(args.output_dir, 'confusion_matrix.png'),
        show=False
    )
    
    # Normalized confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names=class_names,
        save_path=os.path.join(args.output_dir, 'confusion_matrix_normalized.png'),
        show=False,
        normalize=True
    )
    
    # ROC curve (for binary classification)
    if config['model']['num_classes'] == 2:
        plot_roc_curve(
            y_true,
            y_prob[:, 1],  # Probability of positive class
            save_path=os.path.join(args.output_dir, 'roc_curve.png'),
            show=False
        )
    
    # Class-wise metrics
    plot_class_wise_metrics(
        metrics,
        class_names=class_names,
        save_path=os.path.join(args.output_dir, 'class_wise_metrics.png'),
        show=False
    )
    
    # Save metrics to file
    metrics_file = os.path.join(args.output_dir, 'test_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DEEPFAKE DETECTION - TEST SET METRICS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Test samples: {len(test_dataset)}\n\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {metrics['f1_score']:.4f}\n")
        if 'roc_auc' in metrics:
            f.write(f"ROC-AUC:   {metrics['roc_auc']:.4f}\n")
        f.write("\n" + metrics_calculator.get_classification_report(y_true, y_pred))
    
    logger.info(f"\nEvaluation complete!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"  - Metrics: {metrics_file}")
    logger.info(f"  - Confusion Matrix: {os.path.join(args.output_dir, 'confusion_matrix.png')}")
    logger.info(f"  - ROC Curve: {os.path.join(args.output_dir, 'roc_curve.png')}")


if __name__ == '__main__':
    main()
