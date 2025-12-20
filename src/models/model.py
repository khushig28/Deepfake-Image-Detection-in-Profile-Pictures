"""
Deepfake Detection Model Architecture
Based on EfficientNet-B0 for efficient CPU training
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict


class DeepfakeDetector(nn.Module):
    """
    Deepfake detection model using transfer learning
    
    Args:
        architecture (str): Base architecture (efficientnet_b0, resnet50, mobilenet_v2)
        num_classes (int): Number of output classes (default: 2 for binary)
        pretrained (bool): Use pretrained weights
        dropout (float): Dropout rate for regularization
    """
    
    def __init__(
        self, 
        architecture: str = 'efficientnet_b0',
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        super(DeepfakeDetector, self).__init__()
        
        self.architecture = architecture
        self.num_classes = num_classes
        
        # Load pretrained model
        if architecture == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            # Replace classifier
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(num_features, num_classes)
            )
        
        elif architecture == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            # Replace final layer
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(num_features, num_classes)
            )
        
        elif architecture == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            # Replace classifier
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(num_features, num_classes)
            )
        
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        print(f"Initialized {architecture} with {num_classes} output classes")
        print(f"Pretrained: {pretrained}, Dropout: {dropout}")
    
    def forward(self, x):
        """Forward pass"""
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze all layers except the final classifier"""
        for name, param in self.backbone.named_parameters():
            if 'classifier' not in name and 'fc' not in name:
                param.requires_grad = False
        print("Backbone frozen. Only training classifier layers.")
    
    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen. Training all layers.")
    
    def get_num_params(self):
        """Get total number of parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }


def get_model(config: Dict, device: str = 'cpu'):
    """
    Create and initialize model from config
    
    Args:
        config (Dict): Configuration dictionary
        device (str): Device to load model on
    
    Returns:
        DeepfakeDetector: Initialized model
    """
    model_config = config['model']
    
    model = DeepfakeDetector(
        architecture=model_config['architecture'],
        num_classes=model_config['num_classes'],
        pretrained=model_config['pretrained'],
        dropout=model_config['dropout']
    )
    
    # Freeze backbone if specified
    if model_config.get('freeze_backbone', False):
        model.freeze_backbone()
    
    # Move to device
    model = model.to(device)
    
    # Print model info
    param_info = model.get_num_params()
    print(f"\nModel Parameters:")
    print(f"  Total: {param_info['total']:,}")
    print(f"  Trainable: {param_info['trainable']:,}")
    print(f"  Frozen: {param_info['frozen']:,}")
    
    return model


def load_checkpoint(model, checkpoint_path, device='cpu'):
    """
    Load model from checkpoint
    
    Args:
        model: Model instance
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        model, epoch, best_acc
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_accuracy', 0.0)
    
    print(f"Loaded checkpoint from epoch {epoch} with accuracy {best_acc:.4f}")
    
    return model, epoch, best_acc


def save_checkpoint(model, optimizer, epoch, best_accuracy, filepath):
    """
    Save model checkpoint
    
    Args:
        model: Model instance
        optimizer: Optimizer instance
        epoch: Current epoch
        best_accuracy: Best validation accuracy so far
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': best_accuracy
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")
