"""
Explainable AI Module - Grad-CAM Visualization
Generates heatmaps showing which parts of the image influenced the prediction
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    Visualizes which regions of the image the model focuses on
    """
    
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM
        
        Args:
            model: Trained PyTorch model
            target_layer: Layer to visualize (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Save forward activation"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Save backward gradient"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Class Activation Map
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (None = predicted class)
        
        Returns:
            cam: Class activation map (H, W)
        """
        # Forward pass
        output = self.model(input_tensor)
        
        # Get predicted class if not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)
        
        # Weighted combination
        cam = (weights * activations).sum(dim=0)  # (H, W)
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()
    
    def generate_heatmap(self, input_tensor, original_image, target_class=None, alpha=0.4):
        """
        Generate heatmap overlay on original image
        
        Args:
            input_tensor: Preprocessed input tensor
            original_image: Original PIL Image
            target_class: Target class (None = predicted)
            alpha: Overlay transparency (0-1)
        
        Returns:
            overlay_image: PIL Image with heatmap overlay
            heatmap_only: Heatmap as PIL Image
        """
        # Generate CAM
        cam = self.generate_cam(input_tensor, target_class)
        
        # Resize CAM to original image size
        original_size = original_image.size  # (W, H)
        cam_resized = cv2.resize(cam, original_size)
        
        # Normalize to 0-255
        heatmap = np.uint8(255 * cam_resized)
        
        # Apply colormap (RAINBOW or JET)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Convert original image to numpy
        original_np = np.array(original_image)
        
        # Overlay heatmap on original image
        overlay = cv2.addWeighted(original_np, 1 - alpha, heatmap_colored, alpha, 0)
        
        # Convert back to PIL
        overlay_image = Image.fromarray(overlay)
        heatmap_only = Image.fromarray(heatmap_colored)
        
        return overlay_image, heatmap_only


def get_target_layer(model, architecture='efficientnet_b0'):
    """
    Get the target layer for Grad-CAM based on architecture
    
    Args:
        model: PyTorch model
        architecture: Model architecture name
    
    Returns:
        target_layer: Layer to visualize
    """
    if architecture == 'efficientnet_b0':
        # Last convolutional layer in EfficientNet-B0
        return model.backbone.features[-1]
    elif architecture == 'resnet50':
        return model.backbone.layer4[-1]
    elif architecture == 'mobilenet_v2':
        return model.backbone.features[-1]
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def generate_explanation(model, input_tensor, original_image, config, device):
    """
    Generate complete explanation with heatmap
    
    Args:
        model: Trained model
        input_tensor: Preprocessed input tensor
        original_image: Original PIL Image
        config: Configuration dict
        device: Device (cuda/cpu)
    
    Returns:
        dict: Explanation data with heatmap images
    """
    # Get target layer
    architecture = config['model']['architecture']
    target_layer = get_target_layer(model, architecture)
    
    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # Move to device
    input_tensor = input_tensor.to(device)
    
    # Set model to eval mode
    model.eval()
    
    # Generate heatmap
    overlay_image, heatmap_only = grad_cam.generate_heatmap(
        input_tensor, 
        original_image,
        alpha=0.5  # 50% overlay transparency
    )
    
    return {
        'overlay_image': overlay_image,
        'heatmap_only': heatmap_only
    }
