"""
Prediction script for deepfake detection on random images
Predicts whether an image is Real or Fake (AI-generated)

Usage:
    # Single image prediction
    python predict.py --image path/to/image.jpg
    
    # Batch prediction
    python predict.py --image_dir path/to/images/
    
    # With custom checkpoint
    python predict.py --image path/to/image.jpg --checkpoint outputs/checkpoints/best_model.pth
"""

import os
import sys
import yaml
import torch
import argparse
from PIL import Image
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(__file__))

from src.models.model import DeepfakeDetector, load_checkpoint
from src.data.transforms import get_inference_transform


def predict_single_image(model, image_path, transform, device, class_names):
    """
    Predict on a single image
    
    Args:
        model: Trained model
        image_path: Path to image
        transform: Image transformations
        device: Device to use
        class_names: List of class names
    
    Returns:
        Tuple of (predicted_class, confidence, probabilities)
    """
    # Load and transform image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None, None
    
    # Apply transformations
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item()
    all_probs = probabilities[0].cpu().numpy()
    
    return predicted_class, confidence_score, all_probs


def predict_batch(model, image_dir, transform, device, class_names, extensions=['.jpg', '.jpeg', '.png', '.bmp']):
    """
    Predict on multiple images in a directory
    
    Args:
        model: Trained model
        image_dir: Directory containing images
        transform: Image transformations
        device: Device to use
        class_names: List of class names
        extensions: Valid image extensions
    
    Returns:
        List of prediction results
    """
    results = []
    
    # Get all image files
    image_files = []
    for filename in os.listdir(image_dir):
        if any(filename.lower().endswith(ext) for ext in extensions):
            image_files.append(os.path.join(image_dir, filename))
    
    if len(image_files) == 0:
        print(f"No images found in {image_dir}")
        return results
    
    print(f"\nFound {len(image_files)} images")
    print("Processing...\n")
    
    for image_path in image_files:
        pred_class, confidence, probs = predict_single_image(
            model, image_path, transform, device, class_names
        )
        
        if pred_class is not None:
            results.append({
                'image': os.path.basename(image_path),
                'path': image_path,
                'prediction': pred_class,
                'confidence': confidence,
                'probabilities': probs
            })
    
    return results


def main():
    """Main prediction function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Predict deepfake images')
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='Path to a single image for prediction'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default=None,
        help='Directory containing images for batch prediction'
    )
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
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.image is None and args.image_dir is None:
        print("Error: Please provide either --image or --image_dir")
        parser.print_help()
        sys.exit(1)
    
    if args.image is not None and args.image_dir is not None:
        print("Error: Please provide only one of --image or --image_dir")
        sys.exit(1)
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = config['hardware']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available. Using CPU instead.")
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    print(f"Loading model from: {args.checkpoint}")
    
    # Load model
    model = DeepfakeDetector(
        architecture=config['model']['architecture'],
        num_classes=config['model']['num_classes'],
        pretrained=False,
        dropout=config['model']['dropout']
    )
    
    model, epoch, best_acc = load_checkpoint(model, args.checkpoint, device)
    model = model.to(device)
    
    print(f"Model loaded successfully (trained for {epoch} epochs, val acc: {best_acc:.4f})")
    
    # Get transforms
    transform = get_inference_transform(config)
    
    # Get class names
    class_names = config['data'].get('classes', ['Fake', 'Real'])
    
    # Predict
    if args.image:
        # Single image prediction
        print(f"\nProcessing image: {args.image}")
        
        if not os.path.exists(args.image):
            print(f"Error: Image not found: {args.image}")
            sys.exit(1)
        
        pred_class, confidence, probs = predict_single_image(
            model, args.image, transform, device, class_names
        )
        
        if pred_class is not None:
            print("\n" + "="*60)
            print("PREDICTION RESULT")
            print("="*60)
            print(f"Image:       {os.path.basename(args.image)}")
            print(f"Prediction:  {pred_class}")
            print(f"Confidence:  {confidence*100:.2f}%")
            print("\nClass Probabilities:")
            for i, class_name in enumerate(class_names):
                print(f"  {class_name:10s}: {probs[i]*100:.2f}%")
            print("="*60)
            
            # Interpretation
            print("\nInterpretation:")
            if pred_class == 'Fake':
                print("⚠️  This image is likely AI-GENERATED (Deepfake)")
                print("    The image shows characteristics of synthetic/AI-generated content.")
            else:
                print("✓  This image appears to be REAL")
                print("    The image shows characteristics of authentic content.")
            
            if confidence < 0.7:
                print(f"\n⚠️  Note: Confidence is relatively low ({confidence*100:.2f}%)")
                print("    The model is less certain about this prediction.")
    
    else:
        # Batch prediction
        print(f"\nProcessing images in: {args.image_dir}")
        
        if not os.path.exists(args.image_dir):
            print(f"Error: Directory not found: {args.image_dir}")
            sys.exit(1)
        
        results = predict_batch(
            model, args.image_dir, transform, device, class_names
        )
        
        if len(results) == 0:
            print("No images were processed")
            sys.exit(0)
        
        # Print results
        print("\n" + "="*90)
        print("BATCH PREDICTION RESULTS")
        print("="*90)
        print(f"{'Image':40s} {'Prediction':15s} {'Confidence':12s} {'Probabilities':20s}")
        print("-"*90)
        
        for result in results:
            probs_str = " | ".join([f"{p*100:.1f}%" for p in result['probabilities']])
            print(f"{result['image']:40s} {result['prediction']:15s} {result['confidence']*100:10.2f}%  {probs_str}")
        
        print("="*90)
        
        # Summary statistics
        total = len(results)
        fake_count = sum(1 for r in results if r['prediction'] == 'Fake')
        real_count = total - fake_count
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        print("\nSummary:")
        print(f"  Total images:     {total}")
        print(f"  Predicted Fake:   {fake_count} ({fake_count/total*100:.1f}%)")
        print(f"  Predicted Real:   {real_count} ({real_count/total*100:.1f}%)")
        print(f"  Avg confidence:   {avg_confidence*100:.2f}%")
        print("="*90)
        
        # Save results to CSV
        output_csv = os.path.join(args.image_dir, 'predictions.csv')
        with open(output_csv, 'w') as f:
            f.write("Image,Prediction,Confidence," + ",".join(class_names) + "\n")
            for result in results:
                probs_str = ",".join([f"{p:.6f}" for p in result['probabilities']])
                f.write(f"{result['image']},{result['prediction']},{result['confidence']:.6f},{probs_str}\n")
        
        print(f"\nResults saved to: {output_csv}")


if __name__ == '__main__':
    main()
