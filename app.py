"""
Flask Web Application for Deepfake Detection
Provides REST API for image upload and prediction
"""

import os
import sys
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import torch
from PIL import Image
import yaml
import base64
from io import BytesIO

# Add src to path
sys.path.append(os.path.dirname(__file__))

from src.models.model import DeepfakeDetector, load_checkpoint
from src.data.transforms import get_inference_transform
from src.utils.explainable_ai import generate_explanation

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for model
model = None
transform = None
config = None
device = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    """Load the trained model"""
    global model, transform, config, device
    
    # Load config
    config_path = 'config/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    checkpoint_path = 'outputs/checkpoints/best_model.pth'
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model not found at {checkpoint_path}. Please train the model first!")
    
    model = DeepfakeDetector(
        architecture=config['model']['architecture'],
        num_classes=config['model']['num_classes'],
        pretrained=False,
        dropout=config['model']['dropout']
    )
    
    model, epoch, best_acc = load_checkpoint(model, checkpoint_path, device)
    model = model.to(device)
    model.eval()
    
    # Get transform
    transform = get_inference_transform(config)
    
    print(f"Model loaded successfully! (Epoch: {epoch}, Accuracy: {best_acc:.4f})")
    print(f"Device: {device}")


def predict_image(image, generate_heatmap=True):
    """
    Predict if image is real or fake
    
    Args:
        image: PIL Image
        generate_heatmap: Whether to generate Grad-CAM heatmap
    
    Returns:
        dict: Prediction results with optional heatmap
    """
    # Transform image
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # Get class names
    class_names = config['data'].get('classes', ['Fake', 'Real'])
    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item()
    all_probs = probabilities[0].cpu().numpy()
    
    # Prepare result
    result = {
        'prediction': predicted_class,
        'confidence': float(confidence_score),
        'probabilities': {
            class_names[0]: float(all_probs[0]),
            class_names[1]: float(all_probs[1])
        },
        'is_fake': predicted_class == 'Fake',
        'warning_level': get_warning_level(predicted_class, confidence_score)
    }
    
    # Generate explanation heatmap
    if generate_heatmap:
        try:
            explanation = generate_explanation(
                model, image_tensor, image, config, device
            )
            
            # Convert heatmap to base64
            buffered = BytesIO()
            explanation['overlay_image'].save(buffered, format="PNG")
            heatmap_base64 = base64.b64encode(buffered.getvalue()).decode()
            result['heatmap'] = f"data:image/png;base64,{heatmap_base64}"
        except Exception as e:
            print(f"Error generating heatmap: {e}")
            result['heatmap'] = None
    
    return result


def get_warning_level(prediction, confidence):
    """Get warning level based on prediction and confidence"""
    if prediction == 'Fake':
        if confidence > 0.9:
            return 'high'
        elif confidence > 0.7:
            return 'medium'
        else:
            return 'low'
    else:
        return 'safe'


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for prediction"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file type
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, BMP, WEBP'}), 400
        
        # Read and process image
        image = Image.open(file.stream).convert('RGB')
        
        # Get prediction
        result = predict_image(image)
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': device
    }), 200


if __name__ == '__main__':
    print("="*80)
    print("DEEPFAKE DETECTION WEB APP")
    print("="*80)
    
    # Load model
    print("\nLoading model...")
    try:
        load_model()
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nPlease ensure:")
        print("  1. Model is trained (run: python train.py)")
        print("  2. best_model.pth exists in outputs/checkpoints/")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("Starting web server...")
    print("Open your browser and go to: http://localhost:5000")
    print("="*80 + "\n")
    
    # Run app
    app.run(host='0.0.0.0', port=5000, debug=False)
