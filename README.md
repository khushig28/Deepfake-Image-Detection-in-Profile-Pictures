# Deepfake Image Detection

**An AI-Powered Solution for Identifying AI-Generated Faces in Profile Pictures to Prevent Catfish Accounts and Enhance Online Identity Security**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Prediction](#prediction)
- [Configuration](#configuration)
- [Metrics and Outputs](#metrics-and-outputs)
- [Hardware Requirements](#hardware-requirements)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a deep learning-based solution for detecting deepfake (AI-generated) images in profile pictures. The system uses transfer learning with EfficientNet-B0 to achieve high accuracy while maintaining efficient training times, even on CPU-only systems.

**Key Objectives:**
- Identify AI-generated faces in profile pictures
- Prevent catfish accounts using fake identities
- Enhance online identity security
- Provide reliable, real-time predictions

## ✨ Features

- **✅ Efficient Training**: Optimized for CPU training with 10-15 minute epochs (depending on dataset size)
- **📊 Comprehensive Metrics**: Track accuracy, precision, recall, F1-score, ROC-AUC, TPR, FPR, and more after every epoch
- **🎯 High Generalization**: Transfer learning approach ensures good performance on both dataset and random images
- **📁 Modular Architecture**: Clean, well-structured, and maintainable codebase
- **🔧 Configurable**: Easy-to-modify YAML configuration for all hyperparameters
- **📈 Visualization**: Automatic generation of training curves, confusion matrices, and ROC curves
- **💾 Checkpointing**: Automatic saving of best and periodic model checkpoints
- **⚡ Early Stopping**: Prevent overfitting with configurable early stopping
- **🔍 Prediction Tools**: Single image and batch prediction capabilities

## 📂 Project Structure

```
deepfake_detection/
├── config/
│   └── config.yaml                 # Configuration file
├── data/
│   ├── Train/                      # Training data
│   │   ├── Fake/
│   │   └── Real/
│   ├── Validation/                 # Validation data
│   │   ├── Fake/
│   │   └── Real/
│   └── Test/                       # Test data
│       ├── Fake/
│       └── Real/
├── src/
│   ├── data/
│   │   ├── dataset.py             # Dataset class
│   │   └── transforms.py          # Data augmentation
│   ├── models/
│   │   └── model.py               # Model architecture
│   ├── training/
│   │   └── trainer.py             # Training pipeline
│   └── utils/
│       ├── metrics.py             # Metrics calculation
│       ├── logger.py              # Logging utilities
│       └── visualize.py           # Visualization tools
├── outputs/                        # Generated during training
│   ├── checkpoints/               # Model checkpoints
│   ├── logs/                      # Training logs
│   └── metrics/                   # Metrics and plots
├── train.py                       # Main training script
├── evaluate.py                    # Model evaluation script
├── predict.py                     # Prediction script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 7.8GB RAM (minimum)
- Intel CPU (GPU optional but not required)

### Step 1: Clone or Navigate to Project

```bash
cd e:\major_project_implementation\deepfake_detection
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install PyTorch CPU version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

## 📊 Dataset Setup

Your dataset should already be organized in the following structure:

```
data/
├── Train/
│   ├── Fake/        # AI-generated/fake images
│   └── Real/        # Real/authentic images
├── Validation/
│   ├── Fake/
│   └── Real/
└── Test/
    ├── Fake/
    └── Real/
```

**Dataset Requirements:**
- Images should be in common formats (JPG, PNG, BMP, etc.)
- Balanced or near-balanced classes recommended
- Minimum resolution: 224x224 (images will be resized)
- Total: 140K images (as per your dataset)

**Note**: The dataset is already set up in your `data/` directory.

## 🚀 Usage

### Training

Train the model on your dataset:

```bash
python train.py
```

**What happens during training:**
1. ✅ Loads configuration from `config/config.yaml`
2. ✅ Sets up data loaders with augmentation
3. ✅ Initializes EfficientNet-B0 model with pretrained weights
4. ✅ Trains for specified epochs with validation after each epoch
5. ✅ Displays comprehensive metrics after every epoch including:
   - Accuracy, Precision, Recall, F1-Score
   - TPR, TNR, FPR, FNR
   - Confusion Matrix
   - ROC-AUC
6. ✅ Saves best model and periodic checkpoints
7. ✅ Generates training curves and visualizations
8. ✅ Logs all metrics to CSV for analysis

**Expected Training Time:**
- **Per epoch**: 15-25 minutes (CPU, 140K images, batch size 32)
- **Total training**: ~7-12 hours for 30 epochs

**Monitoring Training:**
- Watch console output for real-time metrics
- Check `outputs/logs/training.log` for detailed logs
- View `outputs/metrics/training_metrics.csv` for all epoch metrics
- Monitor `outputs/metrics/training_history.png` for loss/accuracy curves

### Evaluation

Evaluate the trained model on the test set:

```bash
python evaluate.py
```

**Or specify a custom checkpoint:**

```bash
python evaluate.py --checkpoint outputs/checkpoints/best_model.pth
```

**Evaluation outputs:**
- Comprehensive test metrics (accuracy, precision, recall, F1, etc.)
- Confusion matrix (both raw and normalized)
- ROC curve with AUC score
- Class-wise performance metrics
- Detailed classification report
- All visualizations saved in `outputs/evaluation/`

### Prediction

#### Single Image Prediction

Predict whether a single image is real or AI-generated:

```bash
python predict.py --image path/to/image.jpg
```

**Example output:**
```
===========================================================
PREDICTION RESULT
===========================================================
Image:       profile_pic.jpg
Prediction:  Fake
Confidence:  94.32%

Class Probabilities:
  Fake      : 94.32%
  Real      : 5.68%
===========================================================

Interpretation:
⚠️  This image is likely AI-GENERATED (Deepfake)
    The image shows characteristics of synthetic/AI-generated content.
```

#### Batch Prediction

Predict on multiple images in a directory:

```bash
python predict.py --image_dir path/to/images/
```

**Batch output:**
- Predictions for all images in the directory
- Summary statistics (# fake vs real, average confidence)
- Results saved to CSV file in the same directory

#### Custom Checkpoint

Use a specific model checkpoint:

```bash
python predict.py --image path/to/image.jpg --checkpoint outputs/checkpoints/checkpoint_epoch_20.pth
```

## ⚙️ Configuration

All training parameters can be configured in `config/config.yaml`:

### Key Parameters

```yaml
# Training
training:
  epochs: 30                    # Number of training epochs
  batch_size: 32                # Batch size (adjust based on RAM)
  learning_rate: 0.0001         # Initial learning rate
  
# Model
model:
  architecture: efficientnet_b0  # Model architecture
  pretrained: true              # Use pretrained weights
  dropout: 0.3                  # Dropout rate
  
# Data Augmentation
augmentation:
  random_horizontal_flip: true
  random_rotation_degrees: 15
  color_jitter:
    brightness: 0.2
    contrast: 0.2
    
# Hardware
hardware:
  device: cpu                   # cpu or cuda
  num_workers: 2                # Data loading workers
```

**To modify training:**
1. Edit `config/config.yaml`
2. Adjust parameters as needed
3. Run `python train.py` with new settings

## 📈 Metrics and Outputs

### Per-Epoch Metrics

After each epoch, you'll see:

```
==============================================================================
EPOCH 1/30 SUMMARY
==============================================================================
Training   - Loss: 0.4521 | Accuracy: 0.7823
Validation - Loss: 0.3845 | Accuracy: 0.8234
==============================================================================

==============================================================================
Validation Metrics
==============================================================================
Accuracy:  0.8234
Precision: 0.8156
Recall:    0.8312
F1-Score:  0.8233
ROC-AUC:   0.8945

Per-Class Metrics:
  Fake:
    Precision: 0.8234
    Recall:    0.8123
    F1-Score:  0.8178
  Real:
    Precision: 0.8078
    Recall:    0.8501
    F1-Score:  0.8284

Binary Classification Metrics:
  TPR (Sensitivity): 0.8501
  TNR (Specificity): 0.8123
  FPR:               0.1877
  FNR:               0.1499
  PPV (Precision):   0.8078
  NPV:               0.8234

Confusion Matrix:
              Fake        Real
Fake         45821        7234
Real          8743       48202
==============================================================================
```

### Saved Outputs

**Checkpoints** (`outputs/checkpoints/`):
- `best_model.pth` - Best model based on validation accuracy
- `last_model.pth` - Model from last epoch
- `checkpoint_epoch_N.pth` - Periodic checkpoints

**Logs** (`outputs/logs/`):
- `training.log` - Complete training log
- Console output with progress bars

**Metrics** (`outputs/metrics/`):
- `training_metrics.csv` - All metrics for every epoch
- `training_history.png` - Loss and accuracy curves

**Evaluation** (`outputs/evaluation/`):
- `confusion_matrix.png` - Confusion matrix heatmap
- `confusion_matrix_normalized.png` - Normalized confusion matrix
- `roc_curve.png` - ROC curve with AUC
- `class_wise_metrics.png` - Per-class performance
- `test_metrics.txt` - Detailed metrics report

## 💻 Hardware Requirements

### Minimum Requirements
- **CPU**: Intel Core i5 or equivalent (4 cores)
- **RAM**: 7.8 GB available
- **Storage**: 5 GB free space (for models and outputs)
- **OS**: Windows, Linux, or macOS

### Recommended
- **CPU**: Intel Core i7 or better (8+ cores)
- **RAM**: 16 GB
- **GPU**: NVIDIA GPU with 4GB+ VRAM (optional, for faster training)

### Performance Expectations

| Hardware | Epoch Time | Total Training (30 epochs) |
|----------|------------|----------------------------|
| CPU (i5, 4 cores) | 20-25 min | ~10-12 hours |
| CPU (i7, 8 cores) | 15-20 min | ~7-10 hours |
| GPU (RTX 3060) | 3-5 min | ~1.5-2.5 hours |

## 🔧 Troubleshooting

### Common Issues

**1. Out of Memory Error**

```
RuntimeError: [enforce fail at alloc_cpu.cpp:...] DefaultCPUAllocator: can't allocate memory
```

**Solution:**
- Reduce `batch_size` in `config/config.yaml` (try 16 or 8)
- Reduce `num_workers` to 0 or 1
- Close other applications

**2. Slow Training**

**Solutions:**
- Reduce image size (change `image.size` in config to 128 or 192)
- Reduce batch size (will use less memory and may speed up)
- Ensure no other heavy processes are running
- Consider using fewer augmentations

**3. Model Not Improving**

**Solutions:**
- Increase learning rate slightly
- Try different optimizer (Adam, AdamW, SGD)
- Adjust learning rate scheduler
- Check dataset for class imbalance
- Ensure data augmentation is appropriate

**4. Import Errors**

```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**5. Dataset Not Found**

```
FileNotFoundError: [Errno 2] No such file or directory: 'data/Train'
```

**Solution:**
- Verify dataset is in `data/Train`, `data/Validation`, `data/Test`
- Check folder names match configuration exactly (case-sensitive)
- Ensure `Fake` and `Real` subfolders exist

## 🎓 How It Works

### Model Architecture

1. **Backbone**: EfficientNet-B0 (pretrained on ImageNet)
   - Efficient and accurate for image classification
   - Optimized for CPU training
   - 5.3M parameters

2. **Transfer Learning**:
   - Use pretrained weights from ImageNet
   - Fine-tune on deepfake dataset
   - Custom classification head for binary classification

3. **Training Strategy**:
   - Data augmentation for better generalization
   - Learning rate scheduling for optimal convergence
   - Early stopping to prevent overfitting
   - Batch normalization and dropout for regularization

### Metrics Explanation

- **Accuracy**: Overall correctness of predictions
- **Precision**: Of all predicted fakes, how many are actually fake
- **Recall (TPR)**: Of all actual fakes, how many did we catch
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (ability to distinguish classes)
- **TNR (Specificity)**: Of all actual real images, how many did we correctly identify
- **FPR**: False alarm rate (real images incorrectly flagged as fake)

## 📝 Tips for Best Results

1. **Data Quality**:
   - Use high-quality, diverse images
   - Ensure balanced classes (equal fake and real images)
   - Remove corrupted or low-quality images

2. **Training**:
   - Let training run to completion (don't interrupt)
   - Monitor validation metrics, not just training metrics
   - Use early stopping to prevent overfitting

3. **Prediction**:
   - Use high-resolution images for prediction
   - Consider confidence scores (low confidence = uncertain)
   - Test on diverse, unseen images

4. **Optimization**:
   - Adjust batch size based on your RAM
   - Experiment with learning rates
   - Try different augmentation strategies

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 📧 Contact

For questions, issues, or collaboration:
- Open an issue in the repository
- Contact: [Your Email]

## 🙏 Acknowledgments

- EfficientNet architecture by Google Research
- PyTorch framework
- scikit-learn for metrics
- All contributors and users

---

**Built with ❤️ for Online Identity Security**
