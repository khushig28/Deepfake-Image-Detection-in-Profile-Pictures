# Deepfake Detection - Quick Reference Card

## 🚀 Installation (One-Time Setup)

```bash
# Navigate to project
cd e:\major_project_implementation\deepfake_detection

# Install PyTorch (CPU version) - requires internet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install dependencies - requires internet
pip install -r requirements.txt
```

## 📝 Core Commands

### Train Model
```bash
python train.py
```
**Expected time**: 15-25 min/epoch, ~7-12 hours total (30 epochs)  
**Output**: `outputs/checkpoints/best_model.pth`

### Evaluate Model
```bash
python evaluate.py
```
**Output**: Metrics, confusion matrix, ROC curve in `outputs/evaluation/`

### Predict Single Image
```bash
python predict.py --image path/to/image.jpg
```
**Output**: Prediction (Real/Fake) with confidence

### Predict Multiple Images
```bash
python predict.py --image_dir path/to/images/
```
**Output**: Predictions for all images + CSV file

## 📊 What You Get After Each Epoch

```
✅ Training Loss & Accuracy
✅ Validation Loss & Accuracy
✅ Precision (Overall + Per-Class)
✅ Recall (Overall + Per-Class)
✅ F1-Score (Overall + Per-Class)
✅ ROC-AUC Score
✅ TPR (True Positive Rate)
✅ TNR (True Negative Rate)
✅ FPR (False Positive Rate)
✅ FNR (False Negative Rate)
✅ PPV (Positive Predictive Value)
✅ NPV (Negative Predictive Value)
✅ Confusion Matrix (Visual)
✅ Epoch Time + ETA
```

## 📁 Important Files

| File | Purpose |
|------|---------|
| `config/config.yaml` | All settings (batch size, learning rate, etc.) |
| `train.py` | Main training script |
| `evaluate.py` | Test set evaluation |
| `predict.py` | Predict on new images |
| `outputs/checkpoints/best_model.pth` | Best trained model |
| `outputs/metrics/training_metrics.csv` | All epoch metrics |
| `outputs/metrics/training_history.png` | Training curves |
| `README.md` | Complete documentation |

## ⚙️ Key Configuration Parameters

Edit `config/config.yaml`:

```yaml
training:
  epochs: 30              # Number of training epochs
  batch_size: 32          # Batch size (reduce if out of memory)
  learning_rate: 0.0001   # Initial learning rate

model:
  architecture: efficientnet_b0  # Model type
  pretrained: true               # Use ImageNet weights
  dropout: 0.3                   # Regularization

hardware:
  device: cpu             # cpu or cuda
  num_workers: 2          # Data loading workers
```

## 🔧 Common Issues & Solutions

### Out of Memory
```yaml
# In config.yaml, reduce:
batch_size: 16    # or 8
num_workers: 0    # or 1
```

### Slow Training
```yaml
# In config.yaml, reduce:
image:
  size: 128       # or 192 (from 224)
batch_size: 16
```

### Model Not Improving
```yaml
# In config.yaml, try:
learning_rate: 0.00005  # Lower LR
epochs: 40              # More epochs
```

## 📈 Expected Performance

| Metric | Initial | Mid-Training | Well-Trained |
|--------|---------|--------------|--------------|
| Val Accuracy | 70-75% | 85-90% | 90-95% |
| F1-Score | 0.65-0.70 | 0.80-0.85 | 0.88-0.93 |
| ROC-AUC | 0.75-0.80 | 0.88-0.92 | 0.93-0.97 |

## 💡 Quick Tips

1. **First training run**: Downloads pretrained weights (~20MB, internet needed)
2. **Monitor training**: Check console for real-time metrics
3. **Stop training**: Press Ctrl+C (progress is saved)
4. **Resume training**: Re-run `python train.py` (not implemented, will start fresh)
5. **Check results**: Look in `outputs/` folder
6. **Batch size**: Reduce if out of memory, increase if you have spare RAM
7. **Best model**: Automatically saved as `best_model.pth`

## 📞 Getting Help

1. Check `README.md` for detailed documentation
2. Review `outputs/logs/training.log` for detailed error messages
3. See troubleshooting section in README
4. Check configuration in `config/config.yaml`

## ✅ Pre-Flight Checklist

Before training:
- [ ] Dependencies installed
- [ ] Dataset in `data/Train`, `data/Validation`, `data/Test`
- [ ] Each folder has `Fake/` and `Real/` subfolders
- [ ] Config reviewed (`config/config.yaml`)
- [ ] At least 10GB free disk space
- [ ] No other heavy processes running

## 🎯 Success Indicators

Training is going well if:
- ✅ Validation accuracy increases over epochs
- ✅ Validation loss decreases over epochs
- ✅ Confusion matrix shows high diagonal values
- ✅ F1-Score > 0.85 after 15-20 epochs
- ✅ No large gap between train and val accuracy

## 📦 Project Structure Summary

```
deepfake_detection/
├── config/           # Configuration
├── data/            # Your dataset (140K images)
├── src/             # Source code modules
├── outputs/         # Training results (auto-created)
├── train.py         # Main training
├── evaluate.py      # Evaluation
├── predict.py       # Prediction
└── README.md        # Documentation
```

---

**Ready to start?** Run: `python train.py`
