# 🚀 STEP-BY-STEP EXECUTION GUIDE
## Deepfake Detection Training - Complete Process

---

## ✅ STEP 1: Verify Python (DONE ✓)
**Status**: Python 3.13.2 is installed  
**Status**: PyTorch 2.9.1+cpu is installed  

---

## ✅ STEP 2: Install Remaining Dependencies

Open terminal in project folder and run:

```bash
cd e:\major_project_implementation\deepfake_detection
pip install -r requirements.txt
```

**What this does**: Installs all required Python packages (NumPy, Pandas, scikit-learn, matplotlib, etc.)  
**Time**: 2-3 minutes  
**Internet required**: Yes

**Expected output**: You'll see packages being downloaded and installed

---

## ✅ STEP 3: Verify Dataset Structure

Check that your dataset is organized correctly:

```
e:\major_project_implementation\deepfake_detection\data\
├── Train\
│   ├── Fake\     ← Should contain fake/AI-generated images
│   └── Real\     ← Should contain real images
├── Validation\
│   ├── Fake\
│   └── Real\
└── Test\
    ├── Fake\
    └── Real\
```

**What to check**:
- Each folder (Train, Validation, Test) has both "Fake" and "Real" subfolders
- Images are inside these subfolders
- You have a good number of images in each

---

## ✅ STEP 4: Start Training

Run the training command:

```bash
python train.py
```

**What this does**: 
- Loads your dataset (~140K images)
- Downloads pretrained EfficientNetB0 weights (~20MB, only first time)
- Starts training for 30 epochs
- Shows comprehensive metrics after each epoch
- Saves best model automatically

**Time**: 15-25 minutes per epoch, ~7-12 hours total  
**Internet required**: Only for first run (to download pretrained weights)

---

## ✅ STEP 5: Monitor Training Progress

**During training, you'll see:**

```
Epoch 1/30 [Train]: 100%|████████| 1234/1234 [18:34<00:00]
Epoch 1/30 [Val]:   100%|████████| 234/234 [02:15<00:00]

==============================================================================
EPOCH 1/30 SUMMARY
==============================================================================
Training   - Loss: 0.4521 | Accuracy: 0.7823
Validation - Loss: 0.3845 | Accuracy: 0.8234
==============================================================================

Validation Metrics:
- Accuracy:  0.8234
- Precision: 0.8156
- Recall:    0.8312
- F1-Score:  0.8233
- ROC-AUC:   0.8945
- Confusion Matrix: [shown in console]
- TPR, FPR, TNR, FNR: [all displayed]
==============================================================================
```

**What to watch for**:
- ✅ Validation accuracy should increase over epochs
- ✅ Validation loss should decrease
- ✅ Training shouldn't crash or show errors

**Where are results saved**:
- `outputs/checkpoints/best_model.pth` ← Your trained model
- `outputs/logs/training.log` ← Detailed log
- `outputs/metrics/training_metrics.csv` ← All epoch metrics
- `outputs/metrics/training_history.png` ← Training curves

---

## ✅ STEP 6: Wait for Training to Complete

**You can**:
- ✅ Let it run in background
- ✅ Check progress periodically
- ✅ Stop training with Ctrl+C (progress is saved in checkpoints)

**Expected final accuracy**: 90-95% validation accuracy after 30 epochs

---

## ✅ STEP 7: Evaluate Model on Test Set

After training completes, run:

```bash
python evaluate.py
```

**What this does**:
- Loads your best trained model
- Tests it on the test dataset
- Generates comprehensive metrics report
- Creates visualizations (confusion matrix, ROC curve)

**Time**: 5-10 minutes  
**Output location**: `outputs/evaluation/`

---

## ✅ STEP 8: Test on Random Images

Predict if an image is real or fake:

**Single image:**
```bash
python predict.py --image path\to\your\image.jpg
```

**Multiple images:**
```bash
python predict.py --image_dir path\to\your\images_folder
```

**What you'll see**:
```
===========================================================
PREDICTION RESULT
===========================================================
Image:       test_image.jpg
Prediction:  Fake
Confidence:  94.32%

Class Probabilities:
  Fake      : 94.32%
  Real      : 5.68%
===========================================================
```

---

## 🔍 TROUBLESHOOTING

### Problem: Out of Memory Error
**Solution**: Edit `config/config.yaml` and reduce:
```yaml
batch_size: 16    # or 8 (instead of 32)
num_workers: 0    # or 1 (instead of 2)
```

### Problem: Training Too Slow
**Solution**: This is expected on CPU. Each epoch takes 15-25 minutes for 140K images.

### Problem: Package Installation Error
**Solution**: 
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Problem: Can't Find Dataset
**Solution**: Verify folder names are exactly:
- `Train` (not "training" or "train")
- `Fake` and `Real` (capital F and R)

---

## 📊 WHAT SUCCESS LOOKS LIKE

**After 30 epochs, you should have**:
- ✅ Validation accuracy: 90-95%
- ✅ F1-Score: 0.88-0.93
- ✅ ROC-AUC: 0.93-0.97
- ✅ Best model saved at `outputs/checkpoints/best_model.pth`
- ✅ Training curves showing improvement
- ✅ Confusion matrix with high diagonal values

---

## 🎯 SUMMARY: YOUR ACTION ITEMS

**Right now, do this:**

1. ✅ Open terminal
2. ✅ Navigate to: `cd e:\major_project_implementation\deepfake_detection`
3. ✅ Run: `pip install -r requirements.txt`
4. ⏳ Wait 2-3 minutes for installation
5. ✅ Run: `python train.py`
6. ⏳ Wait 7-12 hours for training to complete
7. ✅ Run: `python evaluate.py`
8. ✅ Run: `python predict.py --image your_test_image.jpg`

**That's it! You're done! 🎉**

---

## 📞 NEED HELP?

If you encounter any errors:
1. Check `outputs/logs/training.log` for details
2. Verify dataset structure
3. Check available RAM (should have ~5GB free)
4. Try reducing batch_size in config if out of memory
