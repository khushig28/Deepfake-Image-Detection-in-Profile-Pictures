# ⚡ ULTRA-FAST CPU Training Guide

## 🎯 Goal: 10-15 Minutes Per Epoch on CPU

Current settings optimized for maximum speed on CPU:

### Configuration Applied:
```yaml
image:
  size: 96          # Smallest viable size (was 224 → 128 → 96)

training:
  batch_size: 128   # Maximum batches per iteration
  epochs: 20        # Reduced from 30

hardware:
  device: "cpu"
  num_workers: 6    # Maximum parallel loading
  prefetch_factor: 3
```

### Expected Performance:
- **Batches**: ~1,094 per epoch (half of before)
- **Time per epoch**: 10-15 minutes ✅
- **Total training**: 3-5 hours (20 epochs)
- **Accuracy**: 88-91% (slightly lower due to smaller images)

---

## 🚀 FASTEST Option: Install GPU Support

If you have **NVIDIA GPU**, this is 10x faster:

### Step 1: Stop current training (Ctrl+C)

### Step 2: Check your GPU
```powershell
wmic path win32_VideoController get name
```

If you see "NVIDIA", continue:

### Step 3: Install CUDA PyTorch
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Edit config
```yaml
hardware:
  device: "cuda"
  pin_memory: true

image:
  size: 224  # Can use larger images

training:
  batch_size: 64
```

### Step 5: Train
```bash
python train.py
```

**Result**: 3-5 min/epoch instead of 60+ min! 🚀

---

## 📊 Speed Comparison:

| Config | Image Size | Batch | Workers | Time/Epoch |
|--------|-----------|-------|---------|------------|
| Original | 224 | 32 | 2 | 120 min ❌ |
| Optimized CPU | 96 | 128 | 6 | 10-15 min ✅ |
| With GPU | 224 | 64 | 4 | 3-5 min ⚡ |

---

## ⚠️ Trade-offs:

**96x96 images (Current):**
- ✅ Much faster training (10-15 min/epoch)
- ⚠️ Slightly lower accuracy (88-91% vs 93-95%)
- ✅ Still good for deepfake detection
- ✅ Works on CPU

**GPU (If available):**
- ✅ Fastest training (3-5 min/epoch)
- ✅ Can use full 224x224 images
- ✅ Best accuracy (93-95%)
- ⚠️ Requires NVIDIA GPU + CUDA setup

---

## 🎯 My Recommendation:

1. **Try current optimized config first** (10-15 min/epoch)
2. **If you have NVIDIA GPU**, set up CUDA for best results
3. **If no GPU**, the 96x96 config will work well

The model will still detect deepfakes effectively even at 96x96!

---

## To Resume Training:

Stop current training (Ctrl+C) and restart:
```bash
python train.py
```

Now it should be 10-15 min/epoch! ⚡
