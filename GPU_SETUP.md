# 🚀 GPU Training Setup Guide

## Step 1: Check if GPU is Available

Run this command to check:
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**If you see:**
- ✅ `CUDA Available: True` → You have GPU support! Continue to Step 2
- ❌ `CUDA Available: False` → You need to install CUDA-enabled PyTorch (see Step 3)

---

## Step 2: Enable GPU in Configuration (If CUDA is Available)

Edit `config/config.yaml`:

```yaml
# Hardware Optimization
hardware:
  device: "cuda"  # Change from "cpu" to "cuda"
  num_workers: 4
  pin_memory: true  # Change to true for GPU
  prefetch_factor: 2
```

**That's it!** Now run:
```bash
python train.py
```

### ⚡ Performance with GPU:
- **Epoch time**: 3-5 minutes (instead of 10-15 min on CPU)
- **Total training**: 1.5-2.5 hours (instead of 7-12 hours)
- **Can use larger images**: Set `image.size: 224` for better accuracy
- **Can use larger batches**: Set `batch_size: 128` or `256`

---

## Step 3: Install CUDA-enabled PyTorch (If CUDA Not Available)

**First, check your GPU:**
- Press `Windows + R`, type `dxdiag`, press Enter
- Go to "Display" tab and note your GPU name
- Check if it's NVIDIA GPU (required for CUDA)

### If you have NVIDIA GPU:

**Uninstall current PyTorch:**
```bash
pip uninstall torch torchvision
```

**Install CUDA version:**

For **CUDA 11.8** (most common):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

For **CUDA 12.1** (newer GPUs):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Not sure which CUDA version?** Install CUDA 11.8 (works on most GPUs).

**Verify installation:**
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

Should show: `CUDA Available: True`

---

## Step 4: Optimize Settings for GPU

Edit `config/config.yaml` for maximum GPU performance:

```yaml
# Image size - GPU can handle larger
image:
  size: 224  # or even 256 for better accuracy

# Training - GPU can handle larger batches
training:
  batch_size: 128  # or 64, 256 depending on GPU memory
  
# Hardware
hardware:
  device: "cuda"
  num_workers: 4
  pin_memory: true
```

---

## 🎯 Quick Comparison: CPU vs GPU

| Setting | CPU | GPU (NVIDIA) |
|---------|-----|--------------|
| **Image Size** | 128x128 | 224x224 or 256x256 |
| **Batch Size** | 32-64 | 128-256 |
| **Epoch Time** | 10-15 min | 3-5 min |
| **Total Training** | 7-12 hours | 1.5-2.5 hours |
| **Accuracy** | 91-93% | 93-96% |

---

## ⚠️ Common GPU Issues

### Issue: "CUDA out of memory"

**Solution 1**: Reduce batch size
```yaml
batch_size: 64  # or 32
```

**Solution 2**: Reduce image size
```yaml
image:
  size: 192  # or 160
```

**Solution 3**: Reduce workers
```yaml
num_workers: 2
```

### Issue: "RuntimeError: CUDA error: device-side assert triggered"

**Solution**: This is usually a data issue. Check your dataset for corrupted images.

### Issue: GPU usage shows 0% in Task Manager

**Solution**: 
- Make sure `device: "cuda"` in config
- Verify PyTorch sees GPU: `python -c "import torch; print(torch.cuda.is_available())"`

---

## 📊 Recommended GPU Settings

### For 4GB GPU (like GTX 1650, GTX 1050 Ti):
```yaml
image:
  size: 160
training:
  batch_size: 32
hardware:
  device: "cuda"
  num_workers: 2
  pin_memory: true
```

### For 6-8GB GPU (like RTX 3060, GTX 1660):
```yaml
image:
  size: 224
training:
  batch_size: 64
hardware:
  device: "cuda"
  num_workers: 4
  pin_memory: true
```

### For 10GB+ GPU (like RTX 3080, RTX 4070):
```yaml
image:
  size: 256
training:
  batch_size: 128
hardware:
  device: "cuda"
  num_workers: 4
  pin_memory: true
```

---

## 🚀 Quick Start with GPU

**1. Enable GPU in config:**
```yaml
hardware:
  device: "cuda"
  pin_memory: true
```

**2. Run training:**
```bash
python train.py
```

**3. Watch console - you should see:**
```
Device: cuda
GPU: NVIDIA GeForce GTX/RTX XXXX
GPU Memory: X.XX GB
```

**That's it!** Training will be much faster! ⚡

---

## 💡 Tips for Best GPU Performance

1. **Close other GPU applications** (games, Chrome with hardware acceleration, etc.)
2. **Monitor GPU usage** with Task Manager (Performance → GPU)
3. **Start with lower batch size** and increase if you have spare GPU memory
4. **Use larger images** (224 or 256) for better accuracy on GPU
5. **Keep drivers updated** for best performance

---

## ✅ Summary

**If CUDA is already available:**
1. Edit `config/config.yaml`
2. Change `device: "cuda"` and `pin_memory: true`
3. Run `python train.py`

**If CUDA is not available:**
1. Install CUDA-enabled PyTorch
2. Edit config as above
3. Run `python train.py`

**Enjoy 5-10x faster training! 🚀**
