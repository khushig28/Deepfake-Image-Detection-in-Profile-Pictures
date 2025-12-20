# 📦 DEPLOYMENT GUIDE - Run on Any Device

## 🎯 How to Transfer & Run This Project

This guide shows you how to **zip your project** and **run it on another device** (laptop, PC, server, etc.)

---

## 📋 STEP-BY-STEP DEPLOYMENT

### ✅ OPTION 1: Deploy with Trained Model (Recommended)

**Best for:** Running the web app on another device with your trained model

---

### 📦 **Step 1: Prepare Files for Transfer**

#### **What to INCLUDE in ZIP:**

**Essential Files:**
```
deepfake_detection/
├── app.py                      ✅ Web server
├── config/
│   └── config.yaml            ✅ Configuration
├── src/                       ✅ All source code
│   ├── __init__.py
│   ├── data/
│   ├── models/
│   ├── training/
│   └── utils/
├── templates/                 ✅ Web interface
│   └── index.html
├── static/                    ✅ CSS/JS
│   ├── style.css
│   └── script.js
├── outputs/
│   └── checkpoints/
│       └── best_model.pth    ✅ Trained model (CRITICAL!)
├── requirements.txt          ✅ Dependencies
├── README.md                 ✅ Documentation
├── WEB_APP_GUIDE.md         ✅ Usage instructions
└── MODEL_PERFORMANCE.md     ✅ Performance details
```

**Optional (Helpful):**
```
├── COMPLETE_GUIDE.md
├── EXPLAINABLE_AI.md
├── GPU_SETUP.md
└── outputs/metrics/          # Training metrics (optional)
```

#### **What to EXCLUDE (Save Space):**

**DON'T Include:**
```
❌ data/                    # Dataset (very large ~20GB)
❌ outputs/logs/            # Training logs
❌ outputs/metrics/         # CSVs (unless needed)
❌ outputs/evaluation/      # Evaluation results
❌ uploads/                 # Temporary uploads
❌ __pycache__/             # Python cache
❌ *.pyc                    # Compiled Python
❌ .git/                    # Git repository
```

**Estimated ZIP size:** ~50-100 MB (with model, without dataset)

---

### 📦 **Step 2: Create ZIP File**

#### **Windows:**
1. Navigate to project folder:
   ```
   e:\major_project_implementation\
   ```

2. Right-click `deepfake_detection` folder

3. Select **"Send to" → "Compressed (zipped) folder"**

4. Name it: `deepfake_detection_deploy.zip`

#### **Or Using Command Line:**
```powershell
# Navigate to parent folder
cd e:\major_project_implementation\

# Create zip (requires PowerShell)
Compress-Archive -Path deepfake_detection -DestinationPath deepfake_detection_deploy.zip
```

---

### 📤 **Step 3: Transfer to New Device**

**Methods:**
- USB drive
- Cloud storage (Google Drive, Dropbox, OneDrive)
- Email (if < 25MB)
- File share (network)

---

### 💻 **Step 4: Setup on New Device**

#### **Requirements on New Device:**
- Windows, Linux, or macOS
- Python 3.8 or higher
- 8GB+ RAM recommended
- ~2GB free disk space

#### **Installation Steps:**

**1. Extract ZIP:**
```bash
# Extract to desired location
# Example: C:\Projects\deepfake_detection\
```

**2. Open Terminal/Command Prompt:**
```bash
# Navigate to project
cd C:\Projects\deepfake_detection
```

**3. Create Virtual Environment (Recommended):**
```bash
# Create venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

**4. Install Dependencies:**

**For CPU-only (works everywhere):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**For GPU (if new device has NVIDIA GPU):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**Time:** 3-5 minutes (downloads packages)

**5. Run the Web App:**
```bash
python app.py
```

**6. Open Browser:**
```
http://localhost:5000
```

**Done! ✅**

---

## 🚀 Quick Deployment Commands

**Copy-paste these on the new device:**

```bash
# 1. Navigate to extracted folder
cd path/to/deepfake_detection

# 2. Install PyTorch (CPU version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Install other dependencies
pip install -r requirements.txt

# 4. Run web app
python app.py

# 5. Open http://localhost:5000 in browser
```

**That's it! 5 commands and you're running! 🎉**

---

## 📋 Troubleshooting New Device

### Issue: "Model not found"
**Solution:** Ensure `outputs/checkpoints/best_model.pth` is in the ZIP

### Issue: "Module not found"
**Solution:** 
```bash
pip install -r requirements.txt
```

### Issue: "CUDA not available" (if you want GPU)
**Solution:** Install CUDA PyTorch:
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Port 5000 already in use
**Solution:** Edit `app.py` line ~200:
```python
app.run(host='0.0.0.0', port=5001, debug=False)  # Change port
```

---

## 🔄 OPTION 2: Deploy for Retraining

**If you want to retrain on new device:**

### **What to Include:**
```
✅ All source code (src/, config/, scripts)
✅ requirements.txt
✅ Documentation
❌ Trained model (will retrain)
❌ Dataset (transfer separately if needed)
```

### **On New Device:**
```bash
# 1. Setup as above (Steps 1-4)

# 2. Add your dataset to data/ folder
# 3. Run training
python train.py
```

---

## 🌐 OPTION 3: Deploy to Cloud/Server

### **For Production Deployment:**

**Cloud Platforms:**
- **Heroku** (easy, free tier)
- **AWS EC2** (scalable)
- **Google Cloud Run** (serverless)
- **Azure Web Apps** (enterprise)

**Quick Heroku Deployment:**

1. Create `Procfile`:
```
web: python app.py
```

2. Create `runtime.txt`:
```
python-3.13.2
```

3. Deploy:
```bash
heroku create deepfake-detector
git push heroku main
```

---

## 📊 File Size Breakdown

| What to Transfer | Size |
|------------------|------|
| **Source code** | ~2 MB |
| **Trained model** | ~20 MB |
| **Dependencies** | ~500 MB (installed on device) |
| **Dataset** | ~20 GB (optional, for retraining) |

**Recommended ZIP:** ~25-50 MB (code + model, no dataset)

---

## ✅ Checklist for New Device

**Before zipping:**
- [ ] `best_model.pth` exists in `outputs/checkpoints/`
- [ ] All source code files present
- [ ] `requirements.txt` included
- [ ] `README.md` included
- [ ] Dataset excluded (unless needed)

**On new device:**
- [ ] Python 3.8+ installed
- [ ] ZIP extracted
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] `python app.py` runs successfully
- [ ] Web app accessible at localhost:5000

---

## 🎯 Different Deployment Scenarios

### **Scenario 1: Demo on Laptop**
```
Transfer: Code + Model (~50 MB)
Setup time: 5 minutes
Internet: Needed for dependency install
```

### **Scenario 2: Production Server**
```
Transfer: Full project
Setup time: 10-15 minutes
Internet: Needed
Consider: Docker container
```

### **Scenario 3: Offline Device**
```
Transfer: Code + Model + Pre-downloaded deps
Setup time: 2 minutes
Internet: Not needed
Pre-download: pip download -r requirements.txt
```

---

## 🐳 Docker Deployment (Advanced)

**Create `Dockerfile`:**
```dockerfile
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
```

**Build and Run:**
```bash
docker build -t deepfake-detector .
docker run -p 5000:5000 deepfake-detector
```

---

## 📝 Pre-Transfer Checklist

**Test before zipping:**
```bash
# 1. Test web app works
python app.py
# Visit http://localhost:5000

# 2. Test prediction
python predict.py --image test_image.jpg

# 3. Verify model exists
ls outputs/checkpoints/best_model.pth
```

**If all 3 work → Ready to zip! ✅**

---

## 🎉 Summary

**To deploy to new device:**

1. **ZIP** the project (exclude `data/` folder)
2. **Transfer** ZIP file
3. **Extract** on new device
4. **Install** Python dependencies
5. **Run** `python app.py`
6. **Access** http://localhost:5000

**Done in ~5 minutes! 🚀**

---

## 💡 Pro Tips

### Minimize ZIP Size:
```bash
# Exclude unnecessary files
- Remove __pycache__/
- Remove .pyc files  
- Remove data/ folder
- Remove outputs/logs/
- Keep only best_model.pth
```

### Faster Setup on New Device:
```bash
# Create a setup.bat file
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python app.py
```

Then just run: `setup.bat`

### For Multiple Devices:
- Upload ZIP to Google Drive
- Share link
- Everyone can download and setup

---

## 📞 Support

**If issues on new device:**

1. Check Python version: `python --version`
2. Check dependencies: `pip list`
3. Check model exists: `ls outputs/checkpoints/`
4. Check logs: Look at terminal output
5. Review `README.md` in project

**Most common fix:** Reinstall dependencies
```bash
pip install -r requirements.txt
```

---

## ✅ You're Ready!

**Your project is fully portable!**

Just zip, transfer, and run on any device! 🎊
