# 🚀 Quick Deploy Commands

## GitHub Upload (Copy & Paste)

```bash
# Navigate to project
cd e:\major_project_implementation\deepfake_detection

# Initialize Git
git init

# Configure user
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Add all files
git add .

# Commit
git commit -m "Initial commit: Deepfake Detection System with 98.71% accuracy"

# Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/deepfake-detection.git

# Push
git branch -M main
git push -u origin main
```

## Vercel Deploy

1. Go to https://vercel.com
2. Sign up with GitHub
3. Click "New Project"
4. Import `deepfake-detection` repository
5. Click "Deploy"
6. Done! ✅

## Alternative: Hugging Face Spaces (Recommended for ML)

```bash
# Create space at huggingface.co/spaces
# Then:

git clone https://huggingface.co/spaces/YOUR_USERNAME/deepfake-detector
cd deepfake-detector
cp -r e:\major_project_implementation\deepfake_detection\* .
git lfs install
git lfs track "*.pth"
git add .
git commit -m "Deploy deepfake detector"
git push
```

**Live at:** `https://huggingface.co/spaces/YOUR_USERNAME/deepfake-detector`

## Files Created for Deployment

✅ `.gitignore` - Exclude unnecessary files
✅ `vercel.json` - Vercel configuration  
✅ `runtime.txt` - Python version
✅ `.gitattributes` - Git LFS for large files
✅ `GITHUB_VERCEL_DEPLOY.md` - Full guide

**You're all set to deploy! 🎉**
