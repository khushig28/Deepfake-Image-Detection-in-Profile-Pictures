#!/bin/bash
echo "========================================"
echo "Deepfake Detector - Quick Setup"
echo "========================================"
echo ""

echo "Installing PyTorch (CPU version)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "Starting web application..."
echo "Open your browser and go to: http://localhost:5000"
echo ""

python app.py
