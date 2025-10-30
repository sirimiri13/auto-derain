#!/bin/bash

# Auto-Derain Setup Script
# This script sets up the environment for auto-derain on Kaggle

echo "🚀 Starting Auto-Derain setup..."

# Upgrade pip
pip install --upgrade pip -q

# Uninstall and reinstall transformers to avoid conflicts
echo "📦 Installing transformers (clean install)..."
pip uninstall transformers -y -q 2>/dev/null || true
pip install transformers==4.35.0 --no-cache-dir -q

# Install other dependencies
echo "📦 Installing other dependencies..."
pip install -q --upgrade numpy>=1.24.0 scipy>=1.11.0

# Clone repositories
echo "📥 Cloning model repositories..."
if [ ! -d "RLP" ]; then
    git clone https://github.com/tthieu0901/RLP.git
fi
if [ ! -d "Improve-NeRD-rain" ]; then
    git clone https://github.com/Master-HCMUS/Improve-NeRD-rain.git
fi

# Set PyTorch memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install dependencies for Improve-NeRD-rain
echo "📦 Installing Improve-NeRD-rain dependencies..."
pip install -q -r Improve-NeRD-rain/requirements.txt

echo "✅ Setup complete!"
echo "⚠️  Please RESTART KERNEL before importing modules"