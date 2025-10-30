#!/bin/bash

# Auto-Derain Setup Script
# This script sets up the environment for auto-derain on Kaggle

echo "🚀 Starting Auto-Derain setup..."

# Fix NumPy 2.x compatibility issue
echo "📦 Installing compatible NumPy version..."
pip install -q "numpy<2.0.0,>=1.24.0" --force-reinstall

# Install compatible transformers version for Kaggle
echo "📦 Installing transformers..."
pip install -q transformers==4.41.2 tokenizers==0.19.1 --no-cache-dir

# Clone repositories
echo "📥 Cloning model repositories..."
if [ ! -d "/kaggle/working/auto-rain/RLP" ]; then
    cd /kaggle/working
    git clone https://github.com/tthieu0901/RLP.git
fi
if [ ! -d "/kaggle/working/auto-rain/Improve-NeRD-rain" ]; then
    cd /kaggle/working
    git clone https://github.com/Master-HCMUS/Improve-NeRD-rain.git
fi

# Set PyTorch memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install dependencies for Improve-NeRD-rain
echo "📦 Installing Improve-NeRD-rain dependencies..."
pip install -q -r /kaggle/working/auto-derain/Improve-NeRD-rain/requirements.txt

echo "✅ Setup complete!"
echo "⚠️  Please RESTART KERNEL before importing modules"