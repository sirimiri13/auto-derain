#!/bin/bash

# Auto-Derain Setup Script
# This script sets up the environment for auto-derain on Kaggle

echo "🚀 Starting Auto-Derain setup..."

# Install transformers
echo "📦 Installing transformers..."
pip install -q transformers

# Clone repositories
echo "📥 Cloning model repositories..."
git clone https://github.com/tthieu0901/RLP.git
git clone https://github.com/Master-HCMUS/Improve-NeRD-rain.git

# Set PyTorch memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install dependencies for Improve-NeRD-rain
echo "📦 Installing Improve-NeRD-rain dependencies..."
pip install -q -r Improve-NeRD-rain/requirements.txt

echo "✅ Setup complete!"