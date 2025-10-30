#!/bin/bash
# Auto-Derain Setup Script

# Function to setup auto-derain
setup_autoderain() {
    local base_dir="$1"
    local autoderain_dir="$base_dir"
    
    echo "🚀 Setting up Auto-Derain in $base_dir"

    pip install -q "numpy<2.0.0,>=1.24.0" --force-reinstall

    pip install -q transformers==4.41.2 tokenizers==0.19.1 --no-cache-dir

    # Clone repositories
    if [ ! -d "$autoderain_dir/RLP" ]; then
        cd "$base_dir"
        git clone https://github.com/tthieu0901/RLP.git
    fi

    if [ ! -d "$autoderain_dir/Improve-NeRD-rain" ]; then
        cd "$base_dir"
        git clone https://github.com/Master-HCMUS/Improve-NeRD-rain.git
    fi

    # Set PyTorch memory allocation
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    # Save config
    cat > "$autoderain_dir/config.py" << EOF
AUTODERAIN_BASE_DIR = "$base_dir"
AUTODERAIN_DIR = "$autoderain_dir"
RLP_REPO_PATH = "$autoderain_dir/RLP"
NERD_REPO_PATH = "$autoderain_dir/Improve-NeRD-rain"
EOF

    echo "✅ Setup complete!"
}

# Main execution
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    setup_autoderain "$1"
else
    echo "📚 Auto-derain setup loaded!"
fi