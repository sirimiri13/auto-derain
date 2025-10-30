# Auto-Derain: Automatic Day/Night Rain Removal

Automatic rain removal system that detects day/night scenes and applies the appropriate deraining model:
- **Night scenes**: RLP (Uformer_T_RLP_RPIM) 
- **Day scenes**: Improve-NeRD-Rain


## 📋 Setup Script Usage

### Syntax
```bash
bash setup.sh <base_directory>
```

### Examples
```bash
# Kaggle
bash setup.sh /kaggle/working

# Google Colab
bash setup.sh /content

# Local machine
bash setup.sh /path/to/your/directory
```

## 🧪 Test Script Usage

## 🐍 Python API Usage

### Import and Initialize
```python
from classifier import ImgClassifier

classifier = ImgClassifier(base_dir="/custom/path")
```


### Single Image Deraining
```python
classifier.derain_single(
    image_path="/path/to/rainy_image.jpg",
    output_dir="/path/to/output",
    rlp_weights="/path/to/rlp_weights.pth",
    nerd_weights="/path/to/nerd_weights.pth",
    gt_dir="/path/to/ground_truth"
)
```

### Batch Processing
```python
classifier.derain_auto(
    image_dir="/path/to/rainy_images",
    output_dir="/path/to/output",
    rlp_weights="/path/to/rlp_weights.pth",
    nerd_weights="/path/to/nerd_weights.pth",
    gt_dir="/path/to/ground_truth"
)
```

## 📊 Complete Kaggle Workflow

### Cell 1: Setup
```python
!git clone https://github.com/sirimiri13/auto-derain.git
%cd auto-derain
!bash setup.sh /kaggle/working/auto-derain
```

### Cell 2: Restart Kernel
**⚠️ Important: Restart kernel before proceeding**

### Cell 3: Define Paths and Run Test
```python
%cd /kaggle/working/auto-derain


# Run full pipeline with all parameters
!python test.py {INPUT_DIR} {OUTPUT_DIR} {RLP_WEIGHTS} {NERD_WEIGHTS} {GT_DIR} {DATASET_NAME}
```

## 📁 Project Structure

```
/kaggle/working/
├── auto-derain/           # Cloned repository
├── RLP/                   # Cloned RLP repository
├── Improve-NeRD-rain/     # Cloned NeRD repository
└── config.py              # Auto-generated config
```

## 🔧 Configuration

After running `setup.sh /kaggle/working`, a `config.py` file is created in `/kaggle/working/`:

```python
AUTODERAIN_BASE_DIR = "/kaggle/working"
AUTODERAIN_DIR = "/kaggle/working"
RLP_REPO_PATH = "/kaggle/working/RLP"
NERD_REPO_PATH = "/kaggle/working/Improve-NeRD-rain"
```

## 📋 Requirements

- Python 3.7+
- PyTorch with CUDA support
- transformers==4.41.2
- numpy<2.0.0
- PIL, OpenCV, matplotlib
- CLIP model for day/night classification

## 🎯 Expected Output

### Setup Output
```
🚀 Setting up Auto-Derain in /kaggle/working
✅ Setup complete!
```

### Test Pipeline Output
```
🚀 Auto-Derain Test
📂 Input: /kaggle/input/rain-dataset/rainy_images
📁 Output: /kaggle/working/results
🌙 RLP weights: /kaggle/input/weights/rlp.pth
☀️ NeRD weights: /kaggle/input/weights/nerd.pth
🎯 Ground truth: /kaggle/input/rain-dataset/clean_images
📊 Dataset name: rain100h

🌧️ Deraining...
🔍 Classifying into day/night...
✅ Classification done: 25 day, 25 night
🚀 Running NIGHT model: RLP
✅ Night model done in 45.2s
🚀 Running DAY model: Improve-NeRD-Rain  
✅ Day model done in 38.7s
✅ Results saved to: /kaggle/working/results

📊 Evaluating results for dataset: rain100h
📈 rain100h - Average PSNR: 28.45 dB
📈 rain100h - Average SSIM: 0.8823
```

## 🚨 Error Messages

### Test Script Errors
```bash
❌ Error: Exactly 6 parameters required!
💡 Usage: python test.py <input_dir> <output_dir> <rlp_weights> <nerd_weights> <gt_dir> <dataset_name>
```

### Common Issues
- **"Base directory not found"**: Re-run setup script: `bash setup.sh /kaggle/working`
- **Missing weights**: Ensure model weight files exist at specified paths
- **Memory issues**: System automatically handles tiling for large images

## 📚 References

- RLP: [https://github.com/tthieu0901/RLP](https://github.com/tthieu0901/RLP)
- Improve-NeRD-Rain: [https://github.com/Master-HCMUS/Improve-NeRD-rain](https://github.com/Master-HCMUS/Improve-NeRD-rain)
- CLIP: OpenAI CLIP for scene classification
