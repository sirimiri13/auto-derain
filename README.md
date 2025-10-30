# Auto-Derain: Automatic Day/Night Rain Removal

Automatic rain removal system that detects day/night scenes and applies the appropriate deraining model:

- **Night scenes**: RLP (Uformer_T_RLP_RPIM)
- **Day scenes**: Improve-NeRD-Rain

## � Setup on Kaggle

```python
# Cell 1: Setup
!git clone https://github.com/sirimiri13/auto-derain.git
%cd auto-derain
!bash setup.sh /kaggle/working/auto-derain

# ⚠️ RESTART KERNEL after setup
```

## 🧪 Test Script Usage

```bash
python test.py <input_dir> <output_dir> <rlp_weights> <nerd_weights> <gt_dir>
```

**Example:**

```python
!python test.py \
    /kaggle/input/rain-dataset/images \
    /kaggle/working/results \
    /kaggle/input/weights/rlp.pth \
    /kaggle/input/weights/nerd.pth \
    /kaggle/input/rain-dataset/ground_truth
```

## � Python API Usage

```python
from classifier import ImgClassifier

# Initialize
classifier = ImgClassifier()

# Single image
classifier.derain_single(
    image_path="<image_path>",
    output_dir="<output_dir>",
    rlp_weights="<rlp_weights>",
    nerd_weights="<nerd_weights>",
    gt_dir="<gt_dir>"
)

# Batch processing
classifier.derain_auto(
    image_dir="<input_dir>",
    output_dir="<output_dir>",
    rlp_weights="<rlp_weights>",
    nerd_weights="<nerd_weights>",
    gt_dir="<gt_dir>"
)

# Evaluation
from evaluate import evaluate

avg_psnr, avg_ssim = evaluate(
    root_dir="<output_dir>",
    gt_root_dir="<gt_dir>",
    dataset="<dataset_name>",
    device="cuda"
)

