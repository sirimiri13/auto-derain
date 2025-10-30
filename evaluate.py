"""
Evaluation module for deraining results
Calculate PSNR and SSIM metrics
"""

import os
import torch
import kornia
from tqdm import tqdm
from torchvision.io import read_image


def rgb_to_ycbcr(img: torch.Tensor) -> torch.Tensor:
    if img.ndim == 3:  # Single image
        img = img.unsqueeze(0)

    if img.dtype == torch.uint8:
        img = img.float()

    if img.max() > 1.0:
        img = img / 255.0

    T = torch.tensor([
        [65.481, 128.553, 24.966],
        [-37.797, -74.203, 112.0],
        [112.0, -93.786, -18.214]
    ], dtype=img.dtype, device=img.device) / 255
    
    offset = torch.tensor([16, 128, 128], dtype=img.dtype, device=img.device) / 255

    ycbcr = torch.zeros_like(img)
    for i in range(3):
        ycbcr[:, i, :, :] = (
            T[i, 0] * img[:, 0, :, :]
            + T[i, 1] * img[:, 1, :, :]
            + T[i, 2] * img[:, 2, :, :]
            + offset[i]
        )
    return ycbcr


def evaluate(root_dir: str, gt_root_dir: str, dataset: str, device: str = "cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    print(f"\n📊 Evaluating dataset: {dataset}")
    file_path = os.path.join(root_dir, dataset)

    # Get all result images
    image_files = [
        os.path.join(file_path, f)
        for f in os.listdir(file_path)
        if f.lower().endswith(('.jpg', '.png'))
    ]

    # Match ground truth files
    gt_files = []
    for f in image_files:
        base = os.path.basename(f)
        name, _ = os.path.splitext(base)

        
        # Handle GTAV naming convention
        if dataset.lower() == 'gtav':
            name = name.split('_')[0]
        else:
            gt_name = base
        gt_png = os.path.join(gt_root_dir, name + '.png')
        gt_jpg = os.path.join(gt_root_dir, name + '.jpg')
        if os.path.exists(gt_png):
            gt_files.append(gt_png)
        elif os.path.exists(gt_jpg):
            gt_files.append(gt_jpg)
        else:
            raise FileNotFoundError(f"Ground truth file not found for {name} (tried both .png and .jpg)")

    total_psnr, total_ssim = 0.0, 0.0
    img_num = len(image_files)

    # Process each image pair
    for img_file, gt_file in tqdm(
        zip(image_files, gt_files), 
        total=img_num, 
        desc=f"Processing {dataset}"
    ):
        # Load images
        input_img = read_image(img_file).float().unsqueeze(0).to(device)
        gt_img = read_image(gt_file).float().unsqueeze(0).to(device)

        # Convert to Y channel (luminance)
        input_y = rgb_to_ycbcr(input_img)[:, 0, :, :].unsqueeze(1)
        gt_y = rgb_to_ycbcr(gt_img)[:, 0, :, :].unsqueeze(1)

        # Calculate metrics
        psnr_val = kornia.metrics.psnr(input_y, gt_y, max_val=1.0)
        ssim_val = kornia.metrics.ssim(
            input_y, gt_y, window_size=11, max_val=1.0
        ).mean()

        total_psnr += psnr_val.item()
        total_ssim += ssim_val.item()

    avg_psnr = total_psnr / img_num
    avg_ssim = total_ssim / img_num
    
    print(f"✅ {dataset}: PSNR = {avg_psnr:.4f}, SSIM = {avg_ssim:.4f}")
    
    return avg_psnr, avg_ssim


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate deraining results")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root directory with result folders")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Ground truth directory")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    evaluate(args.root_dir, args.gt_dir, args.dataset, args.device)