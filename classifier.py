"""
Auto Derain - Automatic day/night scene detection and deraining
"""

import os
import time
import warnings
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid")


def get_free_gpu_memory():
    """Return free GPU memory in GiB."""
    if not torch.cuda.is_available():
        return 0
    free, total = torch.cuda.mem_get_info()
    return free / (1024 ** 3)


class ImgClassifier:
    """
    Automatic image classifier and deraining pipeline.
    
    Detects whether images are day or night scenes, then applies
    the appropriate deraining model (RLP for night, Improve-NeRD for day).
    """
    
    def __init__(self, model_id="openai/clip-vit-base-patch32"):
        """Initialize CLIP model for day/night classification."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)

        # Day/night classification prompts
        self.labels = ["night", "day"]
        texts = ["a photo taken at night", "a photo taken in the day"]
        self.text_inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        self.text_inputs = {k: v.to(self.device) for k, v in self.text_inputs.items()}

        # Model configurations
        self.model_map = {
            "night": {
                "name": "RLP",
                "repo_path": "/kaggle/working/auto-derain/RLP",
                "cmd_template": (
                    "python {repo}/test.py "
                    "--gpus 1 "
                    "--input_dir {input_dir} "
                    "--result_dir {result_dir} "
                    "--weights {weights} "
                    "--model_name Uformer_T_RLP_RPIM "
                    "--tile"
                ),
            },
            "day": {
                "name": "Improve-NeRD-Rain",
                "repo_path": "/kaggle/working/auto-derain/Improve-NeRD-rain",
                "cmd_template": (
                    "python {repo}/test.py "
                    "--input_dir {input_dir} "
                    "--gt_dir {gt_dir} "
                    "--output_dir {result_dir} "
                    "--weights {weights} "
                    "--gpus 0 "
                    "--win_size 256 "
                    "--batch_size 1 "
                    "--save_images"
                ),
            },
        }

    def predict_day_night(self, image_path):
        """
        Classify image as 'day' or 'night'.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            str: 'day' or 'night'
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(
                input_ids=self.text_inputs["input_ids"],
                attention_mask=self.text_inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
            )
            probs = torch.softmax(outputs.logits_per_image, dim=-1)[0]
            pred = self.labels[int(probs.argmax())]
        return pred

    def derain_single(
        self,
        image_path,
        output_dir,
        rlp_weights,
        nerd_weights,
        gt_dir=None,
        log=True
    ):
        """
        Process a single image with automatic scene detection.
        
        Args:
            image_path: Path to input rainy image
            output_dir: Directory to save derained result
            rlp_weights: Path to RLP model weights (for night)
            nerd_weights: Path to NeRD model weights (for day)
            gt_dir: Ground truth directory (optional)
            log: Whether to print progress logs
        """
        image_path = Path(image_path)
        scene_type = self.predict_day_night(str(image_path))
        model_info = self.model_map[scene_type]

        # Create temporary folder for single image
        tmp_dir = Path(tempfile.mkdtemp())
        shutil.copy(image_path, tmp_dir / image_path.name)
        
        # Compose command
        if scene_type == "night":
            cmd = model_info["cmd_template"].format(
                repo=model_info["repo_path"],
                input_dir=str(tmp_dir),
                result_dir=output_dir,
                weights=rlp_weights,
            )
        else:
            cmd = model_info["cmd_template"].format(
                repo=model_info["repo_path"],
                input_dir=str(tmp_dir),
                gt_dir=gt_dir or str(tmp_dir),
                result_dir=output_dir,
                weights=nerd_weights,
            )

        # Auto-decide whether to use tiling
        use_tile = False
        if scene_type == "night":
            free_mem = get_free_gpu_memory()
            img = Image.open(image_path)
            w, h = img.size
        
            if max(w, h) > 1280 or free_mem < 3.0:
                use_tile = True
        
            if use_tile and log:
                print(f"⚙️ Enabling tile mode (image={w}x{h}, free_mem={free_mem:.1f}GB)")

        if scene_type == "night" and not use_tile:
            cmd = cmd.replace("--tile", "")
            
        if log:
            print(f"🌗 Detected: {scene_type.upper()} → {model_info['name']}")
            print(f"🚀 Running model on single image\n{cmd}\n")
    
        # Run derain model
        result = subprocess.run(cmd, shell=True, text=True)
    
        # Normalize RLP output structure
        self._normalize_output(output_dir)
    
        if log:
            result_img = next(Path(output_dir).rglob(f"*{image_path.stem}*"), None)
            if result_img:
                self.visualize_img(result_img, title=f"Derained ({scene_type})")
            else:
                print("⚠️ No output files found.")

    def derain_auto(
        self,
        image_dir,
        output_dir,
        rlp_weights,
        nerd_weights,
        gt_dir=None,
    ):
        """
        Batch process multiple images with automatic scene detection.
        
        Pipeline:
        1. Classify all images into day/night
        2. Run appropriate model for each group
        3. Normalize output structure
        
        Args:
            image_dir: Directory containing input rainy images
            output_dir: Directory to save derained results
            rlp_weights: Path to RLP model weights (for night)
            nerd_weights: Path to NeRD model weights (for day)
            gt_dir: Ground truth directory (optional)
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        tmp_root = Path(tempfile.mkdtemp(prefix="derain_auto_"))
        day_dir, night_dir = tmp_root / "day", tmp_root / "night"
        os.makedirs(day_dir, exist_ok=True)
        os.makedirs(night_dir, exist_ok=True)
    
        # Step 1: Classify images
        all_images = sorted([
            p for p in image_dir.glob("*") 
            if p.suffix.lower() in [".jpg", ".png", ".jpeg"]
        ])
        print(f"📂 Found {len(all_images)} images in {image_dir}")
        print("🔍 Classifying into day/night...")
    
        for img_path in tqdm(all_images, desc="Classifying", unit="img"):
            try:
                scene_type = self.predict_day_night(str(img_path))
                target_dir = day_dir if scene_type == "day" else night_dir
                shutil.copy2(img_path, target_dir / img_path.name)
            except Exception as e:
                tqdm.write(f"⚠️ Skipped {img_path.name}: {e}")
    
        print(f"✅ Classification done: {len(list(day_dir.glob('*')))} day, "
              f"{len(list(night_dir.glob('*')))} night")
    
        # Step 2: Run deraining models
        for scene_type, input_folder in [("night", night_dir), ("day", day_dir)]:
            if not any(input_folder.glob("*")):
                print(f"⭐️ Skipping {scene_type} (no images)")
                continue
        
            model_info = self.model_map[scene_type]
            weights = nerd_weights if scene_type == "day" else rlp_weights
        
            cmd_kwargs = {
                "repo": model_info["repo_path"],
                "input_dir": str(input_folder),
                "result_dir": str(output_dir),
                "weights": weights,
            }
            if "{gt_dir}" in model_info["cmd_template"]:
                cmd_kwargs["gt_dir"] = gt_dir or str(input_folder)
        
            cmd = model_info["cmd_template"].format(**cmd_kwargs)
        
            # Auto-tile logic for night images
            if scene_type == "night":
                free_mem = get_free_gpu_memory()
                largest_img = max(input_folder.glob("*"), key=lambda p: p.stat().st_size)
                w, h = Image.open(largest_img).size
        
                use_tile = max(w, h) > 1280 or free_mem < 3.0
                if use_tile:
                    print(f"⚙️ Enabling tile mode for night model "
                          f"(max image {w}x{h}, free {free_mem:.1f}GB)")
                else:
                    cmd = cmd.replace("--tile", "")
                    print(f"⚙️ Running without tiling "
                          f"(max image {w}x{h}, free {free_mem:.1f}GB)")
        
            print(f"\n🚀 Running {scene_type.upper()} model: {model_info['name']}")
            start = time.time()
            result = subprocess.run(cmd, shell=True, text=True)
            elapsed = time.time() - start
        
            if result.returncode == 0:
                print(f"✅ {scene_type.capitalize()} model done in {elapsed:.1f}s")
            else:
                print(f"⚠️ {scene_type.capitalize()} model failed.")
    
        # Step 3: Normalize output structure
        print("\n🧹 Normalizing output...")
        self._normalize_output(output_dir)
        print("✅ All done!")
        
    def _normalize_output(self, output_dir):
        """Move RLP nested outputs to root level."""
        sub_output = Path(output_dir) / "Uformer_T_RLP_RPIM"
        if sub_output.exists():
            for f in sub_output.glob("*"):
                shutil.move(str(f), Path(output_dir) / f.name)
            shutil.rmtree(sub_output)
            print("📁 Normalized RLP output to:", output_dir)

    def visualize_img(self, image, title=""):
        """Display image using matplotlib."""
        img = cv2.imread(str(image))
        if img is None:
            print(f"⚠️ Cannot read image: {image}")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")
        plt.show()


# Test if module can be imported
if __name__ == "__main__":
    print("✅ classifier.py module loaded successfully!")
    print(f"Available classes: ImgClassifier")
    print(f"Available functions: get_free_gpu_memory")