#!/usr/bin/env python3
"""
Simple test script for Auto-Derain system on Kaggle.
Usage: python test.py <input_dir> [output_dir] [rlp_weights] [nerd_weights] [gt_dir] [dataset_name]

"""

import sys
from pathlib import Path

def main():
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "results"
    rlp_weights = sys.argv[3] if len(sys.argv) > 3 else None
    nerd_weights = sys.argv[4] if len(sys.argv) > 4 else None
    gt_dir = sys.argv[5] if len(sys.argv) > 5 else None
    dataset_name = sys.argv[6] if len(sys.argv) > 6 else "test_dataset"
    
    
    try:
        from classifier import ImgClassifier
        classifier = ImgClassifier()
        
        if not rlp_weights and not nerd_weights:
            image_files = []
            for ext in ["*.jpg", "*.png", "*.jpeg"]:
                image_files.extend(list(Path(input_dir).rglob(ext)))
            
            print(f"📁 Found {len(image_files)} images")
            
            for img_path in image_files[:5]:  # Test first 5
                scene_type = classifier.predict_day_night(str(img_path))
                print(f"📸 {img_path.name}: {scene_type.upper()}")
        
        else:
            print("\n🌧️ Deraining...")
            classifier.derain_auto(
                image_dir=input_dir,
                output_dir=output_dir,
                rlp_weights=rlp_weights,
                nerd_weights=nerd_weights,
                gt_dir=gt_dir
            )
            
            print(f"✅ Results saved to: {output_dir}")
            
            # Evaluation if ground truth is provided
            if gt_dir and Path(gt_dir).exists():
                print(f"\n📊 Evaluating results for dataset: {dataset_name}")
                try:
                    from evaluate import evaluate
                    
                    avg_psnr, avg_ssim = evaluate(
                        root_dir=output_dir,
                        gt_root_dir=gt_dir,
                        dataset=dataset_name,
                        device="cuda" if classifier.device == "cuda" else "cpu"
                    )
                    
                    print(f"📈 {dataset_name} - Average PSNR: {avg_psnr:.2f} dB")
                    print(f"📈 {dataset_name} - Average SSIM: {avg_ssim:.4f}")
                    
                except Exception as eval_error:
                    print(f"⚠️ Evaluation failed: {eval_error}")
            else:
                print("⚠️ No ground truth provided, skipping evaluation")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()