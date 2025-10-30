#!/usr/bin/env python3
"""
Simple test script for Auto-Derain system on Kaggle.
Usage: python test.py <input_dir> <output_dir> <rlp_weights> <nerd_weights> <gt_dir>
"""

import sys
from pathlib import Path

def main():
    if len(sys.argv) != 6:
        print("❌ Error: Exactly 5 parameters required!")
        print("💡 Usage: python test.py <input_dir> <output_dir> <rlp_weights> <nerd_weights> <gt_dir>")
        return

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    rlp_weights = sys.argv[3]
    nerd_weights = sys.argv[4]
    gt_dir = sys.argv[5]
    
    # Extract dataset name from output_dir
    dataset_name = Path(output_dir).name
    
    print("🚀 Auto-Derain Test")
    print(f"📂 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"🌙 RLP weights: {rlp_weights}")
    print(f"☀️ NeRD weights: {nerd_weights}")
    print(f"🎯 Ground truth: {gt_dir}")
    print(f"📊 Dataset name: {dataset_name}")
    
    try:
        from classifier import ImgClassifier
        classifier = ImgClassifier()
        
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
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠️ Ground truth directory not found: {gt_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()