#!/usr/bin/env python3
"""
Simple test script for Auto-Derain system on Kaggle.
Usage: python test.py <input_dir> [output_dir] [rlp_weights] [nerd_weights] [gt_dir]

"""

import sys
from pathlib import Path

def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2  else "results"
    rlp_weights = sys.argv[3] if len(sys.argv) > 3 else None
    nerd_weights = sys.argv[4] if len(sys.argv) > 4 else None
    gt_dir = sys.argv[5] if len(sys.argv) > 5 else None
    
    print("🚀 Auto-Derain Test")
    print(f"📂 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"🌙 RLP weights: {rlp_weights or 'None'}")
    print(f"☀️ NeRD weights: {nerd_weights or 'None'}")
    print(f"🎯 Ground truth: {gt_dir or 'None'}")
    
    try:
        from classifier import ImgClassifier
        classifier = ImgClassifier()
        
        if not rlp_weights and not nerd_weights:
            print("\n🔍 Testing classification...")
            
            image_files = []
            for ext in ["*.jpg", "*.png", "*.jpeg"]:
                image_files.extend(list(Path(input_dir).rglob(ext)))
            
            print(f"📁 Found {len(image_files)} images")
            
            for img_path in image_files[:5]:  # Test first 5
                scene_type = classifier.predict_day_night(str(img_path))
                print(f"📸 {img_path.name}: {scene_type.upper()}")
        
        else:
            print("\nDeraining...")
            classifier.derain_auto(
                image_dir=input_dir,
                output_dir=output_dir,
                rlp_weights=rlp_weights,
                nerd_weights=nerd_weights,
                gt_dir=gt_dir
            )
            
            print(f"✅ Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()