import config
from models.loader import load_models
from models.analyzer import ImageAnalyzer
from core.processor import VideoProcessor

def main():
    # 1. Load Models through models/loader.py
    clip_model, processor, yolo_model = load_models()
    
    # 2. Initialize Functional Components from core and models
    video_proc = VideoProcessor(yolo_model)
    analyzer = ImageAnalyzer(clip_model, processor)

    # 3. Step 1: Process Video to find theft
    print(f"\n[STEP 1]   Starting video monitoring: {config.VIDEO_PATH}")
    snapshots = video_proc.process(config.VIDEO_PATH)

    # 4. Step 2: If theft detected, analyze and show images
    if snapshots:
        baseline_img = snapshots['baseline']
        moment_img = snapshots['moment']
        
        print(f"\n[STEP 2]   Theft alert triggered.")
        print(f"[RESULT]   Baseline image: {baseline_img}")
        print(f"[RESULT]   Moment image:   {moment_img}")
        
        # Show both images
        from PIL import Image
        try:
            Image.open(moment_img).show(title="THEFT MOMENT")
            Image.open(baseline_img).show(title="STOLEN ITEM (BEFORE)")
        except:
            pass

        # Detailed Analysis using baseline
        analyzer.analyze_stolen_item(baseline_img)
        
        # Feature Vector Extraction
        vector = analyzer.extract_vector(baseline_img)
        if vector:
            print(f"\n[STEP 3]   Vector extraction completed. Dim: {len(vector)}")
            print(f"[RESULT]   Vector sample: {vector[:5]}...")
    else:
        print("\n[RESULT]   No theft events detected during video processing.")

if __name__ == '__main__':
    main()
