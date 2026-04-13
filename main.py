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
    theft_img = video_proc.process(config.VIDEO_PATH)

    # 4. Step 2: If theft detected, analyze the item
    if theft_img:
        print(f"\n[STEP 2]   Theft alert triggered.")
        print(f"[RESULT]   Baseline image saved at: {theft_img}")
        
        # Detailed Analysis
        analyzer.analyze_stolen_item(theft_img)
        
        # Feature Vector Extraction
        vector = analyzer.extract_vector(theft_img)
        if vector:
            print(f"\n[STEP 3]   Vector extraction completed. Dim: {len(vector)}")
            print(f"[RESULT]   Vector sample: {vector[:5]}...")
    else:
        print("\n[RESULT]   No theft events detected during video processing.")

if __name__ == '__main__':
    main()
