import config
from models.loader import load_models
from models.analyzer import ImageAnalyzer
from core.processor import VideoProcessor
from PIL import Image

def main():
    # 1. models/loader.py를 통해 모델 로드
    clip_model, processor, yolo_model = load_models()
    
    # 2. core 및 models에서 기능 컴포넌트 초기화
    video_proc = VideoProcessor(yolo_model)
    analyzer = ImageAnalyzer(clip_model, processor)

    # 3. 1단계: 도난 확인을 위한 비디오 처리
    print(f"\n[STEP 1]   Starting video monitoring: {config.VIDEO_PATH}")
    snapshots = video_proc.process(config.VIDEO_PATH)

    # 4. 2단계: 도난 탐지 시 이미지 분석 및 출력
    if snapshots:
        baseline_img = snapshots['baseline']
        moment_img = snapshots['moment']
        
        print(f"\n[STEP 2]   Theft alert triggered.")
        print(f"[RESULT]   Baseline image: {baseline_img}")
        print(f"[RESULT]   Moment image:   {moment_img}")
        
        # 두 이미지 모두 표시
        try:
            Image.open(moment_img).show(title="THEFT MOMENT")
            Image.open(baseline_img).show(title="STOLEN ITEM (BEFORE)")
        except:
            pass

        # 베이스라인 이미지를 이용한 상세 분석
        analyzer.analyze_stolen_item(baseline_img)
        
        # 특징 벡터 추출
        vector = analyzer.extract_vector(baseline_img)
        if vector:
            print(f"\n[STEP 3]   Vector extraction completed. Dim: {len(vector)}")
            print(f"[RESULT]   Vector sample: {vector[:5]}...")
    else:
        print("\n[RESULT]   No theft events detected during video processing.")

if __name__ == '__main__':
    main()
