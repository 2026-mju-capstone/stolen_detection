import torch
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
import config

def load_models():
    print("[INFO]     Loading CLIP and YOLO models...")
    
    # MPS (Apple Silicon) 장치 확인
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[INFO]     Computing device: {device.upper()}")

    clip_model = CLIPModel.from_pretrained(config.MODEL_ID)
    processor = CLIPProcessor.from_pretrained(config.MODEL_ID)
    
    yolo_model = YOLO(config.YOLO_MODEL_PATH)
    yolo_model.to(device) # YOLO 모델을 감지된 장치로 이동
    
    return clip_model, processor, yolo_model
