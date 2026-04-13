from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
import config

def load_models():
    print("[INFO]     Loading CLIP and YOLO models...")
    clip_model = CLIPModel.from_pretrained(config.MODEL_ID)
    processor = CLIPProcessor.from_pretrained(config.MODEL_ID)
    yolo_model = YOLO(config.YOLO_MODEL_PATH)
    return clip_model, processor, yolo_model
