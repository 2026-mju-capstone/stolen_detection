# Configuration constants

MODEL_ID = "openai/clip-vit-base-patch32"
YOLO_MODEL_PATH = "yolo11s.pt"
VIDEO_PATH = "video/test.mp4"
SHOW_UI = False  # Toggle UI display (Set to False for maximum performance)
# YOLO tracking targets
VALID_LOST_ITEMS = {
    'backpack', 'umbrella', 'handbag', 
    'bottle', 'cup', 'cell phone', 'book'
}

# CLIP analysis categories
ANALYSIS_CATEGORIES = [
    "smartphone", "earphones", "bag", "wallet", 
    "credit card", "student ID card", "textbook", "notebook", 
    "umbrella", "glasses"
]

# CLIP color prompts
ANALYSIS_COLORS = [
    "black", "white", "gray", "red", "blue", "green", 
    "yellow", "brown", "pink", "purple", "orange", "beige"
]
