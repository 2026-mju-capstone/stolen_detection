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

# --- Improved Theft Detection Settings ---
THEFT_CONFIDENCE_THRESHOLD = 0.7  # Confidence threshold to trigger alert
VERIFICATION_FRAMES = 30          # Consecutive frames an object must be missing before alert
CONTACT_WEIGHT = 0.3              # Weight: Intensity of non-owner contact
OWNER_CLARITY_WEIGHT = 0.5        # Weight: Clarity of initial owner assignment (0.0 if no owner)
STATIONARY_WEIGHT = 0.2           # Weight: Certainty of stationary state
FLEEING_DETECTION = True          # Enable/disable fleeing behavior analysis
FLEEING_SPEED_THRESHOLD = 15.0     # Pixels per frame to be considered fleeing
