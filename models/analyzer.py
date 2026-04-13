import torch
import torch.nn.functional as F
from PIL import Image
import config

class ImageAnalyzer:
    def __init__(self, clip_model, processor):
        self.model = clip_model
        self.processor = processor

    def get_best_match(self, outputs, labels):
        probs = outputs.logits_per_image.softmax(dim=1).squeeze().tolist()
        sorted_output = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)
        return sorted_output[0]

    def analyze_stolen_item(self, img_path):
        if not img_path:
            return None

        image = Image.open(img_path).convert("RGB")
        
        # Category Analysis
        inputs = self.processor(text=config.ANALYSIS_CATEGORIES, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        best_item, prob_item = self.get_best_match(outputs, config.ANALYSIS_CATEGORIES)
        print(f"\n[ANALYSIS] Category : {best_item} ({prob_item*100:.1f}%)")

        # Color Analysis
        color_prompts = [f"{c} {best_item}" for c in config.ANALYSIS_COLORS]
        inputs_color = self.processor(text=color_prompts, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs_color = self.model(**inputs_color)
        best_color, prob_color = self.get_best_match(outputs_color, color_prompts)
        print(f"[ANALYSIS] Color    : {best_color.split()[0]} ({prob_color*100:.1f}%)")

        return best_item, best_color

    def extract_vector(self, image_path):
        if not image_path: return None
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            if hasattr(outputs, 'image_embeds'): features = outputs.image_embeds
            elif hasattr(outputs, 'pooler_output'): features = outputs.pooler_output
            elif isinstance(outputs, torch.Tensor): features = outputs
            else: features = outputs[0]
            features = F.normalize(features, p=2, dim=-1)
            
        return features.squeeze().cpu().numpy().tolist()
