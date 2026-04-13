import cv2
import platform
import config
from core.detector import TheftDetector

class VideoProcessor:
    def __init__(self, yolo_model):
        self.model = yolo_model
        self.detector = TheftDetector(stationary_threshold_frames=50, proximity_pixels=100)
        self._setup_target_classes()

    def _setup_target_classes(self):
        self.target_indices = [0] # person
        for idx, name in self.model.names.items():
            if name in config.VALID_LOST_ITEMS:
                self.target_indices.append(idx)

    def process(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR]    Could not open video file: {video_path}")
            return None
            
        theft_snapshot = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            results = self.model.track(frame, persist=True, verbose=False, classes=self.target_indices, conf=0.5)
            is_theft = self.detector.update(results[0], frame, config.VALID_LOST_ITEMS)
            
            if is_theft:
                last_alert = self.detector.alerts[-1]
                theft_snapshot = last_alert['file']
                print("[INFO]     Theft detected. Stopping video processing.")
                break

            annotated_frame = results[0].plot()
            cv2.imshow("Theft Detection System", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        if platform.system() == 'Darwin': cv2.waitKey(1)
        
        return theft_snapshot
