import cv2
import time
import platform
import config
from core.detector import TheftDetector

class VideoProcessor:
    def __init__(self, yolo_model):
        self.model = yolo_model
        self.detector = TheftDetector(stationary_threshold_frames=50, proximity_pixels=100)
        self.frame_count = 0
        self.start_time = None
        self._setup_target_classes()

    def _setup_target_classes(self):
        self.target_indices = [0] # 사람 (person)
        for idx, name in self.model.names.items():
            if name in config.VALID_LOST_ITEMS:
                self.target_indices.append(idx)

    def process(self, video_path):
        theft_snapshots = None
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR]    Could not open video file: {video_path}")
            return None
            
        theft_snapshot = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            self.frame_count += 1
            if self.start_time is None:
                self.start_time = time.time()
                
            results = self.model.track(frame, persist=True, verbose=False, classes=self.target_indices, conf=0.5)
            is_theft = self.detector.update(results[0], frame, config.VALID_LOST_ITEMS)
            
            if is_theft:
                last_alert = self.detector.alerts[-1]
                theft_snapshots = {
                    'baseline': last_alert['baseline_file'],
                    'moment': last_alert['moment_file']
                }
                print("[INFO]     Theft detected. Stopping video processing.")
                break

            # SHOW_UI가 True인 경우에만 UI를 렌더링하고 표시
            if config.SHOW_UI:
                annotated_frame = results[0].plot()
                
                elapsed_time = time.time() - self.start_time
                avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # UI 오버레이 - 프레임 정보
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (10, 10), (280, 80), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.4, annotated_frame, 0.6, 0, annotated_frame)
                
                cv2.putText(annotated_frame, f"Frame: {self.frame_count} / {total_frames}", 
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Avg FPS: {avg_fps:.2f}", 
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Theft Detection System", annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # 헤드리스 모드일 때 UI 대기를 건너뛰고 프레임만 처리
                if self.frame_count % 100 == 0:  # 100프레임마다 진행 상황 출력
                    print(f"[INFO]     Processing... (Frame: {self.frame_count})")
                
        cap.release()
        cv2.destroyAllWindows()
        if platform.system() == 'Darwin': cv2.waitKey(1)
        
        return theft_snapshots if theft_snapshots else None
