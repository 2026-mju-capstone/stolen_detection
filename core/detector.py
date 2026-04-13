import cv2
import math
import time
import os

class TheftDetector:
    def __init__(self, stationary_threshold_frames=50, proximity_pixels=100, missing_threshold_frames=10, output_dir="output"):
        self.stationary_threshold = stationary_threshold_frames
        self.proximity_pixels = proximity_pixels
        self.missing_threshold = missing_threshold_frames
        self.stationary_objects = {}  # {track_id: {'pos': (x,y), 'frames': N, 'last_seen_frame': F}}
        self.owners = {}              # {track_id: owner_person_id}
        self.near_history = {}        # {track_id: [person_id, ...]}
        self.tracked_items = {}
        self.alerts = []
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def calculate_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def is_touching(self, box1, box2):
        return not (box1[2] < box2[0] or box1[0] > box2[2] or 
                    box1[3] < box2[1] or box1[1] > box2[3])

    def update(self, results, frame, valid_classes):
        current_frame_ids = []
        persons = []
        items_in_frame = []

        if results.boxes.id is None:
            return self._handle_disappearances(current_frame_ids, frame)

        for box in results.boxes:
            if box.id is None: continue
            track_id = int(box.id[0])
            cls_id = int(box.cls[0])
            class_name = results.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            current_frame_ids.append(track_id)
            
            if class_name == 'person':
                persons.append({'id': track_id, 'center': center, 'bbox': (x1, y1, x2, y2)})
            elif class_name in valid_classes:
                items_in_frame.append({'id': track_id, 'center': center, 'bbox': (x1, y1, x2, y2), 'class_name': class_name})

        for item in items_in_frame:
            tid = item['id']
            center = item['center']
            ibox = item['bbox']
            
            if tid not in self.tracked_items:
                self.tracked_items[tid] = {
                    'last_pos': center, 'stay_count': 0, 
                    'is_stationary': False, 'class_name': item['class_name'],
                    'near_history': 0, 'missing_count': 0,
                    'last_person_id': None
                }
            
            self.tracked_items[tid]['missing_count'] = 0
            
            dist_moved = self.calculate_distance(center, self.tracked_items[tid]['last_pos'])
            if dist_moved < 50: 
                self.tracked_items[tid]['stay_count'] += 1
            else:
                self.tracked_items[tid]['stay_count'] = 0 
            
            if not self.tracked_items[tid]['is_stationary']:
                if self.tracked_items[tid]['stay_count'] >= self.stationary_threshold:
                    self.tracked_items[tid]['is_stationary'] = True
                    x1, y1, x2, y2 = ibox
                    h, w = frame.shape[:2]
                    y1, y2, x1, x2 = max(0, y1), min(h, y2), max(0, x1), min(w, x2)
                    self.tracked_items[tid]['baseline_crop'] = frame[y1:y2, x1:x2].copy()
                    print(f"[INFO]     ID {tid}({item['class_name']}) is now stationary.")
            
            self.tracked_items[tid]['last_pos'] = center

            closest_person_id = None
            min_dist = float('inf')
            for p in persons:
                dist = self.calculate_distance(center, p['center'])
                if self.is_touching(ibox, p['bbox']) or dist < self.proximity_pixels:
                    if dist < min_dist:
                        min_dist = dist
                        closest_person_id = p['id']
            
            if closest_person_id is not None:
                # 첫 감지 시 소유자 등록
                if tid not in self.owners:
                    self.owners[tid] = closest_person_id
                    print(f"[INFO]     Owner of ID {tid} assigned to Person {closest_person_id}")
                
                self.tracked_items[tid]['near_history'] = 120
                self.tracked_items[tid]['last_person_id'] = closest_person_id
            else:
                self.tracked_items[tid]['near_history'] = max(0, self.tracked_items[tid]['near_history'] - 1)

        return self._handle_disappearances(current_frame_ids, frame)

    def _handle_disappearances(self, current_frame_ids, frame):
        detected_theft = False
        for tid in list(self.tracked_items.keys()):
            if tid not in current_frame_ids:
                self.tracked_items[tid]['missing_count'] += 1
                if self.tracked_items[tid]['missing_count'] >= self.missing_threshold:
                    item_info = self.tracked_items[tid]
                    if item_info['is_stationary']:
                        if item_info['near_history'] > 0:
                            last_p = item_info.get('last_person_id')
                            owner_p = self.owners.get(tid)
                            
                            # 소유자가 아닌 사람이 가져갔을 때만 알람
                            if last_p is not None and last_p != owner_p:
                                self.trigger_alert(tid, frame)
                                detected_theft = True
                            else:
                                print(f"[INFO]     ID {tid} was retrieved by its owner (Person {owner_p}).")
                    del self.tracked_items[tid]
        return detected_theft

    def trigger_alert(self, track_id, frame):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        moment_file = os.path.join(self.output_dir, f"theft_moment_ID{track_id}_{timestamp}.jpg")
        cv2.imwrite(moment_file, frame)
        
        baseline_file = os.path.join(self.output_dir, f"stolen_item_baseline_ID{track_id}_{timestamp}.jpg")
        if 'baseline_crop' in self.tracked_items[track_id]:
            cv2.imwrite(baseline_file, self.tracked_items[track_id]['baseline_crop'])
            print(f"[SAVE]     Baseline image saved: {baseline_file}")

        print(f"[ALERT]    Theft suspected! Object ID: {track_id}")
        self.alerts.append({
            'id': track_id, 
            'time': timestamp, 
            'baseline_file': baseline_file,
            'moment_file': moment_file
        })
