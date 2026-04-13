import cv2
import math
import time
import os

class TheftDetector:
    def __init__(self, stationary_threshold_frames=50, proximity_pixels=100, missing_threshold_frames=10, output_dir="output"):
        self.stationary_threshold = stationary_threshold_frames
        self.proximity_pixels = proximity_pixels
        self.missing_threshold = missing_threshold_frames
        self.output_dir = output_dir
        
        self.tracked_items = {}  # {track_id: item_data}
        self.alerts = []
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def _calculate_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _is_touching(self, box1, box2):
        # Check for bounding box overlap
        return not (box1[2] < box2[0] or box1[0] > box2[2] or 
                    box1[3] < box2[1] or box1[1] > box2[3])

    def update(self, results, frame, valid_classes):
        """Main update loop for theft detection."""
        current_frame_ids = []
        persons, items_in_frame = self._parse_results(results, valid_classes)
        
        # Track which IDs are present in the current frame
        for item in items_in_frame:
            current_frame_ids.append(item['id'])
            self._process_item_state(item, persons, frame)

        for person in persons:
            current_frame_ids.append(person['id'])

        return self._handle_disappearances(current_frame_ids, frame)

    def _parse_results(self, results, valid_classes):
        """Separate YOLO results into persons and target items."""
        persons = []
        items = []
        
        if results.boxes.id is None:
            return persons, items

        for box in results.boxes:
            if box.id is None: continue
            
            track_id = int(box.id[0])
            cls_id = int(box.cls[0])
            class_name = results.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            data = {'id': track_id, 'center': center, 'bbox': (x1, y1, x2, y2), 'class_name': class_name}
            
            if class_name == 'person':
                persons.append(data)
            elif class_name in valid_classes:
                items.append(data)
                
        return persons, items

    def _process_item_state(self, item, persons, frame):
        """Update the state of a detected item (ownership, movement, proximity)."""
        tid = item['id']
        center = item['center']
        ibox = item['bbox']
        
        # 1. Initialize if new item (Birth-time ownership assignment)
        if tid not in self.tracked_items:
            owner_id = self._find_closest_person(item, persons)
            self.tracked_items[tid] = {
                'class_name': item['class_name'],
                'owner_id': owner_id,
                'last_pos': center,
                'stay_count': 0,
                'is_stationary': False,
                'near_history': 0,
                'missing_count': 0,
                'last_person_id': None
            }
            status = f"Person {owner_id}" if owner_id else "NO owner"
            print(f"[INFO]     ID {tid}({item['class_name']}) appeared. Owner: {status}")

        item_data = self.tracked_items[tid]
        item_data['missing_count'] = 0
        
        # 2. Update Stationary Status
        dist_moved = self._calculate_distance(center, item_data['last_pos'])
        if dist_moved < 50:
            item_data['stay_count'] += 1
        else:
            item_data['stay_count'] = 0
        
        if not item_data['is_stationary'] and item_data['stay_count'] >= self.stationary_threshold:
            item_data['is_stationary'] = True
            self._save_baseline_crop(tid, item_data, ibox, frame)
            print(f"[INFO]     ID {tid}({item_data['class_name']}) is now stationary.")
        
        item_data['last_pos'] = center

        # 3. Update Proximity History (Who was near the object last?)
        closest_person_id = self._find_closest_person(item, persons)
        if closest_person_id is not None:
            item_data['near_history'] = 120 # Maintain history for 120 frames (~4-5 sec)
            item_data['last_person_id'] = closest_person_id
        else:
            item_data['near_history'] = max(0, item_data['near_history'] - 1)

    def _find_closest_person(self, item, persons):
        """Find the ID of the person closest to or touching the item."""
        closest_id = None
        min_dist = float('inf')
        for p in persons:
            dist = self._calculate_distance(item['center'], p['center'])
            if self._is_touching(item['bbox'], p['bbox']) or dist < self.proximity_pixels:
                if dist < min_dist:
                    min_dist = dist
                    closest_id = p['id']
        return closest_id

    def _handle_disappearances(self, current_frame_ids, frame):
        """Detect theft when a stationary object disappears with a non-owner contact."""
        detected_theft = False
        for tid in list(self.tracked_items.keys()):
            if tid not in current_frame_ids:
                item_data = self.tracked_items[tid]
                item_data['missing_count'] += 1
                
                if item_data['missing_count'] >= self.missing_threshold:
                    if item_data['is_stationary'] and item_data['near_history'] > 0:
                        last_p = item_data.get('last_person_id')
                        owner_p = item_data.get('owner_id')
                        
                        # Trigger alert if the last person near the item was NOT the owner
                        if last_p is not None and last_p != owner_p:
                            self._trigger_alert(tid, frame)
                            detected_theft = True
                        else:
                            print(f"[INFO]     ID {tid} was retrieved by its owner (Person {owner_p}).")
                    
                    del self.tracked_items[tid]
        return detected_theft

    def _save_baseline_crop(self, track_id, item_data, bbox, frame):
        """Save a cropped image of the object when it becomes stationary."""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        y1, y2, x1, x2 = max(0, y1), min(h, y2), max(0, x1), min(w, x2)
        item_data['baseline_crop'] = frame[y1:y2, x1:x2].copy()

    def _trigger_alert(self, track_id, frame):
        """Capture evidence and record a suspected theft event."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        item_data = self.tracked_items[track_id]
        
        moment_file = os.path.join(self.output_dir, f"theft_moment_ID{track_id}_{timestamp}.jpg")
        cv2.imwrite(moment_file, frame)
        
        baseline_file = os.path.join(self.output_dir, f"stolen_item_baseline_ID{track_id}_{timestamp}.jpg")
        if 'baseline_crop' in item_data:
            cv2.imwrite(baseline_file, item_data['baseline_crop'])
            print(f"[SAVE]     Baseline image saved: {baseline_file}")

        print(f"[ALERT]    Theft suspected! Object ID: {track_id}")
        self.alerts.append({
            'id': track_id, 
            'time': timestamp, 
            'baseline_file': baseline_file,
            'moment_file': moment_file
        })
