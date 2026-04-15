import cv2
import math
import time
import os
import config
from core.logger import TheftLogger

class TheftDetector:
    def __init__(self, stationary_threshold_frames=50, proximity_pixels=100, missing_threshold_frames=10, output_dir="output"):
        self.stationary_threshold = stationary_threshold_frames
        self.proximity_pixels = proximity_pixels
        self.missing_threshold = missing_threshold_frames
        self.output_dir = output_dir
        
        self.tracked_items = {}  # {track_id: item_data}
        self.person_trajectories = {}  # {person_id: [centers]}
        self.alerts = []
        self.logger = TheftLogger()
        
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
            owner_id, _ = self._find_closest_person(item, persons)
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
        
        # 0. Track Person Trajectories for fleeing analysis
        for p in persons:
            pid = p['id']
            if pid not in self.person_trajectories:
                self.person_trajectories[pid] = []
            self.person_trajectories[pid].append(p['center'])
            if len(self.person_trajectories[pid]) > 60: # Keep 2 seconds
                self.person_trajectories[pid].pop(0)

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
        closest_person_id, is_touching = self._find_closest_person(item, persons)
        if closest_person_id is not None:
            # Maintain history for longer if they actually touched it
            history_frames = 120 if is_touching else 60
            item_data['near_history'] = history_frames
            item_data['last_person_id'] = closest_person_id
        else:
            item_data['near_history'] = max(0, item_data['near_history'] - 1)

    def _find_closest_person(self, item, persons):
        """Find the ID of the person closest to or touching the item."""
        closest_id = None
        min_dist = float('inf')
        is_touching = False
        
        for p in persons:
            if self._is_touching(item['bbox'], p['bbox']):
                is_touching = True
                return p['id'], True # Return immediately if touching
            
            dist = self._calculate_distance(item['center'], p['center'])
            if dist < self.proximity_pixels:
                if dist < min_dist:
                    min_dist = dist
                    closest_id = p['id']
                    
        return closest_id, is_touching

    def _handle_disappearances(self, current_frame_ids, frame):
        """Detect theft using a weighted confidence score and multi-frame verification."""
        detected_theft = False
        for tid in list(self.tracked_items.keys()):
            if tid not in current_frame_ids:
                item_data = self.tracked_items[tid]
                item_data['missing_count'] += 1
                
                # Store the potential theft moment frame at the FIRST disappearance frame
                if item_data['missing_count'] == 1:
                    item_data['potential_moment_frame'] = frame.copy()
                
                # Multi-frame verification
                if item_data['missing_count'] >= config.VERIFICATION_FRAMES:
                    if item_data['is_stationary'] and item_data['near_history'] > 0:
                        score = self._calculate_theft_score(tid, item_data)
                        
                        if score >= config.THEFT_CONFIDENCE_THRESHOLD:
                            # Use the stored potential moment if available
                            theft_frame = item_data.get('potential_moment_frame', frame)
                            self._trigger_alert(tid, theft_frame, score)
                            detected_theft = True
                        else:
                            print(f"[INFO]     ID {tid} missing but confidence too low ({score:.2f}).")
                    
                    del self.tracked_items[tid]
        return detected_theft

    def _calculate_theft_score(self, tid, item_data):
        """Calculate theft confidence based on multiple factors."""
        score = 0.0
        last_p = item_data.get('last_person_id')
        owner_p = item_data.get('owner_id')

        # Critical: If the owner is the one who took it, it's not theft
        if last_p is not None and last_p == owner_p:
            return 0.0

        # 1. Contact Weight (How long was the suspect near the item?)
        if last_p is not None and last_p != owner_p:
            score += config.CONTACT_WEIGHT
            
        # 2. Owner Clarity (Is the thief clearly NOT the owner?)
        if owner_p is not None and last_p != owner_p:
            score += config.OWNER_CLARITY_WEIGHT
            
        # 3. Stationary Degree (Was it firmly planted?)
        if item_data['stay_count'] > 100: # Stayed longer than double threshold
            score += config.STATIONARY_WEIGHT
            
        # 4. Fleeing Detection
        if config.FLEEING_DETECTION and last_p in self.person_trajectories:
            speed = self._calculate_speed(last_p)
            if speed > config.FLEEING_SPEED_THRESHOLD:
                score += 0.2  # Bonus for fleeing
                print(f"[DEBUG]    Suspect {last_p} is fleeing (Speed: {speed:.1f})")

        return min(1.0, score)

    def _calculate_speed(self, person_id):
        """Calculate recent average speed of a person."""
        traj = self.person_trajectories.get(person_id, [])
        if len(traj) < 2: return 0.0
        
        # Calculate distance between last few frames
        total_dist = 0
        samples = min(10, len(traj))
        for i in range(len(traj) - samples, len(traj) - 1):
            p1, p2 = traj[i], traj[i+1]
            total_dist += math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        return total_dist / (samples - 1)

    def _save_baseline_crop(self, track_id, item_data, bbox, frame):
        """Save a cropped image of the object when it becomes stationary."""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        y1, y2, x1, x2 = max(0, y1), min(h, y2), max(0, x1), min(w, x2)
        item_data['baseline_crop'] = frame[y1:y2, x1:x2].copy()

    def _trigger_alert(self, track_id, frame, score):
        """Capture evidence and record a suspected theft event."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        item_data = self.tracked_items[track_id]
        
        moment_file = os.path.join(self.output_dir, f"theft_moment_ID{track_id}_{timestamp}.jpg")
        cv2.imwrite(moment_file, frame)
        
        baseline_file = os.path.join(self.output_dir, f"stolen_item_baseline_ID{track_id}_{timestamp}.jpg")
        if 'baseline_crop' in item_data:
            cv2.imwrite(baseline_file, item_data['baseline_crop'])
            print(f"[SAVE]     Baseline image saved: {baseline_file}")

        print(f"[ALERT]    Theft suspected! ID: {track_id} (Confidence: {score:.2f})")
        alert_data = {
            'id': track_id, 
            'time': timestamp, 
            'confidence': score,
            'baseline_file': baseline_file,
            'moment_file': moment_file
        }
        self.alerts.append(alert_data)
        self.logger.log_event('theft_suspected', alert_data)
