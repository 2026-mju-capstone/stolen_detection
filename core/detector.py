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
        
        self.tracked_items = {}  # {추적_ID: 아이템_데이터}
        self.person_trajectories = {}  # {사람_ID: [중심점_리스트]}
        self.alerts = []
        self.logger = TheftLogger()
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def _calculate_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _is_touching(self, box1, box2):
        # 바운딩 박스 겹침 확인
        return not (box1[2] < box2[0] or box1[0] > box2[2] or 
                    box1[3] < box2[1] or box1[1] > box2[3])

    def update(self, results, frame, valid_classes):
        """도난 탐지를 위한 메인 업데이트 루프."""
        current_frame_ids = []
        persons, items_in_frame = self._parse_results(results, valid_classes)
        
        # 현재 프레임에 존재하는 ID 추적
        for item in items_in_frame:
            current_frame_ids.append(item['id'])
            self._process_item_state(item, persons, frame)

        for person in persons:
            current_frame_ids.append(person['id'])

        return self._handle_disappearances(current_frame_ids, frame)

    def _parse_results(self, results, valid_classes):
        """YOLO 결과를 사람과 대상 물체로 분리."""
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
        """탐지된 물체의 상태 업데이트 (소유권, 이동, 근접도)."""
        tid = item['id']
        center = item['center']
        ibox = item['bbox']
        
        # 1. 새 물체 초기화 (출현 시 소유권 할당)
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
        
        # 0. 도주 분석을 위한 사람 궤적 추적
        for p in persons:
            pid = p['id']
            if pid not in self.person_trajectories:
                self.person_trajectories[pid] = []
            self.person_trajectories[pid].append(p['center'])
            if len(self.person_trajectories[pid]) > 60: # 2초간 유지
                self.person_trajectories[pid].pop(0)

        # 2. 정지 상태 업데이트
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

        # 3. 근접 이력 업데이트 (마지막에 누가 물체 근처에 있었는가?)
        closest_person_id, is_touching = self._find_closest_person(item, persons)
        if closest_person_id is not None:
            # 실제로 만졌을 경우 이력을 더 오래 유지
            history_frames = 120 if is_touching else 60
            item_data['near_history'] = history_frames
            item_data['last_person_id'] = closest_person_id
        else:
            item_data['near_history'] = max(0, item_data['near_history'] - 1)

    def _find_closest_person(self, item, persons):
        """물체에 가장 가깝거나 접촉 중인 사람의 ID를 찾음."""
        closest_id = None
        min_dist = float('inf')
        is_touching = False
        
        for p in persons:
            if self._is_touching(item['bbox'], p['bbox']):
                is_touching = True
                return p['id'], True # 접촉 중이면 즉시 반환
            
            dist = self._calculate_distance(item['center'], p['center'])
            if dist < self.proximity_pixels:
                if dist < min_dist:
                    min_dist = dist
                    closest_id = p['id']
                    
        return closest_id, is_touching

    def _handle_disappearances(self, current_frame_ids, frame):
        """가중치 기반 신뢰도 점수와 다중 프레임 검증을 사용하여 도난 탐지."""
        detected_theft = False
        for tid in list(self.tracked_items.keys()):
            if tid not in current_frame_ids:
                item_data = self.tracked_items[tid]
                item_data['missing_count'] += 1
                
                # 물체가 처음 사라진 프레임에서 잠재적 도난 순간 저장
                if item_data['missing_count'] == 1:
                    item_data['potential_moment_frame'] = frame.copy()
                
                # 다중 프레임 검증
                if item_data['missing_count'] >= config.VERIFICATION_FRAMES:
                    if item_data['is_stationary'] and item_data['near_history'] > 0:
                        score = self._calculate_theft_score(tid, item_data)
                        
                        if score >= config.THEFT_CONFIDENCE_THRESHOLD:
                            # 저장된 잠재적 순간 프레임 사용
                            theft_frame = item_data.get('potential_moment_frame', frame)
                            self._trigger_alert(tid, theft_frame, score)
                            detected_theft = True
                        else:
                            print(f"[INFO]     ID {tid} missing but confidence too low ({score:.2f}).")
                    
                    del self.tracked_items[tid]
        return detected_theft

    def _calculate_theft_score(self, tid, item_data):
        """여러 요인을 기반으로 도난 신뢰도 계산."""
        score = 0.0
        last_p = item_data.get('last_person_id')
        owner_p = item_data.get('owner_id')

        # 중요: 소유자가 직접 가져간 것이라면 도난이 아님
        if last_p is not None and last_p == owner_p:
            return 0.0

        # 1. 접촉 가중치 (의심자가 물체 근처에 얼마나 머물렀는가?)
        if last_p is not None and last_p != owner_p:
            score += config.CONTACT_WEIGHT
            
        # 2. 소유자 명확성 (절도범이 확실히 소유자가 아닌가?)
        if owner_p is not None and last_p != owner_p:
            score += config.OWNER_CLARITY_WEIGHT
            
        # 3. 정지 정도 (물체가 확실히 고정되어 있었는가?)
        if item_data['stay_count'] > 100: # 임계값의 두 배 이상 정지 상태 유지
            score += config.STATIONARY_WEIGHT
            
        # 4. 도주 탐지
        if config.FLEEING_DETECTION and last_p in self.person_trajectories:
            speed = self._calculate_speed(last_p)
            if speed > config.FLEEING_SPEED_THRESHOLD:
                score += 0.2  # 도주 시 가산점
                print(f"[DEBUG]    Suspect {last_p} is fleeing (Speed: {speed:.1f})")

        return min(1.0, score)

    def _calculate_speed(self, person_id):
        """사람의 최근 평균 속도 계산."""
        traj = self.person_trajectories.get(person_id, [])
        if len(traj) < 2: return 0.0
        
        # 최근 몇 프레임 간의 거리 계산
        total_dist = 0
        samples = min(10, len(traj))
        for i in range(len(traj) - samples, len(traj) - 1):
            p1, p2 = traj[i], traj[i+1]
            total_dist += math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        return total_dist / (samples - 1)

    def _save_baseline_crop(self, track_id, item_data, bbox, frame):
        """물체가 정지 상태가 되면 해당 물체의 크롭 이미지를 저장."""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        y1, y2, x1, x2 = max(0, y1), min(h, y2), max(0, x1), min(w, x2)
        item_data['baseline_crop'] = frame[y1:y2, x1:x2].copy()

    def _trigger_alert(self, track_id, frame, score):
        """증거 캡처 및 도난 의심 이벤트 기록."""
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
