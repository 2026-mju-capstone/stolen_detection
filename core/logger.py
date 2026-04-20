import json
import os
from datetime import datetime

class TheftLogger:
    def __init__(self, log_file="output/theft_log.json"):
        self.log_file = log_file
        self.events = []
        self._ensure_output_dir()
        self._load_existing()

    def _ensure_output_dir(self):
        output_dir = os.path.dirname(self.log_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def _load_existing(self):
        """파일이 존재할 경우 기존 로그를 로드함."""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    self.events = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.events = []

    def log_event(self, event_type, data):
        """
        도난 이벤트를 기록함.
        event_type: 'suspected', 'confirmed' 등
        data: 이벤트 상세 내용을 포함하는 딕셔너리
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': data
        }
        self.events.append(entry)
        self._save()
        print(f"[LOGGER]   Event recorded: {event_type}")

    def _save(self):
        """이력을 JSON 파일로 저장함."""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.events, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"[ERROR]    Failed to save log: {e}")
