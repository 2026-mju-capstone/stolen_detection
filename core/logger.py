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
        """Load existing logs if the file exists."""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    self.events = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.events = []

    def log_event(self, event_type, data):
        """
        Log a theft event.
        event_type: 'suspected', 'confirmed', etc.
        data: Dict containing event details
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
        """Save history to JSON file."""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.events, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"[ERROR]    Failed to save log: {e}")
