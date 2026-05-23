"""Progress tracker with on-disk checkpoint to survive interruptions.

Each completed sample is appended to a JSONL file immediately. On resume,
the runner reads which sample_ids are already done and skips them.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import json
import threading


class ProgressTracker:
    """Simple JSONL-based checkpoint.
    
    Usage:
        tracker = ProgressTracker(output_path="results/chrcp/gsm8k.jsonl")
        for sample in samples:
            if tracker.is_done(sample.sample_id):
                continue
            result = run(sample)
            tracker.write(result)
    """
    
    def __init__(self, output_path: str | Path):
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._done_ids: set[str] = set()
        self._load_existing()
    
    def _load_existing(self) -> None:
        """Read existing file (if any) to populate _done_ids."""
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = row.get("sample_id")
                if sid:
                    self._done_ids.add(sid)
    
    def is_done(self, sample_id: str) -> bool:
        return sample_id in self._done_ids
    
    @property
    def done_count(self) -> int:
        return len(self._done_ids)
    
    def write(self, record: dict) -> None:
        """Append a result record (must contain 'sample_id' key)."""
        sid = record.get("sample_id")
        if not sid:
            raise ValueError("record must contain 'sample_id'")
        
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._done_ids.add(sid)
    
    def all_records(self) -> list[dict]:
        """Read all records from disk."""
        records = []
        if not self.path.exists():
            return records
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records