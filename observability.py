from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from models import StageRecord

_event_callback: Callable[[str, dict[str, Any]], None] | None = None


def set_event_callback(callback: Callable[[str, dict[str, Any]], None] | None) -> None:
    global _event_callback
    _event_callback = callback


def emit_event(event: str, data: dict[str, Any] | None = None) -> bool:
    if _event_callback is None:
        return False
    _event_callback(event, data or {})
    return True


class StageLogWriter:
    def __init__(self, logs_dir: Path, run_id: str) -> None:
        self.run_dir = logs_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.stage_records_path = self.run_dir / "stage_records.jsonl"

    def write_stage_record(self, record: StageRecord) -> None:
        value = record.model_dump(mode="json")
        self._append_jsonl(self.stage_records_path, value)
        emit_event("stage_result", value)

    def append_validation(self, value: Any) -> None:
        self._append_jsonl(self.run_dir / "validation.jsonl", _jsonable(value))

    def append_rejection(self, value: Any) -> None:
        self._append_jsonl(self.run_dir / "rejections.jsonl", _jsonable(value))

    @staticmethod
    def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, sort_keys=True) + "\n")


def _jsonable(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    return {"value": value}
