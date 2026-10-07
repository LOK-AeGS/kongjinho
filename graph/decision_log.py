"""Supervisor 결정을 State 밖 JSONL로 기록한다."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class DecisionLogger:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else None

    def log(
        self,
        trace_id: str,
        step: int,
        decision: list[str],
        reason: str,
        ts: str | None = None,
    ) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "trace_id": trace_id,
            "step": step,
            "node": "supervisor",
            "decision": decision,
            "reason": reason,
            "ts": ts or datetime.now(timezone.utc).isoformat(),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
