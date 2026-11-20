"""
Assignment 11 - Audit Log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}       # request_id -> perf_counter at input
        self._pending: dict[str, dict] = {}     # request_id -> entry awaiting its output

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Open an entry: who asked what, and when (latency starts here).

        The input is stored redacted: an attacker who pastes a secret or a
        customer's phone number into a prompt must not turn the audit log into
        a second copy of that data.
        """
        key = request_id or user_id
        self._open[key] = time.perf_counter()
        self._pending[key] = {
            "request_id": key,
            "user_id": user_id,
            "started_at": utc_now_iso(),
            "input": _redact(text),
        }

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Close the entry: the reply, which layer decided, and the latency."""
        key = request_id or user_id
        started = self._open.pop(key, None)
        entry = self._pending.pop(key, None) or {
            "request_id": key, "user_id": user_id, "started_at": None, "input": None,
        }
        entry.update({
            "finished_at": utc_now_iso(),
            "output": _redact(text),
            "blocked": blocked,
            "layer": layer,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1) if started else None,
        })
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(path)


def _redact(text: str | None) -> str:
    from guardrails.output_guardrails import content_filter

    return content_filter(text or "")["redacted"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
