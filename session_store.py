"""
Session Store — saves completed research runs to disk so past sessions survive restarts.
One JSON file per session in config.SESSIONS_DIR:
{"id", "created_at", "updated_at", "runs": [{"question", "timestamp", "verdict", "route", "state"}]}
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config
import memory

_SAFE_ID = re.compile(r"[^A-Za-z0-9_-]")


def _path(session_id: str) -> Path:
    return config.SESSIONS_DIR / f"{_SAFE_ID.sub('_', session_id)}.json"


def _read(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save_run(session_id: str, state: dict[str, Any]) -> None:
    """Append one completed research run to its session file."""
    now = datetime.now(timezone.utc).isoformat()
    path = _path(session_id)
    session = _read(path) or {"id": session_id, "created_at": now, "runs": []}
    session["runs"].append({
        "question": state.get("question", ""),
        "timestamp": now,
        "verdict": state.get("fact_check_verdict", ""),
        "route": state.get("route", ""),
        "state": state,
    })
    session["updated_at"] = now
    config.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(session, ensure_ascii=False, default=str), encoding="utf-8")


def load_session(session_id: str) -> dict[str, Any] | None:
    return _read(_path(session_id))


def list_sessions() -> list[dict[str, Any]]:
    """Summaries of saved sessions, most recently updated first."""
    summaries = []
    for path in config.SESSIONS_DIR.glob("*.json"):
        session = _read(path)
        if not session or not session.get("runs"):
            continue
        summaries.append({
            "id": session["id"],
            "title": session["runs"][0].get("question", "Untitled"),
            "runs": len(session["runs"]),
            "updated_at": session.get("updated_at", ""),
        })
    return sorted(summaries, key=lambda s: s["updated_at"], reverse=True)


def delete_session(session_id: str) -> None:
    _path(session_id).unlink(missing_ok=True)
    memory.reset_session(session_id)


def restore_memory(session_id: str) -> None:
    """Rebuild the agent's conversation memory from a saved session so follow-ups keep their context."""
    session = load_session(session_id)
    sess_mem = memory.get_session_memory(session_id)
    if not session or sess_mem.get_history():
        return
    for run in session["runs"]:
        state = run.get("state", {})
        common = {
            "route": run.get("route", "web"),
            "sources_count": len(state.get("unique_source_urls", [])),
            "fact_check_verdict": run.get("verdict", ""),
        }
        sess_mem.add_turn(role="user", content=run.get("question", ""), **common)
        sess_mem.add_turn(role="assistant", content=state.get("final_report", ""), **common)
