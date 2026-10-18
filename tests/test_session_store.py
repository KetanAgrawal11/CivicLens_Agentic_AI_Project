"""Tests for saving, listing, restoring and deleting research sessions."""

import pytest

import config
import memory
import session_store


@pytest.fixture(autouse=True)
def temp_sessions_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SESSIONS_DIR", tmp_path)


def _state(question, verdict="Supported"):
    return {"question": question, "fact_check_verdict": verdict, "route": "web", "final_report": f"Report: {question}"}


def test_save_and_list_sessions():
    session_store.save_run("s1", _state("First question"))
    session_store.save_run("s1", _state("Follow-up"))
    session_store.save_run("s2", _state("Other topic"))

    sessions = session_store.list_sessions()
    assert [s["id"] for s in sessions] == ["s2", "s1"]  # most recent first
    assert sessions[1]["title"] == "First question"
    assert sessions[1]["runs"] == 2


def test_restore_memory_rebuilds_conversation():
    session_store.save_run("s_restore", _state("What did the court rule?"))
    memory.reset_session("s_restore")
    session_store.restore_memory("s_restore")

    history = memory.get_session_memory("s_restore").get_history()
    assert [h["role"] for h in history] == ["user", "assistant"]
    assert history[1]["content"] == "Report: What did the court rule?"


def test_delete_session_and_unsafe_ids():
    session_store.save_run("../evil id", _state("q"))
    assert all(p.parent == config.SESSIONS_DIR for p in config.SESSIONS_DIR.iterdir())
    session_store.delete_session("../evil id")
    assert session_store.list_sessions() == []
