"""
Conversation & Agent Memory Module — Isolated session memory with sliding-window
token budgets, older turn summarization, and active execution context tracking.
Keeps conversation history distinct from the vector knowledge base.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import config


@dataclass
class ConversationTurn:
    role: str  # "user" | "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    route: str = "web"
    sources_count: int = 0
    fact_check_verdict: str = ""


class SessionMemory:
    """Manages isolated conversation history for a specific conversation session."""

    def __init__(self, conversation_id: str | None = None):
        self.conversation_id: str = conversation_id or str(uuid.uuid4())
        self.turns: list[ConversationTurn] = []
        self.summary: str = ""
        self.created_at: str = datetime.now(timezone.utc).isoformat()

    def add_turn(
        self,
        role: str,
        content: str,
        route: str = "web",
        sources_count: int = 0,
        fact_check_verdict: str = "",
    ) -> None:
        """Add a user or assistant turn to history."""
        self.turns.append(
            ConversationTurn(
                role=role,
                content=content,
                route=route,
                sources_count=sources_count,
                fact_check_verdict=fact_check_verdict,
            )
        )

    def get_history(self) -> list[dict[str, Any]]:
        """Return list of dicts representing conversation turns for UI inspection."""
        return [
            {
                "role": t.role,
                "content": t.content,
                "timestamp": t.timestamp,
                "route": t.route,
                "sources_count": t.sources_count,
                "fact_check_verdict": t.fact_check_verdict,
            }
            for t in self.turns
        ]

    def get_turns(self, max_turns: int = config.MAX_RECENT_TURNS) -> list[ConversationTurn]:
        """Return the most recent turns within max_turns limit."""
        return self.turns[-max_turns:]

    def get_formatted_context(self, max_turns: int = config.MAX_RECENT_TURNS) -> str:
        """
        Produce a compact, token-bounded context string for LLM prompts.
        Includes running summary of older turns if available.
        """
        parts = []
        if self.summary:
            parts.append(f"[Earlier Conversation Summary]: {self.summary}")

        recent = self.turns[-max_turns:]
        for t in recent:
            role_label = "User" if t.role == "user" else "Assistant"
            parts.append(f"{role_label}: {t.content[:800]}")

        return "\n\n".join(parts)

    def summarize_older_turns(self, llm_callable=None) -> None:
        """
        If turns exceed max_recent_turns, distill older turns into a concise summary
        to respect the token budget without losing critical context.
        """
        if len(self.turns) <= config.MAX_RECENT_TURNS:
            return

        older_turns = self.turns[:-config.MAX_RECENT_TURNS]
        text_to_summarize = "\n".join(f"{t.role}: {t.content[:300]}" for t in older_turns)

        if llm_callable is not None:
            try:
                system_prompt = "You are a concise conversation summarizer. Summarize key topics, user queries, and previous conclusions in 2-3 sentences."
                user_prompt = f"Existing summary: {self.summary}\nOlder turns to incorporate:\n{text_to_summarize}"
                self.summary = llm_callable(system_prompt, user_prompt).strip()
            except Exception:
                self.summary = f"User previously researched {len(older_turns)} topics."
        else:
            self.summary = f"Earlier discussed topics: {len(older_turns)} previous interactions."

    def clear(self) -> None:
        """Reset conversation turns and summary."""
        self.turns = []
        self.summary = ""
        self.conversation_id = str(uuid.uuid4())


# Global session registry (isolated per conversation_id)
_sessions: dict[str, SessionMemory] = {}


def get_session_memory(conversation_id: str | None = None) -> SessionMemory:
    """Retrieve or create a SessionMemory instance for a given conversation_id."""
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    if conversation_id not in _sessions:
        _sessions[conversation_id] = SessionMemory(conversation_id)
    return _sessions[conversation_id]


def reset_session(conversation_id: str) -> None:
    """Clear memory for a conversation."""
    if conversation_id in _sessions:
        _sessions[conversation_id].clear()
