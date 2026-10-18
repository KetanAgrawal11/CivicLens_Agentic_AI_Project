"""
Query Router Module — Intelligent classification into 4 distinct routing modes:
1. `direct`: Self-contained or conversational queries answered without retrieval.
2. `rag`: Queries targeting indexed documents in Qdrant.
3. `web`: External queries, live news, and general fact-checking.
4. `rag_and_web`: Comparative or multi-faceted queries combining uploaded docs with external web sources.
Uses fast deterministic heuristics first, falling back to structured LLM classification when ambiguous.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

import config

RouteType = Literal["direct", "rag", "web", "rag_and_web"]


def route_query(
    question: str,
    has_indexed_documents: bool = False,
    llm_callable=None,
    kb_top_score: float | None = None,
) -> dict[str, Any]:
    """
    Classify the query into one of: 'direct', 'rag', 'web', 'rag_and_web'.
    Returns a dict with route, reason, confidence, and bypass_planning flag.
    """
    q_clean = question.strip().lower()

    # 1. Fast deterministic check: Direct / Conversational / Math / Basic definitions
    conversational_patterns = [
        r'^(hi|hello|hey|good\s+(morning|afternoon|evening)|howdy)\b',
        r'^(what can you do|who are you|help|how do you work)\??$',
        r'^(what is|define)\s+(photosynthesis|a noun|gravity|the pythagorean theorem)\??$',
        r'^\d+\s*[\+\-\*\/\^]\s*\d+',
    ]
    for pattern in conversational_patterns:
        if re.search(pattern, q_clean):
            return {
                "route": "direct",
                "reason": "Self-contained conversational or definitional query requiring no external or vector retrieval.",
                "confidence": 0.95,
                "bypass_planning": True,
            }

    # 2. Check for explicit keywords indicating comparison between document and external web
    rag_and_web_indicators = [
        "compare the uploaded",
        "compare with web",
        "compare with news",
        "verify against current news",
        "check if the uploaded document matches",
        "cross-verify the document with",
        "in our documents vs current",
    ]
    if has_indexed_documents and any(ind in q_clean for ind in rag_and_web_indicators):
        return {
            "route": "rag_and_web",
            "reason": "Query explicitly requests comparing uploaded documents against external web sources.",
            "confidence": 0.90,
            "bypass_planning": False,
        }

    # 3. Check for explicit Knowledge Base / Document references
    kb_indicators = [
        "in the uploaded document",
        "in our document",
        "in the pdf",
        "according to the document",
        "from the uploaded file",
        "in our knowledge base",
        "in the file",
        "what does the uploaded",
        "summarize the uploaded",
    ]
    if has_indexed_documents and any(ind in q_clean for ind in kb_indicators):
        return {
            "route": "rag",
            "reason": "Query specifically requests retrieval from uploaded knowledge base documents.",
            "confidence": 0.90,
            "bypass_planning": False,
        }

    # 4. If no documents are indexed, queries requiring search must route to 'web'
    if not has_indexed_documents:
        # Check if it's a simple direct question
        if len(q_clean.split()) <= 4 and ("what is" in q_clean or "meaning of" in q_clean):
            return {
                "route": "direct",
                "reason": "Brief conceptual question answered directly.",
                "confidence": 0.85,
                "bypass_planning": True,
            }
        return {
            "route": "web",
            "reason": "No uploaded documents indexed; routing to live external web research.",
            "confidence": 0.90,
            "bypass_planning": False,
        }

    # 4b. Indexed documents semantically match the query: use them, cross-checked against the web
    if kb_top_score is not None and kb_top_score >= config.MIN_RAG_RELEVANCE_SCORE:
        return {
            "route": "rag_and_web",
            "reason": f"Indexed documents are relevant to this query (top match {kb_top_score:.2f}); combining knowledge base evidence with live web verification.",
            "confidence": 0.90,
            "bypass_planning": False,
        }

    # 5. External research indicators that don't target uploaded documents
    web_indicators = [
        "controversy", "protest", "investigate", "fact-check", "latest news", "recent",
        "election", "policy", "court", "ruling", "who won", "scandal", "market", "trend",
        "verify whether", "is it true that", "who is", "what happened"
    ]
    if any(ind in q_clean for ind in web_indicators):
        return {
            "route": "web",
            "reason": "External research or fact-checking inquiry routed to live web sources.",
            "confidence": 0.90,
            "bypass_planning": False,
        }


    # 5. Use lightweight LLM classification if ambiguous and LLM is supplied
    if llm_callable is not None and config.GROQ_API_KEY:
        try:
            system_prompt = """You are a query routing classifier for an agentic research system.
Classify the user's research query into exactly ONE of four routes:
- "direct": Self-contained, greeting, simple math, or general knowledge that needs NO search.
- "rag": Questions specifically about uploaded internal documents/PDFs.
- "web": Questions about external news, current events, or general fact-checking not referencing internal files.
- "rag_and_web": Questions comparing uploaded documents with current external news, or requiring both internal knowledge and external verification.

Respond strictly with a JSON object:
{"route": "direct"|"rag"|"web"|"rag_and_web", "reason": "brief explanation", "confidence": 0.0 to 1.0}"""

            user_prompt = f"User query: '{question}'\nIndexed documents present: {has_indexed_documents}"
            raw = llm_callable(system_prompt, user_prompt)

            # Parse JSON
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                route = parsed.get("route", "web")
                if route in ("direct", "rag", "web", "rag_and_web"):
                    return {
                        "route": route,
                        "reason": parsed.get("reason", "Classified via structured routing model."),
                        "confidence": float(parsed.get("confidence", 0.85)),
                        "bypass_planning": route == "direct",
                    }
        except Exception:
            pass

    # 6. Default fallback
    return {
        "route": "web",
        "reason": "General research inquiry routed to web research.",
        "confidence": 0.80,
        "bypass_planning": False,
    }
