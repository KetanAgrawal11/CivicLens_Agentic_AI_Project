"""
CivicLens — Research and Fact-Checking Platform.
Streamlit interface for the LangGraph research agent: live multi-step execution, evidence
triangulation, Qdrant knowledge base management, tool testing and citation-backed reports.
"""

from __future__ import annotations

import html
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

import agent
import config
import knowledge_base
import memory
import report_export
import session_store
import tools

# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title=f"{config.APP_TITLE} — Research & Fact-Checking",
    page_icon=":material/verified_user:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design System (fonts and base colours come from .streamlit/config.toml) ──
st.markdown("""
<style>
    code, pre, .mono-font { font-family: 'JetBrains Mono', monospace; }

    /* Top padding must clear Streamlit's fixed toolbar (~3.75rem tall) */
    .block-container { padding-top: 4.25rem; padding-bottom: 3.5rem; max-width: 1480px; }

    /* Command header */
    .cl-navbar {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.2rem 1.6rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 8px 24px -6px rgba(15, 23, 42, 0.35);
        display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;
    }
    .cl-brand { display: flex; align-items: center; gap: 0.85rem; }
    .cl-logo {
        width: 42px; height: 42px; border-radius: 10px; flex-shrink: 0;
        background: linear-gradient(135deg, #3b82f6 0%, #2d5a87 100%); color: #ffffff;
        display: grid; place-items: center;
    }
    .cl-title { font-size: 1.55rem; font-weight: 800; color: #f8fafc; line-height: 1.15; letter-spacing: -0.02em; }
    .cl-subtitle { color: #94a3b8; font-size: 0.88rem; font-weight: 500; margin-top: 0.15rem; }

    .cl-pills { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
    .cl-pill {
        display: inline-flex; align-items: center; gap: 0.45rem;
        padding: 0.32rem 0.8rem; border-radius: 9999px; font-size: 0.77rem; font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.1); background: rgba(255, 255, 255, 0.04); color: #e2e8f0;
        white-space: nowrap;
    }
    .cl-pill svg { opacity: 0.85; }
    .cl-pill.ok { background: rgba(16, 185, 129, 0.12); color: #34d399; border-color: rgba(16, 185, 129, 0.28); }
    .cl-pill.warn { background: rgba(245, 158, 11, 0.12); color: #fbbf24; border-color: rgba(245, 158, 11, 0.28); }
    .cl-pill.err { background: rgba(239, 68, 68, 0.12); color: #f87171; border-color: rgba(239, 68, 68, 0.28); }
    .cl-pill.info { background: rgba(59, 130, 246, 0.12); color: #7dd3fc; border-color: rgba(59, 130, 246, 0.28); }
    .cl-dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; box-shadow: 0 0 8px currentColor; }

    /* Section headings */
    .cl-heading { display: flex; align-items: center; gap: 0.55rem; margin: 0.25rem 0 0.75rem; }
    .cl-step-badge {
        width: 24px; height: 24px; border-radius: 6px; background: #2d5a87; color: #ffffff;
        font-size: 0.78rem; font-weight: 700; display: grid; place-items: center;
    }
    .cl-heading-text { font-size: 1.05rem; font-weight: 700; color: #0f172a; }

    /* Verdict banner */
    .cl-verdict {
        border-radius: 14px; padding: 1.25rem 1.6rem; margin: 1rem 0 1.1rem;
        display: flex; align-items: center; justify-content: space-between; gap: 1.5rem; flex-wrap: wrap;
        box-shadow: 0 6px 20px -6px rgba(15, 23, 42, 0.18);
        border: 1px solid var(--border); border-left: 6px solid var(--accent); background: var(--bg);
    }
    .cl-verdict-title {
        display: flex; align-items: center; gap: 0.6rem;
        font-size: 1.2rem; font-weight: 800; letter-spacing: 0.01em; color: var(--accent); text-transform: uppercase;
    }
    .cl-verdict-desc { font-size: 0.92rem; color: #334155; margin-top: 0.3rem; max-width: 780px; }
    .cl-coverage { text-align: right; }
    .cl-coverage-label { font-size: 0.74rem; color: #64748b; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em; }
    .cl-coverage-value { font-size: 1.85rem; font-weight: 800; color: #0f172a; line-height: 1.15; }
    .cl-coverage-detail { font-size: 0.76rem; color: #64748b; }

    /* Live pipeline stepper */
    .cl-stepper { display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.5rem; margin: 0.5rem 0 1rem; }
    .cl-step {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
        padding: 0.7rem 0.85rem; display: flex; align-items: center; gap: 0.6rem; transition: all 0.2s ease;
    }
    .cl-step-num {
        width: 24px; height: 24px; border-radius: 50%; flex-shrink: 0; display: grid; place-items: center;
        font-size: 0.75rem; font-weight: 700; background: #e2e8f0; color: #64748b;
    }
    .cl-step-name { font-size: 0.84rem; font-weight: 600; color: #475569; }
    .cl-step.active { background: #eff6ff; border-color: #3b82f6; box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12); }
    .cl-step.active .cl-step-num { background: #3b82f6; color: #ffffff; }
    .cl-step.active .cl-step-name { color: #1d4ed8; }
    .cl-step.done { background: #f0fdf4; border-color: #bbf7d0; }
    .cl-step.done .cl-step-num { background: #16a34a; color: #ffffff; }
    .cl-step.done .cl-step-name { color: #15803d; }

    /* Activity terminal */
    .cl-terminal {
        background: #0b1120; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px;
        padding: 0.9rem 1rem; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
        max-height: 300px; overflow-y: auto; color: #e2e8f0; box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.5);
    }
    .cl-terminal.tall { max-height: 520px; }
    .cl-log { padding: 0.32rem 0; border-bottom: 1px solid rgba(255, 255, 255, 0.05); display: flex; align-items: flex-start; gap: 0.6rem; }
    .cl-log-time { color: #64748b; font-size: 0.74rem; white-space: nowrap; padding-top: 0.08rem; }
    .cl-log-badge {
        padding: 0.08rem 0.45rem; border-radius: 4px; font-size: 0.68rem; font-weight: 600;
        text-transform: uppercase; white-space: nowrap; letter-spacing: 0.03em;
    }
    .badge-routing { background: #312e81; color: #a5b4fc; }
    .badge-planning { background: #831843; color: #f9a8d4; }
    .badge-tool { background: #0c4a6e; color: #7dd3fc; }
    .badge-fetch { background: #064e3b; color: #6ee7b7; }
    .badge-rag { background: #581c87; color: #d8b4fe; }
    .badge-conflict { background: #7f1d1d; color: #fca5a5; }
    .badge-synthesis { background: #14532d; color: #86efac; }
    .badge-default { background: #1e293b; color: #cbd5e1; }

    /* Source cards */
    .cl-source {
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 0.95rem 1.15rem; margin-bottom: 0.7rem; transition: all 0.2s ease;
    }
    .cl-source:hover { border-color: #93c5fd; box-shadow: 0 6px 16px -6px rgba(15, 23, 42, 0.15); }
    .cl-source-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 0.8rem; }
    .cl-source-title { font-weight: 700; font-size: 0.92rem; color: #0f172a; }
    .cl-source-title a { color: #1d4ed8; text-decoration: none; }
    .cl-source-title a:hover { text-decoration: underline; }
    .cl-source-meta { font-size: 0.76rem; color: #64748b; margin-top: 0.15rem; word-break: break-all; }
    .cl-source-text { font-size: 0.87rem; color: #334155; margin-top: 0.5rem; line-height: 1.55; }
    .cl-tag {
        padding: 0.18rem 0.6rem; border-radius: 9999px; font-size: 0.73rem; font-weight: 600;
        border: 1px solid; white-space: nowrap;
    }
    .cl-tag.high { background: #f0fdf4; color: #15803d; border-color: #bbf7d0; }
    .cl-tag.mod { background: #eff6ff; color: #1d4ed8; border-color: #bfdbfe; }
    .cl-tag.low { background: #fffbeb; color: #b45309; border-color: #fde68a; }

    /* Knowledge base cluster header */
    .cl-cluster {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 14px;
        padding: 1.3rem 1.5rem; margin-bottom: 1.25rem; box-shadow: 0 8px 24px -6px rgba(15, 23, 42, 0.35);
        display: flex; align-items: center; gap: 0.9rem;
    }
    .cl-cluster h3 { margin: 0; color: #f8fafc; font-size: 1.15rem; font-weight: 700; }
    .cl-cluster p { color: #94a3b8; font-size: 0.88rem; margin: 0.2rem 0 0; }

    @media (max-width: 900px) {
        .cl-stepper { grid-template-columns: 1fr; }
        .cl-coverage { text-align: left; }
    }
</style>
""", unsafe_allow_html=True)


# ── Icons & Constants ────────────────────────────────────────────────────────

def _svg(paths: str, size: int = 18) -> str:
    """Inline line icon for HTML blocks (Streamlit material icons only render in widgets and markdown)."""
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{paths}</svg>'
    )


ICON_SHIELD = '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/>'
ICON_DATABASE = '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3"/>'
ICON_CPU = '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M15 2v2M9 2v2M15 20v2M9 20v2M2 15h2M2 9h2M20 15h2M20 9h2"/>'
ICON_TOOL = '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>'

# Dropdown label -> route passed to the agent (None = let the agent decide)
ROUTE_OPTIONS: dict[str, str | None] = {
    "Automatic routing": None,
    "Rigorous fact-check (documents + web)": "rag_and_web",
    "Internal documents only": "rag",
    "Live web search only": "web",
    "Direct analysis (no retrieval)": "direct",
}

# Quick-start buttons: label, material icon, preset name in config.RESEARCH_PRESETS
QUICK_PRESETS = [
    ("ECI SIR revision", ":material/how_to_vote:", "ECI Electoral Roll Revision (SIR)"),
    ("Grid stability claim", ":material/bolt:", "Fact-Check: Renewable Energy Grid Stability"),
    ("Deepfake regulation", ":material/gavel:", "Deepfake AI Regulation & Enforcement"),
    ("Solid-state batteries", ":material/battery_charging_full:", "Solid-State Battery Commercialization"),
]

VERDICT_STYLES: dict[str, dict[str, str]] = {
    "supported": {
        "title": "Verified & supported",
        "desc": "The central claims are backed by the authoritative sources that were retrieved.",
        "accent": "#15803d", "bg": "linear-gradient(135deg, #f0fdf4 0%, #ffffff 100%)", "border": "#bbf7d0",
        "icon": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    },
    "contradicted": {
        "title": "Contradicted / disputed",
        "desc": "Authoritative evidence contradicts at least one central claim.",
        "accent": "#b91c1c", "bg": "linear-gradient(135deg, #fef2f2 0%, #ffffff 100%)", "border": "#fecaca",
        "icon": '<circle cx="12" cy="12" r="10"/><path d="m15 9-6 6M9 9l6 6"/>',
    },
    "partial": {
        "title": "Partially supported",
        "desc": "Some elements are corroborated, while others lack proof or need context.",
        "accent": "#b45309", "bg": "linear-gradient(135deg, #fffbeb 0%, #ffffff 100%)", "border": "#fde68a",
        "icon": '<circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/>',
    },
    "insufficient": {
        "title": "Insufficient evidence",
        "desc": "The retrieved sources are not enough to confirm or refute the claims.",
        "accent": "#475569", "bg": "linear-gradient(135deg, #f8fafc 0%, #ffffff 100%)", "border": "#e2e8f0",
        "icon": '<circle cx="12" cy="12" r="10"/><path d="M8 12h8"/>',
    },
    "direct": {
        "title": "Answered directly",
        "desc": "No sources were searched for this question, so no fact-check verdict applies.",
        "accent": "#1d4ed8", "bg": "linear-gradient(135deg, #eff6ff 0%, #ffffff 100%)", "border": "#bfdbfe",
        "icon": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
    },
}

# Per-claim verdict -> (material icon, st.badge colour)
CLAIM_VERDICTS: dict[str, tuple[str, str]] = {
    "Supported": (":material/check_circle:", "green"),
    "Partially Supported": (":material/error:", "orange"),
    "Contradicted": (":material/cancel:", "red"),
    "Contradicted / Disputed": (":material/cancel:", "red"),
    "Insufficient Evidence": (":material/help:", "gray"),
}

PIPELINE_STEPS = ["Route & plan", "Gather evidence", "Extract claims", "Compare sources", "Write report"]

# Graph node that just finished -> (steps completed, progress, text for the next step)
NODE_PROGRESS: dict[str, tuple[int, float, str]] = {
    "route_query": (0, 0.15, "Route selected. Formulating the research plan..."),
    "plan_research": (1, 0.30, "Plan ready. Searching the knowledge base, Wikipedia and the web..."),
    "execute_searches": (2, 0.55, "Evidence gathered. Extracting verifiable claims..."),
    "extract_claims": (3, 0.70, "Claims extracted. Cross-examining sources for contradictions..."),
    "compare_and_analyze": (4, 0.85, "Checking whether more evidence is needed..."),
    "decide_followup": (4, 0.90, "Writing the evidence-backed report with citations..."),
    "verify_and_report": (5, 0.98, "Report complete."),
    "direct_response": (5, 0.98, "Direct answer composed."),
}

TOOL_CATALOG = [
    ("search_web", "Live web & news search", "Queries the DuckDuckGo text and news index for current information.", "query, max_results"),
    ("fetch_webpage", "Web page content extractor", "Fetches and cleans the full text of a URL, with SSRF protection.", "url, timeout"),
    ("search_knowledge_base", "Qdrant vector search", "Semantic search over 384-dimensional document vectors with metadata filters.", "query, filters, top_k"),
    ("search_wikipedia", "Encyclopedic grounding", "Retrieves background summaries and definitions from Wikipedia.", "query, max_results"),
    ("search_academic", "arXiv paper search", "Searches arXiv for scientific, technical and medical research.", "query, max_results"),
    ("check_source_credibility", "Source trust evaluator", "Scores domains using TLDs, fact-checker registries and news-wire reputation.", "url_or_domain"),
    ("calculate_statistics", "Safe calculator", "Evaluates mathematical and statistical expressions safely.", "expression"),
    ("summarize_document", "Document summarizer", "Summarizes key passages from an indexed document.", "document_name, focus_query"),
    ("extract_claims", "Claim extractor", "Extracts verifiable claims, dates and attributions from text.", "text"),
    ("compare_evidence", "Evidence triangulation", "Finds which sources corroborate or contradict each claim.", "claims, evidence_items"),
    ("verify_citations", "Citation verification", "Checks that every citation maps to a source that was actually retrieved.", "claims_or_answer, retrieved_sources"),
]

# Live tool tester inputs per tool: (argument, label, input kind, default value)
TOOL_TESTER: dict[str, list[tuple[str, str, str, Any]]] = {
    "search_web": [("query", "Web search query", "text", "Election Commission of India latest order")],
    "fetch_webpage": [("url", "Page URL", "text", "https://en.wikipedia.org/wiki/Election_Commission_of_India")],
    "search_knowledge_base": [
        ("query", "Semantic query", "text", "Supreme Court judgment on the Special Intensive Revision"),
        ("top_k", "Number of passages", "number", 5),
    ],
    "search_wikipedia": [("query", "Wikipedia query", "text", "Supreme Court of India")],
    "search_academic": [("query", "Academic paper query", "text", "transformer attention mechanism")],
    "check_source_credibility": [("source", "Domain or URL", "text", "https://eci.gov.in")],
    "calculate_statistics": [("expression", "Math / statistics expression", "text", "(450000 - 320000) / 320000 * 100")],
    "summarize_document": [
        ("document_name", "Indexed document", "document", None),
        ("focus_query", "Focus (optional)", "text", "deletion of names from the electoral roll"),
    ],
    "extract_claims": [(
        "text", "Text to analyse", "textarea",
        "On 24 June 2025 the Election Commission of India ordered a Special Intensive Revision of electoral rolls "
        "in Bihar. According to the Commission, 65 lakh names were removed from the draft roll published on "
        "1 August 2025. Opposition parties alleged that eligible voters were excluded without notice.",
    )],
    "compare_evidence": [
        ("claims", "Claims (JSON list)", "json", [
            {"claim": "The draft electoral roll for Bihar was published on 1 August 2025", "source_name": "News report"},
        ]),
        ("sources", "Evidence sources (JSON list)", "json", [
            {"source": "ECI press note", "text": "The draft electoral roll for Bihar was published on 1 August 2025 after the enumeration phase."},
            {"source": "Newspaper", "text": "The Election Commission published Bihar's draft electoral roll on 1 August 2025."},
        ]),
    ],
    "verify_citations": [
        ("answer", "Answer text containing citations", "textarea",
         "The draft roll was published on 1 August 2025 [WEB-1] (https://eci.gov.in/press-note). "
         "Turnout data came from https://unknown-site.example/data."),
        ("sources", "Retrieved sources (JSON list of URLs or titles)", "json", ["https://eci.gov.in/press-note"]),
    ],
}


def run_tool_test(name: str, args: dict[str, Any]) -> Any:
    """Call one tool with the tester's inputs; search tools use small result counts to keep output readable."""
    if name == "search_web":
        return tools.search_web(args["query"], max_results=3)
    if name == "fetch_webpage":
        res = tools.fetch_webpage(args["url"])
        if isinstance(res.get("content"), str) and len(res["content"]) > 3000:
            res = {**res, "content": res["content"][:3000] + " … (truncated for display)"}
        return res
    if name == "search_knowledge_base":
        return tools.search_knowledge_base(args["query"], top_k=int(args["top_k"]))
    if name == "search_wikipedia":
        return tools.search_wikipedia(args["query"], max_results=2)
    if name == "search_academic":
        return tools.search_academic(args["query"], max_results=3)
    if name == "check_source_credibility":
        return tools.check_source_credibility(args["source"])
    if name == "calculate_statistics":
        return tools.calculate_statistics(args["expression"])
    if name == "summarize_document":
        return tools.summarize_document(args["document_name"], focus_query=args.get("focus_query", ""))
    if name == "extract_claims":
        return tools.extract_claims(args["text"])
    if name == "compare_evidence":
        return tools.compare_evidence(args["claims"], args["sources"])
    if name == "verify_citations":
        return tools.verify_citations(args["answer"], args["sources"])
    raise ValueError(f"Unknown tool: {name}")


# ── Session State ────────────────────────────────────────────────────────────
PRESET_NAMES = list(config.RESEARCH_PRESETS)
for key, value in {
    "conversation_id": f"session_{int(time.time())}",
    "research_state": None,
    "is_running": False,
    "last_elapsed": None,
    "active_run": None,  # (session_id, run_index) of the results being shown
    "selected_preset": PRESET_NAMES[0],
    "sidebar_preset": PRESET_NAMES[0],
    "question_text": config.RESEARCH_PRESETS[PRESET_NAMES[0]]["query"],
}.items():
    st.session_state.setdefault(key, value)


def _load_preset(name: str) -> None:
    st.session_state.selected_preset = name
    st.session_state.sidebar_preset = name
    st.session_state.question_text = config.RESEARCH_PRESETS[name]["query"]


def _on_sidebar_preset() -> None:
    _load_preset(st.session_state.sidebar_preset)


def _queue_followup() -> None:
    """Move the follow-up into the question box and run it (callbacks run before widgets are drawn)."""
    text = st.session_state.get("followup_text", "").strip()
    if text:
        st.session_state.question_text = text
        st.session_state.followup_text = ""
        st.session_state.auto_run = True


def _reset_session() -> None:
    """Start a new session; the previous one stays saved under Past sessions."""
    st.session_state.conversation_id = f"session_{int(time.time())}"
    st.session_state.research_state = None
    st.session_state.last_elapsed = None
    st.session_state.active_run = None


def _open_run(session_id: str, run_index: int) -> None:
    """Show a saved run's results; switching sessions also restores that session's follow-up memory."""
    session = session_store.load_session(session_id)
    if not session or run_index >= len(session.get("runs", [])):
        return
    if session_id != st.session_state.conversation_id:
        st.session_state.conversation_id = session_id
        session_store.restore_memory(session_id)
    st.session_state.research_state = session["runs"][run_index]["state"]
    st.session_state.last_elapsed = None
    st.session_state.active_run = (session_id, run_index)


def _delete_session(session_id: str) -> None:
    session_store.delete_session(session_id)
    if session_id == st.session_state.conversation_id:
        _reset_session()


# ── Helpers ──────────────────────────────────────────────────────────────────

def esc(value: Any) -> str:
    return html.escape(str(value or ""))


def pdf_for(res_state: dict[str, Any]) -> tuple[bytes | None, str]:
    """Build the run's PDF once and reuse it on later reruns (the same results object stays in session state)."""
    cached = st.session_state.get("pdf_cache")
    if cached and cached[0] is res_state:
        return cached[1], cached[2]
    try:
        pdf_bytes, error = report_export.build_pdf(res_state), ""
    except Exception as ex:
        pdf_bytes, error = None, f"{type(ex).__name__}: {ex}"
    st.session_state.pdf_cache = (res_state, pdf_bytes, error)
    return pdf_bytes, error


def short_text(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def verdict_icon(verdict: str, route: str = "") -> str:
    """Material icon for a saved run's overall verdict (used in the sidebar session list)."""
    v = (verdict or "").lower()
    if route == "direct":
        return ":material/chat_bubble:"
    if "partial" in v:
        return ":material/error:"
    if "contradict" in v:
        return ":material/cancel:"
    if "supported" in v:
        return ":material/check_circle:"
    return ":material/help:"


def local_time(iso_timestamp: str, fmt: str) -> str:
    """Show a stored UTC ISO timestamp in this machine's local time zone (naive values are treated as UTC)."""
    try:
        dt = datetime.fromisoformat(str(iso_timestamp))
    except ValueError:
        return str(iso_timestamp or "")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone().strftime(fmt)


def heading(text: str, number: int | None = None) -> None:
    badge = f'<span class="cl-step-badge">{number}</span>' if number else ""
    st.markdown(f'<div class="cl-heading">{badge}<span class="cl-heading-text">{esc(text)}</span></div>', unsafe_allow_html=True)


def source_card(title: str, meta: str, text: str, tag: str = "", tag_level: str = "mod", url: str = "") -> None:
    title_html = f'<a href="{esc(url)}" target="_blank" rel="noopener">{esc(title)}</a>' if url else esc(title)
    tag_html = f'<span class="cl-tag {tag_level}">{esc(tag)}</span>' if tag else ""
    meta_html = f'<div class="cl-source-meta">{esc(meta)}</div>' if meta else ""
    st.markdown(
        f'<div class="cl-source"><div class="cl-source-header"><div>'
        f'<div class="cl-source-title">{title_html}</div>{meta_html}</div>{tag_html}</div>'
        f'<div class="cl-source-text">{esc(text)}</div></div>',
        unsafe_allow_html=True,
    )


def render_stepper(done: int) -> str:
    items = []
    for i, name in enumerate(PIPELINE_STEPS):
        state = "done" if i < done else "active" if i == done else ""
        mark = "&#10003;" if i < done else str(i + 1)
        items.append(f'<div class="cl-step {state}"><span class="cl-step-num">{mark}</span><span class="cl-step-name">{name}</span></div>')
    return f'<div class="cl-stepper">{"".join(items)}</div>'


def _badge_class(action: str) -> str:
    a = action.lower()
    for needle, css in (
        ("rout", "badge-routing"), ("plan", "badge-planning"), ("rag", "badge-rag"), ("conflict", "badge-conflict"),
        ("fetch", "badge-fetch"), ("source", "badge-fetch"), ("synth", "badge-synthesis"), ("report", "badge-synthesis"),
        ("tool", "badge-tool"),
    ):
        if needle in a:
            return css
    return "badge-default"


def render_terminal(entries: list[dict[str, Any]], tall: bool = False) -> str:
    rows = "".join(
        f'<div class="cl-log"><span class="cl-log-time">{esc(local_time(e.get("timestamp", ""), "%H:%M:%S"))}</span>'
        f'<span class="cl-log-badge {_badge_class(e.get("action", ""))}">{esc(e.get("action", "info").replace("_", " ")[:16])}</span>'
        f'<span>{esc(e.get("detail", ""))}</span></div>'
        for e in entries
    )
    return f'<div class="cl-terminal{" tall" if tall else ""}">{rows}</div>'


def verdict_key(res_state: dict[str, Any]) -> str:
    if res_state.get("route") == "direct":
        return "direct"
    v = (res_state.get("fact_check_verdict") or "").lower()
    if "partial" in v:
        return "partial"
    if "contradict" in v or "disputed" in v or "false" in v:
        return "contradicted"
    if "supported" in v:
        return "supported"
    return "insufficient"


# ── Live status (read once per run) ──────────────────────────────────────────
kb_stats = knowledge_base.get_collection_stats()
total_points_live = kb_stats.get("total_points", 0)
backend_type = kb_stats.get("backend", "local")
kb_docs = knowledge_base.get_ingested_documents()
model_name_short = config.GROQ_MODEL.split("/")[-1]

ready = bool(config.GROQ_API_KEY) and total_points_live > 0
status_pill = (
    '<span class="cl-pill ok"><span class="cl-dot"></span>Ready</span>' if ready else
    '<span class="cl-pill warn"><span class="cl-dot"></span>Setup needed</span>'
)
kb_pill_text = f"{'Qdrant Cloud' if backend_type == 'cloud' else 'Local Qdrant'} · {len(kb_docs)} docs · {total_points_live:,} chunks"

st.markdown(f"""
<div class="cl-navbar">
    <div class="cl-brand">
        <div class="cl-logo">{_svg(ICON_SHIELD, 22)}</div>
        <div>
            <div class="cl-title">{esc(config.APP_TITLE)}</div>
            <div class="cl-subtitle">Agentic research &amp; fact-checking platform &nbsp;·&nbsp; v{esc(config.APP_VERSION)}</div>
        </div>
    </div>
    <div class="cl-pills">
        {status_pill}
        <span class="cl-pill info">{_svg(ICON_DATABASE, 14)}{esc(kb_pill_text)}</span>
        <span class="cl-pill">{_svg(ICON_CPU, 14)}{esc(model_name_short)}</span>
        <span class="cl-pill">{_svg(ICON_TOOL, 14)}{len(TOOL_CATALOG)} tools</span>
    </div>
</div>
""", unsafe_allow_html=True)

tab_research, tab_kb, tab_tools, tab_diagnostics = st.tabs([
    ":material/manage_search: Research & Fact-Checking",
    ":material/database: Knowledge Base",
    ":material/handyman: Tool Registry",
    ":material/monitor_heart: Diagnostics & Memory",
])


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR: System Status, Session & Presets
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("##### :material/sensors: System status")
    if backend_type == "cloud":
        st.success(
            f"**Qdrant Cloud**  \nCollection `{config.QDRANT_COLLECTION_NAME}`  \n"
            f"{len(kb_docs)} documents · {total_points_live:,} vectors · cosine, 384-d",
            icon=":material/cloud_done:",
        )
    else:
        st.warning(f"**Local Qdrant**  \n{total_points_live:,} vectors stored on this machine", icon=":material/storage:")

    if config.GROQ_API_KEY:
        st.info(f"**Groq inference**  \nModel `{config.GROQ_MODEL}`  \nMax tokens `{config.GROQ_MAX_TOKENS}`", icon=":material/memory:")
    else:
        st.error("`GROQ_API_KEY` is missing from `.env`", icon=":material/key_off:")

    st.divider()
    st.markdown("##### :material/forum: Research sessions")
    current_runs = (session_store.load_session(st.session_state.conversation_id) or {}).get("runs", [])
    st.caption(
        f"Current session · {len(current_runs)} question(s) · follow-ups use this context"
        if current_runs else "Current session · no questions yet"
    )
    for i, run in enumerate(current_runs):
        st.button(
            short_text(run.get("question", ""), 60),
            key=f"cur_run_{i}",
            icon=verdict_icon(run.get("verdict", ""), run.get("route", "")),
            type="secondary" if st.session_state.get("active_run") == (st.session_state.conversation_id, i) else "tertiary",
            width="stretch",
            on_click=_open_run,
            args=(st.session_state.conversation_id, i),
            help=run.get("question", ""),
        )
    st.button("New session", icon=":material/add_comment:", width="stretch", on_click=_reset_session)

    past_sessions = [s for s in session_store.list_sessions() if s["id"] != st.session_state.conversation_id]
    if past_sessions:
        with st.expander(f"Past sessions ({len(past_sessions)})", icon=":material/history:"):
            for s in past_sessions[:20]:
                st.button(
                    short_text(s["title"], 60),
                    key=f"past_{s['id']}",
                    icon=":material/chat:",
                    type="tertiary",
                    width="stretch",
                    on_click=_open_run,
                    args=(s["id"], s["runs"] - 1),
                    help=f"{s['title']}\n\n{s['runs']} question(s) · last used {local_time(s['updated_at'], '%d %b %Y, %H:%M')}",
                )
            with st.popover("Delete a session", icon=":material/delete:", width="stretch"):
                to_delete = st.selectbox(
                    "Session", [s["id"] for s in past_sessions],
                    format_func=lambda sid: short_text(next(s["title"] for s in past_sessions if s["id"] == sid), 50),
                    key="delete_session_choice",
                )
                st.button("Delete permanently", type="primary", key="delete_session_btn", on_click=_delete_session, args=(to_delete,))

    st.divider()
    st.markdown("##### :material/target: Investigation presets")
    st.selectbox(
        "Load scenario",
        options=PRESET_NAMES,
        key="sidebar_preset",
        on_change=_on_sidebar_preset,
    )
    active_preset = config.RESEARCH_PRESETS[st.session_state.selected_preset]
    st.caption(f"**Domain:** {active_preset['domain']}")
    st.caption(active_preset["description"])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1: RESEARCH & FACT-CHECKING
# ═══════════════════════════════════════════════════════════════════════════════

with tab_research:
    heading("Formulate a claim or research question", 1)

    query_input = st.text_area(
        "Claim to verify, allegation, or research question",
        key="question_text",
        height=110,
        placeholder="e.g. Has the Supreme Court ruled on the voter roll revision procedure in Bihar?",
        help="Enter any statement, factual question, or political, legal or scientific assertion.",
    )

    chip_cols = st.columns(len(QUICK_PRESETS))
    for col, (label, icon, preset_name) in zip(chip_cols, QUICK_PRESETS):
        if preset_name in config.RESEARCH_PRESETS:
            col.button(label, icon=icon, width="stretch", on_click=_load_preset, args=(preset_name,), key=f"chip_{preset_name}")

    st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)

    col_c1, col_c2, col_c3 = st.columns([2, 1.2, 1.2])
    with col_c1:
        selected_mode = st.selectbox(
            "Investigation routing strategy",
            options=list(ROUTE_OPTIONS),
            help="Automatic checks your documents first and adds the web when needed.",
        )
    with col_c2:
        max_hops = st.slider(
            "Reasoning depth (research rounds)", min_value=1, max_value=4, value=2,
            help="How many times the agent may search again to fill evidence gaps.",
        )
    with col_c3:
        top_k_passages = st.slider(
            "Passage retrieval depth", min_value=3, max_value=12, value=config.MAX_RAG_PASSAGES,
            help="How many passages to pull from the knowledge base.",
        )

    col_btn, _ = st.columns([1.4, 4])
    with col_btn:
        start_investigation = st.button(
            "Execute investigation",
            type="primary",
            icon=":material/play_arrow:",
            width="stretch",
            disabled=st.session_state.is_running or not query_input.strip(),
        )
    if st.session_state.pop("auto_run", False) and query_input.strip():
        start_investigation = True

    # ── Live execution pipeline ─────────────────────────────────────────────
    if start_investigation:
        st.session_state.is_running = True
        # Seed the terminal so it is visible immediately (the first agent log can take a few seconds)
        live_entries: list[dict[str, Any]] = [{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "starting",
            "detail": "Starting the research agent (the first run after a restart also loads the embedding model)...",
        }]

        stepper_box = st.empty()
        stepper_box.markdown(render_stepper(0), unsafe_allow_html=True)
        progress_bar = st.progress(0.05, text="Analyzing the question and selecting a routing strategy...")
        log_box = st.empty()
        log_box.markdown(render_terminal(live_entries), unsafe_allow_html=True)
        plan_box = st.empty()

        def on_live_log(entry: dict[str, Any]) -> None:
            live_entries.append(entry)
            log_box.markdown(render_terminal(live_entries[-9:]), unsafe_allow_html=True)

        def on_node_update(state: dict[str, Any], node_name: str) -> None:
            if node_name in NODE_PROGRESS:
                done, pct, text = NODE_PROGRESS[node_name]
                stepper_box.markdown(render_stepper(done), unsafe_allow_html=True)
                progress_bar.progress(pct, text=text)
            if node_name == "plan_research" and state.get("research_plan"):
                queries = ", ".join(f"`{q}`" for q in state.get("search_queries", []))
                objectives = " · ".join(state.get("sub_objectives", []))
                plan_box.info(
                    f"**Research plan:** {state['research_plan']}\n\n"
                    + (f"**Objectives:** {objectives}\n\n" if objectives else "")
                    + f"**Search queries:** {queries}",
                    icon=":material/assignment:",
                )

        start_time = time.time()
        try:
            final_state = agent.run_research(
                question=query_input,
                conversation_id=st.session_state.conversation_id,
                callback=on_node_update,
                log_callback=on_live_log,
                route_override=ROUTE_OPTIONS[selected_mode],
                max_iterations=max_hops,
                rag_top_k=top_k_passages,
            )
            st.session_state.research_state = final_state
            if not final_state.get("error"):
                session_store.save_run(st.session_state.conversation_id, final_state)
                saved = session_store.load_session(st.session_state.conversation_id) or {}
                st.session_state.active_run = (st.session_state.conversation_id, len(saved.get("runs", [])) - 1)
        except Exception as e:
            st.session_state.research_state = {"error": str(e), "activity_log": live_entries}
        finally:
            st.session_state.last_elapsed = round(time.time() - start_time, 1)
            st.session_state.is_running = False

        st.rerun()

    # ── Completed research results ───────────────────────────────────────────
    res_state = st.session_state.research_state
    if not res_state and not st.session_state.is_running:
        # Idle terminal: always visible, fills with live agent activity once a run starts
        st.markdown(render_terminal([{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "idle",
            "detail": "Waiting for a question. Live agent activity will appear here when you execute an investigation.",
        }]), unsafe_allow_html=True)
    if res_state and not st.session_state.is_running:
        st.divider()

        if res_state.get("error"):
            st.error(f"**Execution error:** {res_state['error']}", icon=":material/error:")

        comp = res_state.get("evidence_comparison", {})
        style = VERDICT_STYLES[verdict_key(res_state)]
        coverage = comp.get("evidence_coverage")
        coverage_html = (
            f'<div class="cl-coverage-label">Evidence coverage</div>'
            f'<div class="cl-coverage-value">{coverage}%</div>'
            f'<div class="cl-coverage-detail">{comp.get("grounded_claims", 0)} of {comp.get("total_claims", 0)} claims matched to sources</div>'
            if coverage is not None else
            '<div class="cl-coverage-label">Evidence coverage</div><div class="cl-coverage-value">N/A</div>'
            '<div class="cl-coverage-detail">No claims were evaluated</div>'
        )
        st.markdown(f"""
        <div class="cl-verdict" style="--accent:{style['accent']};--bg:{style['bg']};--border:{style['border']}">
            <div>
                <div class="cl-verdict-title">{_svg(style['icon'], 22)}{esc(style['title'])}</div>
                <div class="cl-verdict-desc">{esc(style['desc'])}</div>
            </div>
            <div class="cl-coverage">{coverage_html}</div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3, m4, m5 = st.columns(5)
        route_labels = {"direct": "DIRECT", "rag": "RAG", "web": "WEB", "rag_and_web": "RAG + WEB"}
        m1.metric("Routing mode", route_labels.get(res_state.get("route") or "web", "WEB"), border=True)
        m2.metric("Qdrant passages", len(res_state.get("rag_passages", [])), border=True)
        m3.metric("Web sources", len(res_state.get("fetched_sources", [])), border=True)
        m4.metric("Claims evaluated", comp.get("total_claims", len(res_state.get("extracted_claims", []))), border=True)
        m5.metric("Verified citations", res_state.get("citation_verification", {}).get("verified_count", 0), border=True)

        run_facts = [
            f"{res_state.get('iteration', 0)} of {res_state.get('max_iterations', 0)} research round(s)",
            f"{res_state.get('tool_call_count', 0)} tool calls",
            f"{res_state.get('pages_fetched', 0)} pages downloaded",
        ]
        if res_state.get("stop_reason"):
            run_facts.append(f"Stopped: {res_state['stop_reason'].rstrip('.')}")
        if st.session_state.last_elapsed:
            run_facts.insert(0, f"Completed in {st.session_state.last_elapsed}s")
        st.caption(" · ".join(run_facts))

        # Keep the execution log visible after the run (the live view is cleared when results load)
        if res_state.get("activity_log"):
            with st.expander(f"Execution log ({len(res_state['activity_log'])} steps)", icon=":material/terminal:", expanded=True):
                st.markdown(render_terminal(res_state["activity_log"]), unsafe_allow_html=True)

        subtab_report, subtab_matrix, subtab_sources, subtab_trace, subtab_followup = st.tabs([
            ":material/description: Verified Report",
            ":material/fact_check: Evidence & Claims Matrix",
            ":material/travel_explore: Sources & Credibility",
            ":material/timeline: Agent Execution Trace",
            ":material/forum: Follow-up",
        ])

        # Report
        with subtab_report:
            if res_state.get("research_plan"):
                st.info(f"**Research plan executed:** {res_state['research_plan']}", icon=":material/assignment:")

            if res_state.get("final_report"):
                file_stem = f"civiclens_report_{int(time.time())}"
                col_exp, _ = st.columns([1, 4])
                with col_exp:
                    with st.popover("Export", icon=":material/download:", width="stretch"):
                        st.download_button(
                            "Markdown (.md)",
                            data=res_state["final_report"],
                            file_name=f"{file_stem}.md",
                            mime="text/markdown",
                            icon=":material/description:",
                            width="stretch",
                            help="The verified report text only.",
                        )
                        st.download_button(
                            "JSON (.json)",
                            data=json.dumps(res_state, indent=2, default=str),
                            file_name=f"{file_stem}.json",
                            mime="application/json",
                            icon=":material/data_object:",
                            width="stretch",
                            help="Every detail of the run: report, claims, sources and execution log.",
                        )
                        pdf_bytes, pdf_error = pdf_for(res_state)
                        if pdf_bytes:
                            st.download_button(
                                "PDF (.pdf)",
                                data=pdf_bytes,
                                file_name=f"{file_stem}.pdf",
                                mime="application/pdf",
                                icon=":material/picture_as_pdf:",
                                width="stretch",
                                help="Verified Report, Evidence & Claims Matrix and Sources & Credibility in one document.",
                            )
                            st.caption("PDF includes the report, the claims matrix and all sources.")
                        else:
                            st.caption(f"PDF unavailable: {pdf_error}")
                with st.container(border=True):
                    st.markdown(res_state["final_report"])

        # Claims matrix
        with subtab_matrix:
            analysis = comp.get("analysis", [])
            if analysis:
                st.caption(
                    "Each claim was judged by the AI against all retrieved sources."
                    if comp.get("method") == "llm_judgement" else
                    "Claims were matched to sources by keyword overlap (AI judgement was unavailable for this run)."
                )
                counts: dict[str, int] = {}
                for a in analysis:
                    name = a.get("verdict", "Insufficient Evidence")
                    counts[name] = counts.get(name, 0) + 1
                badge_cols = st.columns(max(len(counts), 4))
                for col, (name, n) in zip(badge_cols, counts.items()):
                    icon, color = CLAIM_VERDICTS.get(name, (":material/help:", "gray"))
                    with col:
                        st.badge(f"{name}: {n}", icon=icon, color=color)

                for idx, a in enumerate(analysis, start=1):
                    name = a.get("verdict", "Insufficient Evidence")
                    icon, color = CLAIM_VERDICTS.get(name, (":material/help:", "gray"))
                    claim_text = a.get("claim", "")
                    short = claim_text if len(claim_text) <= 110 else claim_text[:107] + "..."
                    with st.expander(f"{idx}. {short}", icon=icon, expanded=idx <= 3):
                        st.badge(name, icon=icon, color=color)
                        st.markdown(f"**Claim:** {claim_text}")
                        if a.get("verdict_reason"):
                            st.markdown(f"**Why:** {a['verdict_reason']}")
                        st.caption(
                            f"Supporting sources: {a.get('supporting_count', 0)} · "
                            f"Contradicting sources: {a.get('contradicting_count', 0)}"
                            + (f" · Attributed to: {a['attributed_to']}" if a.get("attributed_to") else "")
                        )
                        for ev in a.get("related_evidence", []):
                            source_card(
                                title=ev.get("source", "Source"),
                                meta=ev.get("url", ""),
                                text=ev.get("evidence_text", "")[:300],
                                tag=(
                                    "Contradicts" if ev.get("is_contradiction")
                                    else "Supports" if ev.get("relevance") is None
                                    else f"Match {ev['relevance']:.0%}"
                                ),
                                tag_level="low" if ev.get("is_contradiction") else "mod",
                            )
            elif res_state.get("extracted_claims"):
                for idx, c in enumerate(res_state["extracted_claims"], start=1):
                    st.markdown(f"**{idx}.** {c.get('claim')} — _{c.get('source_name', 'Source')}_")
            else:
                st.info("No claims were evaluated in this run.", icon=":material/info:")

        # Sources & credibility
        with subtab_sources:
            kb_passages = res_state.get("rag_passages", [])
            web_sources = res_state.get("fetched_sources", [])
            wiki_items = res_state.get("wiki_facts", [])
            acad_items = res_state.get("academic_papers", [])

            if not (kb_passages or web_sources or wiki_items or acad_items):
                st.info("No sources were retrieved in this run.", icon=":material/info:")

            if kb_passages:
                st.markdown(f"##### :material/folder_open: Internal knowledge base ({len(kb_passages)} passages)")
                for i, p in enumerate(kb_passages, start=1):
                    source_card(
                        title=f"[KB-{i}] {p.get('document_name', 'Document')}",
                        meta=f"Page {p.get('page_number', '?')}",
                        text=p.get("text", "")[:400],
                        tag=f"Similarity {p.get('raw_score', p.get('relevance_score', 0)):.2f}",
                        tag_level="high",
                    )

            if web_sources:
                st.markdown(f"##### :material/public: Live web sources ({len(web_sources)} pages)")
                for j, ws in enumerate(web_sources, start=1):
                    cred = ws.get("credibility", {})
                    trust_score = cred.get("credibility_score", 60)
                    source_card(
                        title=f"[WEB-{j}] {ws.get('title', 'Web page')[:100]}",
                        meta=ws.get("url", ""),
                        text=(ws.get("snippet") or ws.get("content", ""))[:300],
                        tag=f"{cred.get('trust_tier', 'Standard web')} · {trust_score}/100",
                        tag_level="high" if trust_score >= 85 else "mod" if trust_score >= 60 else "low",
                        url=ws.get("url", ""),
                    )

            if wiki_items:
                st.markdown(f"##### :material/menu_book: Wikipedia grounding ({len(wiki_items)} articles)")
                for wi in wiki_items:
                    source_card(
                        title=wi.get("title", "Article"), meta=wi.get("url", ""),
                        text=(wi.get("extract") or wi.get("snippet", ""))[:300], url=wi.get("url", ""),
                    )

            if acad_items:
                st.markdown(f"##### :material/science: Academic papers ({len(acad_items)})")
                for paper in acad_items:
                    source_card(
                        title=paper.get("title", "Paper"), meta=f"Published {paper.get('published', 'n/a')}",
                        text=paper.get("abstract", "")[:300], url=paper.get("url", ""),
                    )

        # Execution trace
        with subtab_trace:
            logs = res_state.get("activity_log", [])
            if logs:
                st.caption(f"{len(logs)} recorded steps across {res_state.get('iteration', 1)} research round(s)")
                st.markdown(render_terminal(logs, tall=True), unsafe_allow_html=True)
            else:
                st.info("No trace entries recorded.", icon=":material/info:")

        # Follow-up
        with subtab_followup:
            st.caption("The agent remembers previous findings, claims and sources from this research session.")
            f1, f2 = st.columns([4, 1], vertical_alignment="bottom")
            with f1:
                st.text_input(
                    "Follow-up question", key="followup_text",
                    placeholder="e.g. Which statutory sections did the court cite?",
                )
            with f2:
                st.button("Submit follow-up", icon=":material/send:", width="stretch", on_click=_queue_followup, key="btn_followup")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2: KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════════════════

with tab_kb:
    st.markdown(f"""
    <div class="cl-cluster">
        <div class="cl-logo">{_svg(ICON_DATABASE, 22)}</div>
        <div>
            <h3>{'Qdrant Cloud cluster' if backend_type == 'cloud' else 'Local Qdrant store'}</h3>
            <p>Vector database for semantic retrieval over your documents, with metadata filtering.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    c_m1.metric("Vector backend", "Qdrant Cloud" if backend_type == "cloud" else "Local persistent", border=True)
    c_m2.metric("Collection", config.QDRANT_COLLECTION_NAME, border=True)
    c_m3.metric("Documents · vectors", f"{len(kb_docs)} · {total_points_live:,}", border=True)
    c_m4.metric("Embedding model", f"{config.EMBEDDING_MODEL} ({config.EMBEDDING_DIMENSION}-d)", border=True)

    with st.expander("How documents are stored", icon=":material/info:"):
        st.markdown(
            "- Each document is split into overlapping chunks.\n"
            "- Each chunk is embedded locally with Sentence-Transformers.\n"
            "- Chunks are upserted to Qdrant with their document name, page number, SHA-256 hash and text.\n"
            "- Files with identical content are detected by hash and skipped."
        )

    col_sync, _ = st.columns([1.5, 3.5])
    with col_sync:
        if st.button("Re-sync documents folder", type="primary", icon=":material/sync:", width="stretch"):
            with st.spinner("Extracting text, generating embeddings and updating Qdrant..."):
                re_res = knowledge_base.reindex_all_documents()
            st.toast(f"Synchronized · {re_res.get('total_chunks', 0):,} chunks", icon=":material/check_circle:")
            st.rerun()

    st.divider()

    st.markdown("#### :material/upload_file: Upload new documents")
    st.caption("Supported formats: PDF, Word (.docx), Markdown, plain text, CSV and JSON")

    col_up1, col_up2 = st.columns([3, 1])
    with col_up1:
        uploaded_files = st.file_uploader(
            "Drop documents here or browse",
            type=["pdf", "docx", "txt", "md", "markdown", "json", "csv"],
            accept_multiple_files=True,
            key="cloud_file_uploader",
        )
    with col_up2:
        doc_date = st.text_input("Source date", value=datetime.now().strftime("%Y-%m-%d"))

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) ready for ingestion")
        if st.button("Vectorize & upload to Qdrant", icon=":material/cloud_upload:", width="stretch"):
            entries = []
            for f in uploaded_files:
                dest = config.DOCUMENTS_DIR / f.name
                dest.write_bytes(f.getvalue())
                entries.append({"path": str(dest), "name": Path(f.name).stem, "date": doc_date})
            with st.spinner("Extracting text, computing hashes, generating embeddings and uploading..."):
                ingest_res = knowledge_base.ingest_multiple_documents(entries)
            st.toast(
                f"Added {ingest_res.get('added_count', 0)} · skipped {ingest_res.get('skipped_count', 0)} duplicate(s)",
                icon=":material/check_circle:",
            )
            st.rerun()

    st.divider()

    st.markdown(f"#### :material/inventory_2: Indexed documents ({len(kb_docs)})")
    if kb_docs:
        with st.container(border=True):
            h1, h2, h3, h4 = st.columns([3, 1.5, 1.2, 1])
            h1.caption("**Document**")
            h2.caption("**Dates**")
            h3.caption("**Chunks**")
            h4.caption("**Action**")
            for d in kb_docs:
                col_doc1, col_doc2, col_doc3, col_doc4 = st.columns([3, 1.5, 1.2, 1], vertical_alignment="center")
                with col_doc1:
                    st.markdown(f"**{d.get('name', 'Document')}**")
                    st.caption(f"SHA-256 `{(d.get('hash') or 'n/a')[:16]}` · {(d.get('content_type') or '.pdf').lstrip('.').upper()}")
                with col_doc2:
                    st.caption(f"Source: {d.get('date', 'Unknown')}  \nAdded: {local_time(d.get('ingested_at', ''), '%Y-%m-%d')}")
                with col_doc3:
                    st.markdown(f"{d.get('chunks_count', 0):,}")
                with col_doc4:
                    with st.popover("Delete", icon=":material/delete:"):
                        st.caption(f"Permanently remove **{d.get('name')}** and its vectors?")
                        if st.button("Delete permanently", type="primary", key=f"del_{d.get('name')}"):
                            knowledge_base.delete_document(d.get("name"))
                            st.toast(f"Removed '{d.get('name')}'", icon=":material/delete:")
                            st.rerun()
    else:
        st.info("No documents indexed yet.", icon=":material/info:")

    st.divider()

    with st.expander("Vector search sandbox — test retrieval", icon=":material/search:"):
        st.caption("Runs a direct similarity search on Qdrant with live cosine scores.")
        v_query = st.text_input("Semantic query", placeholder="e.g. deletion of voters without notice under ECI guidelines")
        if v_query:
            v_res = knowledge_base.search_knowledge_base(v_query, top_k=5)
            st.caption(
                f"{v_res.get('total_found', 0)} passage(s) above the relevance threshold ({config.MIN_RAG_RELEVANCE_SCORE:.2f})"
                + (f" · best score {v_res['top_raw_score']:.2f}" if v_res.get("top_raw_score") is not None else "")
            )
            for pt in v_res.get("results", []):
                source_card(
                    title=pt.get("document_name", "Document"),
                    meta=f"Page {pt.get('page_number', '?')}",
                    text=pt.get("text", "")[:350],
                    tag=f"Similarity {pt.get('raw_score', 0):.4f}",
                    tag_level="high",
                )


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3: TOOL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

with tab_tools:
    st.markdown(f"#### :material/handyman: Callable tool suite ({len(TOOL_CATALOG)} tools)")
    st.caption("Each tool is a Python function that returns structured, serializable data to the agent.")

    tool_cols = st.columns(2)
    for i, (name, title, desc, args) in enumerate(TOOL_CATALOG):
        with tool_cols[i % 2]:
            with st.expander(f"`{name}` — {title}", icon=":material/settings:"):
                st.markdown(desc)
                st.caption(f"Arguments: `{args}`")

    st.divider()

    st.markdown("#### :material/science: Live tool tester")

    tool_titles = {name: title for name, title, _, _ in TOOL_CATALOG}
    col_t1, col_t2 = st.columns([1.5, 3])
    with col_t1:
        test_tool_name = st.selectbox(
            "Tool to test",
            options=list(TOOL_TESTER),
            format_func=lambda n: f"{n} — {tool_titles.get(n, '')}",
        )
        st.caption(next((desc for name, _, desc, _ in TOOL_CATALOG if name == test_tool_name), ""))

    tool_inputs: dict[str, Any] = {}
    input_error = None
    with col_t2:
        for field, label, kind, default in TOOL_TESTER[test_tool_name]:
            key = f"tool_{test_tool_name}_{field}"
            if kind == "text":
                tool_inputs[field] = st.text_input(label, value=default, key=key)
            elif kind == "number":
                tool_inputs[field] = st.number_input(label, min_value=1, max_value=12, value=default, key=key)
            elif kind == "textarea":
                tool_inputs[field] = st.text_area(label, value=default, height=120, key=key)
            elif kind == "document":
                doc_names = [d.get("name") for d in kb_docs]
                if doc_names:
                    tool_inputs[field] = st.selectbox(label, options=doc_names, key=key)
                else:
                    input_error = "No documents are indexed yet; add one in the Knowledge Base tab."
            elif kind == "json":
                raw = st.text_area(label, value=json.dumps(default, indent=2), height=160, key=key)
                try:
                    tool_inputs[field] = json.loads(raw)
                except json.JSONDecodeError as ex:
                    input_error = f"{label}: invalid JSON ({ex.msg} at line {ex.lineno})."

    if input_error:
        st.warning(input_error, icon=":material/warning:")
    if st.button(f"Run {test_tool_name}", icon=":material/play_arrow:", width="stretch", disabled=bool(input_error)):
        with st.spinner("Executing tool..."):
            try:
                out = run_tool_test(test_tool_name, tool_inputs)
            except Exception as ex:
                out = {"error": f"{type(ex).__name__}: {ex}"}
        st.json(out, expanded=2)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4: DIAGNOSTICS & MEMORY
# ═══════════════════════════════════════════════════════════════════════════════

with tab_diagnostics:
    st.markdown("#### :material/lan: API health & connections")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        with st.container(border=True):
            st.markdown("**Groq language model**")
            st.caption(f"Model `{config.GROQ_MODEL}`")
            if st.button("Test Groq connection", icon=":material/bolt:", width="stretch"):
                try:
                    from langchain_core.messages import HumanMessage
                    started = time.time()
                    res = agent._get_llm().invoke([HumanMessage(content="Hello. Reply with 'OK'.")])
                    st.success(f"Groq responded `{res.content.strip()[:40]}` in {time.time() - started:.1f}s", icon=":material/check_circle:")
                except Exception as ex:
                    st.error(f"Groq connection error: {ex}", icon=":material/error:")

    with col_d2:
        with st.container(border=True):
            st.markdown("**Qdrant vector store**")
            st.caption(f"Backend `{backend_type}` · collection `{config.QDRANT_COLLECTION_NAME}`")
            if st.button("Test Qdrant connection", icon=":material/bolt:", width="stretch"):
                try:
                    client, backend = knowledge_base.get_qdrant_client()
                    c_names = [c.name for c in client.get_collections().collections]
                    st.success(f"Connected ({backend}) · collections: {', '.join(c_names)}", icon=":material/check_circle:")
                except Exception as ex:
                    st.error(f"Qdrant connection error: {ex}", icon=":material/error:")

    st.divider()

    st.markdown("#### :material/history: Active conversation memory")
    history = memory.get_session_memory(st.session_state.conversation_id).get_history()
    if history:
        st.caption(f"{len(history)} messages in the current session")
        for h in history:
            role = "user" if h.get("role") == "user" else "assistant"
            content = h.get("content", "")
            with st.chat_message(role):
                st.markdown(content if len(content) <= 600 else content[:600] + "...")
    else:
        st.info("No conversation turns in the current session yet.", icon=":material/info:")
