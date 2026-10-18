"""
Agent Module — LangGraph stateful workflow with intelligent routing, autonomous planning,
dynamic tool execution, evidence evaluation, contradiction detection, iterative replanning,
self-verification, and citation-backed synthesis.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

import config
import knowledge_base
import memory
import router
import tools

# ═══════════════════════════════════════════════════════════════════════════════
#  RESEARCH STATE DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

class ResearchState(TypedDict, total=False):
    """Full execution state passed through the agent workflow."""
    # Input & Session
    question: str
    conversation_id: str
    conversation_context: str

    # Routing
    route: str
    route_reason: str
    route_override: str  # set when the user forces a route from the UI

    # User-tunable retrieval depth
    rag_top_k: int

    # Planning & Agent Decision
    research_plan: str
    plan_source: str  # "llm" when the planner produced the plan, "fallback" otherwise
    selected_tools: list[str]
    rag_rationale: str
    sub_objectives: list[str]
    search_queries: list[str]

    # Execution tracking
    iteration: int
    max_iterations: int
    tool_call_count: int  # search/lookup tool calls; page downloads are counted in pages_fetched
    pages_fetched: int
    max_tool_calls: int
    activity_log: list[dict[str, Any]]

    # Data collections
    search_results: list[dict[str, Any]]
    fetched_sources: list[dict[str, Any]]
    rag_passages: list[dict[str, Any]]
    wiki_facts: list[dict[str, Any]]
    academic_papers: list[dict[str, Any]]
    unique_source_urls: list[str]

    # Evidence & Verification
    extracted_claims: list[dict[str, Any]]
    evidence_sources: list[dict[str, Any]]  # numbered sources [S1..Sn] shown to the claim judge
    evidence_comparison: dict[str, Any]
    citation_verification: dict[str, Any]
    conflict_detected: bool
    fact_check_verdict: str

    # Iterative reasoning & Replanning
    needs_followup: bool
    followup_query: str
    stop_reason: str

    # Output
    final_report: str
    error: str


def _initial_state(question: str, conversation_id: str | None = None) -> ResearchState:
    conv_id = conversation_id or "session_default"
    sess = memory.get_session_memory(conv_id)
    history_context = sess.get_formatted_context()

    return ResearchState(
        question=question,
        conversation_id=conv_id,
        conversation_context=history_context,
        route="web",
        route_reason="",
        research_plan="",
        plan_source="",
        selected_tools=[],
        rag_rationale="",
        sub_objectives=[],
        search_queries=[],
        iteration=0,
        max_iterations=config.MAX_AGENT_ITERATIONS,
        tool_call_count=0,
        pages_fetched=0,
        max_tool_calls=config.MAX_TOOL_CALLS,
        activity_log=[],
        search_results=[],
        fetched_sources=[],
        rag_passages=[],
        wiki_facts=[],
        academic_papers=[],
        unique_source_urls=[],
        extracted_claims=[],
        evidence_sources=[],
        evidence_comparison={},
        citation_verification={},
        conflict_detected=False,
        fact_check_verdict="",
        needs_followup=False,
        followup_query="",
        stop_reason="",
        final_report="",
        error="",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM CLIENT & LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

_current_log_callback = None

VALID_ROUTES = ("direct", "rag", "web", "rag_and_web")


def _get_llm() -> ChatGroq:
    if not config.GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. Please configure it in your .env file or Streamlit settings."
        )
    return ChatGroq(
        api_key=config.GROQ_API_KEY,
        model=config.GROQ_MODEL,
        temperature=config.GROQ_TEMPERATURE,
        max_tokens=config.GROQ_MAX_TOKENS,
    )


def _log(state: ResearchState, action: str, detail: str, **extra: Any) -> None:
    """Record an execution event in the activity log and stream to live UI callback."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "iteration": state.get("iteration", 0),
        "action": action,
        "detail": detail,
        **extra,
    }
    state.setdefault("activity_log", []).append(entry)
    global _current_log_callback
    if _current_log_callback:
        try:
            _current_log_callback(entry)
        except Exception:
            pass


RATE_LIMIT_RETRIES = 3
RATE_LIMIT_MAX_WAIT = 30.0  # seconds per retry


def _rate_limit_wait(err: str) -> float:
    """Seconds Groq asks us to wait ('try again in 7.5s' / 'in 1m2s'), capped; defaults to 10s."""
    match = re.search(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s", err)
    if not match:
        return 10.0
    return min(RATE_LIMIT_MAX_WAIT, int(match.group(1) or 0) * 60 + float(match.group(2)) + 1.0)


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """Invoke Groq LLM, waiting and retrying on per-minute rate limits, with clear error messages."""
    llm = _get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            err = str(e)
            is_rate_limit = "429" in err or "rate limit" in err.lower()
            if is_rate_limit and attempt < RATE_LIMIT_RETRIES:
                wait = _rate_limit_wait(err)
                if _current_log_callback:
                    try:
                        _current_log_callback({
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "action": "rate_limit_wait",
                            "detail": f"Groq per-minute limit reached; waiting {wait:.0f}s before retrying ({attempt + 1}/{RATE_LIMIT_RETRIES})",
                        })
                    except Exception:
                        pass
                time.sleep(wait)
                continue
            if "401" in err or "api_key" in err.lower() or "unauthorized" in err.lower():
                raise RuntimeError(f"Groq Authentication Error: Invalid or missing GROQ_API_KEY in .env ({err[:160]})")
            if is_rate_limit:
                raise RuntimeError(f"Groq Rate Limit Exceeded: still limited after {RATE_LIMIT_RETRIES} retries. Please wait a minute and try again. ({err[:160]})")
            if "not_found" in err.lower() or "model" in err.lower():
                raise RuntimeError(f"Groq Model Error: Model '{config.GROQ_MODEL}' unavailable. ({err[:160]})")
            raise RuntimeError(f"Groq API Error: {err[:250]}")


_NATIVE_CITATION_RE = re.compile(r"【([^】]*)】")
_SOURCE_TAG_RE = re.compile(r"\b(KB|WEB)-(\d+)\b")


def _normalize_citations(text: str) -> str:
    """
    gpt-oss models sometimes emit their native browsing citations, e.g. 【WEB-1†L1-L4】.
    Rewrite them to the app's [WEB-1] form; drop native markers that name no source tag.
    """
    def repl(match: re.Match) -> str:
        tags = [f"[{kind}-{num}]" for kind, num in _SOURCE_TAG_RE.findall(match.group(1))]
        return "".join(dict.fromkeys(tags))  # keep order, drop duplicates

    return _NATIVE_CITATION_RE.sub(repl, text)


def _parse_json_from_llm(text: str) -> Any:
    """Extract valid JSON from LLM output, handling markdown blocks and raw payloads."""
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1)

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    for pattern in [r'\[.*\]', r'\{.*\}']:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue

    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE 1: QUERY ROUTING
# ═══════════════════════════════════════════════════════════════════════════════

def route_query_node(state: ResearchState) -> ResearchState:
    """Classify the incoming question into direct, rag, web, or rag_and_web."""
    _log(state, "routing", "Analyzing the question and checking the knowledge base for relevant documents")
    if state.get("route_override") in VALID_ROUTES:
        state["route"] = state["route_override"]
        state["route_reason"] = "Routing strategy chosen manually by the user."
        _log(state, "routing", f"Route Selected: '{state['route'].upper()}' — {state['route_reason']}", confidence=1.0)
        return state

    kb_status = knowledge_base.get_index_status()
    has_docs = kb_status.get("document_count", 0) > 0 or kb_status.get("total_chunks", 0) > 0

    # Probe the knowledge base so relevant documents are not skipped by keyword-based routing
    kb_top_score = None
    if has_docs:
        try:
            probe = knowledge_base.search_knowledge_base(state["question"], top_k=1, min_score=0.0)
            kb_top_score = probe.get("top_raw_score")
        except Exception:
            kb_top_score = None

    route_info = router.route_query(
        question=state["question"],
        has_indexed_documents=has_docs,
        llm_callable=_call_llm,
        kb_top_score=kb_top_score,
    )

    state["route"] = route_info.get("route", "web")
    state["route_reason"] = route_info.get("reason", "")
    _log(
        state,
        "routing",
        f"Route Selected: '{state['route'].upper()}' — {state['route_reason']}",
        confidence=route_info.get("confidence", 1.0),
    )
    return state


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE 2A: DIRECT RESPONSE (Bypass retrieval)
# ═══════════════════════════════════════════════════════════════════════════════

def direct_response_node(state: ResearchState) -> ResearchState:
    """Answer self-contained or conversational questions without search overhead."""
    _log(state, "direct_answering", "Synthesizing direct response without external tool calls")

    system_prompt = """You are CivicLens, an intelligent, objective research and fact-checking assistant.
Provide a clear, accurate, helpful, and concise answer to the user's question or greeting.
If the question is conversational, respond pleasantly. If it is definitional or conceptual, explain clearly."""

    context_prefix = ""
    if state.get("conversation_context"):
        context_prefix = f"Conversation History:\n{state['conversation_context']}\n\n"

    user_prompt = f"{context_prefix}User Question: {state['question']}"
    reply = _call_llm(system_prompt, user_prompt)

    state["final_report"] = reply
    state["stop_reason"] = "Self-contained query resolved directly without retrieval."

    # Update conversation memory
    sess = memory.get_session_memory(state.get("conversation_id"))
    sess.add_turn(role="user", content=state["question"], route="direct")
    sess.add_turn(role="assistant", content=reply, route="direct")

    _log(state, "complete", "Direct answer generated successfully")
    return state


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE 2B: AUTONOMOUS RESEARCH PLANNER (Context-Aware Dynamic Tool Selection)
# ═══════════════════════════════════════════════════════════════════════════════

def plan_research_node(state: ResearchState) -> ResearchState:
    """Dynamically select required tools, evaluate KB relevance, and construct targeted search queries."""
    route = state.get("route", "web")
    _log(state, "planning", f"Formulating dynamic research plan for route '{route.upper()}'")

    # Inspect what documents actually exist in the Qdrant Knowledge Base
    kb_docs = knowledge_base.get_ingested_documents()
    kb_doc_names = [d.get("name") for d in kb_docs if d.get("name")]
    kb_summary_text = (
        f"Available Indexed Internal Documents ({len(kb_doc_names)}): {', '.join(kb_doc_names)}"
        if kb_doc_names
        else "No internal documents currently indexed in knowledge base."
    )

    system_prompt = f"""You are CivicLens Autonomous Research Planning Agent. Given a research question, routing mode ('{route}'), and internal document catalog, dynamically construct a structured research plan.

{kb_summary_text}

AVAILABLE TOOLS:
- "search_web": Live DuckDuckGo search for current news, external fact-checking, and general knowledge.
- "fetch_webpage": Scrape full content of authoritative web URLs.
- "search_knowledge_base": Dense vector retrieval in Qdrant for internal documents. ONLY select this if the user's question specifically relates to one of the indexed internal documents.
- "search_wikipedia": Background encyclopedic definitions and institutional overviews.
- "search_academic": arXiv scientific papers for academic, AI architecture, medical, or benchmark research.
- "check_source_credibility": Institutional trust and IFCN certification assessment.
- "calculate_statistics": Mathematical verification for quantitative or statistical assertions.

TOOL SELECTION RULES:
1. If the question does NOT relate to any of the indexed internal documents (e.g. general tech, AI releases like Gemini/Claude, international events), do NOT select "search_knowledge_base" — rely on "search_web" and "search_academic".
2. If route is 'rag', focus strictly on "search_knowledge_base".
3. If route is 'rag_and_web', select both "search_knowledge_base" and "search_web" to cross-verify internal assertions with live external reporting.
4. Select "search_academic" only when peer-reviewed or technical research would help (science, engineering, medicine, AI/ML, benchmarks). Skip it for politics, law, elections and current news.
5. Select "search_wikipedia" when background on organisations, laws, places or concepts would help.
Only the tools you list will run, so choose deliberately.

Respond strictly with a JSON object:
{{
    "plan": "Concise 2-3 sentence overview of the research strategy",
    "selected_tools": ["search_web", "fetch_webpage"],
    "rag_rationale": "Explicit reason why internal KB is searched or skipped for this query",
    "sub_objectives": ["Objective 1", "Objective 2"],
    "queries": ["search query 1", "search query 2"]
}}"""

    context_str = ""
    if state.get("conversation_context"):
        context_str = f"\nPrevious context:\n{state['conversation_context']}\n"

    user_prompt = f"Question: {state['question']}{context_str}"
    response = _call_llm(system_prompt, user_prompt)
    parsed = _parse_json_from_llm(response)

    planned_tools = parsed.get("selected_tools") if isinstance(parsed, dict) else None
    if isinstance(planned_tools, list) and any(t in PLANNER_TOOLS for t in planned_tools):
        state["plan_source"] = "llm"
        state["research_plan"] = parsed.get("plan", f"Investigating: {state['question']}")
        state["selected_tools"] = [t for t in planned_tools if t in PLANNER_TOOLS]
        state["rag_rationale"] = parsed.get("rag_rationale", "")
        state["sub_objectives"] = parsed.get("sub_objectives", [])
        state["search_queries"] = parsed.get("queries") or [state["question"]]
    elif isinstance(parsed, dict):
        # Plan text is usable but no valid tool list: keep the plan, let keyword rules pick tools
        state["plan_source"] = "fallback"
        state["research_plan"] = parsed.get("plan", f"Investigating: {state['question']}")
        state["selected_tools"] = ["search_web", "fetch_webpage"]
        state["rag_rationale"] = parsed.get("rag_rationale", "")
        state["sub_objectives"] = parsed.get("sub_objectives", [])
        state["search_queries"] = parsed.get("queries") or [state["question"]]
    else:
        state["plan_source"] = "fallback"
        state["research_plan"] = f"Investigating: {state['question']}"
        state["selected_tools"] = ["search_web", "fetch_webpage"]
        state["rag_rationale"] = "Default web research path."
        state["sub_objectives"] = ["Identify key claims", "Collect corroborating evidence", "Verify facts"]
        state["search_queries"] = [state["question"]]

    _log(
        state,
        "plan_created",
        f"Plan: {state['research_plan']}",
        selected_tools=state["selected_tools"],
        rag_rationale=state["rag_rationale"],
        sub_objectives=state["sub_objectives"],
        queries=state["search_queries"],
    )
    return state


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE 3: DYNAMIC TOOL EXECUTION (Strictly Respecting Agent Tool Selection)
# ═══════════════════════════════════════════════════════════════════════════════

PLANNER_TOOLS = {
    "search_web", "fetch_webpage", "search_knowledge_base", "search_wikipedia",
    "search_academic", "check_source_credibility", "calculate_statistics",
}

# Word-prefix triggers ("research" also matches "researchers"); short "ai" must be a whole word
_ACADEMIC_PREFIX_RE = re.compile(
    r"\b(?:paper|stud(?:y|ies|ied)|research|scien|benchmark|climat|medic|clinical|trial|arxiv|claude)\w*",
    re.IGNORECASE,
)
_ACADEMIC_WORD_RE = re.compile(r"\b(?:ai|a\.i\.|ml|llms?)(?=\W|$)", re.IGNORECASE)


def _has_academic_trigger(question: str) -> bool:
    """Return True if the question mentions scientific, academic or AI topics."""
    return bool(_ACADEMIC_PREFIX_RE.search(question) or _ACADEMIC_WORD_RE.search(question))


def execute_tools_node(state: ResearchState) -> ResearchState:
    """Execute callable tools dynamically according to the agent's explicit tool plan."""
    state["iteration"] = state.get("iteration", 0) + 1
    route = state.get("route", "web")
    queries = state.get("search_queries", [])
    selected_tools = set(state.get("selected_tools", []))

    _log(state, "executing_tools", f"Iteration {state['iteration']}/{state.get('max_iterations', config.MAX_AGENT_ITERATIONS)}")

    # 1. Knowledge Base Retrieval: ONLY if route is 'rag'/'rag_and_web' OR if agent explicitly chose 'search_knowledge_base'
    should_search_kb = (route in ("rag", "rag_and_web")) or ("search_knowledge_base" in selected_tools)
    if should_search_kb:
        _log(state, "tool_call", "Invoking tool: `search_knowledge_base` on Qdrant")
        rag_query = queries[0] if queries else state["question"]
        rag_res = tools.search_knowledge_base(
            rag_query,
            top_k=state.get("rag_top_k") or config.MAX_RAG_PASSAGES,
            min_score=config.MIN_RAG_RELEVANCE_SCORE,
        )
        state["tool_call_count"] = state.get("tool_call_count", 0) + 1

        passages = rag_res.get("results", [])
        if passages:
            # Keep passages from earlier rounds; add only new ones
            known_ids = {p.get("id") for p in state.get("rag_passages", [])}
            new_passages = [p for p in passages if p.get("id") not in known_ids]
            state["rag_passages"] = state.get("rag_passages", []) + new_passages
            _log(state, "rag_retrieved", f"Retrieved {len(new_passages)} new relevant passages from Qdrant knowledge base (top score: {passages[0].get('raw_score', 0):.2f})")
        else:
            msg = rag_res.get("message") or "No passages met minimum relevance threshold"
            _log(state, "rag_empty", f"Qdrant KB: {msg}. Discarded non-relevant candidate vectors.")
    else:
        _log(state, "tool_bypassed", f"Knowledge base search bypassed ({state.get('rag_rationale') or 'Unrelated to internal documents'})")

    # 2. Live Web Searches: if route in ('web', 'rag_and_web') OR if 'search_web' was selected
    should_search_web = (route in ("web", "rag_and_web")) or ("search_web" in selected_tools)
    if should_search_web:
        # The first round may collect MAX_UNIQUE_SOURCES pages; each follow-up round may add MAX_FOLLOWUP_SOURCES more
        round_quota = config.MAX_UNIQUE_SOURCES if state["iteration"] == 1 else config.MAX_FOLLOWUP_SOURCES
        source_limit = len(state.get("unique_source_urls", [])) + round_quota
        for q in queries:
            if len(state.get("unique_source_urls", [])) >= source_limit:
                _log(state, "source_cap", f"Reached this round's source quota ({round_quota} new pages)")
                break
            if state.get("tool_call_count", 0) >= state.get("max_tool_calls", config.MAX_TOOL_CALLS):
                _log(state, "tool_cap", "Reached maximum tool call quota")
                break

            _log(state, "tool_call", f"Invoking tool: `search_web` for query '{q}'")
            search_res = tools.search_web(q, max_results=6)
            state["tool_call_count"] = state.get("tool_call_count", 0) + 1

            for r in search_res.get("results", []):
                u = r.get("url", "")
                if u and u not in state.get("unique_source_urls", []):
                    state.setdefault("search_results", []).append(r)
                    state.setdefault("unique_source_urls", []).append(u)
                    if len(state["unique_source_urls"]) >= source_limit:
                        break

    # 3. Parallel Webpage Content Fetching
    search_results = state.get("search_results", [])
    already_fetched = {s.get("url") for s in state.get("fetched_sources", [])}
    to_fetch = [r for r in search_results if r.get("url") and r["url"] not in already_fetched][:config.MAX_UNIQUE_SOURCES]

    if to_fetch:
        _log(state, "fetching", f"Fetching full text from {len(to_fetch)} web sources in parallel")
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=min(len(to_fetch), 5)) as executor:
            future_to_item = {
                executor.submit(tools.fetch_webpage, item.get("url", "")): item
                for item in to_fetch
            }
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                url = item.get("url", "")
                try:
                    fetch_res = future.result()
                    # Page downloads don't use the tool-call budget, so they can't starve follow-up rounds
                    state["pages_fetched"] = state.get("pages_fetched", 0) + 1
                    if fetch_res.get("success"):
                        cred = tools.check_source_credibility(url)
                        state.setdefault("fetched_sources", []).append({
                            "url": url,
                            "title": item.get("title") or fetch_res.get("title", "Untitled"),
                            "content": fetch_res.get("content", "")[:3500],
                            "date": item.get("date"),
                            "snippet": item.get("snippet", ""),
                            "credibility": cred,
                        })
                        _log(state, "source_fetched", f"✓ Fetched: {item.get('title', url)[:75]} [{cred.get('trust_tier', 'Standard')}]")
                    else:
                        _log(state, "fetch_notice", f"✗ Notice on {url[:60]}: {fetch_res.get('error', 'Could not read content')}")
                except Exception as ex:
                    _log(state, "fetch_notice", f"✗ Error on {url[:60]}: {str(ex)[:80]}")

    # 4. Background Entity Grounding via Wikipedia (if selected or general encyclopedic inquiry)
    # The planner's tool list decides; the old default rules apply only when no LLM plan exists
    planner_decided = state.get("plan_source") == "llm"
    should_search_wiki = "search_wikipedia" in selected_tools or (
        not planner_decided
        and route in ("web", "rag_and_web") and state.get("iteration", 0) == 1 and not state.get("wiki_facts")
    )
    if should_search_wiki and not state.get("wiki_facts"):
        wiki_query = queries[0] if queries else state["question"]
        _log(state, "tool_call", f"Invoking tool: `search_wikipedia` for '{wiki_query[:60]}'")
        wiki_res = tools.search_wikipedia(wiki_query, max_results=2)
        state["tool_call_count"] = state.get("tool_call_count", 0) + 1
        if wiki_res.get("results"):
            state["wiki_facts"] = wiki_res["results"]
            _log(state, "wiki_retrieved", f"Retrieved {len(wiki_res['results'])} background encyclopedic articles from Wikipedia")

    # 5. Scholarly & Academic Paper Search (if AI/scientific/benchmark context or selected)
    is_academic = "search_academic" in selected_tools or (
        not planner_decided and _has_academic_trigger(state["question"])
    )
    if is_academic and not state.get("academic_papers") and route in ("web", "rag_and_web"):
        acad_query = queries[0] if queries else state["question"]
        _log(state, "tool_call", f"Invoking tool: `search_academic` on arXiv repository for '{acad_query[:60]}'")
        acad_res = tools.search_academic(acad_query, max_results=3)
        state["tool_call_count"] = state.get("tool_call_count", 0) + 1
        if acad_res.get("results"):
            state["academic_papers"] = acad_res["results"]
            _log(state, "academic_retrieved", f"Retrieved {len(acad_res['results'])} scholarly research papers from arXiv")

    return state


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE 4: EXTRACT CLAIMS
# ═══════════════════════════════════════════════════════════════════════════════

def extract_claims_node(state: ResearchState) -> ResearchState:
    """Extract independently verifiable claims, dates, and attributions from collected evidence."""
    _log(state, "extracting", "Extracting key factual claims and stakeholder statements")

    # Numbered source list [S1..Sn]: the LLM cites these IDs when judging each claim
    evidence_sources: list[dict[str, Any]] = []

    def add_source(kind: str, name: str, url: str, text: str, note: str = "") -> None:
        evidence_sources.append({"id": f"S{len(evidence_sources) + 1}", "kind": kind, "name": name, "url": url, "text": text, "note": note})

    for p in state.get("rag_passages", []):
        add_source("Knowledge Base (primary document)", f"{p.get('document_name', 'Doc')} (p.{p.get('page_number', '?')})", p.get("url", ""), p.get("text", "")[:1000])
    for s in state.get("fetched_sources", []):
        cred = s.get("credibility") or {}
        note = f"Credibility: {cred.get('trust_tier', 'Standard')}" if cred else ""
        add_source("Web Source", s.get("title", "Unknown"), s.get("url", ""), s.get("content", s.get("snippet", ""))[:1800], note)
    for w in state.get("wiki_facts", []):
        add_source("Wikipedia Background", w.get("title", "Article"), w.get("url", ""), w.get("extract", w.get("snippet", ""))[:1000])
    for a in state.get("academic_papers", []):
        add_source("Academic Paper", a.get("title", "Paper"), a.get("url", ""), a.get("abstract", "")[:1000])

    evidence_sources = evidence_sources[:16]
    state["evidence_sources"] = evidence_sources
    source_texts = [
        f"[{s['id']}] {s['kind']}: {s['name']} ({s['url']}){' — ' + s['note'] if s['note'] else ''}\nContent: {s['text']}\n"
        for s in evidence_sources
    ]

    if not source_texts and state.get("search_results"):
        for r in state.get("search_results", []):
            if r.get("snippet") or r.get("title"):
                source_texts.append(
                    f"Search Snippet: {r.get('title', 'Unknown')} ({r.get('url', '')})\n"
                    f"Snippet: {r.get('snippet', '')}\n"
                )

    if not source_texts:
        _log(state, "extraction_notice", "No source content available for claim extraction")
        state["extracted_claims"] = [
            {"claim": state["question"], "source_name": "User Query", "source_url": "", "type": "allegation", "attributed_to": "User Query"}
        ]
        return state


    combined = "\n---\n".join(source_texts)

    system_prompt = """You are a forensic research analyst. Extract key verifiable factual claims,
figures, legal developments, and attributed positions from the provided sources, then judge
each claim against ALL the numbered sources [S1], [S2], ...

Respond strictly with a JSON array:
[
    {
        "claim": "Specific factual claim or assertion",
        "source_name": "Source title or publisher",
        "source_url": "URL or Document reference",
        "date": "Date if explicitly stated, else null",
        "type": "fact|allegation|opinion|legal_order|statistic",
        "attributed_to": "Official body, person, or organization making the assertion",
        "supporting_sources": ["S1"],
        "contradicting_sources": [],
        "verdict": "supported|partially_supported|contradicted|insufficient",
        "verdict_reason": "One sentence explaining the verdict"
    }
]

JUDGING RULES:
- Judge the claim exactly as stated. For an attributed claim ("X alleged Y"), judge whether
  sources confirm that X said it, not whether Y is true.
- "contradicting_sources" lists ONLY sources that explicitly state something incompatible with
  the claim (a different number, date, outcome, or an explicit denial of it). A source that merely
  mentions protests, criticism, rejected petitions or disagreement on the topic does NOT contradict it.
- "supported": confirmed by a primary document (Knowledge Base, official order, court judgment)
  or by two or more independent sources.
- "partially_supported": confirmed by only one secondary source, or only part of the claim is confirmed.
- "contradicted": at least one credible source explicitly contradicts it.
- "insufficient": no source confirms or contradicts it.

Extract 4 to 12 precise claims. Distinguish facts from allegations. Do NOT wrap in markdown code blocks."""

    user_prompt = f"Research Question: {state['question']}\n\nEvidence Sources:\n{combined}"
    response = _call_llm(system_prompt, user_prompt)
    parsed = _parse_json_from_llm(response)

    if parsed and isinstance(parsed, list):
        state["extracted_claims"] = parsed
        _log(state, "claims_extracted", f"Extracted {len(parsed)} verifiable claims from collected sources")
    else:
        # Fallback to local heuristic extractor
        fallback_claims = tools.extract_claims(combined)
        state["extracted_claims"] = fallback_claims
        _log(state, "claims_extracted", f"Extracted {len(fallback_claims)} claims using fallback parser")

    return state


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE 5: EVIDENCE EVALUATION & CONFLICT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

_LLM_VERDICTS = {
    "supported": "Supported",
    "partially_supported": "Partially Supported",
    "contradicted": "Contradicted",
    "insufficient": "Insufficient Evidence",
}


def _comparison_from_llm_judgements(claims: list[dict[str, Any]], sources: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Build the evidence comparison from the per-claim verdicts the LLM gave during claim extraction.
    Returns None when any claim lacks a valid verdict, so the caller can fall back to keyword matching.
    """
    if not claims or not all(str(c.get("verdict", "")).lower() in _LLM_VERDICTS for c in claims):
        return None

    by_id = {s["id"]: s for s in sources}
    analysis = []
    for c in claims:
        verdict = _LLM_VERDICTS[str(c["verdict"]).lower()]
        supporting = [by_id[i] for i in c.get("supporting_sources") or [] if i in by_id]
        contradicting = [by_id[i] for i in c.get("contradicting_sources") or [] if i in by_id]
        related = [
            {
                "evidence_text": s["text"][:350],
                "source": f"[{s['id']}] {s['name']}",
                "url": s["url"],
                "relevance": None,  # judged by the LLM, not by word overlap
                "is_contradiction": is_contra,
            }
            for group, is_contra in ((contradicting, True), (supporting, False))
            for s in group
        ]
        analysis.append({
            "claim": c.get("claim", ""),
            "attributed_to": c.get("attributed_to") or c.get("source_name", "Unknown"),
            "source_url": c.get("source_url", ""),
            "status": "contradicted" if verdict == "Contradicted" else "unsupported" if verdict == "Insufficient Evidence" else "has_related_evidence",
            "verdict": verdict,
            "verdict_reason": c.get("verdict_reason", ""),
            "supporting_count": len(supporting),
            "contradicting_count": len(contradicting),
            "related_evidence_count": len(related),
            "related_evidence": related[:4],
        })

    comparison = tools.summarize_claim_verdicts(analysis)
    comparison["method"] = "llm_judgement"
    return comparison

def evaluate_evidence_node(state: ResearchState) -> ResearchState:
    """Triangulate evidence across sources and detect contradictions or gaps."""
    _log(state, "evaluating", "Evaluating evidence corroboration and checking for source contradictions")

    claims = state.get("extracted_claims", [])

    llm_comparison = _comparison_from_llm_judgements(claims, state.get("evidence_sources", []))
    if llm_comparison is not None:
        comparison = llm_comparison
        state["evidence_comparison"] = comparison
        state["conflict_detected"] = comparison["conflict_detected"]
        state["fact_check_verdict"] = comparison["overall_verdict"]
        _log(
            state,
            "conflict_detected" if comparison["conflict_detected"] else "evaluation_complete",
            f"Claims judged against sources: {comparison['supported_claims']} supported, "
            f"{comparison['partially_supported_claims']} partially supported, "
            f"{comparison['contradicted_claims']} contradicted, {comparison['unsupported_claims']} without evidence",
        )
        return state

    _log(state, "evaluation_fallback", "Claim judgements unavailable; using keyword-overlap evidence matching")
    evidence_items = []

    for s in state.get("fetched_sources", []):
        evidence_items.append({
            "text": s.get("content", s.get("snippet", "")),
            "source": s.get("title", "Unknown"),
            "url": s.get("url", ""),
        })

    for p in state.get("rag_passages", []):
        evidence_items.append({
            "text": p.get("text", ""),
            "source": f"{p.get('document_name', 'Doc')} (p.{p.get('page_number', '?')})",
            "url": p.get("url", ""),
        })

    comparison = tools.compare_evidence(claims, evidence_items)
    state["evidence_comparison"] = comparison
    state["conflict_detected"] = comparison.get("conflict_detected", False)
    state["fact_check_verdict"] = comparison.get("overall_verdict", "Insufficient Evidence")

    if state["conflict_detected"]:
        _log(
            state,
            "conflict_detected",
            f"⚠️ Evidence conflict detected on {len(comparison.get('conflicting_claims', []))} claims. Evaluating disagreement.",
            conflicts=comparison.get("conflicting_claims", []),
        )
    else:
        _log(
            state,
            "evaluation_complete",
            f"Evidence Evaluated: {comparison.get('claims_with_evidence', 0)} substantiated, {comparison.get('unsupported_claims', 0)} unsupported",
        )

    return state


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE 6: REPLAN & REFLECT (Iterative loop decision)
# ═══════════════════════════════════════════════════════════════════════════════

def replan_or_reflect_node(state: ResearchState) -> ResearchState:
    """Assess whether sufficient evidence has been gathered or if targeted follow-up is needed."""
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", config.MAX_AGENT_ITERATIONS)
    tool_calls = state.get("tool_call_count", 0)
    max_tools = state.get("max_tool_calls", config.MAX_TOOL_CALLS)

    # Check bounds
    if iteration >= max_iter:
        state["needs_followup"] = False
        state["stop_reason"] = f"Reached maximum iteration bound ({max_iter})"
        _log(state, "reflection", f"Stopping research: iteration limit reached ({iteration}/{max_iter})")
        return state

    if tool_calls >= max_tools:
        state["needs_followup"] = False
        state["stop_reason"] = f"Reached maximum tool call quota ({max_tools})"
        _log(state, "reflection", f"Stopping research: tool call limit reached ({tool_calls}/{max_tools})")
        return state

    comparison = state.get("evidence_comparison", {})
    unsupported = comparison.get("unsupported_claims", 0)
    total = comparison.get("total_claims", 0)
    conflict = state.get("conflict_detected", False)

    # If conflict detected OR over 40% claims unsupported (or no claims substantiated) on iteration 1, formulate follow-up query
    has_gaps = conflict or (total == 0) or (total > 0 and unsupported >= total * 0.4)
    if has_gaps and iteration < max_iter:

        _log(state, "reflection", "Evidence gaps or contradictions identified. Formulating targeted follow-up research.")

        system_prompt = """You are a research reflection agent.
Based on unresolved contradictions or unsupported claims, suggest ONE highly specific, targeted search query to resolve the gap, or reply "SUFFICIENT" if no more searches are needed.

Respond strictly with JSON:
{"decision": "search" or "sufficient", "query": "targeted search query", "reason": "why this search is needed"}"""

        gap_claims = [a["claim"] for a in comparison.get("analysis", []) if a.get("status") in ("unsupported", "contradicted")][:4]
        user_prompt = (
            f"Original Question: {state['question']}\n"
            f"Iteration: {iteration}/{max_iter}\n"
            f"Conflict Detected: {conflict}\n"
            f"Gaps/Disputes:\n" + "\n".join(f"- {c}" for c in gap_claims)
        )

        try:
            resp = _call_llm(system_prompt, user_prompt)
            parsed = _parse_json_from_llm(resp)
            if parsed and parsed.get("decision") == "search" and parsed.get("query"):
                state["needs_followup"] = True
                state["followup_query"] = parsed["query"]
                state["search_queries"] = [parsed["query"]]
                _log(state, "followup_replan", f"Replanning with targeted query: '{parsed['query']}' ({parsed.get('reason', '')})")
                return state
        except Exception:
            pass

    state["needs_followup"] = False
    state["stop_reason"] = "Sufficient evidence collected across authoritative sources."
    _log(state, "reflection", "Evidence collection deemed sufficient. Proceeding to citation verification & synthesis.")
    return state


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE 7: SELF-VERIFICATION & FINAL REPORT SYNTHESIS
# ═══════════════════════════════════════════════════════════════════════════════

def verify_and_report_node(state: ResearchState) -> ResearchState:
    """Verify citations against actually retrieved sources and generate structured Markdown report."""
    _log(state, "verifying_citations", "Validating inline source references against retrieved source pool")

    # Collect all available source records for verification
    retrieved_sources_pool = []
    for s in state.get("fetched_sources", []):
        retrieved_sources_pool.append({
            "url": s.get("url", ""),
            "title": s.get("title", ""),
        })
    for p in state.get("rag_passages", []):
        retrieved_sources_pool.append({
            "url": p.get("url", ""),
            "title": p.get("document_name", ""),
            "document_name": p.get("document_name", ""),
        })

    # Prepare claim citations
    claims_for_verify = [
        {
            "claim": c.get("claim", ""),
            "source_url": c.get("source_url", ""),
            "source_name": c.get("source_name", ""),
        }
        for c in state.get("extracted_claims", [])
    ]

    verification = tools.verify_citations(claims_for_verify, retrieved_sources_pool)
    state["citation_verification"] = verification

    _log(
        state,
        "citations_verified",
        f"Verified: {verification.get('verified_count', 0)} claims backed by retrieved sources ({verification.get('unverified_count', 0)} unverified)",
    )

    # Synthesize Final Report
    _log(state, "synthesizing", "Synthesizing final verified research report with inline citations")

    # Build structured sources index
    sources_catalog = []
    sources_text_block = []

    # Knowledge base passages
    for i, p in enumerate(state.get("rag_passages", []), 1):
        tag = f"[KB-{i}]"
        name = p.get("document_name", "Document")
        page = p.get("page_number", "?")
        sources_catalog.append(f"{tag} **{name}** (Page {page})")
        sources_text_block.append(f"{tag} {name} (Page {page}): {p.get('text', '')[:400]}")

    # Web sources
    for j, s in enumerate(state.get("fetched_sources", []), 1):
        tag = f"[WEB-{j}]"
        title = s.get("title", "Web Source")
        url = s.get("url", "#")
        date_str = f" ({s.get('date')})" if s.get("date") else ""
        sources_catalog.append(f"{tag} [{title}]({url}){date_str}")
        sources_text_block.append(f"{tag} {title} ({url}): {s.get('content', '')[:600]}")

    claims_text = "\n".join(
        f"- [{c.get('type', 'claim')}] {c.get('claim')} (Attributed to: {c.get('attributed_to')}, Source: {c.get('source_name')})"
        for c in state.get("extracted_claims", [])[:12]
    )

    verdict = state.get("fact_check_verdict", "Evidence Analyzed")
    system_prompt = f"""You are CivicLens, an expert agentic research and fact-checking platform.
Write an authoritative, rigorous, objective research report in GitHub-flavored Markdown.

VERDICT POLICY:
If the user's query is or contains a claim to fact-check, you must prominently state the overall verdict:
**Verdict: Supported** | **Contradicted** | **Partially Supported** | **Insufficient Evidence**
Provide clear justification.

CITATION RULES:
- Use inline citation tags such as [KB-1], [WEB-1] directly after factual statements.
- Write citations ONLY in that plain square-bracket form. Never use 【】 brackets, "†" markers or line ranges like L1-L4.
- Every factual assertion MUST cite a retrieved source tag from the provided pool.
- NEVER invent URLs, documents, dates, or quotations.
- Acknowledge contradictory evidence and unverified claims explicitly.

REQUIRED SECTIONS:
## Executive Summary
Concise 3-4 sentence overview of key findings and verdict.

## Key Factual Findings & Timeline
Detailed, chronological presentation of verified events with inline citations [KB-X] / [WEB-X].

## Stakeholder Positions & Claims
Objective comparison of positions (official statements, protesters, regulatory bodies, independent observers).

## Evidence Evaluation & Discrepancies
Highlight corroborated evidence, identified contradictions, and what remains unverified.

## Sources & Citations
List the numbered sources consulted ([KB-1] ..., [WEB-1] ...)."""

    user_prompt = f"""User Research Question: {state['question']}
Overall Fact-Check Assessment: {verdict}

RETRIEVED SOURCES CATALOG:
{chr(10).join(sources_catalog)}

EXTRACTED EVIDENCE PASSAGES:
{chr(10).join(sources_text_block[:10])}

KEY EXTRACTED CLAIMS:
{claims_text}

Stop Reason: {state.get('stop_reason')}
Iterations: {state.get('iteration', 1)}/{state.get('max_iterations')}
Total Sources: {len(state.get('unique_source_urls', []))} web, {len(state.get('rag_passages', []))} KB passages

Generate the complete, verified research report now."""

    report = _normalize_citations(_call_llm(system_prompt, user_prompt))

    # Append provenance footer
    footer = (
        f"\n\n---\n*Research conducted by CivicLens Platform at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
        f"| Route: `{state.get('route', 'web').upper()}` "
        f"| Iterations: {state.get('iteration', 1)}/{state.get('max_iterations', config.MAX_AGENT_ITERATIONS)} "
        f"| Sources: {len(state.get('unique_source_urls', []))} web, {len(state.get('rag_passages', []))} KB "
        f"| Fact-Check Verdict: **{verdict}***"
    )
    report += footer

    state["final_report"] = report

    # Save to conversation memory
    sess = memory.get_session_memory(state.get("conversation_id"))
    sess.add_turn(
        role="user",
        content=state["question"],
        route=state.get("route", "web"),
        sources_count=len(state.get("unique_source_urls", [])),
        fact_check_verdict=verdict,
    )
    sess.add_turn(
        role="assistant",
        content=report,
        route=state.get("route", "web"),
        sources_count=len(state.get("unique_source_urls", [])),
        fact_check_verdict=verdict,
    )
    sess.summarize_older_turns(llm_callable=_call_llm)

    _log(state, "report_synthesized", "Final verified research report synthesized and saved to session memory")
    return state


# ═══════════════════════════════════════════════════════════════════════════════
#  CONDITIONAL EDGES & ALIASES
# ═══════════════════════════════════════════════════════════════════════════════

def routing_edge(state: ResearchState) -> Literal["direct_response", "plan_research"]:
    """Branch to direct answering if query is self-contained; otherwise plan research."""
    if state.get("route") == "direct":
        return "direct_response"
    return "plan_research"


def should_followup(state: ResearchState) -> Literal["execute_searches", "verify_and_report"]:
    """Conditional edge: loop back for follow-up search or proceed to report."""
    if state.get("needs_followup") and state.get("followup_query"):
        return "execute_searches"
    return "verify_and_report"


# Node function aliases
route_query = route_query_node
direct_response = direct_response_node
plan_research = plan_research_node
execute_searches = execute_tools_node
fetch_and_retrieve = execute_tools_node
extract_claims = extract_claims_node
compare_and_analyze = evaluate_evidence_node
decide_followup = replan_or_reflect_node
verify_and_report = verify_and_report_node


# ═══════════════════════════════════════════════════════════════════════════════
#  GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def build_research_graph() -> StateGraph:
    """Build and compile the full LangGraph stateful research workflow."""
    graph = StateGraph(ResearchState)

    # Nodes
    graph.add_node("route_query", route_query)
    graph.add_node("direct_response", direct_response)
    graph.add_node("plan_research", plan_research)
    graph.add_node("execute_searches", execute_searches)
    graph.add_node("extract_claims", extract_claims)
    graph.add_node("compare_and_analyze", compare_and_analyze)
    graph.add_node("decide_followup", decide_followup)
    graph.add_node("verify_and_report", verify_and_report)

    # Entry
    graph.set_entry_point("route_query")

    # Routing conditional edge
    graph.add_conditional_edges(
        "route_query",
        routing_edge,
        {
            "direct_response": "direct_response",
            "plan_research": "plan_research",
        },
    )

    # Direct route terminates
    graph.add_edge("direct_response", END)

    # Research pipeline edges
    graph.add_edge("plan_research", "execute_searches")
    graph.add_edge("execute_searches", "extract_claims")
    graph.add_edge("extract_claims", "compare_and_analyze")
    graph.add_edge("compare_and_analyze", "decide_followup")

    # Replanning reflection conditional edge
    graph.add_conditional_edges(
        "decide_followup",
        should_followup,
        {
            "execute_searches": "execute_searches",
            "verify_and_report": "verify_and_report",
        },
    )

    # Terminal edge
    graph.add_edge("verify_and_report", END)

    return graph.compile()




# ═══════════════════════════════════════════════════════════════════════════════
#  RUN RESEARCH ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════

def run_research(
    question: str,
    conversation_id: str | None = None,
    callback=None,
    log_callback=None,
    route_override: str | None = None,
    max_iterations: int | None = None,
    rag_top_k: int | None = None,
) -> ResearchState:
    """
    Execute the agentic research workflow.
    `callback(state, node_name)`: Called after each graph node executes.
    `log_callback(entry)`: Called immediately whenever an action or tool is invoked.
    `route_override`: Force 'direct', 'rag', 'web' or 'rag_and_web' instead of auto-routing.
    `max_iterations`: Maximum evidence-gathering rounds; `rag_top_k`: passages pulled from Qdrant.
    """
    global _current_log_callback
    _current_log_callback = log_callback
    graph = build_research_graph()
    state = _initial_state(question, conversation_id=conversation_id)
    if route_override in VALID_ROUTES:
        state["route_override"] = route_override
    if max_iterations:
        state["max_iterations"] = max_iterations
    if rag_top_k:
        state["rag_top_k"] = rag_top_k

    try:
        for event in graph.stream(state, stream_mode="updates"):
            for node_name, update in event.items():
                if isinstance(update, dict):
                    state.update(update)
                if callback:
                    try:
                        callback(state, node_name)
                    except Exception:
                        pass
    except Exception as e:
        state["error"] = f"Workflow Error: {str(e)}"
        _log(state, "error", str(e))
        if callback:
            try:
                callback(state, "error")
            except Exception:
                pass
    finally:
        _current_log_callback = None

    return state
