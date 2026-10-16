"""
CivicLens — Autonomous Agentic Research and Fact-Checking Platform.
Streamlit application featuring autonomous multi-step reasoning, dynamic tool execution,
evidence triangulation, live Qdrant Cloud vector storage, and citation-backed synthesis.
"""

from __future__ import annotations

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
import router
import tools

# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title=f"{config.APP_TITLE} — Enterprise Agentic AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── High-Fidelity Modern Design System & Custom CSS ──────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Typography & Reset */
    html, body, [class*="css"], .stMarkdown, p, div, span {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        letter-spacing: -0.01em;
    }
    code, pre, .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Container Spacing */
    .block-container {
        padding-top: 1.0rem;
        padding-bottom: 3.5rem;
        max-width: 1480px;
    }

    /* Modern Command Header */
    .veritas-navbar {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.4rem 2.0rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(16px);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .brand-title {
        font-size: 1.9rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .brand-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        font-weight: 500;
        margin-top: 0.2rem;
    }

    /* Status Pills in Navbar */
    .status-badge-group {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        flex-wrap: wrap;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(15, 23, 42, 0.6);
        color: #e2e8f0;
    }
    .status-pill.online {
        background: rgba(16, 185, 129, 0.12);
        color: #10b981;
        border-color: rgba(16, 185, 129, 0.25);
    }
    .status-pill.qdrant {
        background: rgba(59, 130, 246, 0.12);
        color: #38bdf8;
        border-color: rgba(59, 130, 246, 0.25);
    }
    .status-pill.model {
        background: rgba(168, 85, 247, 0.12);
        color: #c084fc;
        border-color: rgba(168, 85, 247, 0.25);
    }
    .status-pill.tools {
        background: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
        border-color: rgba(245, 158, 11, 0.25);
    }
    .pulse-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 8px #10b981;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); opacity: 0.8; }
        50% { transform: scale(1.25); opacity: 1; }
        100% { transform: scale(0.95); opacity: 0.8; }
    }

    /* Hero Question Box */
    .search-card {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.7) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.25);
    }

    /* Preset Prompt Chips */
    .preset-container {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 0.75rem;
    }
    .preset-chip {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 0.35rem 0.85rem;
        border-radius: 8px;
        font-size: 0.82rem;
        color: #cbd5e1;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .preset-chip:hover {
        background: rgba(59, 130, 246, 0.15);
        border-color: #3b82f6;
        color: #ffffff;
    }

    /* Verdict Banners */
    .verdict-card {
        border-radius: 14px;
        padding: 1.4rem 1.8rem;
        margin: 1.2rem 0;
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.2);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.5rem;
        flex-wrap: wrap;
    }
    .verdict-card.supported {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.4) 0%, rgba(6, 95, 70, 0.2) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-left: 6px solid #10b981;
    }
    .verdict-card.contradicted {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.4) 0%, rgba(153, 27, 27, 0.2) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-left: 6px solid #ef4444;
    }
    .verdict-card.partially_supported {
        background: linear-gradient(135deg, rgba(120, 53, 15, 0.4) 0%, rgba(146, 64, 14, 0.2) 100%);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-left: 6px solid #f59e0b;
    }
    .verdict-card.insufficient {
        background: linear-gradient(135deg, rgba(51, 65, 85, 0.4) 0%, rgba(71, 85, 105, 0.2) 100%);
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-left: 6px solid #94a3b8;
    }
    .verdict-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .verdict-desc {
        font-size: 0.92rem;
        color: #cbd5e1;
        margin-top: 0.3rem;
    }

    /* Live Reasoning Pipeline Stepper */
    .stepper-container {
        display: flex;
        justify-content: space-between;
        margin: 1.2rem 0;
        gap: 0.5rem;
        position: relative;
    }
    .step-item {
        flex: 1;
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.75rem 0.9rem;
        text-align: center;
        transition: all 0.2s ease;
    }
    .step-item.active {
        background: rgba(59, 130, 246, 0.15);
        border-color: #3b82f6;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.2);
    }
    .step-item.completed {
        background: rgba(16, 185, 129, 0.1);
        border-color: rgba(16, 185, 129, 0.3);
    }
    .step-num {
        font-size: 0.75rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
    }
    .step-name {
        font-size: 0.88rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-top: 0.2rem;
    }

    /* Activity Stream Logs */
    .activity-terminal {
        background: #090d16;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.83rem;
        max-height: 280px;
        overflow-y: auto;
        color: #e2e8f0;
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.5);
    }
    .log-item {
        padding: 0.35rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        display: flex;
        align-items: flex-start;
        gap: 0.6rem;
    }
    .log-time { color: #64748b; font-size: 0.76rem; }
    .log-badge {
        padding: 0.1rem 0.45rem;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .badge-routing { background: #312e81; color: #a5b4fc; }
    .badge-planning { background: #831843; color: #f472b6; }
    .badge-tool_call { background: #0c4a6e; color: #38bdf8; }
    .badge-fetching { background: #064e3b; color: #4ade80; }
    .badge-rag_retrieved { background: #581c87; color: #c084fc; }
    .badge-conflict { background: #7f1d1d; color: #fca5a5; }
    .badge-synthesis { background: #14532d; color: #86efac; }

    /* Source Box Cards */
    .source-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
        transition: all 0.2s ease;
    }
    .source-card:hover {
        border-color: rgba(59, 130, 246, 0.4);
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
    }
    .source-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.8rem;
    }
    .trust-tag-high {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .trust-tag-mod {
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Qdrant Cloud Hub Metric Cards */
    .cluster-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ── Session State Initialization ─────────────────────────────────────────────
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = f"session_{int(time.time())}"
if "research_state" not in st.session_state:
    st.session_state.research_state = None
if "activity_log" not in st.session_state:
    st.session_state.activity_log = []
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "selected_preset" not in st.session_state:
    st.session_state.selected_preset = "ECI Electoral Roll Revision (SIR)"
if "current_question" not in st.session_state:
    st.session_state.current_question = config.RESEARCH_PRESETS["ECI Electoral Roll Revision (SIR)"]["query"]
if "manual_route" not in st.session_state:
    st.session_state.manual_route = "Autonomous (Auto-Route)"

# Get live stats from Qdrant Cloud
kb_stats = knowledge_base.get_collection_stats()
total_points_live = kb_stats.get("total_points", 0)
backend_type = kb_stats.get("backend", "local")

# ── Enterprise Top Navigation Bar ────────────────────────────────────────────
cloud_status_text = f"Qdrant Cloud: {total_points_live:,} Vectors" if backend_type == "cloud" else f"Local Qdrant: {total_points_live} Chunks"
model_name_short = config.GROQ_MODEL.split("/")[-1] if "/" in config.GROQ_MODEL else config.GROQ_MODEL

st.markdown(f"""
<div class="veritas-navbar">
    <div>
        <div class="brand-title">🛡️ {config.APP_TITLE} <span style="font-size: 0.9rem; font-weight: 600; color: #64748b; letter-spacing: 0.1em;">ENTERPRISE</span></div>
        <div class="brand-subtitle">{config.APP_SUBTITLE} &nbsp;•&nbsp; v{config.APP_VERSION}</div>
    </div>
    <div class="status-badge-group">
        <div class="status-pill online"><div class="pulse-dot"></div> Active Cluster</div>
        <div class="status-pill qdrant">☁️ {cloud_status_text}</div>
        <div class="status-pill model">⚡ {model_name_short}</div>
        <div class="status-pill tools">🛠️ 11 Tools Registered</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Primary Navigation Tabs ──────────────────────────────────────────────────
tab_research, tab_kb, tab_tools, tab_diagnostics = st.tabs([
    "🔬 Agentic Research & Fact-Checking",
    "📚 Qdrant Cloud Knowledge Base",
    "🛠️ Callable Tool Registry (11 Tools)",
    "⚙️ System Diagnostics & Memory",
])

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR: Quick Actions, Active Preset & Session Control
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚡ Live System Status")

    # Qdrant Card
    if backend_type == "cloud":
        st.success(f"☁️ **Qdrant Cloud Cluster**\n\n• Collection: `{config.QDRANT_COLLECTION_NAME}`\n• Vector Points: **{total_points_live:,}**\n• Metric: Cosine (384-dim)")
    else:
        st.warning(f"💾 **Local Qdrant Mode** ({total_points_live} points)")

    # Model Card
    if config.GROQ_API_KEY:
        st.info(f"🧠 **Groq Inference**\n\n• Model: `{config.GROQ_MODEL}`\n• Max Tokens: `{config.GROQ_MAX_TOKENS}`")
    else:
        st.error("⚠️ `GROQ_API_KEY` missing from `.env`")

    st.divider()

    # Session Control
    st.markdown("### 💬 Research Session")
    st.caption(f"Session Token: `{st.session_state.conversation_id[:16]}...`")
    if st.button("🔄 Fresh Session (Reset Memory)", use_container_width=True):
        st.session_state.conversation_id = f"session_{int(time.time())}"
        st.session_state.research_state = None
        st.session_state.activity_log = []
        memory.reset_session(st.session_state.conversation_id)
        st.toast("Started fresh research session with empty memory buffer.")
        st.rerun()

    st.divider()

    # Presets
    st.markdown("### 🎯 Investigation Presets")
    selected_preset = st.selectbox(
        "Load Domain Scenario:",
        options=list(config.RESEARCH_PRESETS.keys()),
        index=list(config.RESEARCH_PRESETS.keys()).index(st.session_state.selected_preset) if st.session_state.selected_preset in config.RESEARCH_PRESETS else 0,
        key="sidebar_preset_dropdown",
    )
    if selected_preset != st.session_state.selected_preset:
        st.session_state.selected_preset = selected_preset
        st.session_state.current_question = config.RESEARCH_PRESETS[selected_preset]["query"]
        st.rerun()

    st.caption(f"**Domain:** {config.RESEARCH_PRESETS[st.session_state.selected_preset]['domain']}")
    st.caption(f"_{config.RESEARCH_PRESETS[st.session_state.selected_preset]['description']}_")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1: AGENTIC RESEARCH & FACT-CHECKING WORKSPACE
# ═══════════════════════════════════════════════════════════════════════════════

with tab_research:
    # Search Console Card
    st.markdown("#### 1. Formulate Claim or Research Query")
    
    query_input = st.text_area(
        "Enter claim to verify, factual allegation, or research question:",
        value=st.session_state.current_question,
        height=100,
        placeholder="e.g. 'Has the Supreme Court delivered a ruling regarding voter roll revision procedures in Bihar?'",
        help="Input any complex statement, factual query, or political/scientific assertion.",
    )

    # Clickable Preset Quick-Chips
    col_chips = st.columns(4)
    preset_keys = list(config.RESEARCH_PRESETS.keys())
    if len(preset_keys) >= 4:
        with col_chips[0]:
            if st.button("🏛️ ECI SIR Roll Revision", use_container_width=True):
                st.session_state.current_question = config.RESEARCH_PRESETS[preset_keys[0]]["query"]
                st.rerun()
        with col_chips[1]:
            if st.button("⚖️ Supreme Court Ruling", use_container_width=True):
                st.session_state.current_question = config.RESEARCH_PRESETS[preset_keys[1]]["query"]
                st.rerun()
        with col_chips[2]:
            if st.button("🌍 Climate Accord Targets", use_container_width=True):
                st.session_state.current_question = config.RESEARCH_PRESETS[preset_keys[2]]["query"]
                st.rerun()
        with col_chips[3]:
            if st.button("🤖 Quantum & AI Frontier", use_container_width=True):
                st.session_state.current_question = config.RESEARCH_PRESETS[preset_keys[3]]["query"]
                st.rerun()

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    # Control Parameters Grid
    col_c1, col_c2, col_c3 = st.columns([2, 1.2, 1])
    with col_c1:
        route_options = [
            "Autonomous (Auto-Route)",
            "⚖️ Rigorous Fact-Checking (RAG + Web)",
            "📚 Internal Documents Only (Qdrant RAG)",
            "🌐 Live External Web Search",
            "⚡ Direct Analysis (No Retrieval)",
        ]
        selected_mode = st.selectbox("Investigation Routing Strategy:", options=route_options, index=0)
    with col_c2:
        max_hops = st.slider("Autonomous Reasoning Depth (Hops):", min_value=1, max_value=4, value=2, help="Number of reflection and evidence-gathering iterations.")
    with col_c3:
        top_k_passages = st.slider("Passage Retrieval Depth:", min_value=3, max_value=12, value=config.MAX_RAG_PASSAGES)

    # Primary Action Button
    col_btn, col_empty = st.columns([1.5, 4])
    with col_btn:
        start_investigation = st.button(
            "🚀 Execute Autonomous Investigation",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.is_running or not query_input.strip(),
        )

    # ── Live Execution Pipeline & Stepper ─────────────────────────────────────
    if start_investigation:
        st.session_state.is_running = True
        st.session_state.activity_log = []
        st.session_state.current_question = query_input

        # Stepper Banner
        st.markdown("""
        <div class="stepper-container">
            <div class="step-item active" id="step-route"><div class="step-num">Step 1</div><div class="step-name">🎯 Route & Plan</div></div>
            <div class="step-item" id="step-retrieve"><div class="step-num">Step 2</div><div class="step-name">🔍 Retrieve Evidence</div></div>
            <div class="step-item" id="step-tools"><div class="step-num">Step 3</div><div class="step-name">🛠️ Tool Execution</div></div>
            <div class="step-item" id="step-eval"><div class="step-num">Step 4</div><div class="step-name">⚖️ Triangulation</div></div>
            <div class="step-item" id="step-synth"><div class="step-num">Step 5</div><div class="step-name">📄 Synthesis & Report</div></div>
        </div>
        """, unsafe_allow_html=True)

        progress_bar = st.progress(0.05, text="Initiating multi-agent cognitive architecture...")
        status_box = st.empty()
        plan_expander = st.empty()
        live_log_container = st.empty()

        recent_logs: list[dict[str, Any]] = []

        def on_live_log(entry: dict[str, Any]) -> None:
            recent_logs.append(entry)
            st.session_state.activity_log = recent_logs
            lines_html = []
            for item in recent_logs[-8:]:
                act = item.get("action", "INFO").lower()
                badge_class = (
                    "badge-routing" if "route" in act
                    else "badge-planning" if "plan" in act
                    else "badge-tool_call" if "tool" in act
                    else "badge-fetching" if "fetch" in act
                    else "badge-rag_retrieved" if "rag" in act
                    else "badge-conflict" if "conflict" in act
                    else "badge-synthesis" if "synth" in act or "report" in act
                    else "badge-tool_call"
                )
                ts = item.get("timestamp", "")[11:19]
                lines_html.append(
                    f'<div class="log-item">'
                    f'<span class="log-time">{ts}</span>'
                    f'<span class="log-badge {badge_class}">{act[:14]}</span>'
                    f'<span>{item.get("detail", "")}</span>'
                    f'</div>'
                )
            live_log_container.markdown(
                f'<div class="activity-terminal">{"".join(lines_html)}</div>',
                unsafe_allow_html=True,
            )

        def on_node_update(node_name: str, state: dict[str, Any]) -> None:
            node_progress_map = {
                "route_query": (0.15, "🎯 Analyzing intent and selecting multi-source routing path..."),
                "plan_research": (0.30, "📋 Formulating strategic multi-objective research plan..."),
                "execute_tools": (0.55, "🛠️ Executing Qdrant vector retrieval, Wikipedia grounding, and web search..."),
                "extract_claims": (0.70, "🔍 Extracting independently verifiable claims and stakeholder positions..."),
                "evaluate_evidence": (0.85, "⚖️ Cross-examining sources and detecting contradictions..."),
                "replan_or_reflect": (0.90, "🧠 Reflecting on evidence completeness and research sufficiency..."),
                "synthesize_report": (0.98, "📄 Synthesizing evidence-backed report with verified citations..."),
            }
            if node_name in node_progress_map:
                pct, text = node_progress_map[node_name]
                progress_bar.progress(pct, text=text)

            if node_name == "plan_research" and state.get("research_plan"):
                sub_objs = state.get("sub_objectives", [])
                sub_objs_pills = " ".join(f"`{o}`" for o in sub_objs) if sub_objs else ""
                plan_expander.markdown(
                    f"**📋 Autonomous Plan:** {state['research_plan']}\n\n"
                    f"*Target Objectives:* {sub_objs_pills}\n\n"
                    f"*Target Queries:* {', '.join(f'`{q}`' for q in state.get('search_queries', []))}"
                )

        start_time = time.time()
        try:
            final_state = agent.run_research(
                question=query_input,
                conversation_id=st.session_state.conversation_id,
                callback=on_node_update,
                log_callback=on_live_log,
            )
            st.session_state.research_state = final_state
            total_elapsed = round(time.time() - start_time, 1)
            progress_bar.progress(1.0, text=f"✅ Research concluded in {total_elapsed}s!")
            status_box.success(f"🎉 Agent workflow completed successfully in {total_elapsed} seconds.")
        except Exception as e:
            st.session_state.research_state = {"error": str(e), "activity_log": recent_logs}
            progress_bar.progress(1.0, text="❌ Execution halted.")
            status_box.error(f"❌ Execution Error: {str(e)}")
        finally:
            st.session_state.is_running = False

        st.rerun()

    # ── Display Completed Research Results ───────────────────────────────────
    res_state = st.session_state.research_state
    if res_state and not st.session_state.is_running:
        st.divider()

        if res_state.get("error"):
            st.error(f"❌ **Execution Error:** {res_state['error']}")

        # High-Impact Verdict Hero Card (if fact-check verdict exists)
        verdict = res_state.get("fact_check_verdict", "Insufficient Evidence")
        verdict_clean = verdict.lower()

        if "supported" in verdict_clean and "partially" not in verdict_clean and "not" not in verdict_clean:
            v_class = "supported"
            v_icon = "🟢"
            v_title = "VERIFIED & FACTUALLY SUPPORTED"
            v_desc = "The central claims are substantiated by authoritative primary documents and corroborated multi-source evidence."
        elif "contradicted" in verdict_clean or "false" in verdict_clean or "disputed" in verdict_clean:
            v_class = "contradicted"
            v_icon = "🔴"
            v_title = "CONTRADICTED / FACTUALLY DISPUTED"
            v_desc = "Authoritative evidence contradicts the assertions in the claim or significant official disproof was established."
        elif "partially" in verdict_clean:
            v_class = "partially_supported"
            v_icon = "🟡"
            v_title = "PARTIALLY SUPPORTED"
            v_desc = "Certain elements of the claim are corroborated, while critical specifics lack proof or require vital context."
        else:
            v_class = "insufficient"
            v_icon = "⚪"
            v_title = "INSUFFICIENT EVIDENCE"
            v_desc = "Available authoritative sources do not contain sufficient verified data to confirm or refute the assertion."

        # Verdict Hero Banner
        st.markdown(f"""
        <div class="verdict-card {v_class}">
            <div>
                <div class="verdict-title">{v_icon} {v_title}</div>
                <div class="verdict-desc">{v_desc}</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; font-weight: 700;">Engine Confidence</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #ffffff;">{res_state.get('evidence_comparison', {}).get('confidence_score', 85)}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Metrics Row
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        web_cnt = len(res_state.get("unique_source_urls", []))
        rag_cnt = len(res_state.get("rag_passages", []))
        claims_cnt = len(res_state.get("extracted_claims", []))
        verified_cnt = res_state.get("citation_verification", {}).get("verified_count", 0)
        tools_executed = res_state.get("tool_call_count", 0)

        col_m1.metric("Routing Mode", res_state.get("route", "web").upper())
        col_m2.metric("Qdrant Chunks", rag_cnt)
        col_m3.metric("Web Sources", web_cnt)
        col_m4.metric("Claims Evaluated", claims_cnt)
        col_m5.metric("Verified Citations", verified_cnt)

        # Tabbed Detail Presentation
        subtab_report, subtab_matrix, subtab_sources, subtab_deliberation, subtab_followup = st.tabs([
            "📄 Verified Synthesis & Report",
            "⚖️ Evidence & Claims Matrix",
            "🌐 Sources & Credibility Inspector",
            "🧠 Agent Deliberation Trace",
            "💬 Contextual Follow-up",
        ])

        # Subtab 1: Report
        with subtab_report:
            if res_state.get("research_plan"):
                st.info(f"📋 **Autonomous Plan Executed:** {res_state['research_plan']}")

            if res_state.get("final_report"):
                col_exp1, col_exp2 = st.columns([1.5, 4])
                with col_exp1:
                    st.download_button(
                        "📥 Export Markdown Report",
                        data=res_state["final_report"],
                        file_name=f"civiclens_report_{int(time.time())}.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
                with col_exp2:
                    st.download_button(
                        "📦 Export Full JSON State",
                        data=json.dumps(res_state, indent=2, default=str),
                        file_name=f"civiclens_state_{int(time.time())}.json",
                        mime="application/json",
                    )

                st.markdown(res_state["final_report"])

        # Subtab 2: Claims Matrix
        with subtab_matrix:
            claims = res_state.get("extracted_claims", [])
            comp = res_state.get("evidence_comparison", {})
            eval_claims = comp.get("evaluated_claims", [])

            if eval_claims:
                st.markdown(f"**Identified & Evaluated {len(eval_claims)} Key Claims:**")
                for idx, ec in enumerate(eval_claims, start=1):
                    status = ec.get("status", "unsupported")
                    status_badge = (
                        "🟢 Corroborated" if status == "corroborated"
                        else "🔴 Contradicted" if status == "contradicted"
                        else "🟡 Disputed" if status == "disputed"
                        else "⚪ Unsupported"
                    )
                    with st.expander(f"{idx}. {status_badge} — {ec.get('claim', '')[:90]}...", expanded=(idx <= 3)):
                        st.markdown(f"**Exact Claim:** {ec.get('claim')}")
                        st.markdown(f"**Status:** `{status.upper()}` &nbsp;|&nbsp; **Confidence:** `{ec.get('confidence', 'N/A')}`")
                        if ec.get("corroborating_sources"):
                            st.markdown(f"**Corroborating Sources:** {', '.join(f'`{s}`' for s in ec['corroborating_sources'])}")
                        if ec.get("conflicting_sources"):
                            st.markdown(f"**Conflicting Sources:** {', '.join(f'`{s}`' for s in ec['conflicting_sources'])}")
                        if ec.get("verbatim_evidence"):
                            st.markdown(f"**Key Excerpt:** _{ec['verbatim_evidence'][:350]}_")
            elif claims:
                st.markdown(f"**Extracted Claims ({len(claims)} items):**")
                for idx, c in enumerate(claims, start=1):
                    st.markdown(f"**{idx}.** {c.get('claim')} — _({c.get('source_name', 'Source')})_")
            else:
                st.info("No discrete claims recorded in state.")

        # Subtab 3: Sources & Credibility Inspector
        with subtab_sources:
            st.markdown("#### Retrieved Evidence Sources & Institutional Trust")
            
            # 1. Qdrant KB Passages
            kb_passages = res_state.get("rag_passages", [])
            if kb_passages:
                st.markdown(f"##### 📚 Internal Knowledge Base ({len(kb_passages)} chunks from Qdrant Cloud)")
                for p in kb_passages:
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-header">
                            <div><strong>📄 {p.get('document_name', 'Document')}</strong> &nbsp;•&nbsp; Page {p.get('page_number', '?')}</div>
                            <span class="trust-tag-high">Vector Match: {p.get('relevance_score', 0):.3f}</span>
                        </div>
                        <div style="font-size: 0.88rem; color: #cbd5e1; margin-top: 0.5rem;">{p.get('text', '')[:400]}...</div>
                    </div>
                    """, unsafe_allow_html=True)

            # 2. Web Sources with Credibility Badges
            web_sources = res_state.get("fetched_sources", [])
            if web_sources:
                st.markdown(f"##### 🌐 Live Web Sources ({len(web_sources)} fetched pages)")
                for ws in web_sources:
                    cred = ws.get("credibility", {})
                    trust_score = cred.get("credibility_score", 60)
                    trust_tier = cred.get("trust_tier", "Standard Web")
                    tag_class = "trust-tag-high" if trust_score >= 85 else "trust-tag-mod"

                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-header">
                            <div><strong>🔗 <a href="{ws.get('url', '#')}" target="_blank" style="color: #60a5fa; text-decoration: none;">{ws.get('title', 'Webpage')[:80]}</a></strong></div>
                            <span class="{tag_class}">{trust_tier} ({trust_score}/100)</span>
                        </div>
                        <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 0.2rem;">URL: {ws.get('url', '')[:90]}</div>
                        <div style="font-size: 0.86rem; color: #cbd5e1; margin-top: 0.4rem;">{ws.get('snippet', ws.get('content', ''))[:300]}...</div>
                    </div>
                    """, unsafe_allow_html=True)

            # 3. Wikipedia Facts
            wiki_items = res_state.get("wiki_facts", [])
            if wiki_items:
                st.markdown(f"##### 📖 Wikipedia Grounding ({len(wiki_items)} articles)")
                for wi in wiki_items:
                    st.markdown(f"• **[{wi.get('title')}]({wi.get('url')})**: {wi.get('extract', wi.get('snippet', ''))[:300]}")

            # 4. Academic Papers
            acad_items = res_state.get("academic_papers", [])
            if acad_items:
                st.markdown(f"##### 🔬 Academic Papers ({len(acad_items)} papers)")
                for ai in acad_items:
                    st.markdown(f"• **[{ai.get('title')}]({ai.get('url')})** ({ai.get('published', '')}): {ai.get('abstract', '')[:300]}")

        # Subtab 4: Deliberation Trace
        with subtab_deliberation:
            st.markdown("#### Autonomous Multi-Hop Execution Trace")
            logs = res_state.get("activity_log", [])
            if logs:
                for entry in logs:
                    ts = entry.get("timestamp", "")[11:19]
                    act = entry.get("action", "").upper()
                    it = f"[Hop {entry.get('iteration', 1)}]" if "iteration" in entry else ""
                    st.markdown(f"• `{ts}` {it} **{act}**: {entry.get('detail', '')}")
            else:
                st.info("No trace entries recorded.")

        # Subtab 5: Follow-up
        with subtab_followup:
            st.markdown("#### 💬 Ask Contextual Follow-up")
            st.caption("The agent remembers previous findings, claims, and sources from this research session.")
            followup_input = st.text_input("Follow-up question:", placeholder="e.g. 'What specific statutory sections did the court cite?'")
            if st.button("Submit Follow-up", key="btn_followup"):
                if followup_input.strip():
                    st.session_state.current_question = followup_input
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2: QDRANT CLOUD KNOWLEDGE BASE WORKSPACE
# ═══════════════════════════════════════════════════════════════════════════════

with tab_kb:
    # Top Qdrant Cluster Overview Card
    st.markdown("""
    <div class="cluster-card">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
            <div>
                <h3 style="margin: 0; color: #ffffff;">☁️ Qdrant Cloud Cluster</h3>
                <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.3rem;">
                    Enterprise vector database powering hybrid dense neural retrieval with real-time payload filtering.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    backend_label = "☁️ Qdrant Cloud (Live)" if backend_type == "cloud" else "💾 Local Persistent"
    c_m1.metric("Vector Backend", backend_label)
    c_m2.metric("Target Collection", config.QDRANT_COLLECTION_NAME)
    c_m3.metric("Live Vector Points", f"{total_points_live:,}")
    c_m4.metric("Embedding Model", "all-MiniLM-L6-v2 (384-d)")

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # Explanation Alert for user's Qdrant question
    with st.expander("ℹ️ Why Qdrant Cloud points populate upon document synchronization (Click to Expand)", expanded=True):
        st.markdown(f"""
        > **Qdrant Cloud Synchronization Architecture:**
        > - When you inspect your Qdrant Cloud dashboard at `https://cloud.qdrant.io` for cluster `Agentic_AI_CivicLens`, the collection **`{config.QDRANT_COLLECTION_NAME}`** currently holds **{total_points_live:,} vector points**.
        > - Each uploaded PDF or document is split into overlapping chunks, dense embeddings are computed locally using Sentence-Transformers, and points are upserted into Qdrant Cloud with full metadata payloads (document name, page number, SHA-256 hash, and text excerpt).
        > - Click **⚡ Push / Re-Sync All Documents to Qdrant Cloud** below at any time to verify or update vector points.
        """)

    # One-Click Sync & Reindex Action
    col_sync1, col_sync2 = st.columns([1.5, 3.5])
    with col_sync1:
        if st.button("⚡ Push / Re-Sync All Documents to Qdrant Cloud", type="primary", use_container_width=True):
            with st.spinner("Extracting text from local PDFs, generating embeddings, and upserting points to Qdrant Cloud..."):
                re_res = knowledge_base.reindex_all_documents()
                st.success(f"✓ Synchronized! Points in Qdrant: {re_res.get('total_chunks', 0):,}")
                st.rerun()

    st.divider()

    # Upload New Documents Zone
    st.markdown("### 📤 Upload New Documents to Vector Cloud")
    st.caption("Supported formats: PDF (`.pdf`), Word (`.docx`), Markdown (`.md`), Text (`.txt`), CSV (`.csv`), JSON (`.json`)")

    col_up1, col_up2 = st.columns([3, 1])
    with col_up1:
        uploaded_files = st.file_uploader(
            "Drop documents here or click to browse:",
            type=["pdf", "docx", "txt", "md", "markdown", "json", "csv"],
            accept_multiple_files=True,
            key="cloud_file_uploader",
        )
    with col_up2:
        doc_date = st.text_input("Source Date:", value=datetime.now().strftime("%Y-%m-%d"))

    if uploaded_files:
        st.info(f"📁 **{len(uploaded_files)} file(s)** ready for ingestion.")
        if st.button("📥 Vectorize & Upsert to Qdrant Cloud", use_container_width=True):
            entries = []
            for f in uploaded_files:
                dest = config.DOCUMENTS_DIR / f.name
                dest.write_bytes(f.getvalue())
                entries.append({
                    "path": str(dest),
                    "name": Path(f.name).stem,
                    "date": doc_date,
                })
            with st.spinner("Extracting text, computing SHA-256 hashes, generating embeddings, and pushing to Qdrant Cloud..."):
                ingest_res = knowledge_base.ingest_multiple_documents(entries)
            st.success(f"✅ Ingestion complete! Added: {ingest_res.get('added_count', 0)} | Skipped (duplicate hash): {ingest_res.get('skipped_count', 0)}")
            st.rerun()

    st.divider()

    # Document Catalog
    st.markdown("### 📑 Indexed Document Inventory")
    docs = knowledge_base.get_ingested_documents()

    if docs:
        for d in docs:
            col_doc1, col_doc2, col_doc3, col_doc4 = st.columns([3, 1.5, 1.2, 1])
            with col_doc1:
                st.markdown(f"📄 **{d.get('name', 'Document')}**")
                st.caption(f"SHA-256: `{d.get('hash', 'N/A')[:16]}...` | Format: `{d.get('content_type', '.pdf')}`")
            with col_doc2:
                st.caption(f"Date: `{d.get('date', 'Unknown')}`")
                st.caption(f"Ingested: `{str(d.get('ingested_at', ''))[:10]}`")
            with col_doc3:
                st.caption(f"Points: **{d.get('chunks_count', 'N/A')} chunks**")
            with col_doc4:
                if st.button("🗑️ Delete", key=f"del_cloud_{d.get('name')}"):
                    knowledge_base.delete_document(d.get("name"))
                    st.toast(f"Removed '{d.get('name')}' from Qdrant Cloud.")
                    st.rerun()
            st.markdown("<hr style='margin: 0.3rem 0; opacity: 0.1;'/>", unsafe_allow_html=True)
    else:
        st.info("No documents currently indexed.")

    st.divider()

    # Direct Vector Search Playground
    with st.expander("🔍 Interactive Qdrant Vector Search Sandbox (Test Cloud Retrieval)"):
        st.caption("Execute direct vector similarity searches on Qdrant points with live cosine similarity scoring.")
        v_query = st.text_input("Enter semantic query:", placeholder="e.g. 'deletion of voters without notice under ECI guidelines'")
        if v_query:
            v_res = knowledge_base.search_knowledge_base(v_query, top_k=5)
            st.markdown(f"**Found {v_res.get('total_found', 0)} vector points in Qdrant Cloud:**")
            for idx, pt in enumerate(v_res.get("results", []), start=1):
                st.markdown(f"**{idx}. {pt.get('document_name')}** (Page {pt.get('page_number')}) — Similarity: `{pt.get('relevance_score'):.4f}`")
                st.text(pt.get("text", "")[:350])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3: CALLABLE TOOL REGISTRY (11 Tools Interactive Console)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_tools:
    st.markdown("### 🛠️ Production Callable Tool Suite (11 Registered Tools)")
    st.caption("CivicLens tools are autonomous Python executables that return structured, serializable data.")

    tool_cards = [
        ("search_web", "Live Web & News Retrieval", "Queries DuckDuckGo real-time text and news index for fresh factual updates.", "query, max_results"),
        ("fetch_webpage", "Full Webpage Content Extractor", "Fetches and sanitizes full text from target URLs with SSRF protection.", "url, timeout"),
        ("search_knowledge_base", "Qdrant Cloud Vector Search", "Searches 384-dimensional dense vectors with cosine similarity and metadata filters.", "query, filters, top_k"),
        ("search_wikipedia", "MediaWiki Encyclopedic Grounding", "Retrieves background summaries, definitions, and official overviews from Wikipedia.", "query, max_results"),
        ("search_academic", "arXiv & Scholarly Paper Search", "Searches arXiv and academic publications for scientific, AI, and medical research.", "query, max_results"),
        ("check_source_credibility", "Institutional & Domain Trust Evaluator", "Checks TLDs, IFCN fact-checker registries, and news wire reputations.", "url_or_domain"),
        ("calculate_statistics", "Safe Quantitative Fact-Check Sandbox", "Evaluates mathematical and statistical expressions (percentages, growth, means) safely.", "expression"),
        ("summarize_document", "Semantic Document Summarizer", "Synthesizes key excerpts and findings from an indexed Qdrant document.", "document_name, focus_query"),
        ("extract_claims", "Forensic Claim Extractor", "Extracts discrete, verifiable propositions, dates, and attributions from source texts.", "text"),
        ("compare_evidence", "Evidence Triangulation & Cross-Exam", "Cross-references claims against multi-source evidence to detect corroborations & contradictions.", "claims, evidence_items"),
        ("verify_citations", "Citation Grounding & Anti-Hallucination", "Ensures all citations in report correspond to retrieved source URLs and KB points.", "claims_or_answer, retrieved_sources"),
    ]

    for name, title, desc, args in tool_cards:
        with st.expander(f"⚙️ `{name}` — {title}"):
            st.markdown(f"**Description:** {desc}")
            st.markdown(f"**Callable Arguments:** `{args}`")

    st.divider()

    # Interactive Tool Playground
    st.markdown("#### 🧪 Live Tool Tester")
    col_t1, col_t2 = st.columns([1.5, 3])
    with col_t1:
        test_tool_name = st.selectbox(
            "Select Tool to Test:",
            options=["check_source_credibility", "calculate_statistics", "search_wikipedia", "search_academic", "search_web"],
        )
    with col_t2:
        if test_tool_name == "check_source_credibility":
            t_input = st.text_input("Domain or URL:", value="https://eci.gov.in")
        elif test_tool_name == "calculate_statistics":
            t_input = st.text_input("Math / Stat Expression:", value="(450000 - 320000) / 320000 * 100")
        elif test_tool_name == "search_wikipedia":
            t_input = st.text_input("Wikipedia Query:", value="Supreme Court of India")
        elif test_tool_name == "search_academic":
            t_input = st.text_input("Academic Paper Query:", value="neural network transformer attention")
        else:
            t_input = st.text_input("Web Search Query:", value="Election Commission of India latest order")

    if st.button(f"Run `{test_tool_name}`", use_container_width=True):
        with st.spinner("Executing tool..."):
            if test_tool_name == "check_source_credibility":
                out = tools.check_source_credibility(t_input)
            elif test_tool_name == "calculate_statistics":
                out = tools.calculate_statistics(t_input)
            elif test_tool_name == "search_wikipedia":
                out = tools.search_wikipedia(t_input, max_results=2)
            elif test_tool_name == "search_academic":
                out = tools.search_academic(t_input, max_results=3)
            else:
                out = tools.search_web(t_input, max_results=3)

        st.json(out)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4: SYSTEM DIAGNOSTICS & MEMORY
# ═══════════════════════════════════════════════════════════════════════════════

with tab_diagnostics:
    st.markdown("### ⚙️ System Diagnostics & Conversation Memory")

    # Live Connections Check
    st.markdown("#### API Health & Connections")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        if st.button("🔌 Test Groq Connection", use_container_width=True):
            try:
                test_llm = agent._get_llm()
                from langchain_core.messages import HumanMessage
                res = test_llm.invoke([HumanMessage(content="Hello. Reply with 'OK'.")])
                st.success(f"✓ Groq API Responsive: `{res.content.strip()}`")
            except Exception as ex:
                st.error(f"✗ Groq Connection Error: {str(ex)}")

    with col_d2:
        if st.button("☁️ Test Qdrant Cloud Connection", use_container_width=True):
            try:
                client, backend = knowledge_base.get_qdrant_client()
                collections = client.get_collections()
                c_names = [c.name for c in collections.collections]
                st.success(f"✓ Qdrant Cloud Connected! Collections: `{', '.join(c_names)}`")
            except Exception as ex:
                st.error(f"✗ Qdrant Connection Error: {str(ex)}")

    st.divider()

    # Active Session Memory Viewer
    st.markdown("#### Active Conversation Memory")
    sess_mem = memory.get_session_memory(st.session_state.conversation_id)
    history = sess_mem.get_history()

    if history:
        st.markdown(f"**Current Conversation Turns ({len(history)} messages):**")
        for h in history:
            role = h.get("role", "user").upper()
            content = h.get("content", "")
            st.markdown(f"**[{role}]**: {content[:300]}...")
    else:
        st.info("No prior conversation turns in current session.")
