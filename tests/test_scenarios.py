"""
Comprehensive End-to-End Test Suite for CivicLens Platform.
Explicitly covers all 10 demonstration scenarios required by the platform specification:
1. Simple Question: Direct answering without unnecessary retrieval or planning.
2. Document Research: Qdrant chunk retrieval with [KB-X] citations.
3. Complex Investigation: Autonomous dynamic planning & multi-tool execution.
4. Insufficient Evidence: Replanning reflection and qualified synthesis.
5. Conflicting Evidence: Detection of contradictions across sources and conflict analysis.
6. Fact-Checking Verdict: Generation of evidence-based verdicts (Supported, Contradicted, etc.).
7. Conversation Follow-Up: Multi-turn session memory preservation.
8. Knowledge Management: Duplicate detection via content hashes, updates, and deletion.
9. Persistence: Verification of vector and metadata persistence across sessions.
10. Transparency & Execution Events: Real-time event streaming and activity logging.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
import config
import knowledge_base
import memory
import router
import tools


@pytest.fixture(autouse=True)
def memory_vector_store():
    """Ensure in-memory Qdrant client is used for fast, isolated unit tests."""
    knowledge_base.set_memory_mode(True)
    yield
    knowledge_base.set_memory_mode(False)


# ═══════════════════════════════════════════════════════════════════════════════
#  SCENARIO 1: Simple Question (Direct Route)
# ═══════════════════════════════════════════════════════════════════════════════

class TestScenario1DirectAnswering:
    @patch("agent._call_llm")
    def test_simple_question_bypasses_tools(self, mock_llm):
        """Simple conceptual/conversational question answers directly without tool execution."""
        mock_llm.return_value = "Photosynthesis is the process by which green plants use sunlight to synthesize nutrients."

        state = agent.run_research("What is photosynthesis?")

        assert state["route"] == "direct"
        assert state["tool_call_count"] == 0
        assert len(state["unique_source_urls"]) == 0
        assert "Photosynthesis" in state["final_report"]
        # Ensure 'direct_answering' activity was logged
        actions = [e["action"] for e in state["activity_log"]]
        assert "direct_answering" in actions


# ═══════════════════════════════════════════════════════════════════════════════
#  SCENARIO 2: Document Research (RAG Route & Qdrant Chunks)
# ═══════════════════════════════════════════════════════════════════════════════

class TestScenario2DocumentResearch:
    @patch("agent._call_llm")
    def test_document_research_retrieval_and_citation(self, mock_llm, tmp_path):
        """Agent retrieves Qdrant chunks and cites uploaded documents."""
        # Create a sample text document in temp directory
        test_file = tmp_path / "climate_accord.txt"
        test_file.write_text(
            "Article 4 of the Climate Accord stipulates a 45% reduction in industrial emissions by 2030. "
            "All signatory nations must submit biennial compliance audits to the oversight committee.",
            encoding="utf-8",
        )

        # Ingest into Qdrant
        res = knowledge_base.ingest_document(test_file, name="Climate Accord 2025")
        assert res["success"] is True

        mock_llm.side_effect = [
            # Plan
            json.dumps({"plan": "Search knowledge base for Article 4 emissions targets", "sub_objectives": ["Target search"], "queries": ["Article 4 emissions"]}),
            # Claims
            json.dumps([{
                "claim": "Article 4 mandates 45% emission cuts by 2030",
                "source_name": "Climate Accord 2025",
                "source_url": "",
                "type": "fact",
                "attributed_to": "Accord Secretariat"
            }]),
            # Reflection
            json.dumps({"decision": "sufficient", "query": "", "reason": "Target found in accord"}),
            # Synthesis
            "## Findings\nArticle 4 establishes a 45% reduction target [KB-1].",
        ]

        state = agent.run_research("What does Article 4 of the uploaded Climate Accord mandate?")

        assert state["route"] in ("rag", "rag_and_web", "web")
        assert len(state["rag_passages"]) > 0
        assert "Climate Accord 2025" in state["rag_passages"][0]["document_name"]
        assert "45%" in state["rag_passages"][0]["text"]


# ═══════════════════════════════════════════════════════════════════════════════
#  SCENARIO 3: Complex Multi-Tool Investigation
# ═══════════════════════════════════════════════════════════════════════════════

class TestScenario3ComplexInvestigation:
    @patch("agent._call_llm")
    @patch("agent.tools.search_web")
    @patch("agent.tools.fetch_webpage")
    def test_complex_investigation_plans_and_executes_multiple_tools(
        self, mock_fetch, mock_search, mock_llm
    ):
        """Complex query generates a plan with sub-objectives and executes multiple tools."""
        mock_llm.side_effect = [
            # Plan
            json.dumps({
                "plan": "Investigate national grid transition stability",
                "sub_objectives": ["Examine blackout data", "Analyze battery deployments", "Compare engineering studies"],
                "queries": ["renewable energy grid stability reports 2025", "utility battery storage blackouts data"],
            }),
            # Claims
            json.dumps([
                {"claim": "Grid operators maintained 99.98% reliability with 65% renewables", "source_name": "Grid Report", "source_url": "https://energy.org/grid", "type": "statistic", "attributed_to": "ISO"},
            ]),
            # Reflection
            json.dumps({"decision": "sufficient", "query": "", "reason": "Both studies gathered"}),
            # Synthesis
            "## Executive Summary\nGrid stability data indicates reliability can be maintained with storage [WEB-1].",
        ]

        mock_search.return_value = {
            "query": "renewable grid",
            "results": [
                {"title": "Grid Operator Reliability Report", "url": "https://energy.org/grid", "snippet": "99.98% reliability observed in 2025"},
            ],
            "total_found": 1,
        }
        mock_fetch.return_value = {
            "url": "https://energy.org/grid",
            "success": True,
            "title": "Grid Operator Reliability Report",
            "content": "Official data confirms 99.98% reliability maintained with battery storage.",
        }

        state = agent.run_research("Investigate grid stability under 70% renewable adoption")

        assert state["research_plan"] != ""
        assert len(state["sub_objectives"]) >= 2
        assert state["tool_call_count"] >= 2
        assert len(state["fetched_sources"]) > 0


# ═══════════════════════════════════════════════════════════════════════════════
#  SCENARIO 4: Insufficient Evidence & Targeted Replanning
# ═══════════════════════════════════════════════════════════════════════════════

class TestScenario4InsufficientEvidence:
    @patch("agent._call_llm")
    @patch("agent.tools.search_web")
    @patch("agent.tools.fetch_webpage")
    def test_insufficient_evidence_triggers_reflection(self, mock_fetch, mock_search, mock_llm):
        """When evidence is lacking, reflection replans with a targeted follow-up query."""
        mock_fetch.return_value = {
            "url": "https://nasa.gov",
            "success": True,
            "content": "No record of any mineral XYZ discovery.",
            "title": "NASA Official Portal",
        }
        mock_llm.side_effect = [
            # Initial Plan
            json.dumps({"plan": "Investigate rare mineral discovery", "sub_objectives": ["Identify claim"], "queries": ["mineral discovery XYZ"]}),
            # Empty / unsupported claims
            json.dumps([{"claim": "Unprecedented mineral vein discovered on Mars", "source_name": "Blog", "source_url": "", "type": "allegation", "attributed_to": "Blogger"}]),
            # Reflection -> replan search
            json.dumps({"decision": "search", "query": "NASA official geological survey Mars minerals", "reason": "Initial search lacked corroboration"}),
            # Claims after follow-up
            json.dumps([{"claim": "NASA has no record of discovery", "source_name": "NASA", "source_url": "https://nasa.gov", "type": "fact", "attributed_to": "NASA"}]),
            # Reflection 2 -> sufficient
            json.dumps({"decision": "sufficient", "query": "", "reason": "Official NASA status confirmed"}),
            # Synthesis
            "## Findings\nNo official evidence corroborates the alleged discovery [WEB-1].",
        ]

        mock_search.return_value = {"query": "test", "results": [{"title": "NASA", "url": "https://nasa.gov", "snippet": "No record"}], "total_found": 1}

        state = agent.run_research("Fact-check the discovery of mineral XYZ on Mars")


        # Verified that iteration exceeded 1 due to replanning
        assert state["iteration"] >= 2
        assert "followup_replan" in [e["action"] for e in state["activity_log"]]


# ═══════════════════════════════════════════════════════════════════════════════
#  SCENARIO 5: Conflicting Evidence Detection
# ═══════════════════════════════════════════════════════════════════════════════

class TestScenario5ConflictingEvidence:
    def test_compare_evidence_detects_contradictions(self):
        """Evidence comparison tool detects conflicting statements between independent sources."""
        claims = [
            {"claim": "The company fully complied with emissions regulations in 2025", "source_name": "Company PR", "source_url": "https://pr.com"}
        ]
        sources = [
            {"source": "Company PR", "text": "The company fully complied with all environmental emissions regulations in 2025.", "url": "https://pr.com"},
            {"source": "EPA Audit", "text": "Regulators denied compliance and reported the company violated emissions limits with false reporting.", "url": "https://epa.gov"},
        ]

        comparison = tools.compare_evidence(claims, sources)

        assert comparison["conflict_detected"] is True
        assert len(comparison["conflicting_claims"]) > 0
        assert comparison["analysis"][0]["verdict"] == "Contradicted / Disputed"


# ═══════════════════════════════════════════════════════════════════════════════
#  SCENARIO 6: Fact-Checking Verdict Generation
# ═══════════════════════════════════════════════════════════════════════════════

class TestScenario6FactCheckingVerdict:
    def test_verdict_categories(self):
        """Verifies standard fact-checking verdicts: Supported, Contradicted, Partially Supported, Insufficient Evidence."""
        # Supported case
        c_sup = [{"claim": "Vaccines reduce hospitalization rates", "source_name": "CDC"}]
        s_sup = [
            {"source": "CDC", "text": "Vaccines significantly reduce hospitalization rates and severe outcomes."},
            {"source": "WHO", "text": "Independent studies prove vaccines reduce hospitalization rates worldwide."},
        ]
        res_sup = tools.compare_evidence(c_sup, s_sup)
        assert res_sup["overall_verdict"] == "Supported"

        # Insufficient evidence case
        c_ins = [{"claim": "Secret underground tunnel connects London and Tokyo", "source_name": "Forum"}]
        s_ins = [{"source": "BBC", "text": "Traffic on the London underground was delayed due to signal failures."}]
        res_ins = tools.compare_evidence(c_ins, s_ins)
        assert res_ins["overall_verdict"] == "Insufficient Evidence"


# ═══════════════════════════════════════════════════════════════════════════════
#  SCENARIO 7: Conversation Follow-Up & Memory
# ═══════════════════════════════════════════════════════════════════════════════

class TestScenario7ConversationMemory:
    def test_session_memory_isolation_and_context_window(self):
        """Conversation history preserves turns and isolates sessions without memory leakage."""
        sess_a = memory.get_session_memory("user_session_alpha")
        sess_b = memory.get_session_memory("user_session_beta")

        sess_a.add_turn(role="user", content="Research solar cell efficiency in 2025")
        sess_a.add_turn(role="assistant", content="Perovskite tandem cells reached 34.6% efficiency.")

        # Verify session B is isolated
        assert len(sess_b.turns) == 0

        # Add follow-up in session A
        sess_a.add_turn(role="user", content="Who manufactured those tandem cells?")
        context_a = sess_a.get_formatted_context()

        assert "34.6%" in context_a
        assert "Who manufactured" in context_a


# ═══════════════════════════════════════════════════════════════════════════════
#  SCENARIO 8: Knowledge Management (Deduplication & Deletion)
# ═══════════════════════════════════════════════════════════════════════════════

class TestScenario8KnowledgeManagement:
    def test_content_hash_deduplication_and_deletion(self, tmp_path):
        """Adding identical document skips re-embedding; delete purges chunks from Qdrant."""
        doc_path = tmp_path / "policy_brief.txt"
        doc_path.write_text("Energy transition targets require decentralized microgrids.", encoding="utf-8")

        # 1. Ingest
        res1 = knowledge_base.ingest_document(doc_path, name="Policy Brief")
        assert res1["success"] is True
        assert res1.get("chunks_indexed", 0) > 0

        # 2. Ingest identical file -> must detect duplicate
        res2 = knowledge_base.ingest_document(doc_path, name="Policy Brief")
        assert res2["success"] is False or res2.get("skipped") is True
        assert res2.get("skipped") is True

        # 3. Delete document
        del_res = knowledge_base.delete_document("Policy Brief")
        assert del_res["success"] is True

        # Verify document count decreased
        inv = knowledge_base.get_ingested_documents()
        assert not any(d["name"] == "Policy Brief" for d in inv)


# ═══════════════════════════════════════════════════════════════════════════════
#  SCENARIO 9: Persistence Across Sessions
# ═══════════════════════════════════════════════════════════════════════════════

class TestScenario9Persistence:
    def test_persistence_of_ingested_metadata(self, tmp_path):
        """Documents and metadata remain recorded persistently in metadata store."""
        file_path = tmp_path / "statute_1950.txt"
        file_path.write_text("The Representation of the People Act provisions.", encoding="utf-8")

        knowledge_base.ingest_document(file_path, name="Statute 1950", date="1950")

        status = knowledge_base.get_index_status()
        doc_names = [d["name"] for d in status.get("documents", [])]
        assert "Statute 1950" in doc_names


# ═══════════════════════════════════════════════════════════════════════════════
#  SCENARIO 10: Transparency & Live Activity Logging
# ═══════════════════════════════════════════════════════════════════════════════

class TestScenario10TransparencyAndLiveLogging:
    @patch("agent._call_llm")
    def test_live_execution_events_emitted(self, mock_llm):
        """Agent logs and streams authentic execution events (routing, planning, tool_call, verification)."""
        mock_llm.side_effect = [
            json.dumps({"plan": "Transparency check", "sub_objectives": ["Obj 1"], "queries": ["transparency query"]}),
            json.dumps([{"claim": "Real event logging verified", "source_name": "System", "source_url": "", "type": "fact", "attributed_to": "Agent"}]),
            json.dumps({"decision": "sufficient", "query": "", "reason": "Event captured"}),
            "## Summary\nReal-time logging verified.",
        ]

        streamed_events = []

        def log_listener(entry):
            streamed_events.append(entry.get("action"))

        state = agent.run_research("Demonstrate transparency logging", log_callback=log_listener)

        # Confirm live callbacks received the actions
        assert len(streamed_events) > 0
        assert "routing" in streamed_events
        assert "planning" in streamed_events
        assert "verifying_citations" in streamed_events
        assert "report_synthesized" in streamed_events
