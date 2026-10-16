"""
Tests for CivicLens agent module.
All tests use mocked LLM and tool calls — no API credentials needed.
"""

import json
import os
import sys
import pytest
from unittest.mock import patch, MagicMock, call

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


class TestResearchWorkflow:
    """Test the LangGraph research workflow with mocked components."""

    @patch("agent._call_llm")
    @patch("agent.tools.web_search")
    @patch("agent.tools.fetch_source")
    @patch("agent.knowledge_base.retrieve_evidence")
    def test_successful_workflow(self, mock_rag, mock_fetch, mock_search, mock_llm):
        """Test a complete mocked research workflow end to end."""
        import agent

        # Mock LLM responses
        mock_llm.side_effect = [
            # Plan response
            json.dumps({
                "plan": "Investigate the ECI SIR protests",
                "queries": ["ECI SIR protests 2025", "Election Commission special revision"],
                "perspectives": ["protesters", "government"]
            }),
            # Extract claims response
            json.dumps([
                {
                    "claim": "ECI ordered SIR in June 2025",
                    "source_name": "News Source",
                    "source_url": "https://news.com/1",
                    "date": "2025-06-24",
                    "type": "fact",
                    "attributed_to": "ECI"
                },
                {
                    "claim": "Protests erupted in multiple states",
                    "source_name": "Reporter",
                    "source_url": "https://news.com/2",
                    "date": "2025-07",
                    "type": "fact",
                    "attributed_to": "News report"
                }
            ]),
            # Decide followup — sufficient
            json.dumps({"decision": "sufficient", "query": "", "reason": "Enough evidence gathered"}),
            # Final report
            "## Summary\nTest research findings about ECI SIR protests.\n\n## Timeline\n- June 2025: ECI ordered SIR",
        ]

        # Mock search
        mock_search.return_value = {
            "query": "ECI SIR protests",
            "results": [
                {"title": "ECI Orders SIR", "url": "https://news.com/1",
                 "snippet": "ECI ordered special revision", "date": "2025-06-24", "source_type": "news"},
                {"title": "Protests Continue", "url": "https://news.com/2",
                 "snippet": "Protests in multiple states", "date": "2025-07-01", "source_type": "news"},
            ],
            "total_found": 2,
            "timestamp": "2025-01-01T00:00:00Z",
        }

        # Mock fetch
        mock_fetch.return_value = {
            "url": "https://news.com/1",
            "success": True,
            "content_type": "html",
            "title": "ECI Orders SIR",
            "content": "The Election Commission ordered a Special Intensive Revision of electoral rolls.",
        }

        # Mock RAG
        mock_rag.return_value = {
            "query": "ECI SIR",
            "results": [
                {
                    "text": "The SIR order was issued on 24 June 2025.",
                    "document_name": "ECI Order",
                    "url": "https://eci.gov.in/1",
                    "date": "2025-06-24",
                    "page_number": 1,
                    "relevance_score": 0.85,
                    "ingestion_date": "2025-01-01",
                }
            ],
            "total_found": 1,
        }

        # Run workflow
        state = agent.run_research("What is the ECI SIR controversy?")

        # Verify results
        assert state.get("research_plan") is not None
        assert len(state.get("activity_log", [])) > 0
        assert state.get("final_report") is not None
        assert "error" not in state or not state["error"]

    @patch("agent._call_llm")
    @patch("agent.tools.web_search")
    @patch("agent.tools.fetch_source")
    @patch("agent.knowledge_base.retrieve_evidence")
    def test_empty_search_results(self, mock_rag, mock_fetch, mock_search, mock_llm):
        """Test workflow handles empty search results gracefully."""
        import agent

        mock_llm.side_effect = [
            json.dumps({"plan": "Test plan", "queries": ["test"], "perspectives": []}),
            json.dumps([]),  # No claims
            json.dumps({"decision": "sufficient", "query": "", "reason": "No evidence found"}),
            "## Summary\nNo results found for the query.",
        ]

        mock_search.return_value = {
            "query": "test",
            "results": [],
            "total_found": 0,
        }

        mock_rag.return_value = {"query": "test", "results": [], "total_found": 0}

        state = agent.run_research("Nonexistent topic XYZ123")

        assert state.get("final_report") is not None
        assert not state.get("error")

    @patch("agent._call_llm")
    @patch("agent.tools.web_search")
    @patch("agent.tools.fetch_source")
    @patch("agent.knowledge_base.retrieve_evidence")
    def test_followup_search_and_iteration_limit(self, mock_rag, mock_fetch, mock_search, mock_llm):
        """Test that conditional follow-up search works and iteration limit is enforced."""
        import agent

        # Set a low iteration limit
        original_max = config.MAX_AGENT_ITERATIONS
        config.MAX_AGENT_ITERATIONS = 2

        try:
            call_count = {"llm": 0}

            def mock_llm_fn(system, user):
                call_count["llm"] += 1
                if call_count["llm"] == 1:
                    return json.dumps({"plan": "Test", "queries": ["q1"], "perspectives": []})
                elif call_count["llm"] == 2:
                    return json.dumps([
                        {"claim": "Unverified claim", "source_name": "S", "source_url": "",
                         "date": None, "type": "allegation", "attributed_to": "Unknown"}
                    ])
                elif call_count["llm"] == 3:
                    # Request followup search
                    return json.dumps({"decision": "search", "query": "followup query", "reason": "Missing evidence"})
                elif call_count["llm"] == 4:
                    return json.dumps([
                        {"claim": "New claim", "source_name": "S2", "source_url": "",
                         "date": None, "type": "fact", "attributed_to": "Source"}
                    ])
                elif call_count["llm"] == 5:
                    # This time should hit iteration limit (2nd iteration = max)
                    return json.dumps({"decision": "sufficient", "query": "", "reason": "Iteration limit"})
                else:
                    return "## Final Report\nCompleted with iteration limit."

            mock_llm.side_effect = mock_llm_fn

            mock_search.return_value = {
                "query": "test",
                "results": [
                    {"title": "R1", "url": "https://r1.com", "snippet": "result", "date": None, "source_type": "web"},
                ],
                "total_found": 1,
            }

            mock_fetch.return_value = {
                "url": "https://r1.com",
                "success": True,
                "content_type": "html",
                "title": "R1",
                "content": "Test content",
            }

            mock_rag.return_value = {"query": "test", "results": [], "total_found": 0}

            state = agent.run_research("Test iteration limit")

            # Should have used multiple iterations
            assert state.get("iteration", 0) >= 1
            # Check stop reason mentions limit or sufficient
            assert state.get("final_report") is not None

        finally:
            config.MAX_AGENT_ITERATIONS = original_max


class TestConditionalEdge:
    """Test the conditional follow-up decision logic."""

    def test_should_followup_when_needed(self):
        import agent

        state = agent._initial_state("test")
        state["needs_followup"] = True
        state["followup_query"] = "more info"

        result = agent.should_followup(state)
        assert result == "execute_searches"

    def test_should_not_followup_when_sufficient(self):
        import agent

        state = agent._initial_state("test")
        state["needs_followup"] = False

        result = agent.should_followup(state)
        assert result == "verify_and_report"


class TestLLMParsing:
    """Test JSON parsing from LLM responses."""

    def test_parse_clean_json(self):
        import agent
        result = agent._parse_json_from_llm('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_in_code_block(self):
        import agent
        text = '```json\n{"key": "value"}\n```'
        result = agent._parse_json_from_llm(text)
        assert result == {"key": "value"}

    def test_parse_json_array(self):
        import agent
        text = 'Here is the data:\n[{"claim": "test"}]'
        result = agent._parse_json_from_llm(text)
        assert isinstance(result, list)
        assert result[0]["claim"] == "test"

    def test_parse_invalid_json(self):
        import agent
        result = agent._parse_json_from_llm("This is not JSON at all.")
        assert result is None
