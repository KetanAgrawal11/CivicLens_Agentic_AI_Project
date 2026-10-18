"""Tests for claim verdicts: overall verdict rule, contradiction detection and LLM judgements."""

import agent
import tools


class TestOverallVerdict:
    def _claims(self, *verdicts):
        return [{"claim": f"claim {i}", "verdict": v} for i, v in enumerate(verdicts)]

    def test_one_contradiction_does_not_override_many_supported(self):
        res = tools.summarize_claim_verdicts(self._claims("Supported", "Supported", "Supported", "Contradicted"))
        assert res["overall_verdict"] == "Partially Supported"
        assert res["conflict_detected"] is True

    def test_contradicted_when_contradictions_dominate(self):
        res = tools.summarize_claim_verdicts(self._claims("Contradicted", "Contradicted", "Supported"))
        assert res["overall_verdict"] == "Contradicted"

    def test_all_supported_and_empty(self):
        assert tools.summarize_claim_verdicts(self._claims("Supported", "Supported"))["overall_verdict"] == "Supported"
        assert tools.summarize_claim_verdicts([])["overall_verdict"] == "Insufficient Evidence"
        assert tools.summarize_claim_verdicts([])["evidence_coverage"] is None


class TestKeywordFallback:
    def test_unrelated_negative_word_is_not_a_contradiction(self):
        """A page that confirms the claim but mentions 'rejected' elsewhere must not contradict it."""
        claims = [{"claim": "The Election Commission ordered the Special Intensive Revision on 24 June 2025"}]
        sources = [{
            "source": "News",
            "text": (
                "The Election Commission ordered the Special Intensive Revision on 24 June 2025. "
                "Separately, the opposition said several petitions were rejected by the high court last year."
            ),
        }]
        res = tools.compare_evidence(claims, sources)
        assert res["analysis"][0]["contradicting_count"] == 0
        assert res["conflict_detected"] is False


class TestCitationNormalization:
    def test_native_gpt_oss_citations_become_plain_tags(self):
        text = "Hiring fell 80% 【WEB-3†L1–L4】; see 【WEB-1†L1; KB-2†L3】 and 【cursor†L1-L2】."
        assert agent._normalize_citations(text) == "Hiring fell 80% [WEB-3]; see [WEB-1][KB-2] and ."

    def test_plain_citations_unchanged(self):
        assert agent._normalize_citations("Fact [KB-1] and [WEB-2].") == "Fact [KB-1] and [WEB-2]."


class TestLlmJudgements:
    SOURCES = [
        {"id": "S1", "name": "ECI Order (p.1)", "url": "", "text": "Order dated 24 June 2025"},
        {"id": "S2", "name": "News", "url": "https://n.example", "text": "Draft rolls published 1 August"},
    ]

    def test_builds_comparison_from_verdicts(self):
        claims = [
            {"claim": "Order issued 24 June", "verdict": "supported", "supporting_sources": ["S1", "S9"], "verdict_reason": "Primary document"},
            {"claim": "Rolls published 2 August", "verdict": "contradicted", "contradicting_sources": ["S2"]},
        ]
        comp = agent._comparison_from_llm_judgements(claims, self.SOURCES)
        assert comp["method"] == "llm_judgement"
        assert comp["analysis"][0]["supporting_count"] == 1  # unknown id S9 ignored
        assert comp["analysis"][1]["related_evidence"][0]["is_contradiction"] is True
        assert comp["overall_verdict"] == "Contradicted"  # 1 contradicted >= 1 supported

    def test_rate_limit_is_retried(self):
        from unittest.mock import MagicMock, patch

        llm = MagicMock()
        llm.invoke.side_effect = [Exception("Error code: 429 - Rate limit reached. Please try again in 2s."), MagicMock(content="OK")]
        with patch("agent._get_llm", return_value=llm), patch("agent.time.sleep") as sleep:
            assert agent._call_llm("system", "user") == "OK"
        sleep.assert_called_once_with(3.0)

    def test_missing_verdict_falls_back(self):
        claims = [{"claim": "Order issued 24 June", "verdict": "supported"}, {"claim": "No verdict here"}]
        assert agent._comparison_from_llm_judgements(claims, self.SOURCES) is None
