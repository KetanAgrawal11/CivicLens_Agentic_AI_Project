"""Tests for the PDF export of a research run."""

import pymupdf

import report_export

STATE = {
    "question": "Did the ECI order a house‑to‑house revision?",
    "route": "rag_and_web",
    "fact_check_verdict": "Partially Supported",
    "iteration": 1,
    "max_iterations": 2,
    "research_plan": "Check the order and the judgment.",
    "final_report": (
        "## Executive Summary\nThe ECI ordered a **house‑to‑house** revision [KB‑1].\n\n"
        "| Date | Event |\n|---|---|\n| 24 June 2025 | Order issued [KB-1] |\n\n- Point one\n- Point <two>"
    ),
    "evidence_comparison": {
        "method": "llm_judgement", "total_claims": 2, "grounded_claims": 1, "evidence_coverage": 50,
        "analysis": [
            {"claim": "Order issued on 24 June 2025", "verdict": "Supported", "verdict_reason": "Primary document.",
             "supporting_count": 1, "contradicting_count": 0,
             "related_evidence": [{"source": "[S1] ECI Order", "is_contradiction": False}]},
            {"claim": "65 lakh names removed", "verdict": "Insufficient Evidence", "supporting_count": 0, "contradicting_count": 0},
        ],
    },
    "rag_passages": [{"document_name": "ECI SIR Order", "page_number": 1, "text": "House-to-house verification", "raw_score": 0.62}],
    "fetched_sources": [{"title": "ECI press note", "url": "https://eci.gov.in/x", "snippet": "Draft roll on 1 August",
                         "credibility": {"credibility_score": 95, "trust_tier": "Government"}}],
}


def test_pdf_contains_all_three_sections_and_clean_text():
    pdf = report_export.build_pdf(STATE)
    assert pdf.startswith(b"%PDF")
    text = "".join(page.get_text() for page in pymupdf.open(stream=pdf, filetype="pdf"))
    for heading in ("1. Verified Report", "2. Evidence & Claims Matrix", "3. Sources & Credibility"):
        assert heading in text
    assert "house-to-house" in text          # non-breaking hyphens normalised
    assert "Point <two>" in text             # markup-like text escaped, not dropped
    assert "Primary document." in text and "eci.gov.in" in text


def test_pdf_handles_empty_run():
    pdf = report_export.build_pdf({"question": "Hello", "route": "direct"})
    text = "".join(page.get_text() for page in pymupdf.open(stream=pdf, filetype="pdf"))
    assert "No claims were evaluated" in text and "No sources were retrieved" in text
