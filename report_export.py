"""
Report Export — builds a PDF of a research run containing the three result views:
Verified Report, Evidence & Claims Matrix, and Sources & Credibility.
"""

from __future__ import annotations

import html
import io
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import markdown
from bs4 import BeautifulSoup, NavigableString, Tag
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle, XPreformatted,
)

import config

# ── Fonts: a Unicode TTF (₹, accents, dashes) when available, else built-in Helvetica ──
_FONT_CANDIDATES = [
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/ariali.ttf", "C:/Windows/Fonts/arialbi.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf"),
    ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf", "/Library/Fonts/Arial Italic.ttf", "/Library/Fonts/Arial Bold Italic.ttf"),
]


def _register_fonts() -> tuple[str, str]:
    for regular, bold, italic, bold_italic in _FONT_CANDIDATES:
        if all(Path(p).exists() for p in (regular, bold, italic, bold_italic)):
            try:
                pdfmetrics.registerFont(TTFont("CL-Regular", regular))
                pdfmetrics.registerFont(TTFont("CL-Bold", bold))
                pdfmetrics.registerFont(TTFont("CL-Italic", italic))
                pdfmetrics.registerFont(TTFont("CL-BoldItalic", bold_italic))
                pdfmetrics.registerFontFamily("CL", normal="CL-Regular", bold="CL-Bold", italic="CL-Italic", boldItalic="CL-BoldItalic")
                return "CL-Regular", "CL-Bold"
            except Exception:
                continue
    return "Helvetica", "Helvetica-Bold"


FONT, FONT_BOLD = _register_fonts()

# ── Colours (match the app) ──
NAVY = colors.HexColor("#2d5a87")
INK = colors.HexColor("#0f172a")
MUTED = colors.HexColor("#64748b")
BORDER = colors.HexColor("#e2e8f0")
SOFT = colors.HexColor("#f8fafc")
VERDICT_COLORS = {
    "Supported": colors.HexColor("#15803d"),
    "Partially Supported": colors.HexColor("#b45309"),
    "Contradicted": colors.HexColor("#b91c1c"),
    "Contradicted / Disputed": colors.HexColor("#b91c1c"),
    "Insufficient Evidence": colors.HexColor("#475569"),
}

# ── Paragraph styles ──
_base = getSampleStyleSheet()
STYLES = {
    "title": ParagraphStyle("title", parent=_base["Title"], fontName=FONT_BOLD, fontSize=20, leading=24, textColor=INK, alignment=TA_LEFT, spaceAfter=4),
    "subtitle": ParagraphStyle("subtitle", fontName=FONT, fontSize=9.5, leading=13, textColor=MUTED, spaceAfter=10),
    "section": ParagraphStyle("section", fontName=FONT_BOLD, fontSize=15, leading=19, textColor=NAVY, spaceBefore=4, spaceAfter=8),
    "h1": ParagraphStyle("h1", fontName=FONT_BOLD, fontSize=14, leading=18, textColor=INK, spaceBefore=10, spaceAfter=5),
    "h2": ParagraphStyle("h2", fontName=FONT_BOLD, fontSize=12.5, leading=16, textColor=INK, spaceBefore=9, spaceAfter=4),
    "h3": ParagraphStyle("h3", fontName=FONT_BOLD, fontSize=11, leading=14, textColor=INK, spaceBefore=7, spaceAfter=3),
    "body": ParagraphStyle("body", fontName=FONT, fontSize=9.5, leading=13.5, textColor=INK, spaceAfter=5),
    "small": ParagraphStyle("small", fontName=FONT, fontSize=8.5, leading=11.5, textColor=MUTED, spaceAfter=3),
    "cell": ParagraphStyle("cell", fontName=FONT, fontSize=8.5, leading=11, textColor=INK),
    "cell_head": ParagraphStyle("cell_head", fontName=FONT_BOLD, fontSize=8.5, leading=11, textColor=INK),
    "quote": ParagraphStyle("quote", fontName=FONT, fontSize=9.5, leading=13.5, textColor=MUTED, leftIndent=10, spaceAfter=5),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=8, leading=10.5, textColor=INK, backColor=SOFT, leftIndent=6, spaceAfter=6),
    "label": ParagraphStyle("label", fontName=FONT_BOLD, fontSize=8, leading=10, textColor=MUTED, spaceBefore=8, spaceAfter=4),
}

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 18 * mm
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN


# LLM output often uses typographic variants the PDF font lacks (rendered as empty boxes)
_CHAR_REPLACEMENTS = str.maketrans({
    "‐": "-", "‑": "-", "‒": "-", "−": "-",            # hyphen variants, minus
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",  # special spaces
    "​": None, "‌": None, "‍": None, "⁠": None, "﻿": None,       # zero-width
})
_FONT_GLYPHS: set[int] | None = None
if FONT == "CL-Regular":
    try:
        _FONT_GLYPHS = set(pdfmetrics.getFont(FONT).face.charToGlyph)
    except Exception:
        _FONT_GLYPHS = None


def _clean(text: Any) -> str:
    """Normalise typographic characters and replace any the font cannot draw."""
    cleaned = str(text or "").translate(_CHAR_REPLACEMENTS)
    if _FONT_GLYPHS is None:
        return cleaned
    return "".join(ch if ord(ch) in _FONT_GLYPHS or ch in "\n\r\t" else "?" for ch in cleaned)


def _esc(text: Any) -> str:
    return html.escape(_clean(text), quote=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  MARKDOWN → REPORTLAB FLOWABLES
# ═══════════════════════════════════════════════════════════════════════════════

_UNSAFE_TAG_RE = re.compile(r"<(?!/?(?:br|b|i|strong|em|sup|sub|u)\b)", re.IGNORECASE)


def _inline(node: Tag | NavigableString) -> str:
    """Convert inline HTML to ReportLab paragraph markup (b, i, links, code, line breaks)."""
    if isinstance(node, NavigableString):
        return _esc(node)
    inner = "".join(_inline(child) for child in node.children)
    name = node.name
    if name in ("strong", "b"):
        return f"<b>{inner}</b>"
    if name in ("em", "i"):
        return f"<i>{inner}</i>"
    if name == "code":
        return f'<font face="Courier">{inner}</font>'
    if name == "a" and node.get("href"):
        return f'<a href="{html.escape(node["href"])}" color="#1d4ed8">{inner}</a>'
    if name == "br":
        return "<br/>"
    if name in ("ul", "ol", "table"):  # block content inside an inline context: flatten
        return " " + _esc(node.get_text(" ", strip=True)) + " "
    return inner


def _list_flowable(node: Tag) -> ListFlowable:
    items = []
    for li in node.find_all("li", recursive=False):
        parts: list[Any] = []
        text_bits = []
        for child in li.children:
            if isinstance(child, Tag) and child.name in ("ul", "ol"):
                if "".join(text_bits).strip():
                    parts.append(Paragraph("".join(text_bits).strip(), STYLES["body"]))
                    text_bits = []
                parts.append(_list_flowable(child))
            elif isinstance(child, Tag) and child.name == "p":
                text_bits.append(_inline(child) + " ")
            else:
                text_bits.append(_inline(child))
        if "".join(text_bits).strip():
            parts.append(Paragraph("".join(text_bits).strip(), STYLES["body"]))
        items.append(ListItem(parts or [Paragraph("", STYLES["body"])], leftIndent=12))
    ordered = node.name == "ol"
    return ListFlowable(
        items,
        bulletType="1" if ordered else "bullet",
        start="1" if ordered else "•",
        leftIndent=14,
        bulletFontName=FONT,
        bulletFontSize=9,
    )


def _table_flowable(node: Tag) -> Table:
    rows = []
    for tr in node.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        rows.append([
            Paragraph(_inline(c), STYLES["cell_head"] if c.name == "th" else STYLES["cell"]) for c in cells
        ])
    if not rows:
        return Spacer(0, 0)
    n_cols = max(len(r) for r in rows)
    rows = [r + [Paragraph("", STYLES["cell"])] * (n_cols - len(r)) for r in rows]
    has_header = bool(node.find("th"))
    table = Table(rows, colWidths=[CONTENT_WIDTH / n_cols] * n_cols, repeatRows=1 if has_header else 0, hAlign="LEFT")
    style = [
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if has_header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), SOFT))
    table.setStyle(TableStyle(style))
    return table


def markdown_to_flowables(md_text: str) -> list[Any]:
    """Render the agent's Markdown report (headings, lists, tables, quotes, code) as PDF flowables."""
    # Keep simple formatting tags (e.g. <br> in table cells); escape any other "<...>" so its text isn't swallowed
    safe_md = _UNSAFE_TAG_RE.sub("&lt;", _clean(md_text))
    soup = BeautifulSoup(markdown.markdown(safe_md, extensions=["tables", "sane_lists"]), "html.parser")
    flow: list[Any] = []
    for node in soup.children:
        if isinstance(node, NavigableString):
            if node.strip():
                flow.append(Paragraph(_esc(node.strip()), STYLES["body"]))
            continue
        name = node.name
        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            style = STYLES["h1"] if name == "h1" else STYLES["h2"] if name == "h2" else STYLES["h3"]
            flow.append(Paragraph(_inline(node), style))
        elif name == "p":
            flow.append(Paragraph(_inline(node), STYLES["body"]))
        elif name in ("ul", "ol"):
            flow.append(_list_flowable(node))
            flow.append(Spacer(0, 4))
        elif name == "table":
            flow.append(_table_flowable(node))
            flow.append(Spacer(0, 8))
        elif name == "blockquote":
            flow.append(Paragraph(_inline(node), STYLES["quote"]))
        elif name == "pre":
            flow.append(XPreformatted(_esc(node.get_text()), STYLES["code"]))
        elif name == "hr":
            flow.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceBefore=6, spaceAfter=6))
        else:
            flow.append(Paragraph(_inline(node), STYLES["body"]))
    return flow


# ═══════════════════════════════════════════════════════════════════════════════
#  PDF SECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _kv_table(rows: list[tuple[str, str]]) -> Table:
    data = [[Paragraph(f"<b>{_esc(k)}</b>", STYLES["cell"]), Paragraph(_esc(v), STYLES["cell"])] for k, v in rows]
    table = Table(data, colWidths=[40 * mm, CONTENT_WIDTH - 40 * mm], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), SOFT),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def _source_block(title: str, meta: str, text: str, tag: str = "", url: str = "") -> KeepTogether:
    title_markup = f'<a href="{html.escape(url)}" color="#1d4ed8">{_esc(title)}</a>' if url else _esc(title)
    head = f"<b>{title_markup}</b>" + (f'  <font color="#64748b">· {_esc(tag)}</font>' if tag else "")
    parts = [Paragraph(head, STYLES["body"])]
    if meta:
        parts.append(Paragraph(_esc(meta), STYLES["small"]))
    if text:
        parts.append(Paragraph(_esc(text), STYLES["cell"]))
    parts.append(Spacer(0, 6))
    return KeepTogether(parts)


def _overview(state: dict[str, Any]) -> list[Any]:
    comp = state.get("evidence_comparison", {}) or {}
    coverage = comp.get("evidence_coverage")
    route_labels = {"direct": "Direct answer", "rag": "Documents only", "web": "Web only", "rag_and_web": "Documents + web"}
    rows = [
        ("Question", state.get("question", "")),
        ("Verdict", state.get("fact_check_verdict") or ("Answered directly" if state.get("route") == "direct" else "Not available")),
        ("Evidence coverage", f"{coverage}% ({comp.get('grounded_claims', 0)} of {comp.get('total_claims', 0)} claims matched to sources)" if coverage is not None else "N/A"),
        ("Search strategy", route_labels.get(state.get("route", ""), state.get("route", ""))),
        ("Sources", f"{len(state.get('rag_passages', []))} document passages · {len(state.get('fetched_sources', []))} web pages · "
                    f"{len(state.get('wiki_facts', []))} Wikipedia · {len(state.get('academic_papers', []))} papers"),
        ("Research rounds", f"{state.get('iteration', 0)} of {state.get('max_iterations', 0)}"),
    ]
    if state.get("research_plan"):
        rows.append(("Research plan", state["research_plan"]))
    return [_kv_table(rows), Spacer(0, 10)]


def _report_section(state: dict[str, Any]) -> list[Any]:
    flow: list[Any] = [Paragraph("1. Verified Report", STYLES["section"])]
    if state.get("final_report"):
        flow += markdown_to_flowables(state["final_report"])
    else:
        flow.append(Paragraph("No report was produced for this run.", STYLES["small"]))
    return flow


def _claims_section(state: dict[str, Any]) -> list[Any]:
    flow: list[Any] = [Paragraph("2. Evidence &amp; Claims Matrix", STYLES["section"])]
    comp = state.get("evidence_comparison", {}) or {}
    analysis = comp.get("analysis", [])
    if not analysis:
        flow.append(Paragraph("No claims were evaluated in this run.", STYLES["small"]))
        return flow

    method = ("Each claim was judged by the AI against all retrieved sources."
              if comp.get("method") == "llm_judgement" else
              "Claims were matched to sources by keyword overlap.")
    counts: dict[str, int] = {}
    for a in analysis:
        counts[a.get("verdict", "Insufficient Evidence")] = counts.get(a.get("verdict", "Insufficient Evidence"), 0) + 1
    flow.append(Paragraph(f"{_esc(method)} " + " · ".join(f"<b>{_esc(k)}</b>: {v}" for k, v in counts.items()), STYLES["small"]))
    flow.append(Spacer(0, 4))

    header = [Paragraph(h, STYLES["cell_head"]) for h in ("#", "Claim", "Verdict", "Evidence")]
    rows = [header]
    row_styles = []
    for idx, a in enumerate(analysis, start=1):
        verdict = a.get("verdict", "Insufficient Evidence")
        color = VERDICT_COLORS.get(verdict, MUTED)
        claim_html = _esc(a.get("claim", ""))
        if a.get("attributed_to"):
            claim_html += f'<br/><font color="#64748b">Attributed to: {_esc(a["attributed_to"])}</font>'
        if a.get("verdict_reason"):
            claim_html += f'<br/><i>Why: {_esc(a["verdict_reason"])}</i>'
        contradicts_label = '<font color="#b91c1c">Contradicts</font>'
        evidence_lines = [
            f"{contradicts_label if ev.get('is_contradiction') else 'Supports'} — {_esc(ev.get('source', 'Source'))}"
            for ev in a.get("related_evidence", [])
        ]
        evidence_html = (
            f"Supporting: {a.get('supporting_count', 0)} · Contradicting: {a.get('contradicting_count', 0)}"
            + ("<br/>" + "<br/>".join(evidence_lines) if evidence_lines else "")
        )
        rows.append([
            Paragraph(str(idx), STYLES["cell"]),
            Paragraph(claim_html, STYLES["cell"]),
            Paragraph(f'<font color="#{color.hexval()[2:]}"><b>{_esc(verdict)}</b></font>', STYLES["cell"]),
            Paragraph(evidence_html, STYLES["cell"]),
        ])
        if idx % 2 == 0:
            row_styles.append(("BACKGROUND", (0, idx), (-1, idx), SOFT))

    table = Table(rows, colWidths=[8 * mm, CONTENT_WIDTH * 0.50, 28 * mm, CONTENT_WIDTH * 0.50 - 36 * mm], repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f7")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        *row_styles,
    ]))
    flow.append(table)
    return flow


def _sources_section(state: dict[str, Any]) -> list[Any]:
    flow: list[Any] = [Paragraph("3. Sources &amp; Credibility", STYLES["section"])]
    kb = state.get("rag_passages", [])
    web = state.get("fetched_sources", [])
    wiki = state.get("wiki_facts", [])
    papers = state.get("academic_papers", [])
    if not (kb or web or wiki or papers):
        flow.append(Paragraph("No sources were retrieved in this run.", STYLES["small"]))
        return flow

    if kb:
        flow.append(Paragraph(f"INTERNAL KNOWLEDGE BASE ({len(kb)} PASSAGES)", STYLES["label"]))
        for i, p in enumerate(kb, start=1):
            flow.append(_source_block(
                f"[KB-{i}] {p.get('document_name', 'Document')}", f"Page {p.get('page_number', '?')}",
                p.get("text", "")[:500], tag=f"Similarity {p.get('raw_score', p.get('relevance_score', 0)):.2f}",
            ))
    if web:
        flow.append(Paragraph(f"LIVE WEB SOURCES ({len(web)} PAGES)", STYLES["label"]))
        for j, ws in enumerate(web, start=1):
            cred = ws.get("credibility", {}) or {}
            flow.append(_source_block(
                f"[WEB-{j}] {ws.get('title', 'Web page')}", ws.get("url", ""),
                (ws.get("snippet") or ws.get("content", ""))[:400],
                tag=f"{cred.get('trust_tier', 'Standard web')} · {cred.get('credibility_score', 60)}/100", url=ws.get("url", ""),
            ))
    if wiki:
        flow.append(Paragraph(f"WIKIPEDIA ({len(wiki)} ARTICLES)", STYLES["label"]))
        for w in wiki:
            flow.append(_source_block(w.get("title", "Article"), w.get("url", ""), (w.get("extract") or w.get("snippet", ""))[:400], url=w.get("url", "")))
    if papers:
        flow.append(Paragraph(f"ACADEMIC PAPERS ({len(papers)})", STYLES["label"]))
        for paper in papers:
            flow.append(_source_block(paper.get("title", "Paper"), f"Published {paper.get('published', 'n/a')}", paper.get("abstract", "")[:400], url=paper.get("url", "")))
    return flow


def build_pdf(state: dict[str, Any]) -> bytes:
    """Return a PDF containing the overview, verified report, claims matrix and sources for one research run."""
    buffer = io.BytesIO()
    generated = datetime.now().astimezone().strftime("%d %b %Y, %H:%M %Z")
    question = " ".join(_clean(state.get("question", "")).split())
    running_title = question if len(question) <= 90 else question[:89] + "…"

    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, PAGE_HEIGHT - 12 * mm, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 12 * mm)
        canvas.setFont(FONT_BOLD, 8)
        canvas.setFillColor(NAVY)
        canvas.drawString(MARGIN, PAGE_HEIGHT - 10 * mm, config.APP_TITLE)
        canvas.setFont(FONT, 8)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 10 * mm, running_title)
        canvas.drawString(MARGIN, 10 * mm, f"Generated {generated}")
        canvas.drawRightString(PAGE_WIDTH - MARGIN, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=18 * mm, bottomMargin=16 * mm,
        title=f"{config.APP_TITLE} research report", author=config.APP_TITLE, subject=question,
    )
    story: list[Any] = [
        Paragraph(f"{_esc(config.APP_TITLE)} Research Report", STYLES["title"]),
        Paragraph(f"Generated {_esc(generated)} · {_esc(config.APP_TITLE)} v{_esc(config.APP_VERSION)}", STYLES["subtitle"]),
        *_overview(state),
        *_report_section(state),
        PageBreak(),
        *_claims_section(state),
        PageBreak(),
        *_sources_section(state),
    ]
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    return buffer.getvalue()
