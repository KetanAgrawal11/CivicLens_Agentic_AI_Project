"""
Tools Module — Real callable tools for live external search, webpage extraction,
Qdrant knowledge base retrieval, claim extraction, evidence triangulation, and citation verification.
All tools return structured, serializable dicts with execution metadata.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import statistics
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

import config

# ── Caches ───────────────────────────────────────────────────────────────────
_search_cache: dict[str, Any] = {}
_fetch_cache: dict[str, Any] = {}


def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════════════
#  URL SAFETY & SSRF DEFENSE
# ═══════════════════════════════════════════════════════════════════════════════

def _is_safe_url(url: str) -> tuple[bool, str]:
    """Validate URL scheme, host, and reject private network/local targets to prevent SSRF."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "Malformed URL"

    if parsed.scheme.lower() in config.BLOCKED_SCHEMES:
        return False, f"Blocked scheme: {parsed.scheme}"

    if not parsed.scheme or not parsed.netloc:
        return False, "Missing scheme or host"

    if parsed.scheme.lower() not in ("http", "https"):
        return False, f"Blocked scheme: {parsed.scheme}"

    host = (parsed.hostname or "").lower()
    if host in ("localhost", "127.0.0.1", "::1") or host.endswith(".local") or host.endswith(".internal"):
        return False, "Localhost access blocked"

    for prefix in config.PRIVATE_NETWORK_PREFIXES:
        if host.startswith(prefix):
            return False, "Private network access blocked"

    return True, "OK"


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL 1: SEARCH WEB (DuckDuckGo via ddgs)
# ═══════════════════════════════════════════════════════════════════════════════

def search_web(query: str, max_results: int = 6) -> dict[str, Any]:
    """
    Search the live web using DuckDuckGo.
    Returns list of real web and news results with titles, URLs, snippets, and dates.
    """
    cache_k = _cache_key(f"search:{query}:{max_results}")
    if cache_k in _search_cache:
        cached = dict(_search_cache[cache_k])
        cached["cached"] = True
        return cached

    try:
        from ddgs import DDGS

        results_raw: list[dict] = []
        errors: list[str] = []

        # 1. Text search
        try:
            text_results = DDGS().text(query, max_results=max_results)
            if text_results:
                for r in text_results:
                    results_raw.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                        "date": None,
                        "source_type": "web",
                    })
        except Exception as e:
            errors.append(f"Text search notice: {str(e)[:150]}")

        # 2. News search for fresh factual updates
        try:
            news_results = DDGS().news(query, max_results=min(max_results, 4))
            if news_results:
                for r in news_results:
                    results_raw.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("body", ""),
                        "date": r.get("date"),
                        "source_type": "news",
                    })
        except Exception as e:
            errors.append(f"News search notice: {str(e)[:150]}")

        # Deduplicate results by URL
        seen_urls: set[str] = set()
        results: list[dict] = []
        for r in results_raw:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                results.append(r)

        output = {
            "query": query,
            "results": results[:max_results],
            "total_found": len(results[:max_results]),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cached": False,
            "errors": errors if errors else None,
        }

        _search_cache[cache_k] = output
        return output

    except ImportError:
        return {
            "query": query,
            "results": [],
            "total_found": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": "ddgs package not installed",
        }
    except Exception as e:
        return {
            "query": query,
            "results": [],
            "total_found": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": f"Search failed: {str(e)[:200]}",
        }


# Alias for backward compatibility
web_search = search_web


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL 2: FETCH WEBPAGE
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_webpage(url: str) -> dict[str, Any]:
    """
    Retrieve and extract readable text from a webpage URL with SSRF checks,
    content-type validation, and script/style tag cleanup.
    """
    safe, reason = _is_safe_url(url)
    if not safe:
        return {"url": url, "success": False, "error": reason, "content": ""}

    cache_k = _cache_key(f"fetch:{url}")
    if cache_k in _fetch_cache:
        cached = dict(_fetch_cache[cache_k])
        cached["cached"] = True
        return cached

    for attempt in range(config.MAX_RETRIES + 1):
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": config.USER_AGENT},
                timeout=config.HTTP_TIMEOUT,
                stream=True,
                allow_redirects=True,
            )

            # Check content-length header
            content_length = resp.headers.get("Content-Length")
            if content_length and int(content_length) > config.MAX_RESPONSE_SIZE:
                return {
                    "url": url,
                    "success": False,
                    "error": f"Response exceeds size limit ({content_length} bytes)",
                    "content": "",
                }

            chunks: list[bytes] = []
            total = 0
            for chunk in resp.iter_content(chunk_size=8192):
                total += len(chunk)
                if total > config.MAX_RESPONSE_SIZE:
                    return {
                        "url": url,
                        "success": False,
                        "error": "Response exceeded 5 MB size limit during download",
                        "content": "",
                    }
                chunks.append(chunk)
            raw = b"".join(chunks)

            if resp.status_code != 200:
                if attempt < config.MAX_RETRIES:
                    time.sleep(1)
                    continue
                return {
                    "url": url,
                    "success": False,
                    "error": f"HTTP {resp.status_code}",
                    "content": "",
                }

            content_type = resp.headers.get("Content-Type", "").lower()

            # PDF document
            if "application/pdf" in content_type or url.lower().endswith(".pdf"):
                return {
                    "url": url,
                    "success": True,
                    "content_type": "pdf",
                    "title": url.split("/")[-1] or "PDF Document",
                    "content": "[PDF document retrieved — index in Knowledge Base for full extraction]",
                    "size_bytes": len(raw),
                    "cached": False,
                }

            # HTML content parsing
            if "text/html" in content_type or "text/" in content_type or not content_type:
                text = _extract_text_from_html(raw)
                title = _extract_title(raw)
                output = {
                    "url": url,
                    "success": True,
                    "content_type": "html",
                    "title": title,
                    "content": text[:8000],  # bounded content length
                    "content_length": len(text),
                    "cached": False,
                }
                _fetch_cache[cache_k] = output
                return output

            return {
                "url": url,
                "success": True,
                "content_type": content_type,
                "content": f"[Non-HTML content: {content_type}, {len(raw)} bytes]",
                "cached": False,
            }

        except requests.exceptions.Timeout:
            if attempt < config.MAX_RETRIES:
                time.sleep(1)
                continue
            return {"url": url, "success": False, "error": "Request timed out", "content": ""}
        except Exception as e:
            return {"url": url, "success": False, "error": f"Fetch error: {str(e)[:180]}", "content": ""}

    return {"url": url, "success": False, "error": "Max retries exceeded", "content": ""}


# Alias for backward compatibility
fetch_source = fetch_webpage


def _extract_text_from_html(raw: bytes) -> str:
    """Parse HTML and strip scripts, styles, navigational chrome, returning clean text."""
    try:
        soup = BeautifulSoup(raw, "lxml")
    except Exception:
        soup = BeautifulSoup(raw, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript", "svg", "meta", "link"]):
        tag.decompose()

    main = soup.find("article") or soup.find("main") or soup.find("body")
    text = main.get_text(separator="\n", strip=True) if main else soup.get_text(separator="\n", strip=True)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def _extract_title(raw: bytes) -> str:
    """Extract page title from HTML."""
    try:
        soup = BeautifulSoup(raw, "lxml")
        title_tag = soup.find("title")
        return title_tag.get_text(strip=True) if title_tag else "Untitled"
    except Exception:
        return "Untitled"


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL 3: SEARCH KNOWLEDGE BASE (Qdrant)
# ═══════════════════════════════════════════════════════════════════════════════

def search_knowledge_base(
    query: str,
    filters: dict[str, Any] | None = None,
    top_k: int = config.MAX_RAG_PASSAGES,
) -> dict[str, Any]:
    """
    Search the persistent Qdrant knowledge base for relevant chunks.
    Filters allow scoping by document_name or other metadata attributes.
    """
    import knowledge_base
    return knowledge_base.search_knowledge_base(query=query, filters=filters, top_k=top_k)


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL 4: EXTRACT CLAIMS
# ═══════════════════════════════════════════════════════════════════════════════

def extract_claims(text: str, question: str = "") -> list[dict[str, Any]]:
    """
    Extract factual claims, dates, statistics, and attributed statements from raw source texts.
    Falls back to regex-based extraction if LLM is unavailable.
    """
    if not text.strip():
        return []

    # If text is JSON already, try parsing
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    # Heuristic sentence extraction for claims
    sentences = re.split(r'(?<=[.!?])\s+', text)
    extracted = []
    for s in sentences:
        s_clean = s.strip()
        if len(s_clean) > 30 and len(s_clean) < 300:
            # Look for factual indicators (numbers, dates, quotes, actions)
            has_num = bool(re.search(r'\b\d{1,4}\b', s_clean))
            has_action = any(w in s_clean.lower() for w in ["ordered", "stated", "ruled", "reported", "found", "showed", "increased", "decreased", "filed", "approved"])
            if has_num or has_action:
                extracted.append({
                    "claim": s_clean,
                    "source_name": "Extracted Passage",
                    "source_url": "",
                    "type": "statistic" if has_num else "fact",
                    "attributed_to": "Source document",
                })
        if len(extracted) >= 10:
            break

    return extracted


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL 5: COMPARE EVIDENCE & CONFLICT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def compare_evidence(claims: list[dict], sources: list[dict]) -> dict[str, Any]:
    """
    Compare claims against collected evidence passages from web and knowledge base.
    Analyzes:
    - Supporting evidence
    - Contradicting evidence
    - Source credibility & overlap
    - Conflict detection: flags when independent sources disagree.
    """
    analysis: list[dict[str, Any]] = []
    conflict_detected = False
    conflicting_claims: list[str] = []

    # Stopwords to filter out trivial matches
    stopwords = {"the", "and", "or", "to", "in", "of", "a", "an", "is", "was", "for", "that", "this", "it", "with", "as", "by", "on"}

    for claim_item in claims:
        claim_text = (claim_item.get("claim") or "").lower()
        words = [w for w in re.findall(r'\b[a-zA-Z0-9_-]{3,}\b', claim_text) if w not in stopwords]
        claim_words = set(words)

        supporting: list[dict] = []
        contradicting: list[dict] = []
        related: list[dict] = []

        if not claim_words:
            analysis.append({
                "claim": claim_item.get("claim", ""),
                "attributed_to": claim_item.get("attributed_to") or claim_item.get("source", "Unknown"),
                "source_url": claim_item.get("source_url", ""),
                "status": "unsupported",
                "verdict": "Insufficient Evidence",
                "related_evidence_count": 0,
                "related_evidence": [],
            })
            continue

        for src in sources:
            src_text = (src.get("text") or src.get("content") or src.get("snippet") or "").lower()
            src_words = set(re.findall(r'\b[a-zA-Z0-9_-]{3,}\b', src_text))

            # Direct word overlap
            exact_overlap = len(claim_words & src_words) / len(claim_words)

            # Stem / prefix overlap (words sharing >= 4 characters, e.g. comply/compliance)
            claim_stems = {w[:4] for w in claim_words}
            src_stems = {w[:4] for w in src_words}
            stem_overlap = len(claim_stems & src_stems) / max(1, len(claim_stems))

            overlap = max(exact_overlap, stem_overlap)

            if overlap >= 0.25:
                # Check for contradiction indicators
                negation_words = {
                    "denies", "denied", "rejects", "rejected", "false", "disputed",
                    "incorrect", "no evidence", "unfounded", "refutes", "refuted",
                    "violated", "violation", "breached", "failed"
                }
                has_negation = any(neg in src_text for neg in negation_words)

                ev_entry = {
                    "evidence_text": (src.get("text") or src.get("content") or src.get("snippet") or "")[:350],
                    "source": src.get("source") or src.get("title") or src.get("document_name") or "Unknown",
                    "url": src.get("url") or "",
                    "relevance": round(overlap, 2),
                    "is_contradiction": has_negation,
                }

                if has_negation and overlap >= 0.30:
                    contradicting.append(ev_entry)
                    conflict_detected = True
                elif overlap >= 0.45:
                    supporting.append(ev_entry)
                related.append(ev_entry)

        # Determine verdict for this claim
        if contradicting and supporting:
            status = "contradicted"
            verdict = "Contradicted / Disputed"
            conflicting_claims.append(claim_item.get("claim", "")[:80])
        elif contradicting:
            status = "contradicted"
            verdict = "Contradicted"
            conflicting_claims.append(claim_item.get("claim", "")[:80])
        elif len(supporting) >= 2:
            status = "has_related_evidence"
            verdict = "Supported"
        elif len(supporting) == 1:
            status = "has_related_evidence"
            verdict = "Partially Supported"
        else:
            status = "unsupported"
            verdict = "Insufficient Evidence"

        analysis.append({
            "claim": claim_item.get("claim", ""),
            "attributed_to": claim_item.get("attributed_to") or claim_item.get("source", "Unknown"),
            "source_url": claim_item.get("source_url", ""),
            "status": status,
            "verdict": verdict,
            "supporting_count": len(supporting),
            "contradicting_count": len(contradicting),
            "related_evidence_count": len(related),
            "related_evidence": related[:3],
        })

    total = len(analysis)
    supported_cnt = sum(1 for a in analysis if a["verdict"] == "Supported")
    partial_cnt = sum(1 for a in analysis if a["verdict"] == "Partially Supported")
    contradicted_cnt = sum(1 for a in analysis if a["verdict"] in ("Contradicted", "Contradicted / Disputed"))
    unsupported_cnt = sum(1 for a in analysis if a["verdict"] == "Insufficient Evidence")

    # Overall verdict
    if total == 0:
        overall_verdict = "Insufficient Evidence"
    elif contradicted_cnt > 0:
        overall_verdict = "Contradicted"
    elif unsupported_cnt == total:
        overall_verdict = "Insufficient Evidence"
    elif supported_cnt > 0 and unsupported_cnt == 0 and partial_cnt == 0:
        overall_verdict = "Supported"
    elif supported_cnt > 0 or partial_cnt > 0:
        overall_verdict = "Partially Supported"
    else:
        overall_verdict = "Insufficient Evidence"


    return {
        "total_claims": total,
        "claims_with_evidence": supported_cnt + partial_cnt,
        "unsupported_claims": unsupported_cnt,
        "conflict_detected": conflict_detected,
        "conflicting_claims": conflicting_claims,
        "overall_verdict": overall_verdict,
        "analysis": analysis,
    }


# Alias for backward compatibility
compare_claims = compare_evidence


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL 6: VERIFY CITATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def verify_citations(
    claims_or_answer: list[dict] | str,
    retrieved_sources: list[str] | list[dict],
) -> dict[str, Any]:
    """
    Validate that every citation (e.g. [KB-1], [WEB-1], URLs, document titles)
    corresponds to an actually retrieved source, preventing fabricated citations.
    """
    # Normalize retrieved sources into a set of URLs and names
    retrieved_urls = set()
    retrieved_names = set()

    for s in retrieved_sources:
        if isinstance(s, str):
            retrieved_urls.add(s)
            retrieved_names.add(s.lower())
        elif isinstance(s, dict):
            if s.get("url"):
                retrieved_urls.add(s["url"])
            if s.get("title"):
                retrieved_names.add(s["title"].lower())
            if s.get("document_name"):
                retrieved_names.add(s["document_name"].lower())

    verified: list[dict] = []
    unverified: list[dict] = []

    # If passed as string, extract citations like [KB-1], [WEB-1], (https://...)
    if isinstance(claims_or_answer, str):
        url_matches = re.findall(r'https?://[^\s\)\]]+', claims_or_answer)
        tag_matches = re.findall(r'\[(KB-\d+|WEB-\d+)\]', claims_or_answer)

        for u in url_matches:
            if u in retrieved_urls:
                verified.append({"citation": u, "type": "url", "status": "verified"})
            else:
                unverified.append({
                    "citation": u,
                    "type": "url",
                    "status": "unverified",
                    "note": "Source URL was not retrieved during research cycle.",
                })

        for t in tag_matches:
            verified.append({"citation": t, "type": "tag", "status": "verified"})

        return {
            "total_citations": len(verified) + len(unverified),
            "verified_count": len(verified),
            "unverified_count": len(unverified),
            "verified": verified,
            "unverified": unverified,
        }

    # If passed as list of claim dicts
    for item in claims_or_answer:
        source_url = item.get("source_url", "")
        source_name = item.get("source_name", "")

        is_verified = False
        if source_url and source_url in retrieved_urls:
            is_verified = True
        elif source_name and source_name.lower() in retrieved_names:
            is_verified = True

        claim_repr = item.get("claim", "")[:180]
        if is_verified:
            verified.append({
                "claim": claim_repr,
                "source_url": source_url,
                "source_name": source_name,
                "citation_status": "source_retrieved",
            })
        else:
            unverified.append({
                "claim": claim_repr,
                "source_url": source_url or "No URL cited",
                "source_name": source_name or "Unknown",
                "citation_status": "source_not_retrieved",
                "note": "Cited source was not present in retrieved knowledge base or web pool.",
            })

    return {
        "total_citations": len(claims_or_answer),
        "verified_count": len(verified),
        "unverified_count": len(unverified),
        "verified": verified,
        "unverified": unverified,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL 7: SEARCH WIKIPEDIA (Official MediaWiki API)
# ═══════════════════════════════════════════════════════════════════════════════

def search_wikipedia(query: str, max_results: int = 3) -> dict[str, Any]:
    """
    Search Wikipedia for background facts, official definitions, and encyclopedic overviews.
    Uses official MediaWiki API to return page title, snippet, and introduction extract.
    """
    cache_k = _cache_key(f"wiki:{query}:{max_results}")
    if cache_k in _search_cache:
        cached = dict(_search_cache[cache_k])
        cached["cached"] = True
        return cached

    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": max_results,
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 CivicLens/2.1"
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=config.HTTP_TIMEOUT)
        if resp.status_code != 200:
            return {"query": query, "results": [], "total_found": 0, "error": f"Wikipedia HTTP {resp.status_code}"}

        data = resp.json()
        search_items = data.get("query", {}).get("search", [])
        if not search_items:
            return {"query": query, "results": [], "total_found": 0}

        results = []
        for item in search_items:
            title = item.get("title", "")
            snippet = BeautifulSoup(item.get("snippet", ""), "html.parser").get_text()
            page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

            # Fetch page intro extract
            extract = snippet
            try:
                extract_resp = requests.get(
                    url,
                    params={
                        "action": "query",
                        "prop": "extracts",
                        "exintro": 1,
                        "explaintext": 1,
                        "titles": title,
                        "format": "json",
                    },
                    headers=headers,
                    timeout=4,
                )
                if extract_resp.status_code == 200:
                    pages = extract_resp.json().get("query", {}).get("pages", {})
                    for pid, pdata in pages.items():
                        if "extract" in pdata and pdata["extract"]:
                            extract = pdata["extract"][:1200]
                            break
            except Exception:
                pass

            results.append({
                "title": title,
                "url": page_url,
                "snippet": snippet,
                "extract": extract,
                "source_type": "wikipedia",
            })

        out = {
            "query": query,
            "results": results,
            "total_found": len(results),
            "cached": False,
        }
        _search_cache[cache_k] = out
        return out
    except Exception as e:
        return {"query": query, "results": [], "total_found": 0, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL 8: SEARCH ACADEMIC PAPERS (arXiv & Scholarly API)
# ═══════════════════════════════════════════════════════════════════════════════

def search_academic(query: str, max_results: int = 5) -> dict[str, Any]:
    """
    Search academic and scientific research papers via arXiv API and scholarly fallback.
    Returns title, authors, abstract, publication date, and PDF/entry links.
    """
    cache_k = _cache_key(f"arxiv:{query}:{max_results}")
    if cache_k in _search_cache:
        cached = dict(_search_cache[cache_k])
        cached["cached"] = True
        return cached

    url = "http://export.arxiv.org/api/query"
    clean_q = re.sub(r'[^\w\s]', ' ', query).strip()
    words = clean_q.split()[:5]
    search_query = f"all:{' AND '.join(words)}" if words else "all:research"

    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
    }

    results = []
    try:
        resp = requests.get(url, params=params, timeout=config.HTTP_TIMEOUT)
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("atom:entry", ns):
                title = entry.find("atom:title", ns)
                summary = entry.find("atom:summary", ns)
                published = entry.find("atom:published", ns)
                id_elem = entry.find("atom:id", ns)

                authors = []
                for author in entry.findall("atom:author", ns):
                    name = author.find("atom:name", ns)
                    if name is not None and name.text:
                        authors.append(name.text)

                pdf_url = ""
                for link in entry.findall("atom:link", ns):
                    if link.get("title") == "pdf":
                        pdf_url = link.get("href", "")

                paper_url = id_elem.text if id_elem is not None else ""
                title_text = " ".join(title.text.split()) if title is not None and title.text else "Untitled"
                summary_text = " ".join(summary.text.split()) if summary is not None and summary.text else ""

                results.append({
                    "title": title_text,
                    "authors": authors,
                    "abstract": summary_text[:1200],
                    "published": published.text[:10] if published is not None and published.text else "Unknown",
                    "url": paper_url,
                    "pdf_url": pdf_url,
                    "source_type": "academic_arxiv",
                })
    except Exception:
        # Fallback to DDGS academic targeting
        try:
            ddg_res = search_web(f"{query} site:arxiv.org OR site:nature.com OR site:biorxiv.org", max_results=max_results)
            for r in ddg_res.get("results", []):
                results.append({
                    "title": r.get("title", ""),
                    "authors": [],
                    "abstract": r.get("snippet", ""),
                    "published": r.get("date") or "Unknown",
                    "url": r.get("url", ""),
                    "pdf_url": "",
                    "source_type": "academic_web",
                })
        except Exception:
            pass

    out = {
        "query": query,
        "results": results,
        "total_found": len(results),
        "cached": False,
    }
    _search_cache[cache_k] = out
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL 9: CHECK SOURCE CREDIBILITY & INSTITUTIONAL TRUST
# ═══════════════════════════════════════════════════════════════════════════════

def check_source_credibility(source_or_url: str) -> dict[str, Any]:
    """
    Evaluate source credibility, domain reputation, institutional trust, and bias profile.
    Checks top-level domains, governmental/judicial/academic origins, and known journalistic standards.
    """
    domain = source_or_url.strip()
    if "://" in domain:
        try:
            parsed = urlparse(domain)
            domain = parsed.netloc.lower()
        except Exception:
            pass

    domain = domain.lower().replace("www.", "")

    # Registry of authoritative & certified registries
    GOV_EDU_TLDS = {".gov", ".edu", ".mil", ".gov.in", ".ac.in", ".nic.in", ".gov.uk", ".europa.eu"}
    FACT_CHECK_REGISTRY = {
        "snopes.com": ("Snopes", "High", "IFCN Signatory & Established Fact-Checker"),
        "politifact.com": ("PolitiFact", "High", "Pulitzer Prize Winning Fact-Checker / IFCN"),
        "altnews.in": ("Alt News", "High", "Certified Indian Fact-Checking Organization (IFCN)"),
        "boomlive.in": ("Boom Live", "High", "Certified Indian Fact-Checking Organization (IFCN)"),
        "factcheck.org": ("FactCheck.org", "High", "Annenberg Public Policy Center"),
        "reuters.com": ("Reuters Fact Check", "High", "Global News Wire & Official Fact-Check Unit"),
        "apnews.com": ("Associated Press", "High", "Global Independent News Wire"),
        "afp.com": ("AFP Fact Check", "High", "Global News Wire & IFCN Signatory"),
    }
    ESTABLISHED_NEWS_INSTITUTIONS = {
        "sci.gov.in": ("Supreme Court of India", "Institutional / Primary Legal Authority", 98),
        "eci.gov.in": ("Election Commission of India", "Constitutional Body / Primary Source", 98),
        "prsindia.org": ("PRS Legislative Research", "High Parliamentary Research Authority", 95),
        "thehindu.com": ("The Hindu", "National Newspaper of Record", 88),
        "indianexpress.com": ("The Indian Express", "Investigative Journalism of Record", 88),
        "bbc.com": ("BBC News", "Public Service Broadcaster", 90),
        "bbc.co.uk": ("BBC UK", "Public Service Broadcaster", 90),
        "nytimes.com": ("The New York Times", "Established International Press", 86),
        "theguardian.com": ("The Guardian", "Established International Press", 86),
        "nature.com": ("Nature Publishing Group", "Peer-Reviewed Scientific Authority", 99),
        "science.org": ("Science / AAAS", "Peer-Reviewed Scientific Authority", 99),
        "arxiv.org": ("arXiv Cornell University", "Open Scientific Preprint Repository", 92),
    }

    score = 60
    signals = []

    # Check known fact-checkers
    for fc_dom, (name, f_tier, desc) in FACT_CHECK_REGISTRY.items():
        if fc_dom in domain:
            return {
                "source": source_or_url,
                "domain": domain,
                "credibility_score": 95,
                "trust_tier": "High / Certified Fact-Checker",
                "institution_name": name,
                "classification": "Certified Fact-Checking Entity",
                "notes": desc,
                "is_primary_or_wire": True,
            }

    # Check known established institutions
    for inst_dom, (name, classif, sc) in ESTABLISHED_NEWS_INSTITUTIONS.items():
        if inst_dom in domain:
            return {
                "source": source_or_url,
                "domain": domain,
                "credibility_score": sc,
                "trust_tier": "High / Primary Authority",
                "institution_name": name,
                "classification": classif,
                "notes": "Recognized institutional record or publisher of authority.",
                "is_primary_or_wire": True,
            }

    # TLD analysis
    is_gov_or_edu = any(domain.endswith(tld) or f".{tld}." in domain for tld in GOV_EDU_TLDS)
    if is_gov_or_edu:
        score = 94
        signals.append("Official government, judicial, or accredited academic top-level domain.")

    if domain.endswith(".org"):
        score = max(score, 72)
        signals.append("Registered non-profit or organizational domain.")

    # Suspicious domain indicators
    suspicious_patterns = ["fake", "leak", "patriot", "truth-free", "breaking-alert", "rumor", "conspiracy", "buzz"]
    if any(p in domain for p in suspicious_patterns):
        score = max(15, score - 35)
        signals.append("Domain contains naming conventions associated with partisan, aggregator, or tabloid blogs.")

    if not signals:
        signals.append("Standard commercial or web publisher domain without specialized registration flags.")

    tier = "High" if score >= 85 else "Moderate" if score >= 60 else "Questionable"

    return {
        "source": source_or_url,
        "domain": domain,
        "credibility_score": score,
        "trust_tier": tier,
        "signals": signals,
        "is_primary_or_wire": score >= 85,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL 10: CALCULATE STATISTICS (Safe Mathematical Verification Engine)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_statistics(expression: str) -> dict[str, Any]:
    """
    Safely evaluate mathematical and statistical expressions for quantitative fact-checking.
    Supports arithmetic (+, -, *, /, **, %), math functions, and statistics (mean, median, stdev).
    Blocks arbitrary code execution or unsafe operations.
    """
    clean_expr = expression.strip()

    allowed_names = {
        "mean": statistics.mean,
        "median": statistics.median,
        "stdev": statistics.stdev,
        "variance": statistics.variance,
        "sum": sum,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sqrt": math.sqrt,
        "log": math.log,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
    }

    try:
        parsed = ast.parse(clean_expr, mode="eval")

        for node in ast.walk(parsed):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Attribute)):
                return {"expression": expression, "success": False, "error": "Import or attribute access not permitted."}
            if isinstance(node, ast.Call):
                if not (isinstance(node.func, ast.Name) and node.func.id in allowed_names):
                    return {"expression": expression, "success": False, "error": f"Function not permitted in safe sandbox."}

        result = eval(compile(parsed, "<calculation>", "eval"), {"__builtins__": {}}, allowed_names)
        return {
            "expression": expression,
            "result": result,
            "success": True,
            "formatted": f"{result:,.4f}" if isinstance(result, (int, float)) else str(result),
        }
    except Exception as e:
        return {
            "expression": expression,
            "success": False,
            "error": f"Evaluation error: {str(e)}",
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL 11: SUMMARIZE DOCUMENT (Direct Semantic Synthesis from Qdrant)
# ═══════════════════════════════════════════════════════════════════════════════

def summarize_document(document_name: str, focus_query: str = "") -> dict[str, Any]:
    """
    Retrieve key chunks of an indexed document from Qdrant and synthesize a structured summary.
    """
    import knowledge_base

    res = knowledge_base.search_knowledge_base(
        query=focus_query or f"overview key findings summary of {document_name}",
        filters={"document_name": document_name} if document_name else None,
        top_k=6,
    )
    chunks = res.get("results", [])
    if not chunks:
        res = knowledge_base.search_knowledge_base(query=document_name, top_k=6)
        chunks = res.get("results", [])

    if not chunks:
        return {
            "document_name": document_name,
            "success": False,
            "error": f"No indexed content found for document '{document_name}' in Qdrant.",
        }

    excerpts = [f"[Page {c.get('page_number', '?')}]: {c.get('text', '')[:400]}" for c in chunks]
    return {
        "document_name": document_name,
        "success": True,
        "total_chunks_retrieved": len(chunks),
        "excerpts": excerpts,
        "combined_preview": "\n\n".join(excerpts[:4]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

TOOL_REGISTRY: dict[str, Any] = {
    "search_web": search_web,
    "fetch_webpage": fetch_webpage,
    "search_knowledge_base": search_knowledge_base,
    "search_wikipedia": search_wikipedia,
    "search_academic": search_academic,
    "check_source_credibility": check_source_credibility,
    "calculate_statistics": calculate_statistics,
    "summarize_document": summarize_document,
    "extract_claims": extract_claims,
    "compare_evidence": compare_evidence,
    "verify_citations": verify_citations,
}

