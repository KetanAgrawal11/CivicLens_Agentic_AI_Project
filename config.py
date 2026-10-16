"""
Agentic AI Research and Fact-Checking Platform — Configuration & Environment Settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")

# ── Application Branding ─────────────────────────────────────────────────────
APP_TITLE: str = "CivicLens"
APP_SUBTITLE: str = "Autonomous Agentic AI Fact-Checking, Research & Verification Platform"
APP_VERSION: str = "2.1.0"

# ── Groq LLM ────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
GROQ_TEMPERATURE: float = 0.2
GROQ_MAX_TOKENS: int = 4096

# ── Qdrant Vector Store (Cloud + Local Fallback) ──────────────────────────────
QDRANT_URL: str = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "").strip()
QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "agentic_research_kb").strip()

# ── Agent limits & execution bounds ──────────────────────────────────────────
MAX_AGENT_ITERATIONS: int = int(os.getenv("MAX_AGENT_ITERATIONS", "5"))
MAX_UNIQUE_SOURCES: int = int(os.getenv("MAX_UNIQUE_SOURCES", "8"))
MAX_RAG_PASSAGES: int = int(os.getenv("MAX_RAG_PASSAGES", "6"))
MIN_RAG_RELEVANCE_SCORE: float = float(os.getenv("MIN_RAG_RELEVANCE_SCORE", "0.40"))
MAX_TOOL_CALLS: int = int(os.getenv("MAX_TOOL_CALLS", "12"))

# ── Conversation Memory ──────────────────────────────────────────────────────
MAX_CONVERSATION_TOKENS: int = int(os.getenv("MAX_CONVERSATION_TOKENS", "3000"))
MAX_RECENT_TURNS: int = int(os.getenv("MAX_RECENT_TURNS", "6"))

# ── HTTP & Safety ────────────────────────────────────────────────────────────
HTTP_TIMEOUT: int = 8          # seconds per request for snappy fetching
MAX_RESPONSE_SIZE: int = 5_000_000  # 5 MB max page download
MAX_RETRIES: int = 1
USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 CivicLens/2.1"
)

# Blocked URL schemes and private networks (SSRF prevention)
BLOCKED_SCHEMES: set[str] = {"file", "ftp", "data", "javascript", "vbscript"}
PRIVATE_NETWORK_PREFIXES: list[str] = [
    "10.", "172.16.", "172.17.", "172.18.", "172.19.",
    "172.20.", "172.21.", "172.22.", "172.23.", "172.24.",
    "172.25.", "172.26.", "172.27.", "172.28.", "172.29.",
    "172.30.", "172.31.", "192.168.", "127.", "0.",
]

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR: Path = _PROJECT_ROOT / "data"
DOCUMENTS_DIR: Path = DATA_DIR / "documents"
INDEX_DIR: Path = DATA_DIR / "index"
CACHE_DIR: Path = DATA_DIR / "cache"
QDRANT_LOCAL_DIR: Path = DATA_DIR / "qdrant_storage"
METADATA_DB: Path = DATA_DIR / "metadata.json"

# Create directories on import
for _d in (DOCUMENTS_DIR, INDEX_DIR, CACHE_DIR, QDRANT_LOCAL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Embeddings & Document Ingestion ──────────────────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = 384
CHUNK_SIZE: int = 512           # characters per chunk
CHUNK_OVERLAP: int = 64         # overlap between chunks
SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".docx", ".txt", ".md", ".markdown", ".json", ".csv"}

# ── Diverse Research Presets ─────────────────────────────────────────────────
RESEARCH_PRESETS: dict[str, dict[str, str]] = {
    "ECI Electoral Roll Revision (SIR)": {
        "domain": "Electoral Governance & Public Law",
        "description": "Investigates protests, official orders, and legal developments concerning the Election Commission of India's SIR process.",
        "query": (
            "Investigate the protests and legal controversies concerning the Election Commission of India's "
            "Special Intensive Revision (SIR) of electoral rolls. What triggered the protests, what do protesters "
            "and the Election Commission/government say, what judicial developments have occurred, and which important "
            "claims remain unverified?"
        ),
    },
    "Fact-Check: Renewable Energy Grid Stability": {
        "domain": "Energy & Climate Fact-Checking",
        "description": "Evaluates the claim that high solar and wind penetration causes chronic grid collapse without nuclear baseload.",
        "query": (
            "Fact-check the claim: 'Transitioning a national power grid to over 70% solar and wind energy "
            "inevitably causes chronic grid instability and blackouts unless backed by new nuclear plants.' "
            "Examine recent empirical engineering data, grid operator reports, and battery storage deployments."
        ),
    },
    "Deepfake AI Regulation & Enforcement": {
        "domain": "Artificial Intelligence & Public Policy",
        "description": "Investigates legislative approaches, watermarking standards, and enforcement hurdles for generative AI media.",
        "query": (
            "Analyze current regulatory frameworks and enforcement mechanisms for generative AI deepfakes "
            "in major democracies. What specific detection standards (e.g., C2PA) are mandated, what legal penalties exist, "
            "and what evidence shows whether these laws are effectively deterring disinformation?"
        ),
    },
    "Solid-State Battery Commercialization": {
        "domain": "Emerging Science & Technology",
        "description": "Evaluates automaker claims regarding mass production timelines and cycle durability of solid-state EV batteries.",
        "query": (
            "Investigate the current commercialization status of solid-state lithium metal batteries for electric vehicles. "
            "Which manufacturers have reached automotive-grade testing, what are the verified energy density and cycle-life figures, "
            "and what production bottlenecks remain unresolved?"
        ),
    },
    "Knowledge Base Document Synthesis": {
        "domain": "Document Analysis & Cross-Verification",
        "description": "Compares uploaded documents against current web sources to check for factual alignment or updates.",
        "query": (
            "Review the key assertions in the uploaded documents in the knowledge base. Are there any inconsistencies, "
            "contradictions, or recent developments from external authoritative sources that update or challenge these assertions?"
        ),
    },
}

# Preserve INITIAL_DOCUMENTS for ECI reference downloads
INITIAL_DOCUMENTS: list[dict] = [
    {
        "name": "ECI SIR Order 24 June 2025",
        "url": "https://www.eci.gov.in/eci/public/api/document?id=17345",
        "date": "2025-06-24",
    },
    {
        "name": "Bihar SIR Press Notes 2025",
        "url": "https://ceoelection.bihar.gov.in/PDF/Year_2025/PressRelease2025/All%20Press%20Notes%20Related%20To%20SIR-2025.pdf",
        "date": "2025",
    },
    {
        "name": "Supreme Court SIR Judgment 27 May 2026",
        "url": "https://api.sci.gov.in/supremecourt/2025/35785/35785_2025_1_1501_71617_Judgement_27-May-2026.pdf",
        "date": "2026-05-27",
    },
    {
        "name": "Representation of the People Act 1950",
        "url": "https://upload.indiacode.nic.in/showfile?actid=AC_CEN_3_20_00016_195043_1517807321506&filename=a1950-43.pdf&type=actfile",
        "date": "1950",
    },
    {
        "name": "Registration of Electors Rules 1960",
        "url": "https://ceopuducherry.py.gov.in/actandrules.php",
        "date": "1960",
        "note": "Landing page — user may need to download the PDF manually from this page.",
    },
]

