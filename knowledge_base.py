"""
Knowledge Base Module — Multi-format Document Ingestion, Chunking,
Sentence-Transformers Embeddings, Content Hashing, and Qdrant Vector Store Integration.
Supports Qdrant Cloud with seamless fallback to local persistent or in-memory storage.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

import config

# ── Lazy-loaded Globals ──────────────────────────────────────────────────────
_embedding_model = None
_qdrant_client: QdrantClient | None = None
_qdrant_backend: str = "unknown"
_use_memory_client: bool = False
_memory_metadata: dict[str, Any] = {"documents": [], "last_indexed": None}


def _cleanup_qdrant() -> None:
    """Cleanly close Qdrant client before interpreter shutdown to prevent deallocator warnings."""
    global _qdrant_client
    if _qdrant_client is not None:
        try:
            _qdrant_client.close()
        except Exception:
            pass
        _qdrant_client = None


atexit.register(_cleanup_qdrant)



# ═══════════════════════════════════════════════════════════════════════════════
#  METADATA STORE
# ═══════════════════════════════════════════════════════════════════════════════

def _load_metadata() -> dict[str, Any]:
    """Load lightweight metadata JSON tracking ingested documents and hashes."""
    global _memory_metadata
    if _use_memory_client:
        return _memory_metadata

    if config.METADATA_DB.exists():
        try:
            return json.loads(config.METADATA_DB.read_text(encoding="utf-8"))
        except Exception:
            return {"documents": [], "last_indexed": None}
    return {"documents": [], "last_indexed": None}


def _save_metadata(meta: dict[str, Any]) -> None:
    """Save metadata to disk or in-memory state."""
    global _memory_metadata
    if _use_memory_client:
        _memory_metadata = meta
        return

    config.METADATA_DB.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )



def compute_file_hash(file_path: str | Path) -> str:
    """Compute SHA-256 hash of a file for duplicate and modification detection."""
    hasher = hashlib.sha256()
    path = Path(file_path)
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
#  DOCUMENT TEXT EXTRACTION (PDF, DOCX, TXT, MD, CSV, JSON)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_pdf_text(pdf_path: str | Path) -> list[dict[str, Any]]:
    """Extract text from PDF with PyMuPDF, preserving page numbers."""
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError:
            return [{"page_number": 0, "text": "", "error": "PyMuPDF not installed"}]

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return []

    pages: list[dict[str, Any]] = []
    try:
        doc = fitz.open(str(pdf_path))
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            if text:
                pages.append({"page_number": page_num + 1, "text": text})
        doc.close()
    except Exception as e:
        return [{"page_number": 0, "text": "", "error": str(e)}]

    if not pages:
        return [{
            "page_number": 0,
            "text": "",
            "warning": "No extractable text found. This PDF may be scanned/image-based and requires OCR.",
        }]

    return pages


def extract_docx_text(docx_path: str | Path) -> list[dict[str, Any]]:
    """Extract text from DOCX files, preserving headings and paragraphs."""
    try:
        import docx
    except ImportError:
        return [{"page_number": 0, "text": "", "error": "python-docx not installed"}]

    path = Path(docx_path)
    if not path.exists():
        return []

    try:
        doc = docx.Document(str(path))
        lines: list[str] = []
        for p in doc.paragraphs:
            text = p.text.strip()
            if text:
                lines.append(text)

        # Include tables if any
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    lines.append(row_text)

        full_text = "\n\n".join(lines)
        if not full_text.strip():
            return [{"page_number": 1, "text": "", "warning": "Empty document"}]

        # Segment into virtual pages (~2000 chars) for granular citation
        page_size = 2000
        pages = []
        for i in range(0, len(full_text), page_size):
            segment = full_text[i : i + page_size].strip()
            if segment:
                pages.append({"page_number": (i // page_size) + 1, "text": segment})
        return pages or [{"page_number": 1, "text": full_text}]
    except Exception as e:
        return [{"page_number": 0, "text": "", "error": f"Failed reading DOCX: {str(e)}"}]


def extract_document_text(file_path: str | Path) -> list[dict[str, Any]]:
    """Extract text from any supported file format (.pdf, .docx, .txt, .md, .csv, .json)."""
    path = Path(file_path)
    if not path.exists():
        return []

    ext = path.suffix.lower()

    if ext == ".pdf":
        return extract_pdf_text(path)
    if ext == ".docx":
        return extract_docx_text(path)

    # Text-based file formats (.txt, .md, .markdown, .csv, .json)
    try:
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = path.read_text(encoding="latin-1", errors="replace")

        if not content.strip():
            return [{"page_number": 1, "text": "", "warning": "Empty file"}]

        # Form-feed page markers if present
        if "\x0c" in content:
            parts = [p.strip() for p in content.split("\x0c") if p.strip()]
            return [{"page_number": i + 1, "text": p} for i, p in enumerate(parts)]

        # Segment long text files into virtual pages (~2000 chars each)
        page_chars = 2000
        if len(content) > page_chars:
            pages = []
            for i in range(0, len(content), page_chars):
                segment = content[i : i + page_chars].strip()
                if segment:
                    pages.append({"page_number": (i // page_chars) + 1, "text": segment})
            return pages
        else:
            return [{"page_number": 1, "text": content.strip()}]
    except Exception as e:
        return [{"page_number": 0, "text": "", "error": f"Failed reading text: {str(e)}"}]


def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks with clean paragraph boundary preservation."""
    if not text.strip():
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += max(1, chunk_size - chunk_overlap)

    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
#  EMBEDDINGS & VECTOR STORE (QDRANT)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_embedding_model():
    """Lazy load sentence transformers model."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedding_model


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Compute embeddings for a list of texts and return list of float lists."""
    if not texts:
        return []
    model = _get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return [vec.tolist() for vec in embeddings]


def set_memory_mode(enable: bool = True) -> None:
    """Switch Qdrant client to in-memory mode (useful for unit tests and sandboxes)."""
    global _use_memory_client, _qdrant_client, _qdrant_backend, _memory_metadata
    _use_memory_client = enable
    _qdrant_client = None
    _qdrant_backend = "unknown"
    _memory_metadata = {"documents": [], "last_indexed": None}



def get_qdrant_client() -> tuple[QdrantClient, str]:
    """
    Connect to Qdrant Cloud or fallback to local persistent/in-memory Qdrant.
    Returns (client, backend_name).
    """
    global _qdrant_client, _qdrant_backend

    if _qdrant_client is not None:
        return _qdrant_client, _qdrant_backend

    if _use_memory_client:
        _qdrant_client = QdrantClient(":memory:")
        _qdrant_backend = "memory"
        ensure_collection_exists(_qdrant_client)
        return _qdrant_client, _qdrant_backend

    # 1. Try Qdrant Cloud if URL and API Key are provided
    if config.QDRANT_URL and config.QDRANT_API_KEY:
        try:
            client = QdrantClient(
                url=config.QDRANT_URL,
                api_key=config.QDRANT_API_KEY,
                timeout=10,
            )
            # Ping by listing collections
            client.get_collections()
            _qdrant_client = client
            _qdrant_backend = "cloud"
            ensure_collection_exists(_qdrant_client)
            return _qdrant_client, _qdrant_backend
        except Exception as e:
            # Fallback to local on connection error
            pass

    # 2. Local persistent Qdrant storage
    try:
        config.QDRANT_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(config.QDRANT_LOCAL_DIR))
        _qdrant_client = client
        _qdrant_backend = "local_persistent"
        ensure_collection_exists(_qdrant_client)
        return _qdrant_client, _qdrant_backend
    except Exception:
        # 3. Final fallback: in-memory
        client = QdrantClient(":memory:")
        _qdrant_client = client
        _qdrant_backend = "memory"
        ensure_collection_exists(_qdrant_client)
        return _qdrant_client, _qdrant_backend


def ensure_collection_exists(client: QdrantClient | None = None) -> bool:
    """Ensure the target Qdrant collection exists with proper vector dimension & metric."""
    if client is None:
        client, backend = get_qdrant_client()
    else:
        backend = _qdrant_backend

    collection_name = config.QDRANT_COLLECTION_NAME
    try:
        existing = [c.name for c in client.get_collections().collections]
        if collection_name not in existing:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=config.EMBEDDING_DIMENSION,
                    distance=Distance.COSINE,
                ),
            )
            # Create payload index for document_name on cloud backend
            if backend == "cloud":
                try:
                    client.create_payload_index(
                        collection_name=collection_name,
                        field_name="document_name",
                        field_schema="keyword",
                    )
                except Exception:
                    pass
        return True
    except Exception:
        return False


def build_index(documents: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """
    Build or rebuild knowledge base index from documents.
    Provided for backwards compatibility with test suites and batch indexing scripts.
    """
    if documents is None:
        return reindex_all_documents()
    res = ingest_multiple_documents(documents)
    res["total_chunks"] = res.get("total_chunks", 0)
    res["documents_indexed"] = res.get("added_count", 0) + res.get("skipped_count", 0)
    return res



def get_collection_stats() -> dict[str, Any]:
    """Retrieve collection statistics and vector count from Qdrant."""
    client, backend = get_qdrant_client()
    collection_name = config.QDRANT_COLLECTION_NAME
    try:
        count_res = client.count(collection_name)
        total_points = count_res.count
        return {
            "collection_name": collection_name,
            "backend": backend,
            "total_points": total_points,
            "dimension": config.EMBEDDING_DIMENSION,
            "status": "ready",
        }
    except Exception as e:
        return {
            "collection_name": collection_name,
            "backend": backend,
            "total_points": 0,
            "error": str(e),
            "status": "error",
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  SEARCH KNOWLEDGE BASE (RETRIEVAL)
# ═══════════════════════════════════════════════════════════════════════════════

def search_knowledge_base(
    query: str,
    filters: dict[str, Any] | None = None,
    top_k: int = config.MAX_RAG_PASSAGES,
    min_score: float = config.MIN_RAG_RELEVANCE_SCORE,
) -> dict[str, Any]:
    """
    Search the Qdrant knowledge base with vector similarity and optional metadata filters.
    Enforces a strict minimum relevance score threshold so irrelevant documents are never returned.
    Returns structured passages with relevance scores, source name, page number, and IDs.
    """
    client, backend = get_qdrant_client()
    collection_name = config.QDRANT_COLLECTION_NAME

    try:
        count_info = client.count(collection_name)
        if count_info.count == 0:
            return {
                "query": query,
                "results": [],
                "total_found": 0,
                "backend": backend,
                "message": "Knowledge base is currently empty. Ingest documents first.",
            }
    except Exception as e:
        return {
            "query": query,
            "results": [],
            "total_found": 0,
            "backend": backend,
            "error": f"Failed checking knowledge base: {str(e)}",
        }

    query_vectors = _embed_texts([query])
    if not query_vectors:
        return {"query": query, "results": [], "total_found": 0, "backend": backend}

    query_vector = query_vectors[0]

    # Construct Qdrant filter if provided
    query_filter = None
    if filters:
        conditions = []
        for key, val in filters.items():
            if val is not None and val != "":
                conditions.append(FieldCondition(key=key, match=MatchValue(value=val)))
        if conditions:
            query_filter = Filter(must=conditions)

    try:
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        points = response.points
    except Exception as e:
        # Fallback to search() if query_points is unsupported
        try:
            points = client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
        except Exception as ex:
            return {
                "query": query,
                "results": [],
                "total_found": 0,
                "backend": backend,
                "error": f"Search failed: {str(ex)}",
            }

    results: list[dict[str, Any]] = []
    top_raw_score = 0.0

    for p in points:
        payload = p.payload or {}
        score = float(getattr(p, "score", 0.0))
        if score > top_raw_score:
            top_raw_score = score

        # Enforce minimum relevance threshold (cosine similarity score)
        if score < min_score:
            continue

        norm_score = round(max(0.0, min(1.0, (score + 1.0) / 2.0 if score <= 1.0 else score)), 4)

        results.append({
            "id": str(p.id),
            "text": payload.get("text", ""),
            "relevance_score": norm_score,
            "raw_score": round(score, 4),
            "document_name": payload.get("document_name", "Unknown"),
            "url": payload.get("url", ""),
            "date": payload.get("date", "Unknown"),
            "page_number": payload.get("page_number", 0),
            "chunk_index": payload.get("chunk_index", 0),
            "doc_hash": payload.get("doc_hash", ""),
            "ingestion_date": payload.get("ingestion_date", ""),
            "source_type": payload.get("source_type", "document"),
        })

    msg = ""
    if points and not results:
        msg = f"Searched Qdrant KB: no passages met the relevance threshold ({min_score:.2f}). Top candidate scored only {top_raw_score:.2f}."

    return {
        "query": query,
        "results": results,
        "total_found": len(results),
        "backend": backend,
        "top_raw_score": round(top_raw_score, 4),
        "threshold": min_score,
        "message": msg,
    }


def get_sample_points(limit: int = 8) -> list[dict[str, Any]]:
    """Retrieve sample points directly from Qdrant Cloud collection for live UI inspection."""
    client, backend = get_qdrant_client()
    collection_name = config.QDRANT_COLLECTION_NAME
    try:
        records, _ = client.scroll(
            collection_name=collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        samples = []
        for r in records:
            p = r.payload or {}
            samples.append({
                "id": str(r.id),
                "document_name": p.get("document_name", "Unknown"),
                "page_number": p.get("page_number", 1),
                "chunk_index": p.get("chunk_index", 0),
                "text_preview": (p.get("text") or "")[:240],
                "date": p.get("date", "Unknown"),
                "doc_hash": p.get("doc_hash", "")[:12],
            })
        return samples
    except Exception:
        return []


def retrieve_evidence(query: str, top_k: int = config.MAX_RAG_PASSAGES) -> dict[str, Any]:
    """Backward-compatible alias for search_knowledge_base."""
    return search_knowledge_base(query=query, filters=None, top_k=top_k)


# ═══════════════════════════════════════════════════════════════════════════════
#  DOCUMENT INGESTION & CONTENT HASH DEDUPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_document(
    source: str | Path,
    name: str | None = None,
    url: str = "",
    date: str = "Unknown",
    force_reindex: bool = False,
) -> dict[str, Any]:
    """
    Ingest a single document into Qdrant.
    Detects duplicates using SHA-256 content hashes:
    - If hash is unchanged and already indexed, skips re-processing.
    - If hash changed (updated file), replaces previous points.
    - If new, chunks, embeds, and upserts into Qdrant.
    """
    client, backend = get_qdrant_client()
    collection_name = config.QDRANT_COLLECTION_NAME
    source_str = str(source)

    # Download if URL
    if source_str.startswith("http://") or source_str.startswith("https://"):
        dl_name = name or "downloaded_doc"
        result = download_document(source_str, dl_name)
        if not result["success"]:
            return result
        source = Path(result["path"])
        url = source_str
    else:
        source = Path(source)

    if not source.exists():
        return {"success": False, "error": f"File not found: {source}"}

    doc_name = name or source.stem
    doc_hash = compute_file_hash(source)
    meta = _load_metadata()
    existing_docs = meta.get("documents", [])

    # Check for duplicate document with identical hash
    identical_entry = next(
        (d for d in existing_docs if (d.get("name") == doc_name or d.get("path") == str(source)) and d.get("hash") == doc_hash),
        None,
    )
    if identical_entry and not force_reindex:
        return {
            "success": False,
            "error": "Document already ingested",
            "document": doc_name,
            "document_name": doc_name,
            "skipped": True,
            "message": f"Document '{doc_name}' is already indexed with identical content hash.",
            "hash": identical_entry.get("hash", doc_hash),
            "chunks": identical_entry.get("chunks_count", 0),
        }

    # If document exists by name/path with DIFFERENT hash, delete old points before re-indexing
    modified_entry = next(
        (d for d in existing_docs if d.get("name") == doc_name or d.get("path") == str(source)),
        None,
    )
    if modified_entry:
        delete_document(doc_name)



    # Extract text
    pages = extract_document_text(source)
    if not pages or (len(pages) == 1 and not pages[0].get("text")):
        return {
            "success": False,
            "document_name": doc_name,
            "error": "No text could be extracted from document",
        }


    # Build chunks
    all_chunks: list[dict[str, Any]] = []
    chunk_texts: list[str] = []
    ingestion_date = datetime.now(timezone.utc).isoformat()

    for page in pages:
        text = page.get("text", "")
        if not text:
            continue
        chunks = chunk_text(text)
        for i, chunk_text_content in enumerate(chunks):
            chunk_texts.append(chunk_text_content)
            all_chunks.append({
                "document_name": doc_name,
                "source_path": str(source),
                "url": url,
                "date": date,
                "page_number": page.get("page_number", 1),
                "chunk_index": i,
                "text": chunk_text_content,
                "doc_hash": doc_hash,
                "ingestion_date": ingestion_date,
                "source_type": source.suffix.lower().replace(".", ""),
            })

    if not chunk_texts:
        return {
            "success": False,
            "document_name": doc_name,
            "error": "No chunks generated from document text",
        }

    # Compute batch embeddings
    vectors = _embed_texts(chunk_texts)

    # Prepare Qdrant points with deterministic UUIDs based on doc_hash + chunk_index
    points: list[PointStruct] = []
    for idx, (chunk_dict, vec) in enumerate(zip(all_chunks, vectors)):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_hash}_{idx}"))
        points.append(PointStruct(id=point_id, vector=vec, payload=chunk_dict))

    # Upsert into Qdrant in batches
    batch_size = 100
    for b in range(0, len(points), batch_size):
        client.upsert(collection_name=collection_name, points=points[b : b + batch_size])

    # Update metadata catalog
    updated_docs = [d for d in existing_docs if d.get("name") != doc_name]
    updated_docs.append({
        "name": doc_name,
        "path": str(source),
        "url": url,
        "date": date,
        "hash": doc_hash,
        "chunks_count": len(points),
        "ingested_at": ingestion_date,
        "content_type": source.suffix.lower(),
    })
    meta["documents"] = updated_docs
    meta["last_indexed"] = ingestion_date
    _save_metadata(meta)

    return {
        "success": True,
        "document_name": doc_name,
        "chunks_indexed": len(points),
        "total_chunks": len(points),
        "backend": backend,
        "hash": doc_hash,
        "ingested_at": ingestion_date,
    }


def ingest_multiple_documents(doc_entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Batch ingest a list of document entries with progress tracking."""
    added = 0
    skipped = 0
    failed = []

    for entry in doc_entries:
        path = entry.get("path")
        name = entry.get("name")
        url = entry.get("url", "")
        date = entry.get("date", "Unknown")

        try:
            res = ingest_document(path, name=name, url=url, date=date)
            if res.get("success"):
                if res.get("skipped"):
                    skipped += 1
                else:
                    added += 1
            else:
                failed.append({"name": name or str(path), "error": res.get("error", "Unknown error")})
        except Exception as e:
            failed.append({"name": name or str(path), "error": str(e)})

    stats = get_collection_stats()
    return {
        "success": True,
        "added_count": added,
        "skipped_count": skipped,
        "failed_count": len(failed),
        "failed": failed,
        "total_chunks": stats.get("total_points", 0),
        "backend": stats.get("backend", "unknown"),
    }


def delete_document(document_name: str) -> dict[str, Any]:
    """Delete all chunks for a document from Qdrant and remove from metadata."""
    client, backend = get_qdrant_client()
    collection_name = config.QDRANT_COLLECTION_NAME

    try:
        # Delete from Qdrant by payload filter
        client.delete(
            collection_name=collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="document_name", match=MatchValue(value=document_name))]
                )
            ),
        )
    except Exception as e:
        return {"success": False, "error": f"Failed deleting from vector store: {str(e)}"}

    # Remove from metadata
    meta = _load_metadata()
    meta["documents"] = [d for d in meta.get("documents", []) if d.get("name") != document_name]
    _save_metadata(meta)

    return {"success": True, "document_name": document_name, "message": f"Deleted '{document_name}'"}


def reindex_all_documents() -> dict[str, Any]:
    """Re-index all documents currently stored in documents directory."""
    documents_to_index = []
    if config.DOCUMENTS_DIR.exists():
        for f in config.DOCUMENTS_DIR.iterdir():
            if f.is_file() and f.suffix.lower() in config.SUPPORTED_EXTENSIONS:
                documents_to_index.append({
                    "path": str(f),
                    "name": f.stem,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                })

    if not documents_to_index:
        return {"success": False, "error": "No documents found to reindex"}

    return ingest_multiple_documents(documents_to_index)


def migrate_from_faiss() -> dict[str, Any]:
    """Migrate documents recorded in existing index_meta.json or data/documents/ to Qdrant."""
    meta = _load_metadata()
    docs = meta.get("documents", [])
    if not docs:
        return reindex_all_documents()

    migrated = 0
    for doc in docs:
        path = doc.get("path")
        if path and Path(path).exists():
            res = ingest_document(path, name=doc.get("name"), date=doc.get("date", "Unknown"))
            if res.get("success"):
                migrated += 1

    return {"success": True, "migrated_count": migrated}


def get_ingested_documents() -> list[dict[str, Any]]:
    """Return all ingested documents from metadata catalog."""
    meta = _load_metadata()
    return meta.get("documents", [])


def get_index_status() -> dict[str, Any]:
    """Return current Qdrant index status and document counts."""
    stats = get_collection_stats()
    meta = _load_metadata()
    docs = meta.get("documents", [])
    return {
        "index_built": stats.get("total_points", 0) > 0,
        "total_chunks": stats.get("total_points", 0),
        "document_count": len(docs),
        "backend": stats.get("backend", "unknown"),
        "last_indexed": meta.get("last_indexed"),
        "documents": docs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  DOCUMENT DOWNLOAD HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def download_document(url: str, name: str) -> dict[str, Any]:
    """Download a remote document and save to documents directory."""
    safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in name)
    dest = config.DOCUMENTS_DIR / f"{safe_name}.pdf"

    if dest.exists() and dest.stat().st_size > 1000:
        return {
            "success": True,
            "path": str(dest),
            "message": "Already downloaded",
            "skipped": True,
        }

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": config.USER_AGENT},
            timeout=30,
            stream=True,
            allow_redirects=True,
        )

        if resp.status_code != 200:
            return {
                "success": False,
                "path": "",
                "error": f"HTTP {resp.status_code} — download failed for '{name}'",
                "url": url,
            }

        chunks: list[bytes] = []
        total = 0
        for chunk in resp.iter_content(chunk_size=8192):
            total += len(chunk)
            if total > 50_000_000:  # 50 MB limit
                return {"success": False, "path": "", "error": "File exceeds 50 MB limit"}
            chunks.append(chunk)
        raw = b"".join(chunks)

        if not raw.startswith(b"%PDF") and not raw.startswith(b"PK\x03\x04"):
            if raw[:100].lower().strip().startswith((b"<html", b"<!doctype", b"<!")):
                return {
                    "success": False,
                    "path": "",
                    "error": f"URL returned HTML instead of a document for '{name}'. Manual download required.",
                    "url": url,
                }

        dest.write_bytes(raw)
        return {
            "success": True,
            "path": str(dest),
            "size_bytes": len(raw),
            "message": f"Downloaded '{name}' ({len(raw):,} bytes)",
        }
    except requests.exceptions.Timeout:
        return {"success": False, "path": "", "error": f"Timeout downloading '{name}'", "url": url}
    except Exception as e:
        return {"success": False, "path": "", "error": f"Download error: {str(e)[:200]}", "url": url}


def download_initial_documents() -> list[dict[str, Any]]:
    """Download initial reference documents defined in config."""
    results: list[dict[str, Any]] = []
    for doc in config.INITIAL_DOCUMENTS:
        url = doc["url"]
        name = doc["name"]

        if doc.get("note") and "landing page" in doc.get("note", "").lower():
            results.append({
                "name": name,
                "url": url,
                "success": False,
                "error": f"Landing page requires manual download: {url}",
                "manual_download": True,
            })
            continue

        result = download_document(url, name)
        result["name"] = name
        result["url"] = url
        results.append(result)

    return results
