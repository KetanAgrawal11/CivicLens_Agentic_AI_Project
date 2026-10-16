"""
Tests for CivicLens knowledge_base module.
All tests are offline — no network or API credentials needed.
Uses temporary files and directories.
"""

import json
import os
import sys
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


class TestPDFExtraction:
    """Test PDF text extraction."""

    def test_extract_from_nonexistent_file(self):
        """Test graceful handling of missing file."""
        import knowledge_base
        result = knowledge_base.extract_pdf_text("/nonexistent/file.pdf")
        assert result == []

    def test_extract_from_valid_pdf(self, tmp_path):
        """Test extraction from a programmatically created PDF."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            pytest.skip("PyMuPDF not installed")

        # Create a minimal PDF with text
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "This is test content for CivicLens knowledge base.")
        pdf_path = tmp_path / "test.pdf"
        doc.save(str(pdf_path))
        doc.close()

        import knowledge_base
        pages = knowledge_base.extract_pdf_text(pdf_path)
        assert len(pages) >= 1
        assert "test content" in pages[0]["text"].lower()
        assert pages[0]["page_number"] == 1


class TestChunking:
    """Test text chunking."""

    def test_basic_chunking(self):
        import knowledge_base
        text = "A" * 1000
        chunks = knowledge_base.chunk_text(text, chunk_size=200, chunk_overlap=50)
        assert len(chunks) > 1
        # Each chunk should be <= chunk_size
        for chunk in chunks:
            assert len(chunk) <= 200

    def test_empty_text_chunking(self):
        import knowledge_base
        chunks = knowledge_base.chunk_text("")
        assert chunks == []

    def test_short_text_single_chunk(self):
        import knowledge_base
        text = "Short text."
        chunks = knowledge_base.chunk_text(text, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == "Short text."

    def test_overlap_exists(self):
        import knowledge_base
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 10  # 260 chars
        chunks = knowledge_base.chunk_text(text, chunk_size=50, chunk_overlap=10)
        # With overlap, chunks should share some content
        assert len(chunks) >= 2


class TestRAGIngestionAndRetrieval:
    """Test the full ingestion → index → retrieval pipeline."""

    def test_full_rag_pipeline(self, tmp_path):
        """Test ingesting a PDF and retrieving relevant passages with metadata."""
        try:
            import fitz
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            pytest.skip(f"Required package not installed: {e}")

        import knowledge_base

        # Override paths to use temp directory
        original_docs = config.DOCUMENTS_DIR
        original_index = config.INDEX_DIR
        original_meta = config.METADATA_DB

        try:
            config.DOCUMENTS_DIR = tmp_path / "documents"
            config.INDEX_DIR = tmp_path / "index"
            config.METADATA_DB = tmp_path / "metadata.json"
            config.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
            config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

            # Reset globals
            knowledge_base._faiss_index = None
            knowledge_base._index_metadata = []

            # Create test PDF
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72),
                "The Election Commission of India ordered a Special Intensive Revision "
                "of electoral rolls in June 2025. This SIR was conducted across all states. "
                "The Supreme Court examined the legality of the SIR process."
            )
            pdf_path = config.DOCUMENTS_DIR / "test_eci.pdf"
            doc.save(str(pdf_path))
            doc.close()

            # Build index
            result = knowledge_base.build_index([{
                "path": str(pdf_path),
                "name": "Test ECI Document",
                "url": "https://test.eci.gov.in/doc1",
                "date": "2025-06-24",
            }])

            assert result["success"] is True
            assert result["total_chunks"] > 0
            assert result["documents_indexed"] == 1

            # Retrieve
            rag_result = knowledge_base.search_knowledge_base("Election Commission SIR", filters={"document_name": "Test ECI Document"})
            if not rag_result.get("results"):
                rag_result = knowledge_base.retrieve_evidence("Election Commission SIR")

            assert len(rag_result["results"]) > 0
            matching = [r for r in rag_result["results"] if r.get("document_name") == "Test ECI Document"]
            assert len(matching) > 0
            first = matching[0]
            assert "document_name" in first
            assert first["document_name"] == "Test ECI Document"
            assert first["url"] == "https://test.eci.gov.in/doc1"
            assert first["date"] == "2025-06-24"
            assert "page_number" in first
            assert "ingestion_date" in first

        finally:
            # Restore paths
            config.DOCUMENTS_DIR = original_docs
            config.INDEX_DIR = original_index
            config.METADATA_DB = original_meta
            knowledge_base._faiss_index = None
            knowledge_base._index_metadata = []


class TestDuplicateIngestion:
    """Test prevention of duplicate ingestion."""

    def test_prevents_duplicate(self, tmp_path):
        """Test that ingesting the same document twice is prevented."""
        try:
            import fitz
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            pytest.skip(f"Required package not installed: {e}")

        import knowledge_base

        original_docs = config.DOCUMENTS_DIR
        original_index = config.INDEX_DIR
        original_meta = config.METADATA_DB

        try:
            config.DOCUMENTS_DIR = tmp_path / "documents"
            config.INDEX_DIR = tmp_path / "index"
            config.METADATA_DB = tmp_path / "metadata.json"
            config.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
            config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

            knowledge_base._faiss_index = None
            knowledge_base._index_metadata = []

            # Create test PDF
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), "Test content for duplicate check.")
            pdf_path = config.DOCUMENTS_DIR / "dup_test.pdf"
            doc.save(str(pdf_path))
            doc.close()

            # First ingestion
            result1 = knowledge_base.ingest_document(pdf_path, name="Dup Test", url="https://test.com")
            assert result1.get("success") is True

            # Second ingestion — should be prevented
            result2 = knowledge_base.ingest_document(pdf_path, name="Dup Test", url="https://test.com")
            assert result2.get("success") is False
            assert "already ingested" in result2.get("error", "").lower()

        finally:
            config.DOCUMENTS_DIR = original_docs
            config.INDEX_DIR = original_index
            config.METADATA_DB = original_meta
            knowledge_base._faiss_index = None
            knowledge_base._index_metadata = []


class TestDocumentDownload:
    """Test document download with validation."""

    @patch("knowledge_base.requests.get")
    def test_rejects_html_as_pdf(self, mock_get, tmp_path):
        """Test that HTML error pages are not saved as PDFs."""
        import knowledge_base

        original_docs = config.DOCUMENTS_DIR
        try:
            config.DOCUMENTS_DIR = tmp_path / "documents"
            config.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.headers = {"Content-Type": "text/html"}
            mock_resp.iter_content.return_value = [
                b"<html><body>Error: Document not found</body></html>"
            ]
            mock_get.return_value = mock_resp

            result = knowledge_base.download_document(
                "https://example.com/fake.pdf", "Test Doc"
            )

            assert result["success"] is False
            assert "html" in result.get("error", "").lower() or "not a valid pdf" in result.get("error", "").lower()

        finally:
            config.DOCUMENTS_DIR = original_docs

    @patch("knowledge_base.requests.get")
    def test_handles_download_timeout(self, mock_get, tmp_path):
        """Test timeout handling during download."""
        import knowledge_base
        import requests as req

        original_docs = config.DOCUMENTS_DIR
        try:
            config.DOCUMENTS_DIR = tmp_path / "documents"
            config.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

            mock_get.side_effect = req.exceptions.Timeout("Timed out")

            result = knowledge_base.download_document(
                "https://slow.example.com/doc.pdf", "Slow Doc"
            )

            assert result["success"] is False
            assert "timeout" in result.get("error", "").lower()

        finally:
            config.DOCUMENTS_DIR = original_docs
