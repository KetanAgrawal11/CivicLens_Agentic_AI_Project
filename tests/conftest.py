"""Shared pytest fixtures for the CivicLens test suite."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import knowledge_base


@pytest.fixture(autouse=True)
def memory_vector_store():
    """Run every test against an isolated in-memory Qdrant and catalog, never the live collection."""
    knowledge_base.set_memory_mode(True)
    yield
    knowledge_base.set_memory_mode(False)
