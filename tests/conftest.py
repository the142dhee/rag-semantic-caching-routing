"""
tests/conftest.py
=================
pytest configuration file.

This file is automatically discovered by pytest. Things defined here
(fixtures, marks, plugins) are available to ALL test files without importing.

KEY CONCEPT — pytest.ini_options:
  We register custom markers here so pytest doesn't warn about unknown marks.
  The `integration` marker lets us run only integration tests:
    pytest -m integration    → only integration tests
    pytest -m "not integration" → skip integration tests (CI default)
"""

import pytest


def pytest_configure(config):
    """Register custom markers to avoid PytestUnknownMarkWarning."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test requiring real API calls"
    )


@pytest.fixture(autouse=True)
def clean_test_state():
    """Ensure in-memory caches, BM25 indices, and sessions are fresh for each test."""
    from app.services.bm25 import bm25_index
    from app.services.query_rewriter import session_manager
    yield
    bm25_index.clear()
    session_manager.clear_all()
