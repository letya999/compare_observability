"""Tests for the RAG pipeline components."""

import pytest
from pathlib import Path

from src.config import Config
from src.models import QueryIntent, QueryAnalysis


class TestConfig:
    """Test configuration loading."""

    def test_default_config(self):
        config = Config()
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.top_k == 5

    def test_paths_created(self, tmp_path):
        config = Config()
        config.pdf_dir = tmp_path / "pdfs"
        config.chroma_dir = tmp_path / "chroma"
        config.__post_init__()
        assert config.pdf_dir.exists()
        assert config.chroma_dir.exists()


class TestModels:
    """Test data models."""

    def test_query_analysis(self):
        qa = QueryAnalysis(
            original_query="What is attention?",
            intent=QueryIntent.FACTUAL,
            entities=["attention"],
            keywords=["attention", "mechanism"],
        )
        assert qa.original_query == "What is attention?"
        assert qa.intent == QueryIntent.FACTUAL
        assert "attention" in qa.entities


class TestPipelineIntegration:
    """Integration tests for the full pipeline."""

    @pytest.mark.skipif(
        not Path(".env").exists(),
        reason="No .env file - skipping integration tests"
    )
    def test_query_without_documents(self):
        """Test querying when no documents are indexed."""
        from src.pipeline.orchestrator import RAGOrchestrator

        orchestrator = RAGOrchestrator()
        result = orchestrator.query("What is attention?")

        # Should complete without error even with no docs
        assert result.query == "What is attention?"
        assert result.query_analysis is not None
