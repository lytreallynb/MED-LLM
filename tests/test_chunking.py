"""Tests for the chunking module."""
import pytest
from medllm.chunking import (
    ChunkMetadata,
    ChunkingConfig,
    chunk_text,
    _split_into_sentences,
)


class TestChunkMetadata:
    def test_to_json(self):
        meta = ChunkMetadata(
            chunk_id="test_001",
            document_id="doc_001",
            drug_name="Ibuprofen",
            section="warnings",
            text="This is a test warning.",
        )
        json_str = meta.to_json()
        assert "test_001" in json_str
        assert "Ibuprofen" in json_str

    def test_from_dict(self):
        data = {
            "chunk_id": "test_002",
            "document_id": "doc_002",
            "drug_name": "Aspirin",
            "section": "dosage",
            "text": "Take as directed.",
        }
        meta = ChunkMetadata(**data)
        assert meta.drug_name == "Aspirin"
        assert meta.section == "dosage"


class TestSentenceSplitting:
    def test_simple_sentences(self):
        text = "This is sentence one. This is sentence two. And sentence three."
        sentences = _split_into_sentences(text)
        assert len(sentences) >= 3

    def test_empty_text(self):
        sentences = _split_into_sentences("")
        assert sentences == []

    def test_abbreviations(self):
        text = "Dr. Smith prescribed 500 mg. daily. Take with food."
        sentences = _split_into_sentences(text)
        # Should handle abbreviations properly
        assert len(sentences) >= 1


class TestChunking:
    def test_short_text_single_chunk(self):
        config = ChunkingConfig(window_size=100, overlap=20)
        text = "Short text that fits in one chunk."
        chunks = list(chunk_text(
            text=text,
            drug_name="TestDrug",
            section="test",
            document_id="doc_001",
            config=config,
        ))
        assert len(chunks) == 1
        assert chunks[0].drug_name == "TestDrug"

    def test_long_text_multiple_chunks(self):
        config = ChunkingConfig(window_size=50, overlap=10)
        text = " ".join(["Word"] * 200)  # Long text
        chunks = list(chunk_text(
            text=text,
            drug_name="TestDrug",
            section="test",
            document_id="doc_001",
            config=config,
        ))
        assert len(chunks) > 1

    def test_empty_text(self):
        config = ChunkingConfig()
        chunks = list(chunk_text(
            text="",
            drug_name="TestDrug",
            section="test",
            document_id="doc_001",
            config=config,
        ))
        assert len(chunks) == 0


class TestChunkingConfig:
    def test_default_values(self):
        config = ChunkingConfig()
        assert config.window_size == 768
        assert config.overlap == 100

    def test_custom_values(self):
        config = ChunkingConfig(window_size=512, overlap=50)
        assert config.window_size == 512
        assert config.overlap == 50
