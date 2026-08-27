"""Tests for the chunking module."""
from medllm.chunking import (
    ChunkMetadata,
    ChunkingConfig,
    TARGET_SECTIONS,
    _chunk_tokens,
    _normalize_text,
    _resolve_drug_name,
)


class FakeTokenizer:
    """Whitespace tokenizer standing in for TokenizerWrapper in tests."""

    def encode(self, text: str) -> list[str]:
        return text.split()

    def tokens_to_text(self, tokens) -> str:
        return " ".join(str(token) for token in tokens)


class TestChunkMetadata:
    def test_to_json(self):
        meta = ChunkMetadata(
            chunk_id="test_001",
            document_id="doc_001",
            drug_name="Ibuprofen",
            section="warnings",
            text="This is a test warning.",
            token_count=5,
            source_file="batch_0.parquet",
        )
        json_str = meta.to_json()
        assert "test_001" in json_str
        assert "Ibuprofen" in json_str

    def test_from_dict(self):
        data = {
            "chunk_id": "test_002",
            "document_id": "doc_002",
            "drug_name": "Aspirin",
            "section": "dosage_and_administration",
            "text": "Take as directed.",
            "token_count": 3,
            "source_file": "batch_1.parquet",
        }
        meta = ChunkMetadata(**data)
        assert meta.drug_name == "Aspirin"
        assert meta.section == "dosage_and_administration"


class TestNormalizeText:
    def test_plain_string(self):
        assert _normalize_text("  hello  ") == "hello"

    def test_none(self):
        assert _normalize_text(None) is None

    def test_list_of_strings(self):
        result = _normalize_text(["first part", "second part"])
        assert "first part" in result
        assert "second part" in result

    def test_empty_list(self):
        assert _normalize_text([]) is None


class TestResolveDrugName:
    def test_brand_name_preferred(self):
        record = {"openfda.brand_name": "Advil", "openfda.generic_name": "Ibuprofen"}
        assert _resolve_drug_name(record) == "Advil"

    def test_fallback_to_generic(self):
        record = {"openfda.generic_name": "Ibuprofen"}
        assert _resolve_drug_name(record) == "Ibuprofen"

    def test_unknown_when_missing(self):
        assert _resolve_drug_name({}) == "unknown"


class TestChunkTokens:
    def _chunks(self, text: str, chunk_size: int, overlap: int):
        tokenizer = FakeTokenizer()
        return list(
            _chunk_tokens(
                tokens=tokenizer.encode(text),
                tokenizer=tokenizer,
                chunk_size=chunk_size,
                overlap=overlap,
                document_id="doc_001",
                section="warnings",
                drug_name="TestDrug",
                source_file="batch_0.parquet",
            )
        )

    def test_short_text_single_chunk(self):
        chunks = self._chunks("Short text that fits in one chunk.", chunk_size=100, overlap=20)
        assert len(chunks) == 1
        assert chunks[0].drug_name == "TestDrug"

    def test_long_text_multiple_chunks(self):
        chunks = self._chunks(" ".join(["word"] * 200), chunk_size=50, overlap=10)
        assert len(chunks) > 1

    def test_overlap_between_chunks(self):
        chunks = self._chunks(" ".join(str(i) for i in range(100)), chunk_size=50, overlap=10)
        first_tokens = chunks[0].text.split()
        second_tokens = chunks[1].text.split()
        assert first_tokens[-10:] == second_tokens[:10]

    def test_empty_text(self):
        chunks = self._chunks("", chunk_size=100, overlap=20)
        assert chunks == []


class TestChunkingConfig:
    def test_default_values(self):
        config = ChunkingConfig()
        assert config.chunk_size == 768
        assert config.chunk_overlap == 100
        assert tuple(config.sections) == tuple(TARGET_SECTIONS)

    def test_custom_values(self):
        config = ChunkingConfig(chunk_size=512, chunk_overlap=50)
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
