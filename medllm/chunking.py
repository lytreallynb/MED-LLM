"""Chunk FDA label Parquet files into overlapping passages for retrieval."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Iterable, Iterator, List, Sequence

import pandas as pd
from tqdm import tqdm

from .tokenization import TokenizerConfig, TokenizerWrapper

TARGET_SECTIONS = [
    "indications_and_usage",
    "warnings",
    "contraindications",
    "adverse_reactions",
    "dosage_and_administration",
    "clinical_pharmacology",
]


@dataclass
class ChunkMetadata:
    chunk_id: str
    document_id: str
    drug_name: str
    section: str
    text: str
    token_count: int
    source_file: str

    def to_json(self) -> str:
        payload = asdict(self)
        return json.dumps(payload, ensure_ascii=False)


@dataclass
class ChunkingConfig:
    parquet_dir: Path = Path("data/clean/parquet")
    output_path: Path = Path("data/clean/chunks.jsonl")
    sections: Sequence[str] = tuple(TARGET_SECTIONS)
    chunk_size: int = 768
    chunk_overlap: int = 100
    min_characters: int = 64
    tokenizer_name: str | None = "Qwen/Qwen2.5-7B"
    max_files: int | None = None


def _normalize_text(value) -> str | None:
    """Flatten nested structures (lists, sets) into a clean paragraph."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        try:
            return _normalize_text(to_list())
        except Exception:  # pragma: no cover - fallback for unexpected objects
            pass
    if isinstance(value, (list, tuple, set)):
        pieces: List[str] = []
        for entry in value:
            normalized = _normalize_text(entry)
            if normalized:
                pieces.append(normalized)
        collapsed = "\n\n".join(pieces).strip()
        return collapsed or None
    if isinstance(value, dict):
        pieces = []
        for key in sorted(value):
            normalized = _normalize_text(value[key])
            if normalized:
                pieces.append(f"{key}: {normalized}")
        collapsed = "\n".join(pieces).strip()
        return collapsed or None
    text = str(value).strip()
    return text or None


def _resolve_drug_name(record: dict) -> str:
    for key in (
        "openfda.brand_name",
        "openfda.generic_name",
        "openfda.substance_name",
        "proprietary_name",
    ):
        value = record.get(key)
        normalized = _normalize_text(value)
        if normalized:
            return normalized.split("\n")[0]
    return record.get("id") or record.get("set_id") or "unknown"


def _chunk_tokens(
    *,
    tokens: Sequence[int | str],
    tokenizer: TokenizerWrapper,
    chunk_size: int,
    overlap: int,
    document_id: str,
    section: str,
    drug_name: str,
    source_file: str,
) -> Iterator[ChunkMetadata]:
    if not tokens:
        return
    start = 0
    chunk_idx = 0
    n_tokens = len(tokens)

    while start < n_tokens:
        end = min(n_tokens, start + chunk_size)
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.tokens_to_text(chunk_tokens).strip()
        if chunk_text:
            chunk_id = f"{document_id}-{section}-{chunk_idx}"
            yield ChunkMetadata(
                chunk_id=chunk_id,
                document_id=document_id,
                drug_name=drug_name,
                section=section,
                text=chunk_text,
                token_count=len(chunk_tokens),
                source_file=source_file,
            )
        chunk_idx += 1
        if end == n_tokens:
            break
        start = max(0, end - overlap)


def _records_from_parquet(file_path: Path) -> Iterable[dict]:
    df = pd.read_parquet(file_path)
    yield from df.to_dict(orient="records")


def chunk_fda_parquet(config: ChunkingConfig) -> tuple[Path, int]:
    """Chunk every Parquet batch into overlapping passages and persist JSONL."""
    parquet_dir = Path(config.parquet_dir)
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = TokenizerWrapper(TokenizerConfig(tokenizer_name=config.tokenizer_name))

    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {parquet_dir}")

    total_chunks = 0
    with output_path.open("w", encoding="utf-8") as sink:
        iterable = parquet_files
        if config.max_files is not None:
            iterable = iterable[: config.max_files]
        for file_path in tqdm(iterable, desc="Chunking FDA labels"):
            for record in _records_from_parquet(file_path):
                document_id = record.get("id") or record.get("set_id") or record.get("spl_id") or "unknown"
                drug_name = _resolve_drug_name(record)
                for section in config.sections:
                    text = _normalize_text(record.get(section))
                    if not text or len(text) < config.min_characters:
                        continue
                    tokens = tokenizer.encode(text)
                    if not tokens:
                        continue
                    for chunk in _chunk_tokens(
                        tokens=tokens,
                        tokenizer=tokenizer,
                        chunk_size=config.chunk_size,
                        overlap=config.chunk_overlap,
                        document_id=document_id,
                        section=section,
                        drug_name=drug_name,
                        source_file=file_path.name,
                    ):
                        sink.write(chunk.to_json())
                        sink.write("\n")
                        total_chunks += 1
    return output_path, total_chunks


__all__ = ["ChunkMetadata", "ChunkingConfig", "chunk_fda_parquet", "TARGET_SECTIONS"]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", default="data/clean/parquet", help="Directory with batch_*.parquet files")
    parser.add_argument("--output", default="data/clean/chunks.jsonl", help="Where to store generated chunks")
    parser.add_argument("--chunk-size", type=int, default=768, help="Number of tokens per chunk")
    parser.add_argument("--overlap", type=int, default=100, help="Token overlap between chunks")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B", help="Tokenizer used for counting tokens")
    parser.add_argument("--max-files", type=int, default=None, help="Optionally limit the number of Parquet files processed")
    return parser


if __name__ == "__main__":
    cli_args = _build_arg_parser().parse_args()
    cfg = ChunkingConfig(
        parquet_dir=Path(cli_args.parquet_dir),
        output_path=Path(cli_args.output),
        chunk_size=cli_args.chunk_size,
        chunk_overlap=cli_args.overlap,
        tokenizer_name=cli_args.tokenizer,
        max_files=cli_args.max_files,
    )
    output_path, total = chunk_fda_parquet(cfg)
    print(f"Wrote {total} chunk(s) to {output_path}")
