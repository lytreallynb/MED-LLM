"""Normalize FDA label text fields and emit chunked passages for embeddings."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from medllm.chunking import TARGET_SECTIONS, ChunkingConfig, chunk_fda_parquet


def _parse_sections(raw: str | None) -> Sequence[str]:
    if not raw:
        return tuple(TARGET_SECTIONS)
    items = [entry.strip() for entry in raw.split(",") if entry.strip()]
    if not items:
        return tuple(TARGET_SECTIONS)
    return tuple(items)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", default="data/clean/parquet", help="Directory that stores batch_*.parquet files")
    parser.add_argument("--output", default="data/clean/chunks.jsonl", help="Destination for the cleaned chunk JSONL")
    parser.add_argument("--sections", default=None, help="Comma-separated list of label sections to keep")
    parser.add_argument("--chunk-size", type=int, default=768, help="Number of tokens per chunk")
    parser.add_argument("--overlap", type=int, default=100, help="Token overlap between chunks")
    parser.add_argument("--min-chars", type=int, default=64, help="Skip sections shorter than this many characters")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B", help="Tokenizer used to count tokens (leave blank to use regex fallback)")
    parser.add_argument("--regex-only", action="store_true", help="Force regex tokenization even if transformers are installed")
    parser.add_argument("--max-files", type=int, default=None, help="Optionally limit the number of parquet files processed")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tokenizer_name = None if args.regex_only else args.tokenizer
    sections = _parse_sections(args.sections)
    cfg = ChunkingConfig(
        parquet_dir=Path(args.parquet_dir),
        output_path=Path(args.output),
        sections=sections,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        min_characters=args.min_chars,
        tokenizer_name=tokenizer_name,
        max_files=args.max_files,
    )
    output_path, total = chunk_fda_parquet(cfg)
    print(f"Wrote {total} cleaned chunk(s) to {output_path}")


if __name__ == "__main__":
    main()
