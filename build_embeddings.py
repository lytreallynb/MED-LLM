"""Generate dense embeddings + metadata for cleaned FDA label chunks."""
from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", default="data/clean/chunks.jsonl", help="Path to the chunk JSONL produced by clean_text_fields.py")
    parser.add_argument("--meta-output", default="data/clean/fda_meta.jsonl", help="Where to write aligned chunk metadata")
    parser.add_argument("--embeddings", default="data/clean/fda_embeddings.npy", help="Destination for the numpy embedding matrix")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Embedding-1.8B", help="Hugging Face embedding model identifier")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for embedding computation")
    parser.add_argument("--max-length", type=int, default=1024, help="Token truncation length")
    parser.add_argument("--device", default=None, help="Optional torch device override (cpu/cuda)")
    parser.add_argument("--no-normalize", action="store_true", help="Disable L2 normalization on embeddings")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from medllm.embeddings import EmbeddingConfig, generate_embeddings  # imported lazily to allow --help without torch
    cfg = EmbeddingConfig(
        chunk_path=Path(args.chunks),
        metadata_output=Path(args.meta_output),
        embedding_output=Path(args.embeddings),
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        normalize=not args.no_normalize,
    )
    emb_path, meta_path, total, dim = generate_embeddings(cfg)
    print(f"Wrote {total} embedding(s) with dimension {dim} to {emb_path}")
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
