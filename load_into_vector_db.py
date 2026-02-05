"""Load FDA embeddings into a FAISS vector database for retrieval."""
from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", default="data/clean/fda_embeddings.npy", help="Path to the numpy embedding matrix")
    parser.add_argument("--metadata", default="data/clean/fda_meta.jsonl", help="Aligned metadata JSONL file")
    parser.add_argument("--index", default="data/clean/fda.index", help="Destination for the FAISS index")
    parser.add_argument("--metric", choices=["cosine", "l2"], default="cosine", help="Similarity metric for FAISS")
    parser.add_argument("--batch-size", type=int, default=2048, help="Number of embeddings to add per FAISS call")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from medllm.indexer import IndexBuilderConfig, build_faiss_index  # defer heavy faiss import until needed
    cfg = IndexBuilderConfig(
        embedding_path=Path(args.embeddings),
        metadata_path=Path(args.metadata),
        index_path=Path(args.index),
        metric=args.metric,
        add_batch_size=args.batch_size,
    )
    index_path, total = build_faiss_index(cfg)
    print(f"FAISS index with {total} vector(s) saved to {index_path}")


if __name__ == "__main__":
    main()
