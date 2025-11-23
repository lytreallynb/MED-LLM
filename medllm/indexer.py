"""Build a FAISS index from persisted embeddings."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm


@dataclass
class IndexBuilderConfig:
    embedding_path: Path = Path("data/clean/fda_embeddings.npy")
    index_path: Path = Path("data/clean/fda.index")
    metadata_path: Path = Path("data/clean/fda_meta.jsonl")
    metric: str = "cosine"
    add_batch_size: int = 2048


def build_faiss_index(config: IndexBuilderConfig) -> tuple[Path, int]:
    embedding_path = Path(config.embedding_path)
    if not embedding_path.exists():
        raise FileNotFoundError(f"Embedding matrix not found: {embedding_path}")
    meta_path = Path(config.metadata_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    embeddings = np.load(embedding_path, mmap_mode="r")
    if embeddings.ndim != 2:
        raise ValueError("Embedding matrix must be 2-dimensional")
    num_vectors, dim = embeddings.shape
    if config.metric == "cosine":
        index = faiss.IndexFlatIP(dim)
    elif config.metric == "l2":
        index = faiss.IndexFlatL2(dim)
    else:
        raise ValueError(f"Unsupported metric: {config.metric}")
    for start in tqdm(range(0, num_vectors, config.add_batch_size), desc="Building FAISS index"):
        end = min(num_vectors, start + config.add_batch_size)
        batch = np.asarray(embeddings[start:end], dtype=np.float32)
        index.add(batch)
    faiss.write_index(index, str(config.index_path))
    return config.index_path, num_vectors


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", default="data/clean/fda_embeddings.npy", help="Path to the numpy embedding matrix")
    parser.add_argument("--metadata", default="data/clean/fda_meta.jsonl", help="Aligned metadata JSONL")
    parser.add_argument("--index", default="data/clean/fda.index", help="Where to store the FAISS index")
    parser.add_argument("--metric", choices=["cosine", "l2"], default="cosine", help="Similarity metric")
    parser.add_argument("--batch-size", type=int, default=2048, help="Number of embeddings to add per FAISS call")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    cfg = IndexBuilderConfig(
        embedding_path=Path(args.embeddings),
        metadata_path=Path(args.metadata),
        index_path=Path(args.index),
        metric=args.metric,
        add_batch_size=args.batch_size,
    )
    index_path, total = build_faiss_index(cfg)
    print(f"FAISS index with {total} vector(s) saved to {index_path}")
