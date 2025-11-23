"""Generate dense embeddings for FDA chunks using Qwen models."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

import numpy as np
from tqdm import tqdm
import math

from .chunking import ChunkMetadata

try:  # Heavy imports are optional until embeddings are actually generated
    import torch
except ImportError as exc:  # pragma: no cover - torch is required at runtime
    raise RuntimeError("PyTorch must be installed to compute embeddings") from exc

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - fallback to raw transformers
    SentenceTransformer = None  # type: ignore

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("transformers must be installed to load Qwen models") from exc


@dataclass
class EmbeddingModelConfig:
    model_name: str = "Qwen/Qwen2.5-Embedding-1.8B"
    batch_size: int = 8
    max_length: int = 1024
    device: str | None = None
    normalize: bool = True


@dataclass
class EmbeddingConfig:
    chunk_path: Path = Path("data/clean/chunks.jsonl")
    metadata_output: Path = Path("data/clean/fda_meta.jsonl")
    embedding_output: Path = Path("data/clean/fda_embeddings.npy")
    model_name: str = "Qwen/Qwen2.5-Embedding-1.8B"
    batch_size: int = 8
    max_length: int = 1024
    device: str | None = None
    normalize: bool = True

    def model_config(self) -> EmbeddingModelConfig:
        return EmbeddingModelConfig(
            model_name=self.model_name,
            batch_size=self.batch_size,
            max_length=self.max_length,
            device=self.device,
            normalize=self.normalize,
        )


class QwenEmbeddingBackend:
    """Wrapper around a SentenceTransformer or raw HF Qwen embedding model."""

    def __init__(self, config: EmbeddingModelConfig):
        self.config = config
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._mode = "hf"
        self._sentence_transformer: SentenceTransformer | None = None
        self._tokenizer = None
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        model_name = self.config.model_name
        if SentenceTransformer is not None:
            try:
                self._sentence_transformer = SentenceTransformer(model_name, device=self.device)
                self._mode = "sentence-transformer"
                return
            except Exception:
                self._sentence_transformer = None
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self._model.to(self.device)
        self._model.eval()

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        if self._mode == "sentence-transformer" and self._sentence_transformer is not None:
            embeddings = self._sentence_transformer.encode(
                texts,
                batch_size=min(self.config.batch_size, len(texts)),
                normalize_embeddings=self.config.normalize,
                convert_to_numpy=True,
            )
            if isinstance(embeddings, list):
                embeddings = np.asarray(embeddings, dtype=np.float32)
            return embeddings.astype(np.float32)
        assert self._tokenizer is not None and self._model is not None
        tokenized = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        with torch.no_grad():
            outputs = self._model(**tokenized)
        if hasattr(outputs, "embeddings"):
            pooled = outputs.embeddings
        elif isinstance(outputs, tuple):
            pooled = outputs[0]
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:  # mean pool last hidden state
            hidden = outputs.last_hidden_state
            attention_mask = tokenized.get("attention_mask")
            mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            summed = torch.sum(hidden * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = summed / counts
        pooled = pooled.detach().float().cpu().numpy()
        if self.config.normalize:
            norms = np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-12
            pooled = pooled / norms
        return pooled.astype(np.float32)


def _iter_chunks(chunk_path: Path) -> Iterator[ChunkMetadata]:
    with chunk_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            yield ChunkMetadata(**payload)


def _batched(iterable: Iterable[ChunkMetadata], batch_size: int) -> Iterator[List[ChunkMetadata]]:
    batch: List[ChunkMetadata] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def generate_embeddings(config: EmbeddingConfig) -> tuple[Path, Path, int, int]:
    """Embed chunks and persist aligned numpy + metadata files."""
    chunk_path = Path(config.chunk_path)
    if not chunk_path.exists():
        raise FileNotFoundError(f"Chunk file not found: {chunk_path}")
    config.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    config.embedding_output.parent.mkdir(parents=True, exist_ok=True)

    backend = QwenEmbeddingBackend(config.model_config())
    total_records = _count_lines(chunk_path)
    if total_records == 0:
        raise ValueError("Chunk file is empty")

    memmap = None
    embedding_dim = None
    written = 0

    total_batches = max(1, math.ceil(total_records / config.batch_size))
    with config.metadata_output.open("w", encoding="utf-8") as meta_sink:
        for batch in tqdm(_batched(_iter_chunks(chunk_path), config.batch_size), total=total_batches, desc="Embedding chunks"):
            texts = [item.text for item in batch]
            vectors = backend.embed_texts(texts)
            if vectors.size == 0:
                continue
            if memmap is None:
                embedding_dim = vectors.shape[1]
                memmap = np.memmap(
                    config.embedding_output,
                    dtype=np.float32,
                    mode="w+",
                    shape=(total_records, embedding_dim),
                )
            assert memmap is not None and embedding_dim is not None
            end = written + len(batch)
            memmap[written:end] = vectors[: len(batch)]
            for item in batch:
                meta_sink.write(item.to_json())
                meta_sink.write("\n")
            written = end
    if memmap is not None:
        memmap.flush()
    if written != total_records:
        raise RuntimeError(
            f"Mismatch between metadata ({written}) and expected chunk count ({total_records})"
        )
    assert embedding_dim is not None
    return config.embedding_output, config.metadata_output, written, embedding_dim


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", default="data/clean/chunks.jsonl", help="Path to chunk JSONL file")
    parser.add_argument("--meta-output", default="data/clean/fda_meta.jsonl", help="Where to save chunk metadata aligned with embeddings")
    parser.add_argument("--embeddings", default="data/clean/fda_embeddings.npy", help="Path for the numpy embedding matrix")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Embedding-1.8B", help="Hugging Face model id for embeddings")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for embedding computation")
    parser.add_argument("--max-length", type=int, default=1024, help="Token truncation length")
    parser.add_argument("--device", default=None, help="Torch device override (cpu/cuda)")
    parser.add_argument("--no-normalize", action="store_true", help="Disable L2 normalization")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
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
    print(f"Wrote {total} embedding(s) with dimension {dim} to {emb_path}\nMetadata saved to {meta_path}")
