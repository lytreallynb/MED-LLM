"""Lightweight evaluation harness for MED-LLM RAG pipeline."""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

from tqdm import tqdm

from .embeddings import EmbeddingModelConfig
from .retrieval import QwenChatClient, RagQueryEngine, RetrievalConfig, SafetyConfig


@dataclass
class DatasetConfig:
    name: str
    path: Path
    max_questions: int | None = None


@dataclass
class EvaluationResult:
    dataset: str
    total: int
    accuracy: float
    hallucination_rate: float
    grounding_correctness: float
    completeness: float
    retrieval_hit_rate: float
    recall_at_k: float
    mrr: float
    avg_top_score: float
    avg_latency_ms: float


class EvaluationSuite:
    def __init__(self, engine: RagQueryEngine):
        self.engine = engine
        # Per-question rows collected during evaluate(), keyed by dataset name
        self.details: Dict[str, List[Dict[str, object]]] = {}

    def _load_records(self, config: DatasetConfig) -> Iterator[Dict[str, str]]:
        path = config.path
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        ext = path.suffix.lower()
        count = 0
        if ext in {".jsonl", ".json"}:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    yield record
                    count += 1
                    if config.max_questions and count >= config.max_questions:
                        break
        elif ext == ".csv":
            with path.open("r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    yield row
                    count += 1
                    if config.max_questions and count >= config.max_questions:
                        break
        else:
            raise ValueError(f"Unsupported dataset extension: {ext}")

    def evaluate(self, config: DatasetConfig) -> EvaluationResult:
        total = 0
        correct = 0
        hallucinations = 0
        grounded = 0
        complete = 0
        retrieval_hits = 0
        chunk_recalls = 0
        chunk_labeled = 0
        reciprocal_ranks: List[float] = []
        top_scores: List[float] = []
        latencies_ms: List[float] = []
        detail_rows: List[Dict[str, object]] = []
        self.details[config.name] = detail_rows
        for record in tqdm(self._load_records(config), desc=f"Evaluating {config.name}"):
            question = record.get("question") or record.get("prompt")
            if not question:
                continue
            answer_key = (record.get("answer") or record.get("output") or "").strip()
            expected_chunk_id = (record.get("chunk_id") or "").strip()
            started = time.perf_counter()
            response = self.engine.query(question)
            latencies_ms.append((time.perf_counter() - started) * 1000.0)
            total += 1
            normalized_answer = response.answer.lower()
            if answer_key and answer_key.lower() in normalized_answer:
                correct += 1
            if "does not contain that information" in normalized_answer or any(
                note in {"low_similarity", "no_hits"} for note in response.safety_notes
            ):
                hallucinations += 1
            if not any(note in {"low_similarity", "no_hits"} for note in response.safety_notes):
                grounded += 1
            if len(response.retrieved) >= 2:
                complete += 1
            retrieved_text = " ".join(item.metadata.text.lower() for item in response.retrieved)
            if answer_key and answer_key.lower() in retrieved_text:
                retrieval_hits += 1
            if response.retrieved:
                top_scores.append(response.retrieved[0].score)
            rank = 0
            if expected_chunk_id:
                chunk_labeled += 1
                for position, item in enumerate(response.retrieved, start=1):
                    if item.metadata.chunk_id == expected_chunk_id:
                        rank = position
                        break
                if rank:
                    chunk_recalls += 1
                    reciprocal_ranks.append(1.0 / rank)
                else:
                    reciprocal_ranks.append(0.0)
            detail_rows.append(
                {
                    "question": question,
                    "section": record.get("section") or "",
                    "drug_name": record.get("drug_name") or "",
                    "hit": bool(answer_key) and answer_key.lower() in retrieved_text,
                    "rank": rank,
                    "top_score": response.retrieved[0].score if response.retrieved else 0.0,
                    "latency_ms": latencies_ms[-1],
                }
            )
            if config.max_questions and total >= config.max_questions:
                break
        if total == 0:
            raise RuntimeError(f"No evaluable questions found in {config.path}")
        return EvaluationResult(
            dataset=config.name,
            total=total,
            accuracy=correct / total,
            hallucination_rate=hallucinations / total,
            grounding_correctness=grounded / total,
            completeness=complete / total,
            retrieval_hit_rate=retrieval_hits / total,
            recall_at_k=(chunk_recalls / chunk_labeled) if chunk_labeled else 0.0,
            mrr=(sum(reciprocal_ranks) / len(reciprocal_ranks)) if reciprocal_ranks else 0.0,
            avg_top_score=(sum(top_scores) / len(top_scores)) if top_scores else 0.0,
            avg_latency_ms=(sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0,
        )


def _parse_dataset_arg(raw: str) -> DatasetConfig:
    try:
        name_part, rest = raw.split("=", 1)
    except ValueError as exc:
        raise ValueError("Datasets must be formatted as name=/path/to/file[:limit]") from exc
    if ":" in rest:
        path_part, limit_part = rest.rsplit(":", 1)
        max_questions = int(limit_part)
    else:
        path_part = rest
        max_questions = None
    return DatasetConfig(name=name_part, path=Path(path_part), max_questions=max_questions)


def run_cli(engine: RagQueryEngine, dataset_args: Sequence[str]) -> tuple[List[EvaluationResult], EvaluationSuite]:
    configs = [_parse_dataset_arg(arg) for arg in dataset_args]
    suite = EvaluationSuite(engine)
    return [suite.evaluate(cfg) for cfg in configs], suite


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Evaluation dataset spec formatted as name=/path/to/file[:max_questions]",
    )
    parser.add_argument("--index", default="data/clean/fda.index")
    parser.add_argument("--metadata", default="data/clean/fda_meta.jsonl")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--qwen-model", default="qwen2.5-72b-instruct")
    parser.add_argument("--no-qwen", action="store_true")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write results as JSON (consumed by the metrics dashboard)",
    )
    return parser


def write_results(results: Sequence[EvaluationResult], output_path: Path, extra: Dict[str, object] | None = None) -> None:
    payload: Dict[str, object] = {"results": [asdict(result) for result in results]}
    if extra:
        payload.update(extra)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as sink:
        json.dump(payload, sink, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    retrieval_cfg = RetrievalConfig(index_path=Path(args.index), metadata_path=Path(args.metadata))
    embed_cfg = EmbeddingModelConfig(model_name=args.embedding_model, device=args.device)
    safety_cfg = SafetyConfig()
    qwen_client = None
    if not args.no_qwen:
        qwen_client = QwenChatClient(model_name=args.qwen_model)
    engine = RagQueryEngine(retrieval_cfg, embed_cfg, safety_cfg, qwen_client)
    results, suite = run_cli(engine, args.dataset)
    for result in results:
        print(
            f"Dataset: {result.dataset}\n"
            f"  Total: {result.total}\n"
            f"  Accuracy: {result.accuracy:.3f}\n"
            f"  Hallucination Rate: {result.hallucination_rate:.3f}\n"
            f"  Grounding Correctness: {result.grounding_correctness:.3f}\n"
            f"  Completeness: {result.completeness:.3f}\n"
            f"  Retrieval Hit Rate: {result.retrieval_hit_rate:.3f}\n"
            f"  Recall@k: {result.recall_at_k:.3f}\n"
            f"  MRR: {result.mrr:.3f}\n"
            f"  Avg Top Score: {result.avg_top_score:.3f}\n"
            f"  Avg Latency: {result.avg_latency_ms:.0f} ms\n"
        )
    if args.output:
        run_config = {
            "embedding_model": args.embedding_model,
            "qwen_model": None if args.no_qwen else args.qwen_model,
            "index": args.index,
            "top_k": retrieval_cfg.top_k,
        }
        write_results(results, Path(args.output), extra={"config": run_config, "details": suite.details})
        print(f"Results written to {args.output}")
