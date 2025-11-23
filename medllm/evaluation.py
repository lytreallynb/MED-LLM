"""Lightweight evaluation harness for MED-LLM RAG pipeline."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
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


class EvaluationSuite:
    def __init__(self, engine: RagQueryEngine):
        self.engine = engine

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
        for record in tqdm(self._load_records(config), desc=f"Evaluating {config.name}"):
            question = record.get("question") or record.get("prompt")
            if not question:
                continue
            answer_key = (record.get("answer") or record.get("output") or "").strip()
            response = self.engine.query(question)
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


def run_cli(engine: RagQueryEngine, dataset_args: Sequence[str]) -> List[EvaluationResult]:
    configs = [_parse_dataset_arg(arg) for arg in dataset_args]
    suite = EvaluationSuite(engine)
    return [suite.evaluate(cfg) for cfg in configs]


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
    parser.add_argument("--embedding-model", default="Qwen/Qwen2.5-Embedding-1.8B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--qwen-model", default="qwen2.5-72b-instruct")
    parser.add_argument("--no-qwen", action="store_true")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    retrieval_cfg = RetrievalConfig(index_path=Path(args.index), metadata_path=Path(args.metadata))
    embed_cfg = EmbeddingModelConfig(model_name=args.embedding_model, device=args.device)
    safety_cfg = SafetyConfig()
    qwen_client = None
    if not args.no_qwen:
        qwen_client = QwenChatClient(model_name=args.qwen_model)
    engine = RagQueryEngine(retrieval_cfg, embed_cfg, safety_cfg, qwen_client)
    results = run_cli(engine, args.dataset)
    for result in results:
        print(
            f"Dataset: {result.dataset}\n"
            f"  Total: {result.total}\n"
            f"  Accuracy: {result.accuracy:.3f}\n"
            f"  Hallucination Rate: {result.hallucination_rate:.3f}\n"
            f"  Grounding Correctness: {result.grounding_correctness:.3f}\n"
            f"  Completeness: {result.completeness:.3f}\n"
        )
