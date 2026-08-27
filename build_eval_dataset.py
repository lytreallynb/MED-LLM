"""Build a retrieval-grounded evaluation dataset from FDA chunk metadata.

Each record pairs a templated question about a drug section with the chunk
that answers it, so the evaluation suite can measure retrieval quality
(recall@k, MRR) without requiring an LLM API key.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

QUESTION_TEMPLATES: Dict[str, str] = {
    "indications_and_usage": "What is {drug} used for?",
    "warnings": "What warnings are associated with {drug}?",
    "contraindications": "When should {drug} not be used?",
    "adverse_reactions": "What are the adverse reactions of {drug}?",
    "dosage_and_administration": "What is the recommended dosage of {drug}?",
    "clinical_pharmacology": "How does {drug} work?",
}


def build_dataset(meta_path: Path, output_path: Path, max_questions: int, seed: int) -> int:
    candidates: List[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    with meta_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            drug = (payload.get("drug_name") or "").strip()
            section = (payload.get("section") or "").strip()
            if not drug or drug.lower() in {"unknown", "n/a", ""}:
                continue
            template = QUESTION_TEMPLATES.get(section)
            if template is None:
                continue
            # One question per (drug, section) pair, anchored to its first chunk
            key = (drug.lower(), section)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            candidates.append(
                {
                    "question": template.format(drug=drug),
                    "answer": drug,
                    "chunk_id": payload.get("chunk_id"),
                    "drug_name": drug,
                    "section": section,
                }
            )
    if not candidates:
        raise RuntimeError(f"No usable (drug, section) pairs found in {meta_path}")
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:max_questions]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as sink:
        for record in selected:
            sink.write(json.dumps(record, ensure_ascii=False))
            sink.write("\n")
    return len(selected)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="data/clean/fda_meta.jsonl", help="Chunk metadata JSONL")
    parser.add_argument("--output", default="data/eval/fda_eval.jsonl", help="Where to write the eval dataset")
    parser.add_argument("--max-questions", type=int, default=100, help="Number of questions to generate")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for reproducibility")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    total = build_dataset(Path(args.metadata), Path(args.output), args.max_questions, args.seed)
    print(f"Wrote {total} evaluation question(s) to {args.output}")
