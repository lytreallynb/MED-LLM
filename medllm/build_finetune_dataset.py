"""Build an instruction fine-tuning dataset from FDA label chunks.

Generates training examples for the four behaviors named in finetune.py's
module docstring: evidence citation, structured response format, safety
disclaimers, and refusal when the evidence does not answer the question.
Medical facts are NOT a training target; answers are extractive from the
provided evidence so the model learns form, not facts.

Each example presents numbered evidence passages plus a question. Positive
examples cite the passage that answers the question. Negative examples pair a
question with evidence that does not answer it, and the correct output is an
abstention. Roughly one third of examples are negative so the model learns to
decline, not just to answer.

Half of the examples (--hard-fraction) are HARD: the distractor passage comes
from the SAME drug's label but a different section. An easy negative can be
solved by matching drug names; a hard negative shows the right drug and still
does not answer the question, so the model must read the section content.
Sections whose content overlaps (warnings / contraindications / adverse
reactions, and indications / clinical pharmacology) are never used as hard
distractors for each other, to keep the abstention label unambiguous.

The train/test split is by document_id BEFORE generation, so no drug's label
appears on both sides.

Usage:
    python -m medllm.build_finetune_dataset \
        --chunks data/clean/chunks_full.jsonl \
        --output-dir data/finetune \
        --seed 17
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

QUESTION_TEMPLATES: Dict[str, str] = {
    "indications_and_usage": "What is {drug} used for?",
    "warnings": "What warnings are associated with {drug}?",
    "contraindications": "When should {drug} not be used?",
    "adverse_reactions": "What are the adverse reactions of {drug}?",
    "dosage_and_administration": "What is the recommended dosage of {drug}?",
    "clinical_pharmacology": "How does {drug} work?",
}

DISCLAIMER = (
    "Disclaimer: This information is for educational purposes only. "
    "Consult a healthcare provider for medical advice."
)

ABSTAIN_TEXT = (
    "The provided evidence does not contain the answer to this question. "
    "I cannot answer without supporting FDA label evidence.\n\n" + DISCLAIMER
)

INSTRUCTION = (
    "Answer the question using ONLY the numbered evidence passages. "
    "Cite the passage you used as [n]. If the evidence does not contain "
    "the answer, say so and decline to answer."
)

# Sections in the same group overlap in content (a warnings passage often
# repeats contraindications), so a same-drug chunk may only serve as a hard
# distractor for a question about a section in a DIFFERENT group.
SECTION_GROUP: Dict[str, str] = {
    "warnings": "safety",
    "contraindications": "safety",
    "adverse_reactions": "safety",
    "indications_and_usage": "purpose",
    "clinical_pharmacology": "purpose",
    "dosage_and_administration": "dosage",
}

# Evidence shorter than this is too thin to anchor an extractive answer.
MIN_CHUNK_CHARS = 120
# Some FDA records carry a set-id UUID instead of a drug name; those chunks
# cannot produce a readable question or answer and are skipped.
UUID_NAME = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}", re.IGNORECASE)
# Answers are cut at a sentence boundary at or before this length.
MAX_ANSWER_CHARS = 400


def load_chunks(path: Path) -> List[dict]:
    chunks: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("section") not in QUESTION_TEMPLATES:
                continue
            if len(record.get("text", "")) < MIN_CHUNK_CHARS:
                continue
            name = record.get("drug_name") or ""
            if not name or UUID_NAME.match(name):
                continue
            chunks.append(record)
    return chunks


def split_by_document(
    chunks: List[dict], test_fraction: float, rng: random.Random
) -> Tuple[List[dict], List[dict]]:
    """Split chunks so no document_id crosses the train/test boundary."""
    by_doc: Dict[str, List[dict]] = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk["document_id"]].append(chunk)
    doc_ids = sorted(by_doc)
    rng.shuffle(doc_ids)
    n_test = max(1, int(len(doc_ids) * test_fraction))
    test_ids = set(doc_ids[:n_test])
    train = [c for d in doc_ids[n_test:] for c in by_doc[d]]
    test = [c for d in doc_ids[:n_test] for c in by_doc[d]]
    return train, test


def extractive_answer(drug: str, ref: int, text: str) -> str:
    """Condense the chunk into an answer that cites it, cut at a sentence end."""
    body = re.sub(r"\s+", " ", text).strip()
    if len(body) > MAX_ANSWER_CHARS:
        cut = body.rfind(". ", 0, MAX_ANSWER_CHARS)
        body = body[: cut + 1] if cut > 0 else body[:MAX_ANSWER_CHARS]
    return (
        f"According to the FDA label for {drug} [{ref}]: {body}\n\n{DISCLAIMER}"
    )


def build_input(passages: List[dict], question: str) -> str:
    lines = []
    for i, p in enumerate(passages):
        text = re.sub(r"\s+", " ", p["text"])[:600]
        lines.append(f"[{i + 1}] ({p['drug_name']}, {p['section']}) {text}")
    return "Evidence:\n" + "\n".join(lines) + f"\n\nQuestion: {question}"


def build_examples(
    chunks: List[dict],
    negative_fraction: float,
    rng: random.Random,
    hard_fraction: float = 0.5,
) -> List[dict]:
    """Generate one example per usable chunk.

    Positive: the gold chunk plus one distractor; the output cites the gold
    passage. Negative: distractors only; the output abstains. Hard examples
    (hard_fraction of those where the label has a usable other section) draw
    the distractor from the same document, so drug-name matching alone cannot
    solve them.
    """
    by_doc: Dict[str, List[dict]] = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk["document_id"]].append(chunk)

    examples: List[dict] = []
    for chunk in chunks:
        others = [
            c
            for c in chunks
            if c["document_id"] != chunk["document_id"]
        ]
        if len(others) < 2:
            continue
        same_doc = [
            c
            for c in by_doc[chunk["document_id"]]
            if c is not chunk
            and SECTION_GROUP[c["section"]] != SECTION_GROUP[chunk["section"]]
        ]
        question = QUESTION_TEMPLATES[chunk["section"]].format(
            drug=chunk["drug_name"].title()
        )
        is_negative = rng.random() < negative_fraction
        is_hard = bool(same_doc) and rng.random() < hard_fraction
        if is_negative:
            if is_hard:
                passages = [rng.choice(same_doc), rng.choice(others)]
            else:
                passages = rng.sample(others, 2)
            rng.shuffle(passages)
            output = ABSTAIN_TEXT
            gold_ref = None
        else:
            distractor = rng.choice(same_doc) if is_hard else rng.choice(others)
            passages = [chunk, distractor]
            rng.shuffle(passages)
            gold_ref = passages.index(chunk) + 1
            output = extractive_answer(
                chunk["drug_name"].title(), gold_ref, chunk["text"]
            )
        examples.append(
            {
                "instruction": INSTRUCTION,
                "input": build_input(passages, question),
                "output": output,
                "meta": {
                    "document_id": chunk["document_id"],
                    "section": chunk["section"],
                    "is_negative": is_negative,
                    "is_hard": is_hard,
                    "gold_ref": gold_ref,
                    "n_passages": len(passages),
                },
            }
        )
    return examples


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/finetune"))
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--negative-fraction", type=float, default=0.33)
    parser.add_argument("--hard-fraction", type=float, default=0.5,
                        help="fraction of examples whose distractor comes from "
                             "the same drug's label (0 = old easy-only behavior)")
    parser.add_argument("--max-train", type=int, default=1500)
    parser.add_argument("--max-test", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    chunks = load_chunks(args.chunks)
    train_chunks, test_chunks = split_by_document(chunks, args.test_fraction, rng)

    train_docs = {c["document_id"] for c in train_chunks}
    test_docs = {c["document_id"] for c in test_chunks}
    assert not (train_docs & test_docs), "document_id leaked across the split"

    train = build_examples(
        train_chunks, args.negative_fraction, rng, args.hard_fraction
    )
    test = build_examples(
        test_chunks, args.negative_fraction, rng, args.hard_fraction
    )
    rng.shuffle(train)
    rng.shuffle(test)
    train = train[: args.max_train]
    test = test[: args.max_test]

    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "test.jsonl", test)

    def stats(rows: List[dict]) -> str:
        neg = sum(1 for r in rows if r["meta"]["is_negative"])
        hard = sum(1 for r in rows if r["meta"]["is_hard"])
        return (
            f"{len(rows)} examples, {neg} negative "
            f"({neg / max(len(rows), 1):.0%}), {hard} hard "
            f"({hard / max(len(rows), 1):.0%})"
        )

    print(f"chunks used: {len(chunks)} across {len(train_docs | test_docs)} documents")
    print(f"train: {stats(train)}  ({len(train_docs)} documents)")
    print(f"test:  {stats(test)}  ({len(test_docs)} documents)")


if __name__ == "__main__":
    main()
