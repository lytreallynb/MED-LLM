#!/usr/bin/env python3
"""Generate a markdown spot-check review sheet from eval_behavior.py outputs.

Samples a seeded, stratified subset of test-set indices (preserving roughly
the negative/positive ratio of the test set, read from the first eval file's
records) and writes one review block per sampled index: the question, then
for each eval file the model's verbatim output, the automated judgments, and
a checkbox for the human reviewer. The same indices are used across all eval
files so outputs are directly comparable.

Usage:
    python scripts/sample_spot_check.py \
        --eval-files results/behavior_base.json results/behavior_tuned.json \
        --test-data data/finetune/test.jsonl \
        --n 50 --seed 17 \
        --output results/spot_check.md
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, NoReturn

JUDGMENT_KEYS = (
    "abstained",
    "citations",
    "citations_all_valid",
    "cited_gold",
    "has_disclaimer",
)


def fail(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(1)


def load_eval_file(path: Path) -> dict:
    if not path.exists():
        fail(
            f"eval file not found: {path}\n"
            "  This file is produced by 'python -m medllm.eval_behavior'.\n"
            "  If the evaluation run is still in progress, wait for it to\n"
            "  finish, then run this script again."
        )
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        fail(
            f"eval file is not valid JSON: {path} ({exc})\n"
            "  If the evaluation run is still writing this file, wait for it\n"
            "  to finish, then run this script again."
        )
    if not isinstance(data, dict) or not isinstance(data.get("records"), list):
        fail(
            f"eval file has no 'records' list: {path}\n"
            "  Expected the JSON structure written by medllm/eval_behavior.py."
        )
    return data


def load_test_inputs(path: Path) -> List[str]:
    if not path.exists():
        fail(f"test data not found: {path}")
    inputs: List[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            inputs.append(str(row.get("input", "")))
    return inputs


def question_line(input_text: str) -> str:
    for line in input_text.splitlines():
        if line.strip().startswith("Question:"):
            return line.strip()
    return "(no 'Question:' line found in this input)"


def model_label(data: dict, path: Path) -> str:
    model = str(data.get("model") or path.stem)
    adapter = data.get("adapter")
    if adapter:
        return f"{model} + adapter {adapter}"
    return f"{model} (base, no adapter)"


def stratified_indices(records: List[dict], n: int, seed: int) -> List[int]:
    """Sample n indices, preserving the negative/positive ratio of records."""
    negatives = [int(r["index"]) for r in records if r.get("should_abstain")]
    positives = [int(r["index"]) for r in records if not r.get("should_abstain")]
    total = len(records)
    if total == 0:
        fail("first eval file has zero records")
    n = min(n, total)
    n_neg = min(len(negatives), round(n * len(negatives) / total))
    n_pos = min(len(positives), n - n_neg)
    n_neg = min(len(negatives), n - n_pos)
    rng = random.Random(seed)
    chosen = rng.sample(negatives, n_neg) + rng.sample(positives, n_pos)
    return sorted(chosen)


def fence_for(text: str) -> str:
    """Return a backtick fence longer than any backtick run in text."""
    fence = "```"
    while fence in text:
        fence += "`"
    return fence


def judgments_line(record: dict) -> str:
    parts = [f"{key}={json.dumps(record.get(key))}" for key in JUDGMENT_KEYS]
    return ", ".join(parts)


def build_sheet(
    eval_paths: List[Path],
    eval_datas: List[dict],
    test_inputs: List[str],
    test_data_path: Path,
    indices: List[int],
    seed: int,
) -> str:
    record_maps: List[Dict[int, dict]] = [
        {int(r["index"]): r for r in data["records"]} for data in eval_datas
    ]
    labels = [model_label(d, p) for d, p in zip(eval_datas, eval_paths)]

    n_neg = sum(1 for i in indices if record_maps[0][i].get("should_abstain"))
    n_pos = len(indices) - n_neg

    lines: List[str] = []
    lines.append("# Spot-check review sheet")
    lines.append("")
    lines.append(
        f"Sampled {len(indices)} test indices "
        f"({n_neg} negative, {n_pos} positive), seed {seed}."
    )
    lines.append("")
    lines.append("Eval files:")
    for path, label in zip(eval_paths, labels):
        lines.append(f"- `{path}`: {label}")
    lines.append("")
    lines.append(f"Test data: `{test_data_path}`")
    lines.append("")
    lines.append(
        "For each output, check the box if the automated judgments match "
        "your own reading of the output."
    )
    lines.append("")

    for idx in indices:
        expected = "abstain" if record_maps[0][idx].get("should_abstain") else "answer"
        lines.append(f"## Index {idx} (expected: {expected})")
        lines.append("")
        if 0 <= idx < len(test_inputs):
            lines.append(question_line(test_inputs[idx]))
        else:
            lines.append(f"(index {idx} is out of range for {test_data_path})")
        lines.append("")
        for record_map, label in zip(record_maps, labels):
            lines.append(f"### {label}")
            lines.append("")
            record = record_map.get(idx)
            if record is None:
                lines.append("(no record with this index in this eval file)")
                lines.append("")
                continue
            output = str(record.get("output", ""))
            fence = fence_for(output)
            lines.append(fence)
            lines.append(output)
            lines.append(fence)
            lines.append("")
            lines.append(f"Automated judgments: {judgments_line(record)}")
            lines.append("")
            lines.append("- [ ] human agrees")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a markdown spot-check review sheet from "
        "eval_behavior.py output files.",
    )
    parser.add_argument(
        "--eval-files",
        type=Path,
        nargs="+",
        required=True,
        help="one or more behavior_*.json files written by eval_behavior.py",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        required=True,
        help="test.jsonl with 'input' fields (data/finetune/test.jsonl)",
    )
    parser.add_argument(
        "--n", type=int, default=50, help="number of indices to sample"
    )
    parser.add_argument("--seed", type=int, default=17, help="sampling seed")
    parser.add_argument(
        "--output", type=Path, required=True, help="markdown output path"
    )
    args = parser.parse_args()

    if args.n <= 0:
        fail("--n must be a positive integer")

    eval_datas = [load_eval_file(path) for path in args.eval_files]
    test_inputs = load_test_inputs(args.test_data)
    indices = stratified_indices(eval_datas[0]["records"], args.n, args.seed)

    sheet = build_sheet(
        args.eval_files, eval_datas, test_inputs, args.test_data, indices, args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(sheet, encoding="utf-8")
    print(
        f"wrote {args.output}: {len(indices)} sampled indices across "
        f"{len(eval_datas)} eval file(s)"
    )


if __name__ == "__main__":
    main()
