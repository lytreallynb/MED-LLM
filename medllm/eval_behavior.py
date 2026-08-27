"""Behavioral evaluation for the grounding fine-tune, base vs adapter.

Measures the behaviors the fine-tune targets, on the held-out test split, with
retrieval untouched. Every metric is mechanically checkable; no LLM judge and
no API key.

  * Abstention precision / recall: on questions whose evidence does not
    contain the answer (is_negative), does the model decline? Reported
    separately so "abstain on everything" cannot score well.
  * Citation validity: every [n] in the output must point at a passage that
    was actually provided. For positives, gold-citation accuracy additionally
    checks that the cited passage is the one that answers the question.
  * Format compliance: the response carries the disclaimer line.
  * Answer faithfulness: the fraction of the answer's content words that
    appear in the cited evidence (see faithfulness.py). Catches answers whose
    citations are formally valid but whose content is fabricated.
  * Hard-subset breakdown: metrics repeated on examples whose distractor
    comes from the same drug's label (meta.is_hard), where drug-name matching
    alone cannot produce the right behavior.

Decoding is greedy and the prompt template matches training exactly, so base
and tuned models are compared under identical conditions.

Usage:
    python -m medllm.eval_behavior \
        --test-data data/finetune/test.jsonl \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter output/lora_grounding_v1/lora_adapter \
        --output results/behavior_tuned.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

try:
    from medllm.faithfulness import faithfulness_score, parse_passages
except ImportError:  # run as a standalone script from inside medllm/
    from faithfulness import faithfulness_score, parse_passages

PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

# Model-agnostic abstention detector. The base model will not use our trained
# phrasing, so any reasonable way of declining must count for both models.
ABSTAIN_PATTERNS = [
    r"does not contain",
    r"cannot answer",
    r"can't answer",
    r"cannot provide",
    r"unable to answer",
    r"unable to provide",
    r"no (?:relevant )?(?:information|evidence)",
    r"not (?:contained|provided|available|found) in",
    r"insufficient (?:information|evidence)",
    r"decline to answer",
    r"outside (?:my|the) (?:scope|area)",
]
ABSTAIN_RE = re.compile("|".join(ABSTAIN_PATTERNS), re.IGNORECASE)

CITATION_RE = re.compile(r"\[(\d{1,2})\]")
DISCLAIMER_RE = re.compile(r"disclaimer\s*:", re.IGNORECASE)


def detect_abstention(text: str) -> bool:
    return bool(ABSTAIN_RE.search(text))


def extract_citations(text: str) -> List[int]:
    return [int(m) for m in CITATION_RE.findall(text)]


def score_example(
    output: str, meta: dict, passages: Optional[List[str]] = None
) -> Dict[str, object]:
    """Score one generated response against the example's metadata."""
    abstained = detect_abstention(output)
    citations = extract_citations(output)
    n_passages = meta["n_passages"]
    valid_citations = [c for c in citations if 1 <= c <= n_passages]
    faithfulness = None
    if passages and not abstained:
        faithfulness = faithfulness_score(output, passages, citations)
    return {
        "abstained": abstained,
        "should_abstain": meta["is_negative"],
        "is_hard": bool(meta.get("is_hard", False)),
        "citations": citations,
        "citations_all_valid": bool(citations) and len(valid_citations) == len(citations),
        "cited_gold": (
            meta["gold_ref"] in citations if meta["gold_ref"] is not None else None
        ),
        "faithfulness": faithfulness,
        "has_disclaimer": bool(DISCLAIMER_RE.search(output)),
    }


def aggregate(rows: List[Dict[str, object]]) -> Dict[str, object]:
    """Aggregate per-example scores into the reported metrics."""
    negatives = [r for r in rows if r["should_abstain"]]
    positives = [r for r in rows if not r["should_abstain"]]
    predicted_abstain = [r for r in rows if r["abstained"]]
    true_abstain = [r for r in negatives if r["abstained"]]

    def safe_div(a: int, b: int) -> Optional[float]:
        return round(a / b, 4) if b else None

    pos_with_citation = [r for r in positives if r["citations"]]
    faith_scores = [
        r["faithfulness"] for r in positives if r.get("faithfulness") is not None
    ]
    metrics = {
        "n": len(rows),
        "n_negative": len(negatives),
        "n_positive": len(positives),
        "abstention_precision": safe_div(len(true_abstain), len(predicted_abstain)),
        "abstention_recall": safe_div(len(true_abstain), len(negatives)),
        "false_abstain_rate_on_positives": safe_div(
            sum(1 for r in positives if r["abstained"]), len(positives)
        ),
        "positive_citation_rate": safe_div(len(pos_with_citation), len(positives)),
        "citation_validity": safe_div(
            sum(1 for r in pos_with_citation if r["citations_all_valid"]),
            len(pos_with_citation),
        ),
        "gold_citation_accuracy": safe_div(
            sum(1 for r in positives if r["cited_gold"]), len(positives)
        ),
        "answer_faithfulness": (
            round(sum(faith_scores) / len(faith_scores), 4) if faith_scores else None
        ),
        "low_faithfulness_rate": safe_div(
            sum(1 for s in faith_scores if s < 0.5), len(faith_scores)
        ),
        "disclaimer_rate": safe_div(
            sum(1 for r in rows if r["has_disclaimer"]), len(rows)
        ),
    }

    # Hard subset: distractor from the same drug's label, so drug-name
    # matching cannot solve it. Only present when the dataset marks it.
    hard = [r for r in rows if r.get("is_hard")]
    if hard:
        hard_neg = [r for r in hard if r["should_abstain"]]
        hard_pos = [r for r in hard if not r["should_abstain"]]
        metrics["n_hard"] = len(hard)
        metrics["abstention_recall_hard"] = safe_div(
            sum(1 for r in hard_neg if r["abstained"]), len(hard_neg)
        )
        metrics["gold_citation_accuracy_hard"] = safe_div(
            sum(1 for r in hard_pos if r["cited_gold"]), len(hard_pos)
        )
    return metrics


def load_model(model_name: str, adapter_path: Optional[str], load_in_4bit: bool = False,
               base_adapter: Optional[str] = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.bfloat16
    else:
        device, dtype = "cpu", torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if load_in_4bit:
        if device != "cuda":
            raise RuntimeError("--load-in-4bit requires a CUDA GPU")
        from transformers import BitsAndBytesConfig

        # A 7B model in fp16 does not fit a 16GB GPU alongside its KV cache,
        # so both base and tuned are evaluated in the same NF4 quantization.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            ),
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, dtype=dtype
        )
    if base_adapter:
        # A second-stage adapter (e.g. GRPO) is trained relative to the base
        # with the first-stage (SFT) adapter already merged in. Reproduce that
        # exact stack: merge the first adapter, then attach the second.
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, base_adapter)
        model = model.merge_and_unload()
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    if not load_in_4bit:
        model.to(device)
    model.eval()
    return model, tokenizer, device


def generate(model, tokenizer, device: str, prompt: str, max_new_tokens: int) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--base-adapter", default=None,
                        help="first-stage adapter to merge into the base before --adapter")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=140)
    parser.add_argument("--limit", type=int, default=0, help="0 = all examples")
    parser.add_argument(
        "--load-in-4bit", action="store_true",
        help="Evaluate with the base model quantized to 4-bit NF4 (CUDA only)"
    )
    args = parser.parse_args()

    examples = [json.loads(l) for l in args.test_data.open()]
    if args.limit:
        examples = examples[: args.limit]

    model, tokenizer, device = load_model(args.model, args.adapter, args.load_in_4bit, args.base_adapter)
    print(f"model={args.model} adapter={args.adapter or 'none'} device={device}")

    records: List[dict] = []
    for i, ex in enumerate(examples):
        prompt = PROMPT_TEMPLATE.format(
            instruction=ex["instruction"], input=ex["input"]
        )
        output = generate(model, tokenizer, device, prompt, args.max_new_tokens)
        scores = score_example(output, ex["meta"], parse_passages(ex["input"]))
        records.append({"index": i, "output": output, **scores})
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(examples)}")

    metrics = aggregate(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "model": args.model,
                "adapter": args.adapter,
                "base_adapter": args.base_adapter,
                "test_data": str(args.test_data),
                "decoding": {"greedy": True, "max_new_tokens": args.max_new_tokens},
                "load_in_4bit": args.load_in_4bit,
                "metrics": metrics,
                "records": records,
            },
            fh,
            ensure_ascii=False,
            indent=1,
        )
    print(json.dumps(metrics, indent=1))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
