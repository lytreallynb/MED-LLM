"""GRPO reinforcement learning on top of the SFT grounding adapter.

Stage two of the post-training pipeline: after supervised LoRA instruction
tuning (finetune.py), this script applies GRPO (Group Relative Policy
Optimization) with PROGRAMMATIC rewards. The model samples multiple
completions per prompt; each is scored by the same mechanical checks the
evaluation harness uses (citation validity, gold-passage citation, abstention
correctness, disclaimer); the policy is updated toward higher-scoring
completions. No reward model, no human labels, no LLM judge.

Reward design (per completion, summed):
  * Negative prompts (evidence does not answer): abstain +1.0, answer -1.0
  * Positive prompts: answer +0.5, abstain -1.0
  * Positive and cites the gold passage: +1.0
  * Positive and answered: faithfulness in [-1.0, +1.0], linear in the
    fraction of the answer's content words supported by the cited evidence
    (2 * score - 1). Fully fabricated content costs as much as abstaining.
  * Cites any passage number that was not provided: -0.5
  * Disclaimer line present: +0.25

The asymmetry (false abstention punished as hard as false answering) is
deliberate: SFT already taught the format, and GRPO must not collapse into
"always refuse". The faithfulness term is the only continuous one, and it
matters for GRPO specifically: once the binary checks saturate, every sample
in a group earns the same reward, the group advantage is zero, and the step
produces no gradient. A continuous term keeps within-group variance alive.

Usage (Colab T4, after the SFT adapter exists):
    python grpo_finetune.py \
        --train-data data/train.jsonl \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --sft-adapter output/lora_grounding_v1/lora_adapter \
        --output-dir output/grpo_grounding_v1 \
        --max-steps 200

Requires: pip install trl (with transformers==4.57.6 already pinned).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional

try:
    from medllm.faithfulness import faithfulness_score, parse_passages
except ImportError:  # run as a standalone script (e.g. copied to Colab)
    from faithfulness import faithfulness_score, parse_passages

PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

ABSTAIN_RE = re.compile(
    r"does not contain|cannot answer|can't answer|cannot provide|unable to answer"
    r"|unable to provide|no (?:relevant )?(?:information|evidence)"
    r"|not (?:contained|provided|available|found) in|insufficient (?:information|evidence)"
    r"|decline to answer|outside (?:my|the) (?:scope|area)",
    re.IGNORECASE,
)
CITATION_RE = re.compile(r"\[(\d{1,2})\]")
DISCLAIMER_RE = re.compile(r"disclaimer\s*:", re.IGNORECASE)


def score_completion(
    text: str,
    is_negative: bool,
    gold_ref: Optional[int],
    n_passages: int,
    passages: Optional[List[str]] = None,
) -> float:
    """Mechanical reward. Mirrors eval_behavior.py's checks."""
    reward = 0.0
    abstained = bool(ABSTAIN_RE.search(text))
    citations = [int(m) for m in CITATION_RE.findall(text)]

    if is_negative:
        reward += 1.0 if abstained else -1.0
    else:
        reward += -1.0 if abstained else 0.5
        if not abstained and gold_ref is not None and gold_ref in citations:
            reward += 1.0
        if not abstained and passages:
            faith = faithfulness_score(text, passages, citations)
            if faith is not None:
                reward += 2.0 * faith - 1.0
    if any(c < 1 or c > n_passages for c in citations):
        reward -= 0.5
    if DISCLAIMER_RE.search(text):
        reward += 0.25
    return reward


def grounding_reward(
    prompts, completions, is_negative, gold_ref, n_passages, passages, **kwargs
) -> List[float]:
    """TRL reward function: extra dataset columns arrive as parallel lists."""
    rewards = []
    for text, neg, gold, n_p, psgs in zip(
        completions, is_negative, gold_ref, n_passages, passages
    ):
        rewards.append(score_completion(text, bool(neg), gold, int(n_p), psgs))
    return rewards


def build_dataset(path: Path, limit: int):
    from datasets import Dataset

    rows = []
    with path.open() as fh:
        for line in fh:
            ex = json.loads(line)
            rows.append(
                {
                    "prompt": PROMPT_TEMPLATE.format(
                        instruction=ex["instruction"], input=ex["input"]
                    ),
                    "is_negative": ex["meta"]["is_negative"],
                    # datasets cannot hold None in an int column; -1 = no gold.
                    "gold_ref": ex["meta"]["gold_ref"] or -1,
                    "n_passages": ex["meta"]["n_passages"],
                    # Passage texts, recovered from the prompt, so the reward
                    # can score answer faithfulness against the evidence.
                    "passages": parse_passages(ex["input"]),
                }
            )
    if limit:
        rows = rows[:limit]
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--sft-adapter", default=None,
                        help="SFT LoRA adapter to merge into the base before GRPO")
    parser.add_argument("--output-dir", default="output/grpo_grounding_v1")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--limit", type=int, default=600,
                        help="cap on training prompts (0 = all)")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, TaskType
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.float16, device_map="auto"
    )
    if args.sft_adapter:
        # Start the policy from the SFT checkpoint: merge the adapter weights
        # into the base, then train a fresh LoRA on top for the GRPO stage.
        print(f"Merging SFT adapter from {args.sft_adapter}")
        model = PeftModel.from_pretrained(model, args.sft_adapter)
        model = model.merge_and_unload()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )

    dataset = build_dataset(args.train_data, args.limit)
    print(f"GRPO prompts: {len(dataset)}")

    import inspect
    import trl as _trl
    print(f"trl version: {getattr(_trl, '__version__', 'unknown')}")
    # GRPOConfig's accepted arguments drift across trl releases; pass only the
    # ones this version's signature actually supports.
    wanted = dict(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.num_generations,
        num_generations=args.num_generations,
        max_prompt_length=640,
        max_completion_length=140,
        logging_steps=5,
        save_steps=50,
        report_to="none",
        fp16=True,
    )
    accepted = set(inspect.signature(GRPOConfig.__init__).parameters)
    dropped = sorted(set(wanted) - accepted)
    if dropped:
        print(f"GRPOConfig does not accept, dropping: {dropped}")
    config = GRPOConfig(**{k: v for k, v in wanted.items() if k in accepted})

    trainer = GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=grounding_reward,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()

    adapter_path = Path(args.output_dir) / "grpo_adapter"
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"Saved GRPO adapter to {adapter_path}")


if __name__ == "__main__":
    main()
