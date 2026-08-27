"""LoRA fine-tuning scaffolding for Qwen medical assistant.

This module provides scaffolding for instruction fine-tuning using LoRA adapters.
NOTE: Fine-tuning should ONLY target:
- Structured reasoning format
- Safety/refusal behavior
- Output formatting

Medical facts should NOT be fine-tuned - they come from RAG retrieval.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    raise RuntimeError("PyTorch required for fine-tuning")

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
    )
except ImportError:
    raise RuntimeError("transformers required for fine-tuning")

try:
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError:
    LoraConfig = None
    get_peft_model = None
    TaskType = None


@dataclass
class FinetuneExample:
    """Single instruction fine-tuning example."""
    instruction: str
    input: str  # Evidence/context
    output: str  # Expected response

    def to_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FinetuneExample":
        return cls(
            instruction=data["instruction"],
            input=data.get("input", ""),
            output=data["output"],
        )


@dataclass
class LoraFinetuneConfig:
    """Configuration for LoRA fine-tuning."""
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    output_dir: str = "output/lora_medical"

    # LoRA parameters
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Training parameters
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    max_length: int = 768

    # Device
    device: Optional[str] = None
    # 4-bit NF4 quantized base (QLoRA). Lets a 7B model train on a 16GB GPU.
    load_in_4bit: bool = False


class MedicalInstructDataset(Dataset):
    """Dataset for medical instruction fine-tuning."""

    PROMPT_TEMPLATE = (
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input}\n\n"
        "### Response:\n{output}"
    )

    def __init__(
        self,
        examples: List[FinetuneExample],
        tokenizer,
        max_length: int = 768,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]

        text = self.PROMPT_TEMPLATE.format(
            instruction=example.instruction,
            input=example.input,
            output=example.output,
        )

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


def collate_variable_length(features: List[dict], pad_token_id: int) -> dict:
    """Pad a batch to its own longest sequence; pad positions get label -100.

    Length is rounded up to a multiple of 64. On Apple MPS the allocator caches
    blocks per tensor shape, so unbucketed per-example lengths make cached
    memory grow with every new length until the machine swaps. Bucketing caps
    the number of distinct shapes.
    """
    max_len = max(f["input_ids"].size(0) for f in features)
    max_len = ((max_len + 63) // 64) * 64
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for f in features:
        n_pad = max_len - f["input_ids"].size(0)
        batch["input_ids"].append(
            torch.nn.functional.pad(f["input_ids"], (0, n_pad), value=pad_token_id)
        )
        batch["attention_mask"].append(
            torch.nn.functional.pad(f["attention_mask"], (0, n_pad), value=0)
        )
        batch["labels"].append(
            torch.nn.functional.pad(f["labels"], (0, n_pad), value=-100)
        )
    return {k: torch.stack(v) for k, v in batch.items()}


def load_examples(path: Path) -> List[FinetuneExample]:
    """Load fine-tuning examples from JSONL file."""
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            examples.append(FinetuneExample.from_dict(data))
    return examples


def create_sample_training_data(output_path: Path) -> None:
    """Create sample training data for medical formatting/safety.

    These examples train the model on:
    1. Proper citation of FDA evidence
    2. Safety disclaimers
    3. Refusal for out-of-scope questions
    4. Structured response format
    """
    examples = [
        FinetuneExample(
            instruction="Summarize the warnings for this medication using the provided evidence.",
            input="Evidence: BOXED WARNING: Cardiovascular Risk. NSAIDs cause an increased risk of serious cardiovascular thrombotic events, myocardial infarction, and stroke, which can be fatal.",
            output="Based on the FDA label, this medication carries a boxed warning about cardiovascular risk. NSAIDs may increase the risk of serious cardiovascular events including heart attacks and strokes, which can be fatal.\n\nDisclaimer: This information is for educational purposes only. Consult a healthcare provider for medical advice.",
        ),
        FinetuneExample(
            instruction="Answer the user's question about medication side effects.",
            input="Evidence: Common adverse reactions include headache (15%), nausea (10%), and dizziness (8%). Rare but serious reactions include liver damage and allergic reactions.",
            output="According to the FDA label, common side effects include:\n- Headache (15% of patients)\n- Nausea (10%)\n- Dizziness (8%)\n\nRare but serious reactions to be aware of include liver damage and allergic reactions. Contact your healthcare provider if you experience any concerning symptoms.\n\nDisclaimer: This information is for educational purposes only.",
        ),
        FinetuneExample(
            instruction="Provide dosage information based on the evidence.",
            input="Evidence: [No relevant dosage information found in the retrieved documents]",
            output="I cannot find specific dosage information for this medication in the FDA evidence provided. Please consult the official FDA label or speak with your healthcare provider or pharmacist for accurate dosing instructions.\n\nDisclaimer: Never adjust medication dosages without consulting a healthcare professional.",
        ),
        FinetuneExample(
            instruction="Answer an emergency medical question.",
            input="User is asking about symptoms of a heart attack.",
            output="If you or someone else may be experiencing a heart attack, call emergency services (911 in the US) immediately. Do not wait to see if symptoms improve.\n\nI am an AI assistant and cannot provide emergency medical care. Time-sensitive emergencies require immediate professional medical attention.",
        ),
        FinetuneExample(
            instruction="Answer a question outside the scope of FDA drug information.",
            input="Evidence: [No FDA drug label information is relevant to this question about surgery procedures]",
            output="This question is outside my area of expertise. I am designed to provide information based on FDA drug labels. For questions about surgical procedures or other medical topics not related to medications, please consult with a qualified healthcare professional.\n\nDisclaimer: This AI assistant is limited to FDA drug information.",
        ),
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict()) + "\n")

    print(f"Created {len(examples)} sample training examples at {output_path}")


def finetune_with_lora(
    config: LoraFinetuneConfig,
    train_data_path: Path,
    eval_data_path: Optional[Path] = None,
) -> str:
    """Fine-tune Qwen model with LoRA adapters.

    Args:
        config: Fine-tuning configuration
        train_data_path: Path to training JSONL file
        eval_data_path: Optional path to evaluation JSONL file

    Returns:
        Path to saved LoRA adapter weights
    """
    if get_peft_model is None:
        raise RuntimeError("peft library required for LoRA fine-tuning: pip install peft")

    if config.device:
        device = config.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load tokenizer and model
    print(f"Loading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.load_in_4bit:
        if device != "cuda":
            raise RuntimeError("--load-in-4bit requires a CUDA GPU")
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            ),
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        if device == "cuda":
            model_dtype = torch.float16
        elif device == "mps":
            model_dtype = torch.bfloat16
        else:
            model_dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            dtype=model_dtype,
            device_map="auto" if device == "cuda" else None,
            # torch 2.4 MPS leaks memory through the SDPA attention path during
            # backward; eager attention is slower per step but stays flat.
            attn_implementation="eager" if device == "mps" else None,
        )

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    print(f"Loading training data from {train_data_path}")
    train_examples = load_examples(train_data_path)
    train_dataset = MedicalInstructDataset(train_examples, tokenizer, config.max_length)

    eval_dataset = None
    if eval_data_path and eval_data_path.exists():
        eval_examples = load_examples(eval_data_path)
        eval_dataset = MedicalInstructDataset(eval_examples, tokenizer, config.max_length)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset else "no",
        fp16=device == "cuda",
        gradient_checkpointing=config.load_in_4bit,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
    )

    # Train
    callbacks = []
    if device == "mps":
        from transformers import TrainerCallback

        class MpsCacheCallback(TrainerCallback):
            """Release cached MPS blocks periodically so allocator growth
            does not push the host into swap."""

            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step % 25 == 0:
                    torch.mps.empty_cache()

        callbacks.append(MpsCacheCallback())

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda feats: collate_variable_length(
            feats, tokenizer.pad_token_id
        ),
        callbacks=callbacks or None,
    )

    print("Starting LoRA fine-tuning...")
    trainer.train()

    # Save adapter
    adapter_path = Path(config.output_dir) / "lora_adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Saved LoRA adapter to {adapter_path}")

    return str(adapter_path)


def main():
    """CLI for fine-tuning operations."""
    import argparse

    parser = argparse.ArgumentParser(description="LoRA fine-tuning for medical QA")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Create sample data command
    sample_parser = subparsers.add_parser("create-samples", help="Create sample training data")
    sample_parser.add_argument(
        "--output", default="data/finetune/samples.jsonl",
        help="Output path for sample data"
    )

    # Fine-tune command
    train_parser = subparsers.add_parser("train", help="Run LoRA fine-tuning")
    train_parser.add_argument("--train-data", required=True, help="Training JSONL file")
    train_parser.add_argument("--eval-data", help="Evaluation JSONL file")
    train_parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Base model")
    train_parser.add_argument("--output-dir", default="output/lora_medical", help="Output directory")
    train_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    train_parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    train_parser.add_argument(
        "--load-in-4bit", action="store_true",
        help="QLoRA: quantize the base model to 4-bit NF4 (CUDA only)"
    )

    args = parser.parse_args()

    if args.command == "create-samples":
        create_sample_training_data(Path(args.output))
    elif args.command == "train":
        config = LoraFinetuneConfig(
            model_name=args.model,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            lora_r=args.lora_r,
            load_in_4bit=args.load_in_4bit,
        )
        finetune_with_lora(
            config,
            Path(args.train_data),
            Path(args.eval_data) if args.eval_data else None,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
