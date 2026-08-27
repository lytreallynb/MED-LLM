# MED-LLM: Medical RAG Pipeline with Qwen

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Qwen](https://img.shields.io/badge/Qwen-2.5-purple.svg)](https://github.com/QwenLM/Qwen)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Retrieval-Augmented Generation (RAG) pipeline for medical Q&A, powered by **Qwen** models. Ingests FDA drug label data, generates embeddings with **Qwen3-Embedding**, builds a FAISS vector index, and serves queries through a FastAPI backend with Qwen LLM responses.

## Key Features

- **FDA Drug Data Ingestion** - Fetches and processes drug labels from openFDA API
- **Qwen Embeddings** - Uses Qwen3-Embedding-0.6B for dense vector representations
- **FAISS Vector Search** - Fast similarity search over the ingested drug-label corpus
- **Qwen LLM Integration** - Generates responses via DashScope API
- **Optional Fine-Tuning** - LoRA scaffolding for safety/formatting tuning
- **Safety Layer** - Hallucination detection and medical disclaimers
- **FastAPI Backend** - RESTful API for web/mobile integration

## Architecture

```
                                    MED-LLM Architecture

    [openFDA API] ──> [JSON Batches] ──> [Parquet Files] ──> [Chunking]
                           │                    │                │
                           │                                     │
                           ▼                    ▼                ▼
                      data/meta/          data/clean/      chunks.jsonl
                                               │                │
                                               │                ▼
                                               │         [Qwen Embeddings]
                                               │                │
                                               ▼                ▼
                                          [FAISS Index] <── fda_embeddings.npy
                                               │
                                               ▼
                                    [RAG Query Engine]
                                    ├── Retrieval (top-k)
                                    ├── Safety Checker
                                    └── LLM Integration
                                               │
                                               ▼
                                      [FastAPI Server]
                                       POST /query
```

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Data Processing** | Python, PyArrow, Pandas |
| **Embeddings** | Qwen3-Embedding-0.6B, Sentence-Transformers, PyTorch |
| **LLM** | Qwen (via DashScope API) |
| **Vector Database** | FAISS (cosine similarity) |
| **Fine-Tuning** | LoRA (optional, for safety/formatting) |
| **Backend** | FastAPI, Uvicorn |
| **Evaluation** | MedMCQA, PubMedQA, MMLU-Medical |

## Quick Start

### Prerequisites

- Python 3.11+
- 8GB+ RAM (for embeddings)
- GPU optional but recommended

### Installation

```bash
git clone https://github.com/lytreallynb/MED-LLM.git
cd MED-LLM
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# Full pipeline: download -> parquet -> chunk -> embed -> index
make pipeline
make chunk
make embeddings
make index

# Or run individual steps
make download            # Fetch JSON batches from openFDA
make parquet             # Convert to Parquet
make chunk               # Tokenize into 768-token windows
make embeddings          # Generate Qwen embeddings
make index               # Build FAISS index
```

### Start the API Server

```bash
export DASHSCOPE_API_KEY=your_key  # Optional for LLM responses
uvicorn medllm.server:app --host 0.0.0.0 --port 8000
```

### Query the API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the side effects of ibuprofen?"}'
```

Response:
```json
{
  "answer": "Common side effects include...",
  "sources": ["NDC-12345", "NDC-67890"],
  "safety_notes": ["Medical disclaimer applied"],
  "chunks_retrieved": 5
}
```

## Evaluation Status

The evaluation harness (see Evaluation section below) is implemented, but benchmark
results have not yet been published; numbers will be added here once a recorded run
of `make evaluate` is committed under `results/`.

Current retrieval uses dense embeddings (Qwen3-Embedding-0.6B) over FAISS.
Hybrid BM25 + dense retrieval is planned but not yet implemented.

## Project Structure

```
MED-LLM/
├── medllm/                    # Core package
│   ├── chunking.py           # Text chunking (768-token windows)
│   ├── embeddings.py         # Qwen embedding generation
│   ├── indexer.py            # FAISS index builder
│   ├── retrieval.py          # RAG query engine + safety checks
│   ├── evaluation.py         # Benchmark evaluation
│   ├── finetune.py           # LoRA fine-tuning scaffolding
│   └── server.py             # FastAPI endpoints
├── tests/                     # Unit tests
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   └── test_llm.py
├── llm.py                     # Standalone Qwen LLM client
├── Dockerfile                 # Container deployment
├── docker-compose.yml         # Multi-service orchestration
├── Makefile                   # Pipeline automation
├── requirements.txt           # Dependencies
└── README.md
```

## Evaluation

Build a retrieval-grounded eval set from the ingested FDA data, run the suite,
and render the dashboard:

```bash
make eval-dataset   # generates data/eval/fda_eval.jsonl from chunk metadata
make evaluate       # writes results/metrics.json (works without an API key)
make dashboard      # renders results/dashboard.html (standalone, no deps)
```

External medical QA datasets (MedMCQA, PubMedQA, MMLU-Medical) can be passed
as additional JSONL/CSV files with `question` and `answer` fields:

```bash
python -m medllm.evaluation \
  --dataset fda=data/eval/fda_eval.jsonl \
  --dataset medmcqa=data/eval/medmcqa.jsonl:200 \
  --output results/metrics.json
```

Metrics reported:
- Accuracy (answer contains the expected key; requires DASHSCOPE_API_KEY)
- Retrieval hit rate (expected answer text appears in retrieved evidence)
- Recall@k and MRR (expected chunk retrieved, and at what rank)
- Hallucination rate (refusals for weak evidence)
- Grounding correctness (queries answered from evidence)
- Completeness (>=2 chunks retrieved)
- Average top-1 similarity and per-query latency

## Configuration

Override defaults via Makefile or CLI:

```bash
# Process more records
make MAX_RECORDS=10000 DOWNLOAD_LIMIT=100 pipeline

# Use different embedding model
python -m medllm.embeddings --model sentence-transformers/all-MiniLM-L6-v2

# Adjust chunk size
python -m medllm.chunking --chunk-size 512 --overlap 50
```

## Safety Features

The RAG engine includes:
- **Keyword filtering** for harmful queries
- **Cosine-threshold hallucination detection**
- **Medical disclaimers** appended to all responses
- **Source citations** for verification

## Docker Deployment

### Using Docker Compose

```bash
# Build and run the API server
docker-compose up -d medllm-api

# Check logs
docker-compose logs -f medllm-api
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DASHSCOPE_API_KEY` | Qwen API key | Required for LLM |
| `MEDLLM_INDEX_PATH` | Path to FAISS index | `data/clean/fda.index` |
| `MEDLLM_TOP_K` | Number of chunks to retrieve | `4` |
| `MEDLLM_EMBED_MODEL` | Embedding model | `Qwen/Qwen3-Embedding-0.6B` |

## Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
python -m pytest tests/test_retrieval.py -v
```

## LoRA Instruction Fine-Tune: Grounding Behavior

### Motivation: behavior, not knowledge

The pipeline keeps medical facts where they belong: in retrieval over FDA
drug labels. The fine-tune therefore targets only three behaviors: structured
response format (including the safety disclaimer), evidence citation, and
refusal when the provided evidence does not answer the question. Training
answers are extractive from the provided passages, so the adapter learns
form, not facts; medical content continues to come from retrieval at
inference time.

### Dataset construction

The dataset is generated by `medllm/build_finetune_dataset.py` from the
cleaned label chunks in `data/clean/chunks_full.jsonl` (seed 17):

- **Filtering.** Only chunks from the six label sections with question
  templates are used (indications and usage, warnings, contraindications,
  adverse reactions, dosage and administration, clinical pharmacology).
  Chunks shorter than 120 characters are dropped as too thin to anchor an
  extractive answer, and 1,622 chunks whose `drug_name` is an FDA set-id
  UUID rather than a readable drug name are filtered out (they cannot
  produce a readable question or answer).
- **Questions.** One templated question per usable chunk, keyed to its
  section (for example, "What warnings are associated with {drug}?").
- **Positive examples.** The gold chunk plus one distractor passage from a
  different document, in shuffled order. The target output answers
  extractively from the gold passage, cites it as `[n]`, and ends with a
  fixed safety disclaimer.
- **Negative examples.** Two distractor passages from other documents and no
  gold passage. The target output is a fixed abstention plus the same
  disclaimer. Roughly one third of the examples are negative (311 of the 894
  training examples, 63 of the 179 test examples) so the model learns to
  decline, not just to answer.
- **Split.** Documents are split into train and test by `document_id` before
  any examples are generated, so no drug label contributes to both sides; an
  assertion enforces zero overlap. Final sizes: 894 training examples and
  179 test examples.

### Training setup

Run by `medllm/finetune.py`:

- Base model `Qwen/Qwen2.5-1.5B-Instruct` with weights frozen; LoRA adapters
  (r=8, alpha 32, dropout 0.1) on the attention `q_proj` and `v_proj`
  projections.
- 2 epochs, learning rate 2e-4 with warmup ratio 0.1, per-device batch size 2
  with gradient accumulation 4, maximum sequence length 768 tokens.
- fp16 mixed precision on an NVIDIA T4 (Google Colab), transformers 4.57.6.
  Earlier attempts on Apple MPS hit allocator memory growth under
  variable-length batches and sustained-load throttling; the collator's
  shape bucketing in `finetune.py` comes from that debugging.
- Prompt template `### Instruction:` / `### Input:` / `### Response:`,
  identical to the template used at evaluation time.
- Trainable parameters: 1,089,536 (0.0705% of 1.54B). Training loss fell
  from 2.03 to 1.05 (reported `train_loss` 1.225 averaged over the run).
  Wall-clock training time: 509 seconds.

### Evaluation protocol

`medllm/eval_behavior.py` generates from the base model and from the base
model plus adapter on the held-out test split, under identical conditions:
greedy decoding, a cap of 140 new tokens, and the training prompt template.
Retrieval is untouched. Every metric is mechanically checkable with regular
expressions: no LLM judge, no API call. Abstention is detected with a
model-agnostic pattern list, so any reasonable phrasing of declining counts
for both models, not just the trained wording.

### Metrics and results

Evaluated examples: 179 total (63 negative, 116 positive).

| Metric | Definition | Base | Tuned |
|---|---|---|---|
| Abstention precision | Of all responses that abstained, the fraction whose evidence truly did not contain the answer. | 0.763 | 0.969 |
| Abstention recall | Of all questions whose evidence did not contain the answer, the fraction where the model abstained. | 0.714 | 0.984 |
| False-abstain rate on positives | The fraction of answerable questions where the model abstained anyway. | 0.121 | 0.017 |
| Positive citation rate | The fraction of answerable questions whose response contains at least one `[n]` citation. | 0.603 | 0.991 |
| Citation validity | Among positive responses that cite, the fraction whose citations all point at passages actually provided. | 0.986 | 1.000 |
| Gold-citation accuracy | The fraction of answerable questions where the response cites the specific passage that answers the question. | 0.578 | 0.991 |
| Disclaimer rate | The fraction of all responses that carry the disclaimer line. | 0.000 | 0.939 |

Abstention precision and recall are reported separately so a model that
abstains on everything cannot score well.

#### Stage two: GRPO with programmatic rewards (null result, explained)

A GRPO reinforcement-learning stage (TRL, 200 steps, ~42 min on a T4) was run
on top of the merged 1.5B SFT checkpoint: 4 completions sampled per prompt,
each scored by the same mechanical checks the evaluation uses (citation
validity, gold-passage grounding, correct refusal, disclaimer), and the policy
updated from group-relative advantages. No reward model, no LLM judge.

Evaluated with the exact training stack reproduced (SFT adapter merged, then
the GRPO adapter attached), SFT+GRPO matches SFT on six of seven metrics to
four decimal places; the disclaimer rate moves 0.9385 to 0.9497 (2 of 179
examples), within single-run noise.

The mechanism is visible in the training telemetry: the SFT checkpoint already
saturates the programmatic reward, so most sampled groups receive identical
scores (the fraction of zero-variance groups runs 0.6 to 1.0) and the
group-relative advantage, and with it the gradient, vanishes. For GRPO to bite
here it would need headroom the reward no longer offers: harder negatives, a
finer-grained reward, or evaluation prompts the SFT stage fails on.

One measurement error is kept on record: an earlier evaluation loaded the GRPO
adapter onto the raw base model without merging the SFT adapter it was trained
on, producing base-level numbers
(`results/behavior_grpo_INVALID_missing_sft_stack.json`). Those numbers are
invalid and quoted nowhere; the corrected file is
`results/behavior_sft_grpo.json`.

#### Stage three: restoring reward headroom (hard negatives + faithfulness)

Both missing ingredients named above are now implemented, and the earlier
results in this section were measured on the easy-only dataset, so they are
not comparable with runs on the regenerated one.

Retraining the full stack on the regenerated dataset (Colab T4: SFT 12 min,
GRPO 200 steps 32 min; metrics in `results/behavior_v2_summary.json`) gives:

| Metric | Base | SFT | SFT+GRPO |
| --- | --- | --- | --- |
| Abstention recall | 0.585 | 0.939 | 0.939 |
| Abstention recall (hard) | 0.387 | 0.871 | 0.871 |
| Gold-citation accuracy | 0.491 | 0.991 | 0.991 |
| Gold-citation accuracy (hard) | 0.400 | 0.982 | 0.982 |
| Answer faithfulness | 0.582 | 0.996 | 0.996 |
| Low-faithfulness rate | 0.355 | 0.000 | 0.000 |

Two findings. First, the hard examples work as designed: the base model,
which could previously shortcut negatives by drug-name matching, drops to
0.387 abstention recall on the hard subset, and a third of its answers fall
below 0.5 faithfulness; SFT on the hard training split recovers 0.871 and
0.982 there. Second, GRPO is a null result a second time, and for the same
mechanism at a deeper level: the SFT policy's own samples are already
extractive copies with correct citations, so even the continuous
faithfulness term saturates near 1.0 on-policy, group advantages stay zero
(train_loss ~1e-8), and 200 steps produce no measurable change. The honest
conclusion for this task scale: behavior alignment here is bought by harder
data and SFT, not by RL on top of it. The remaining headroom (the 13% of
hard negatives still answered) would need targeted data or sampling that
concentrates on those failures, not more GRPO steps on prompts the policy
already solves.

* Hard examples (`build_finetune_dataset.py --hard-fraction`, default 0.5):
  the distractor passage comes from the same drug's label but a different
  section, so drug-name matching alone can no longer solve the example.
  Sections with overlapping content (warnings / contraindications / adverse
  reactions; indications / clinical pharmacology) are never paired as hard
  distractors, keeping the abstention label unambiguous. The eval reports
  `abstention_recall_hard` and `gold_citation_accuracy_hard` on this subset.
* Faithfulness reward (`medllm/faithfulness.py`, shared by training and
  eval): the fraction of an answer's content words supported by the cited
  evidence, mapped to a continuous reward term in [-1, +1]. It catches
  answers whose citations are formally valid but whose content is fabricated,
  which the binary checks cannot see, and being continuous it keeps
  within-group reward variance (and therefore the GRPO gradient) alive after
  the binary checks saturate. Gold extractive answers score 0.999 mean on the
  test split; fabricated content scores near zero. The eval reports
  `answer_faithfulness` and `low_faithfulness_rate`.

#### Scale comparison: Qwen2.5-7B-Instruct (QLoRA)

The same experiment was repeated on Qwen2.5-7B-Instruct with the base model
quantized to 4-bit NF4 (QLoRA), trained for 2 epochs on an NVIDIA T4
(3,455 s, 2,523,136 trainable parameters, 0.033%), and evaluated under the
same protocol with both base and tuned in the same NF4 quantization.

| Metric | 7B base | 7B tuned |
|---|---|---|
| Abstention precision | 0.630 | 0.940 |
| Abstention recall | 1.000 | 1.000 |
| False-abstain rate on positives | 0.319 | 0.034 |
| Positive citation rate | 0.862 | 1.000 |
| Citation validity | 1.000 | 1.000 |
| Gold-citation accuracy | 0.853 | 1.000 |
| Disclaimer rate | 0.000 | 0.939 |

The two base models fail in opposite directions: 1.5B under-refuses
(abstention recall 0.714) while 7B over-refuses (a 0.319 false-abstain rate
on answerable questions). The same training data corrects both, teaching the
small model to decline and the large model to stop declining, which suggests
the tuning calibrates grounding behavior rather than pushing uniformly toward
caution or confidence. The tuned model improved both
while its false-abstain rate fell from 0.121 to 0.017, which rules out
"learned to always refuse" as an explanation. The format metrics
(citation rate, disclaimer rate) partly reflect learned output formatting,
as expected; the more substantive results are the abstention decision and
gold-passage selection on documents never seen in training.

### Scale comparison: Qwen2.5-7B (QLoRA)

The same recipe was rerun on `Qwen/Qwen2.5-7B-Instruct` with the base model
quantized to 4-bit NF4 (QLoRA, double quantization, fp16 compute, gradient
checkpointing) so it trains on a 16GB T4. Training completed: 2 epochs over
the same 894 examples, 2,523,136 trainable parameters (0.033%), 57.6 minutes,
train_loss 0.932 (vs 1.225 for 1.5B, so the larger base fits the task better).

The 7B BASE model evaluates strikingly differently from the 1.5B base on the
same held-out set (both in NF4): abstention recall 1.000 but false-abstain
rate 0.319, meaning the untuned 7B refuses almost a third of answerable
questions; its citation behavior is much stronger out of the box (positive
citation rate 0.862, gold-citation accuracy 0.853, citation validity 1.000,
disclaimer rate 0.000). The tuned-7B evaluation is pending (GPU quota); the
open question it will answer is whether tuning removes the over-refusal
while keeping recall, as it did at 1.5B scale.

### Limitations

- Single seed, single run: no variance estimate across seeds or repeats.
- Small base model (1.5B parameters): findings may not transfer to larger
  models.
- Templated, extractive training data: questions come from six fixed
  templates and target answers are condensed spans of the evidence, so the
  trained behavior is narrower than open-ended medical QA.
- The abstention detector is a mechanical regex shared by both models. It is
  deliberately model-agnostic, but any phrasing of declining that it misses
  is scored as an answer for both models, and an answer that merely mentions
  one of the pattern phrases could be miscounted as an abstention.
- The behaviors evaluated are exactly the behaviors trained. These numbers
  measure whether the tuning worked, not medical capability or answer
  correctness.
- Spot check completed on a stratified 50-example sample
  (`scripts/sample_spot_check.py`, seed 17): independent recomputation of the
  abstention and citation judgments from the raw outputs agreed with the
  stored automated scores on 100 of 100 model-output pairs, and direct
  reading confirmed cited drug, cited passage, abstentions, and disclaimers.
  One of the two false abstentions involves a source record whose drug_name
  field is a supplier name rather than a drug, where declining is arguably
  the right call.

### Reproduce

```bash
python -m medllm.build_finetune_dataset --chunks data/clean/chunks_full.jsonl \
    --output-dir data/finetune --seed 17
python -m medllm.finetune train --train-data data/finetune/train.jsonl \
    --model Qwen/Qwen2.5-1.5B-Instruct --output-dir output/lora_grounding_v1 \
    --epochs 2 --batch-size 2 --lr 2e-4 --lora-r 8
python -m medllm.eval_behavior --test-data data/finetune/test.jsonl \
    --model Qwen/Qwen2.5-1.5B-Instruct --output results/behavior_base.json
python -m medllm.eval_behavior --test-data data/finetune/test.jsonl \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter output/lora_grounding_v1/lora_adapter \
    --output results/behavior_tuned.json
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [openFDA](https://open.fda.gov/) for providing drug label data
- [Qwen](https://github.com/QwenLM/Qwen) for embedding models
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
