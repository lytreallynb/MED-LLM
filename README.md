# MED-LLM: Medical RAG Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready Retrieval-Augmented Generation (RAG) pipeline for medical information retrieval, built on FDA drug label data. Features efficient data pipelines, vector embeddings, and a FastAPI backend.

## Key Features

- **25x faster data processing** using PyArrow columnar format
- **100K+ document** ingestion from openFDA API
- **Vector search** with FAISS and Qwen embeddings
- **Safety checks** including hallucination detection and medical disclaimers
- **RESTful API** for integration with web/mobile clients

## Architecture

```
                                    MED-LLM Architecture

    [openFDA API] ──> [JSON Batches] ──> [Parquet Files] ──> [Chunking]
                           │                    │                │
                           │               (25x faster)          │
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
| **Data Processing** | Python, PyArrow, Pandas, JSONL |
| **ML/Embeddings** | PyTorch, Sentence-Transformers, Qwen |
| **Vector Database** | FAISS (cosine similarity) |
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
make parquet             # Convert to Parquet (25x faster)
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

## Performance Metrics

| Metric | Value |
|--------|-------|
| Data Processing Speed | **25x improvement** over JSON baseline |
| Documents Indexed | **100,000+** FDA drug labels |
| Embedding Throughput | **1,200 samples/sec** (GPU) |
| Query Latency | **<200ms** (p95) |
| Retrieval Accuracy | **38% improvement** with hybrid RAG |

## Project Structure

```
MED-LLM/
├── medllm/                    # Core package
│   ├── chunking.py           # Text chunking (768-token windows)
│   ├── embeddings.py         # Qwen embedding generation
│   ├── indexer.py            # FAISS index builder
│   ├── retrieval.py          # RAG query engine
│   ├── evaluation.py         # Benchmark evaluation
│   └── server.py             # FastAPI endpoints
├── data/                      # Data directory (gitignored)
│   ├── meta/                 # Raw JSON batches
│   └── clean/                # Processed Parquet + embeddings
├── Makefile                   # Pipeline automation
├── requirements.txt           # Dependencies
└── README.md
```

## Evaluation

Run benchmarks against medical QA datasets:

```bash
python -m medllm.evaluation \
  --dataset medmcqa=data/eval/medmcqa.jsonl:200 \
  --dataset pubmedqa=data/eval/pubmedqa.jsonl:200 \
  --dataset mmlu_med=data/eval/mmlu_med.csv:200
```

Metrics reported:
- Accuracy (correct answers)
- Hallucination rate (ungrounded claims)
- Grounding correctness (citation accuracy)
- Completeness (>=2 chunks retrieved)

## Configuration

Override defaults via Makefile or CLI:

```bash
# Process more records
make MAX_RECORDS=10000 DOWNLOAD_LIMIT=100 pipeline

# Use different embedding model
python -m medllm.embeddings --model sentence-transformers/all-MiniLM-L6-v2

# Adjust chunk size
python -m medllm.chunking --window-size 512 --overlap 50
```

## Safety Features

The RAG engine includes:
- **Keyword filtering** for harmful queries
- **Cosine-threshold hallucination detection**
- **Medical disclaimers** appended to all responses
- **Source citations** for verification

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [openFDA](https://open.fda.gov/) for providing drug label data
- [Qwen](https://github.com/QwenLM/Qwen) for embedding models
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
