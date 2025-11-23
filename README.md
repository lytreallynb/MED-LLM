# Med-LLM

Small utilities to download FDA drug label data from the [openFDA](https://open.fda.gov/) API and convert the JSON responses into columnar Parquet files for downstream processing.

## Environment setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data pipeline

The workflow is powered by the provided `Makefile` and writes intermediate files to `data/`.

```bash
make download            # fetch JSON batches into data/meta
make parquet             # convert existing JSON batches into Parquet (data/clean/parquet)
make pipeline            # run download + parquet in one command
make clean               # drop generated Parquet files

# Chunk + embed + index for RAG
make chunk                # tokenize sections (512-1024 token windows w/ overlap)
make embeddings           # run Qwen embeddings + export fda_meta.jsonl
make index                # build FAISS index (fda.index)
```

You can override defaults on the fly, for example:

```bash
make MAX_RECORDS=5000 DOWNLOAD_LIMIT=75
```

The download step respects openFDA's pagination and writes newline-delimited JSON files such as `data/meta/batch_0.json`. The conversion step flattens nested JSON via `pandas.json_normalize` and stores `batch_*.parquet`, which can be loaded efficiently in later stages of the project.

## MED-LLM RAG artifacts

The `medllm` package provides chunking, embedding, indexing, retrieval, evaluation, and server utilities that follow the project instructions in `steps.md`:

1. **Chunk medical sections** — `make chunk` flattens the six required sections into 768-token windows with ~100 token overlap and writes `data/clean/chunks.jsonl`.
2. **Generate Qwen embeddings** — `make embeddings` loads each chunk, feeds it to the specified Qwen embedding model (default `Qwen/Qwen2.5-Embedding-1.8B`), and persists both the numpy matrix (`fda_embeddings.npy`) and aligned metadata (`fda_meta.jsonl`).
3. **Build a FAISS index** — `make index` adds every embedding to a cosine-similarity FAISS index and saves it as `data/clean/fda.index`. The metadata file is kept side-by-side to hydrate prompt contexts later.

> **Dependencies:** Torch, sentence-transformers, and `faiss-cpu` are declared in `requirements.txt`. Install them before running embeddings or indexing.

## Retrieval + FastAPI backend

* The `medllm.retrieval.RagQueryEngine` loads the FAISS index + metadata, embeds incoming questions with the same Qwen encoder, applies a rule-based safety checker (keyword filter, cosine-threshold hallucination detector, medical disclaimer), and optionally calls the Qwen chat model through DashScope.
* Launch the HTTP backend for integration with Xcode using FastAPI:

```bash
export DASHSCOPE_API_KEY=...              # optional; prompt-only mode if omitted
uvicorn medllm.server:app --host 0.0.0.0 --port 8000
```

`POST /query` accepts `{ "question": "..." }` and returns the answer, prompt, safety notes, and retrieved chunks so the iOS client can display citations.

## Evaluation utilities

`python -m medllm.evaluation --dataset medmcqa=data/eval/medmcqa.jsonl:200 --dataset pubmedqa=data/eval/pubmedqa.jsonl:200 --dataset mmlu_med=data/eval/mmlu_med.csv:200`

The runner executes each dataset through the live RAG engine and reports accuracy, hallucination rate, grounding correctness, and completeness (>=2 chunks). Provide Qwen credentials for real answers or pass `--no-qwen` to review prompts only.

## Optional: instruction fine-tuning scaffolding

While medical facts must not be fine-tuned, you can curate additional training data for reasoning/safety using the JSON structure specified in `steps.md`. Keep evidence in the `input` field and the expected structured answer in `output`. Feed the resulting JSONL into your preferred instruction-tuning pipeline (e.g., LoRA adapters for Qwen) to align refusal behavior and formatting while keeping factual knowledge frozen.
