# MED-LLM RAG Pipeline Instructions

## 1. Data Preparation

1. Fetch FDA JSON data.
2. Convert JSON to Parquet using Python and pandas.
3. Extract relevant text fields, including:

   * indications_and_usage
   * warnings
   * contraindications
   * adverse_reactions
   * dosage_and_administration
   * clinical_pharmacology

## 2. Chunking

1. Break long sections into 512–1024 token chunks.
2. Use overlap of approximately 100 tokens.
3. For each chunk, store metadata:

   * drug_name
   * section
   * chunk_id
   * text

## 3. Embedding Generation

1. Use a Qwen embedding model such as:

   * Qwen2.5-Embedding-1.8B
   * Qwen2.5-Embedding-7B
2. For each chunk, generate a dense vector embedding.
3. Save embeddings and metadata.

## 4. Vector Database Construction

1. Use FAISS to build a vector index.
2. Add all embeddings to the FAISS index.
3. Save both the index file and metadata:

   * fda.index
   * fda_meta.jsonl

## 5. Retrieval Pipeline

1. User submits a query.
2. Generate an embedding for the query.
3. Perform FAISS kNN search (k = 3–6).
4. Load corresponding metadata and chunk texts.
5. Combine retrieved texts into the final prompt.

## 6. RAG Prompt Template

Use the following prompt structure:

```
You are a medical assistant model.
Use only the evidence provided. If the evidence does not contain the answer,
respond: "The evidence does not contain that information."

User question:
{question}

Relevant FDA Evidence:
{chunk_1}
{chunk_2}
{chunk_3}

Provide an accurate, evidence-grounded answer.
```

## 7. Optional Model Fine-Tuning

1. Do not fine-tune on medical facts.
2. Fine-tune only for:

   * structured reasoning
   * safety and refusal behavior
   * formatting expected for medical output
3. Use instruction-tuning format:

```
{
  "instruction": "Summarize warnings using provided evidence.",
  "input": "Evidence: ...",
  "output": "..."
}
```

## 8. Safety Layer

1. Implement a rule-based safety checker.
2. Include disclaimers for medical content.
3. Add a hallucination detector or facts-not-found rule.

## 9. Integration with Xcode

1. Use a Python backend (FastAPI or Flask).
2. Steps executed by the backend:

   * embed query
   * perform FAISS search
   * assemble prompt
   * call Qwen model
3. Xcode app sends queries and displays responses.

## 10. Evaluation

Evaluate the system using:

* MedMCQA
* PubMedQA
* MMLU Medical subsets

Track:

* accuracy
* hallucination rate
* grounding correctness
* completeness

