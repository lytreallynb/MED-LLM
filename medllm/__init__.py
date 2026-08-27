"""MED-LLM: Medical RAG Pipeline with Qwen.

A Retrieval-Augmented Generation pipeline for medical Q&A using:
- FDA drug label data from openFDA API
- Qwen embeddings (Qwen3-Embedding-0.6B)
- FAISS vector index for similarity search
- Qwen LLM for response generation via DashScope
- Optional LoRA fine-tuning for safety/formatting
"""

__version__ = "0.2.0"

__all__ = [
    "chunking",
    "embeddings",
    "evaluation",
    "finetune",
    "indexer",
    "retrieval",
    "server",
    "tokenization",
]
