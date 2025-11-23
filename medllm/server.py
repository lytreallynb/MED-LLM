"""FastAPI server that exposes the MED-LLM RAG pipeline."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .embeddings import EmbeddingModelConfig
from .retrieval import (
    PROMPT_TEMPLATE,
    QwenChatClient,
    RagQueryEngine,
    RetrievalConfig,
    SafetyConfig,
)

app = FastAPI(title="MED-LLM RAG Backend", version="0.1.0")


class QueryRequest(BaseModel):
    question: str = Field(..., description="User medical question")


class RetrievedChunkModel(BaseModel):
    chunk_id: str
    document_id: str
    drug_name: str
    section: str
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    prompt: str
    safety_notes: List[str]
    chunks: List[RetrievedChunkModel]


@lru_cache(maxsize=1)
def _build_engine() -> RagQueryEngine:
    index_path = Path(os.getenv("MEDLLM_INDEX_PATH", "data/clean/fda.index"))
    metadata_path = Path(os.getenv("MEDLLM_META_PATH", "data/clean/fda_meta.jsonl"))
    top_k = int(os.getenv("MEDLLM_TOP_K", "4"))
    hallucination_threshold = float(os.getenv("MEDLLM_FACT_THRESHOLD", "0.35"))
    embed_model = os.getenv("MEDLLM_EMBED_MODEL", "Qwen/Qwen2.5-Embedding-1.8B")
    device = os.getenv("MEDLLM_DEVICE")
    retrieval_cfg = RetrievalConfig(index_path=index_path, metadata_path=metadata_path, top_k=top_k, prompt_template=PROMPT_TEMPLATE)
    embed_cfg = EmbeddingModelConfig(model_name=embed_model, device=device)
    safety_cfg = SafetyConfig(hallucination_threshold=hallucination_threshold)
    qwen_client: QwenChatClient | None = None
    dashscope_key = os.getenv("DASHSCOPE_API_KEY")
    if dashscope_key:
        qwen_client = QwenChatClient(
            model_name=os.getenv("MEDLLM_QWEN_MODEL", "qwen2.5-72b-instruct"),
            api_key=dashscope_key,
        )
    return RagQueryEngine(retrieval_cfg, embed_cfg, safety_cfg, qwen_client)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question may not be empty")
    try:
        engine = _build_engine()
        result = engine.query(question)
    except Exception as exc:  # pragma: no cover - surfaced via API response
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    chunks = [
        RetrievedChunkModel(
            chunk_id=item.metadata.chunk_id,
            document_id=item.metadata.document_id,
            drug_name=item.metadata.drug_name,
            section=item.metadata.section,
            text=item.metadata.text,
            score=item.score,
        )
        for item in result.retrieved
    ]
    return QueryResponse(
        answer=result.answer,
        prompt=result.prompt,
        safety_notes=result.safety_notes,
        chunks=chunks,
    )
