"""Query the FAISS index and assemble RAG prompts with safety checks."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import faiss
import numpy as np
import requests

from .chunking import ChunkMetadata
from .embeddings import EmbeddingModelConfig, QwenEmbeddingBackend

PROMPT_TEMPLATE = (
    "You are a medical assistant model.\n"
    "Use only the evidence provided. If the evidence does not contain the answer,\n"
    'respond: "The evidence does not contain that information."\n\n'
    "User question:\n{question}\n\n"
    "Relevant FDA Evidence:\n{evidence}\n\n"
    "Provide an accurate, evidence-grounded answer."
)


@dataclass
class RetrievalConfig:
    index_path: Path = Path("data/clean/fda.index")
    metadata_path: Path = Path("data/clean/fda_meta.jsonl")
    top_k: int = 4
    prompt_template: str = PROMPT_TEMPLATE


@dataclass
class SafetyConfig:
    blocked_keywords: Sequence[str] = ("emergency", "suicide", "kill yourself", "abuse")
    disclaimer_text: str = (
        "This information is for educational purposes and is not a substitute for care from a licensed clinician."
    )
    hallucination_threshold: float = 0.35
    include_disclaimer: bool = True


@dataclass
class SafetyDecision:
    blocked: bool
    message: str
    notes: List[str] = field(default_factory=list)


@dataclass
class RetrievedChunk:
    metadata: ChunkMetadata
    score: float


@dataclass
class RagResponse:
    question: str
    prompt: str
    answer: str
    retrieved: List[RetrievedChunk]
    safety_notes: List[str]


class SafetyChecker:
    def __init__(self, config: SafetyConfig):
        self.config = config
        self._blocked = [kw.lower() for kw in config.blocked_keywords]

    def check_query(self, text: str) -> SafetyDecision | None:
        lowered = text.lower()
        for keyword in self._blocked:
            if keyword in lowered:
                message = (
                    "I cannot provide guidance on emergencies or self-harm. Please contact a licensed professional or "
                    "the appropriate local emergency services immediately."
                )
                return SafetyDecision(blocked=True, message=message, notes=["blocked_keyword"])
        return None

    def needs_fact_block(self, best_score: float) -> bool:
        return best_score < self.config.hallucination_threshold

    def disclaimer(self) -> str:
        return self.config.disclaimer_text if self.config.include_disclaimer else ""


class QwenChatClient:
    """Minimal client for DashScope Qwen text generation."""

    def __init__(
        self,
        model_name: str = "qwen2.5-72b-instruct",
        api_key: str | None = None,
        endpoint: str | None = None,
        timeout: int = 60,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.endpoint = endpoint or "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generate"
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is required to call Qwen models")

    def generate(self, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "input": {
                "messages": [
                    {"role": "system", "content": "You are a helpful medical assistant."},
                    {"role": "user", "content": prompt},
                ]
            },
        }
        response = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        if "output" in data:
            output = data["output"]
            if isinstance(output, dict):
                if "text" in output and isinstance(output["text"], str):
                    return output["text"]
                choices = output.get("choices")
                if choices:
                    message = choices[0].get("message", {})
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, str):
                            return content
        if "choices" in data:
            choice = data["choices"][0]
            message = choice.get("message", {})
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
        raise RuntimeError(f"Unexpected Qwen response: {data}")


class RagQueryEngine:
    def __init__(
        self,
        retrieval_config: RetrievalConfig,
        embed_config: EmbeddingModelConfig,
        safety_config: SafetyConfig | None = None,
        qwen_client: QwenChatClient | None = None,
    ) -> None:
        self.retrieval_config = retrieval_config
        self.embedder = QwenEmbeddingBackend(embed_config)
        self.safety_checker = SafetyChecker(safety_config or SafetyConfig())
        self.index = faiss.read_index(str(retrieval_config.index_path))
        self.metadata = self._load_metadata(retrieval_config.metadata_path)
        self.qwen_client = qwen_client

    @staticmethod
    def _load_metadata(path: Path) -> List[ChunkMetadata]:
        chunks: List[ChunkMetadata] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                chunks.append(ChunkMetadata(**payload))
        return chunks

    def _format_evidence(self, retrieved: List[RetrievedChunk]) -> str:
        lines = []
        for idx, item in enumerate(retrieved, start=1):
            lines.append(
                f"[{idx}] {item.metadata.drug_name} | {item.metadata.section} (score={item.score:.3f})\n{item.metadata.text}"
            )
        return "\n\n".join(lines)

    def _build_prompt(self, question: str, retrieved: List[RetrievedChunk]) -> str:
        evidence = self._format_evidence(retrieved)
        return self.retrieval_config.prompt_template.format(question=question.strip(), evidence=evidence)

    def _retrieve(self, query_vector: np.ndarray, top_k: int) -> List[RetrievedChunk]:
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, k)
        retrieved: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            retrieved.append(RetrievedChunk(metadata=self.metadata[idx], score=float(score)))
        return retrieved

    def query(self, question: str) -> RagResponse:
        safety = self.safety_checker.check_query(question)
        if safety is not None and safety.blocked:
            return RagResponse(
                question=question,
                prompt="",
                answer=safety.message,
                retrieved=[],
                safety_notes=safety.notes,
            )
        query_vec = self.embedder.embed_texts([question])
        if query_vec.size == 0:
            raise RuntimeError("Embedding backend returned an empty vector for the query")
        # faiss expects shape (1, dim)
        query_matrix = np.asarray(query_vec, dtype=np.float32)
        if query_matrix.ndim == 1:
            query_matrix = query_matrix.reshape(1, -1)
        retrieved = self._retrieve(query_matrix, self.retrieval_config.top_k)
        if not retrieved:
            return RagResponse(
                question=question,
                prompt="",
                answer="The evidence does not contain that information.",
                retrieved=[],
                safety_notes=["no_hits"],
            )
        best_score = retrieved[0].score
        if self.safety_checker.needs_fact_block(best_score):
            return RagResponse(
                question=question,
                prompt="",
                answer="The evidence does not contain that information.",
                retrieved=retrieved,
                safety_notes=["low_similarity"],
            )
        prompt = self._build_prompt(question, retrieved[:3])
        if self.qwen_client is not None:
            completion = self.qwen_client.generate(prompt)
        else:
            completion = "Qwen client not configured. Provide this prompt to your model manually."
        disclaimer = self.safety_checker.disclaimer()
        final_answer = completion
        safety_notes: List[str] = []
        if disclaimer:
            final_answer = f"{completion}\n\n{disclaimer}"
            safety_notes.append("disclaimer")
        return RagResponse(
            question=question,
            prompt=prompt,
            answer=final_answer,
            retrieved=retrieved,
            safety_notes=safety_notes,
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="User query to run through the RAG pipeline")
    parser.add_argument("--index", default="data/clean/fda.index", help="Path to FAISS index")
    parser.add_argument("--metadata", default="data/clean/fda_meta.jsonl", help="Chunk metadata JSONL")
    parser.add_argument("--top-k", type=int, default=4, help="Number of neighbors to retrieve")
    parser.add_argument("--embedding-model", default="Qwen/Qwen2.5-Embedding-1.8B", help="Embedding model for queries")
    parser.add_argument("--device", default=None, help="Torch device override")
    parser.add_argument("--no-qwen", action="store_true", help="Skip calling the Qwen generator and print the prompt only")
    parser.add_argument("--qwen-model", default="qwen2.5-72b-instruct", help="DashScope Qwen generation model id")
    parser.add_argument("--hallucination-threshold", type=float, default=0.35, help="Minimum FAISS score required to answer")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    retrieval_cfg = RetrievalConfig(
        index_path=Path(args.index),
        metadata_path=Path(args.metadata),
        top_k=args.top_k,
    )
    embed_cfg = EmbeddingModelConfig(
        model_name=args.embedding_model,
        device=args.device,
        normalize=True,
    )
    safety_cfg = SafetyConfig(hallucination_threshold=args.hallucination_threshold)
    qwen = None
    if not args.no_qwen:
        qwen = QwenChatClient(model_name=args.qwen_model)
    engine = RagQueryEngine(retrieval_cfg, embed_cfg, safety_cfg, qwen)
    response = engine.query(args.question)
    print("Prompt:\n" + response.prompt)
    print("\nAnswer:\n" + response.answer)
