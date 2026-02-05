"""Standalone Qwen LLM module for medical Q&A generation."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import requests


@dataclass
class QwenConfig:
    """Configuration for Qwen LLM client."""
    model_name: str = "qwen2.5-72b-instruct"
    api_key: Optional[str] = None
    endpoint: str = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generate"
    timeout: int = 60
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9


class QwenLLM:
    """Client for Qwen LLM via DashScope API."""

    SYSTEM_PROMPT = (
        "You are a medical assistant AI. You provide accurate, evidence-based "
        "medical information. Always cite your sources when available. "
        "If you're unsure or the information is not in your training data, "
        "say so clearly. Never provide emergency medical advice - direct users "
        "to call emergency services for urgent situations."
    )

    def __init__(self, config: Optional[QwenConfig] = None):
        self.config = config or QwenConfig()
        self.api_key = self.config.api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY environment variable or api_key config required"
            )

    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[dict]] = None
    ) -> List[dict]:
        """Build message list for the API call."""
        messages = []

        # System message
        messages.append({
            "role": "system",
            "content": system_prompt or self.SYSTEM_PROMPT
        })

        # Conversation history
        if history:
            messages.extend(history)

        # Current user message
        messages.append({
            "role": "user",
            "content": prompt
        })

        return messages

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[dict]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate a response from the Qwen model.

        Args:
            prompt: User's question or input
            system_prompt: Optional custom system prompt
            history: Optional conversation history
            max_tokens: Override max tokens for this call
            temperature: Override temperature for this call

        Returns:
            Generated text response
        """
        messages = self._build_messages(prompt, system_prompt, history)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.config.model_name,
            "input": {"messages": messages},
            "parameters": {
                "max_tokens": max_tokens or self.config.max_tokens,
                "temperature": temperature or self.config.temperature,
                "top_p": self.config.top_p,
            }
        }

        response = requests.post(
            self.config.endpoint,
            headers=headers,
            json=payload,
            timeout=self.config.timeout
        )
        response.raise_for_status()

        return self._parse_response(response.json())

    def _parse_response(self, data: dict) -> str:
        """Extract generated text from API response."""
        # Handle DashScope response format
        if "output" in data:
            output = data["output"]
            if isinstance(output, dict):
                # Format 1: {"output": {"text": "..."}}
                if "text" in output:
                    return output["text"]
                # Format 2: {"output": {"choices": [{"message": {"content": "..."}}]}}
                if "choices" in output:
                    return output["choices"][0]["message"]["content"]

        # Format 3: {"choices": [{"message": {"content": "..."}}]}
        if "choices" in data:
            return data["choices"][0]["message"]["content"]

        raise ValueError(f"Unexpected response format: {data}")

    def chat(
        self,
        question: str,
        context: Optional[str] = None,
    ) -> str:
        """Simple chat interface for medical Q&A.

        Args:
            question: User's medical question
            context: Optional context (e.g., retrieved FDA evidence)

        Returns:
            Generated answer
        """
        if context:
            prompt = (
                f"Based on the following FDA drug information:\n\n"
                f"{context}\n\n"
                f"Please answer this question: {question}"
            )
        else:
            prompt = question

        return self.generate(prompt)


def main():
    """CLI interface for testing Qwen LLM."""
    import argparse

    parser = argparse.ArgumentParser(description="Query Qwen LLM for medical Q&A")
    parser.add_argument("question", help="Medical question to ask")
    parser.add_argument("--model", default="qwen2.5-72b-instruct", help="Qwen model name")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens to generate")
    args = parser.parse_args()

    config = QwenConfig(
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    llm = QwenLLM(config)
    response = llm.chat(args.question)
    print(response)


if __name__ == "__main__":
    main()
