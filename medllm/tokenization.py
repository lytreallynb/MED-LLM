"""Reusable tokenizer helper that prioritizes Qwen models with a regex fallback."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Sequence

try:  # transformers is optional during import so unit tests without weights still run
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - handled in runtime
    AutoTokenizer = None  # type: ignore

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass
class TokenizerConfig:
    """Configuration for tokenizer loading."""

    tokenizer_name: str | None = "Qwen/Qwen2.5-7B"
    add_special_tokens: bool = False


class TokenizerWrapper:
    """Thin wrapper around Hugging Face tokenizers with regex fallback."""

    def __init__(self, config: TokenizerConfig | None = None):
        self.config = config or TokenizerConfig()
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None or AutoTokenizer is None:
            return
        if not self.config.tokenizer_name:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[call-arg]
            self.config.tokenizer_name,
            trust_remote_code=True,
        )

    def encode(self, text: str) -> List[Any]:
        """Convert *text* into a sequence of token identifiers."""
        if not text:
            return []
        self._ensure_loaded()
        if self._tokenizer is not None:
            tokens = self._tokenizer.encode(text, add_special_tokens=self.config.add_special_tokens)
            return tokens
        return _TOKEN_PATTERN.findall(text)

    def tokens_to_text(self, tokens: Sequence[Any]) -> str:
        """Convert encoded tokens back into a human readable chunk."""
        if not tokens:
            return ""
        if self._tokenizer is not None:
            return self._tokenizer.decode(tokens, skip_special_tokens=True)
        return " ".join(str(tok) for tok in tokens)

    def token_count(self, text: str) -> int:
        return len(self.encode(text))


__all__ = ["TokenizerConfig", "TokenizerWrapper"]
