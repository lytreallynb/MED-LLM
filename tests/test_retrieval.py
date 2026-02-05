"""Tests for the retrieval and safety modules."""
import pytest
from medllm.retrieval import (
    SafetyConfig,
    SafetyChecker,
    SafetyDecision,
    RetrievalConfig,
    PROMPT_TEMPLATE,
)


class TestSafetyConfig:
    def test_default_values(self):
        config = SafetyConfig()
        assert "emergency" in config.blocked_keywords
        assert config.hallucination_threshold == 0.35
        assert config.include_disclaimer is True

    def test_custom_keywords(self):
        config = SafetyConfig(blocked_keywords=("test", "custom"))
        assert "test" in config.blocked_keywords
        assert "emergency" not in config.blocked_keywords


class TestSafetyChecker:
    def test_blocks_emergency_keywords(self):
        checker = SafetyChecker(SafetyConfig())

        # Should block
        decision = checker.check_query("I'm having an emergency!")
        assert decision is not None
        assert decision.blocked is True

        # Should also block with different case
        decision = checker.check_query("EMERGENCY help needed")
        assert decision is not None
        assert decision.blocked is True

    def test_allows_normal_queries(self):
        checker = SafetyChecker(SafetyConfig())

        decision = checker.check_query("What are the side effects of ibuprofen?")
        assert decision is None

    def test_blocks_suicide_keywords(self):
        checker = SafetyChecker(SafetyConfig())

        decision = checker.check_query("information about suicide")
        assert decision is not None
        assert decision.blocked is True

    def test_hallucination_threshold(self):
        checker = SafetyChecker(SafetyConfig(hallucination_threshold=0.5))

        # Score below threshold - should need blocking
        assert checker.needs_fact_block(0.3) is True

        # Score above threshold - should not need blocking
        assert checker.needs_fact_block(0.6) is False

        # Score at threshold - should not need blocking
        assert checker.needs_fact_block(0.5) is False

    def test_disclaimer(self):
        checker = SafetyChecker(SafetyConfig(include_disclaimer=True))
        disclaimer = checker.disclaimer()
        assert "educational purposes" in disclaimer.lower()

    def test_no_disclaimer(self):
        checker = SafetyChecker(SafetyConfig(include_disclaimer=False))
        disclaimer = checker.disclaimer()
        assert disclaimer == ""


class TestRetrievalConfig:
    def test_default_values(self):
        config = RetrievalConfig()
        assert config.top_k == 4
        assert "fda.index" in str(config.index_path)

    def test_custom_values(self):
        config = RetrievalConfig(top_k=6)
        assert config.top_k == 6


class TestPromptTemplate:
    def test_template_has_placeholders(self):
        assert "{question}" in PROMPT_TEMPLATE
        assert "{evidence}" in PROMPT_TEMPLATE

    def test_template_formatting(self):
        formatted = PROMPT_TEMPLATE.format(
            question="What is aspirin?",
            evidence="Aspirin is a pain reliever."
        )
        assert "What is aspirin?" in formatted
        assert "Aspirin is a pain reliever." in formatted
