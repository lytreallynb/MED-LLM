"""Answer-to-evidence faithfulness scoring, shared by training and eval.

Measures how much of a generated answer is actually supported by the evidence
passages it was given: the fraction of the answer's content words that appear
in the cited passages (or in any provided passage when no valid citation
exists). A fabricated answer scores near 0 even when its citation markers are
formally valid, which is exactly the failure mode the citation checks in
eval_behavior.py cannot see.

The score is deliberately mechanical (lexical overlap, no model calls) so it
can run inside the GRPO reward function on every sampled completion. It is a
precision-style measure: it punishes content the evidence does not support,
and does not require the answer to cover the whole passage.
"""
from __future__ import annotations

import re
from typing import List, Optional, Set

# Function words carry no medical claim and are ignored on both sides.
STOPWORDS = frozenset(
    """a an and are as at be but by can could do does for from had has have if
    in into is it its may might must no not of on or should such than that the
    their then there these they this those to was were which will with would
    you your""".split()
)

# Framing vocabulary from the answer template itself. These words describe the
# act of answering, not the medical content, so they never count as
# unsupported even though they rarely appear in a label passage.
BOILERPLATE = frozenset(
    """according based fda label evidence passage passages provided question
    answer information""".split()
)

WORD_RE = re.compile(r"[a-z]+|\d+(?:\.\d+)?%?")
CITATION_RE = re.compile(r"\[(\d{1,2})\]")
DISCLAIMER_SPLIT_RE = re.compile(r"disclaimer\s*:", re.IGNORECASE)
PASSAGE_LINE_RE = re.compile(r"^\[\d{1,2}\]\s*(.+)$")


def content_words(text: str) -> Set[str]:
    """Lowercase content words: numbers always kept, short/function words dropped."""
    kept: Set[str] = set()
    for word in WORD_RE.findall(text.lower()):
        if word[0].isdigit():
            kept.add(word)
        elif len(word) > 2 and word not in STOPWORDS and word not in BOILERPLATE:
            kept.add(word)
    return kept


def parse_passages(input_text: str) -> List[str]:
    """Recover the numbered evidence passages from an example's input field.

    Matches the format written by build_finetune_dataset.build_input: one
    passage per line, "[n] (drug, section) text". The "(drug, section)" header
    is kept so drug and section names count as supported vocabulary.
    """
    passages: List[str] = []
    for line in input_text.splitlines():
        match = PASSAGE_LINE_RE.match(line.strip())
        if match:
            passages.append(match.group(1))
    return passages


def faithfulness_score(
    answer: str, passages: List[str], citations: Optional[List[int]] = None
) -> Optional[float]:
    """Fraction of the answer's content words found in the evidence, 0.0-1.0.

    The answer is scored against the passages it cites when at least one
    citation is valid, otherwise against all provided passages. The disclaimer
    line and citation markers are stripped first. Returns None when there is
    nothing to score (no content words, or no passages).
    """
    body = DISCLAIMER_SPLIT_RE.split(answer)[0]
    body = CITATION_RE.sub(" ", body)
    answer_words = content_words(body)
    if not answer_words or not passages:
        return None

    cited = [
        passages[c - 1] for c in (citations or []) if 1 <= c <= len(passages)
    ]
    evidence = cited if cited else passages
    evidence_words: Set[str] = set()
    for passage in evidence:
        evidence_words |= content_words(passage)
    return len(answer_words & evidence_words) / len(answer_words)
