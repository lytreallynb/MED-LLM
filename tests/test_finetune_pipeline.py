"""Tests for the fine-tuning dataset builder and behavioral metrics."""
import random

from medllm.build_finetune_dataset import (
    SECTION_GROUP,
    build_examples,
    extractive_answer,
    split_by_document,
)
from medllm.eval_behavior import (
    aggregate,
    detect_abstention,
    extract_citations,
    score_example,
)
from medllm.faithfulness import faithfulness_score, parse_passages
from medllm.grpo_finetune import score_completion


def make_chunks(n_docs: int = 10, per_doc: int = 2,
                sections=("warnings", "dosage_and_administration")):
    chunks = []
    for d in range(n_docs):
        for s, section in zip(range(per_doc), sections):
            chunks.append(
                {
                    "document_id": f"doc-{d}",
                    "drug_name": f"drug{d}",
                    "section": section,
                    "text": f"Section text for doc {d} part {s}. " * 12,
                }
            )
    return chunks


class TestSplit:
    def test_no_document_crosses_split(self):
        chunks = make_chunks(20)
        train, test = split_by_document(chunks, 0.25, random.Random(0))
        train_docs = {c["document_id"] for c in train}
        test_docs = {c["document_id"] for c in test}
        assert train_docs and test_docs
        assert not (train_docs & test_docs)

    def test_split_is_deterministic(self):
        chunks = make_chunks(20)
        a = split_by_document(chunks, 0.25, random.Random(7))
        b = split_by_document(chunks, 0.25, random.Random(7))
        assert a == b


class TestBuildExamples:
    def test_negative_ratio_and_gold_refs(self):
        chunks = make_chunks(15)
        rows = build_examples(chunks, 0.33, random.Random(1))
        assert rows
        negatives = [r for r in rows if r["meta"]["is_negative"]]
        assert 0.15 < len(negatives) / len(rows) < 0.55
        for r in rows:
            if r["meta"]["is_negative"]:
                assert r["meta"]["gold_ref"] is None
                assert "does not contain" in r["output"]
            else:
                ref = r["meta"]["gold_ref"]
                assert 1 <= ref <= r["meta"]["n_passages"]
                assert f"[{ref}]" in r["output"]

    def test_easy_negative_evidence_excludes_gold_document(self):
        chunks = make_chunks(15)
        rows = build_examples(chunks, 1.0, random.Random(2), hard_fraction=0.0)
        for r in rows:
            doc = r["meta"]["document_id"]
            drug = doc.replace("doc-", "drug")
            assert f"({drug}," not in r["input"]

    def test_hard_negative_shows_gold_drug_but_other_section(self):
        chunks = make_chunks(15)
        rows = build_examples(chunks, 1.0, random.Random(3), hard_fraction=1.0)
        hard = [r for r in rows if r["meta"]["is_hard"]]
        assert hard
        for r in hard:
            drug = r["meta"]["document_id"].replace("doc-", "drug")
            # The gold drug appears in the evidence, so name matching cannot
            # solve the example, but the questioned section itself is absent.
            assert f"({drug}," in r["input"]
            assert f"({drug}, {r['meta']['section']})" not in r["input"]

    def test_hard_distractor_never_from_confusable_section_group(self):
        chunks = make_chunks(15, per_doc=3,
                             sections=("warnings", "contraindications",
                                       "dosage_and_administration"))
        rows = build_examples(chunks, 1.0, random.Random(4), hard_fraction=1.0)
        for r in rows:
            if not r["meta"]["is_hard"]:
                continue
            drug = r["meta"]["document_id"].replace("doc-", "drug")
            gold_group = SECTION_GROUP[r["meta"]["section"]]
            for section in SECTION_GROUP:
                if f"({drug}, {section})" in r["input"]:
                    assert SECTION_GROUP[section] != gold_group

    def test_extractive_answer_cites_and_disclaims(self):
        out = extractive_answer("Aspirin", 2, "Take with water. " * 40)
        assert "[2]" in out
        assert "Aspirin" in out
        assert "Disclaimer:" in out


class TestMetrics:
    def test_abstention_detector_variants(self):
        assert detect_abstention("The provided evidence does not contain the answer.")
        assert detect_abstention("I cannot answer this without more information.")
        assert detect_abstention("There is no relevant information in the passages.")
        assert not detect_abstention("According to the FDA label for X [1]: take daily.")

    def test_citation_extraction_and_validity(self):
        meta = {"is_negative": False, "gold_ref": 2, "n_passages": 2}
        good = score_example("According to the label [2]: fine.\n\nDisclaimer: x", meta)
        assert good["citations"] == [2]
        assert good["citations_all_valid"]
        assert good["cited_gold"]
        bad = score_example("See [7] for details.", meta)
        assert not bad["citations_all_valid"]
        assert not bad["cited_gold"]

    def test_aggregate_separates_precision_and_recall(self):
        # A model that abstains on everything: recall 1.0, poor precision.
        rows = []
        for i in range(4):
            rows.append(
                score_example(
                    "I cannot answer this.",
                    {"is_negative": i < 2, "gold_ref": None if i < 2 else 1,
                     "n_passages": 2},
                )
            )
        m = aggregate(rows)
        assert m["abstention_recall"] == 1.0
        assert m["abstention_precision"] == 0.5
        assert m["false_abstain_rate_on_positives"] == 1.0

    def test_aggregate_reports_faithfulness_and_hard_subset(self):
        passages = ["(drugA, warnings) Causes severe headache and nausea."]
        meta_pos = {"is_negative": False, "gold_ref": 1, "n_passages": 1,
                    "is_hard": True}
        meta_neg = {"is_negative": True, "gold_ref": None, "n_passages": 1,
                    "is_hard": True}
        rows = [
            score_example(
                "Causes severe headache and nausea [1].",
                meta_pos, passages,
            ),
            score_example("I cannot answer this.", meta_neg, passages),
        ]
        m = aggregate(rows)
        assert m["answer_faithfulness"] is not None
        assert m["answer_faithfulness"] > 0.8
        assert m["n_hard"] == 2
        assert m["abstention_recall_hard"] == 1.0
        assert m["gold_citation_accuracy_hard"] == 1.0


class TestFaithfulness:
    PASSAGES = [
        "(drugA, warnings) May cause severe liver damage and dizziness in "
        "elderly patients.",
        "(drugB, dosage_and_administration) Take 20 mg twice daily with food.",
    ]

    def test_parse_passages_recovers_numbered_lines(self):
        input_text = (
            "Evidence:\n"
            "[1] (drugA, warnings) May cause liver damage.\n"
            "[2] (drugB, dosage_and_administration) Take 20 mg daily.\n"
            "\nQuestion: What are the warnings for DrugA?"
        )
        passages = parse_passages(input_text)
        assert len(passages) == 2
        assert passages[0].startswith("(drugA, warnings)")
        assert "Question:" not in passages[0] + passages[1]

    def test_extractive_answer_scores_high(self):
        answer = (
            "According to the FDA label for DrugA [1]: may cause severe liver "
            "damage and dizziness in elderly patients.\n\nDisclaimer: consult "
            "a healthcare provider."
        )
        score = faithfulness_score(answer, self.PASSAGES, [1])
        assert score is not None and score > 0.9

    def test_fabricated_answer_scores_low(self):
        answer = (
            "According to the FDA label for DrugA [1]: completely safe during "
            "pregnancy, recommended for children under five."
        )
        score = faithfulness_score(answer, self.PASSAGES, [1])
        assert score is not None and score < 0.3

    def test_scores_against_cited_passage_only(self):
        # Content copied from passage 2 while citing passage 1 is unfaithful.
        answer = "According to the label [1]: take 20 mg twice daily with food."
        cited_wrong = faithfulness_score(answer, self.PASSAGES, [1])
        cited_right = faithfulness_score(answer, self.PASSAGES, [2])
        assert cited_right is not None and cited_right > 0.9
        assert cited_wrong is not None and cited_wrong < cited_right

    def test_returns_none_when_nothing_to_score(self):
        assert faithfulness_score("[1]", self.PASSAGES, [1]) is None
        assert faithfulness_score("some answer text here", [], None) is None


class TestGrpoReward:
    PASSAGES = [
        "(drugA, warnings) May cause severe liver damage and dizziness in "
        "elderly patients.",
    ]

    def test_faithful_answer_beats_fabricated_answer(self):
        faithful = (
            "According to the FDA label for DrugA [1]: may cause severe liver "
            "damage and dizziness in elderly patients.\n\nDisclaimer: x"
        )
        fabricated = (
            "According to the FDA label for DrugA [1]: completely safe, no "
            "known risks, suitable for everyone.\n\nDisclaimer: x"
        )
        r_faithful = score_completion(faithful, False, 1, 1, self.PASSAGES)
        r_fabricated = score_completion(fabricated, False, 1, 1, self.PASSAGES)
        assert r_faithful > r_fabricated
        # The gap must be big enough to matter next to the binary terms.
        assert r_faithful - r_fabricated > 1.0

    def test_reward_without_passages_matches_old_behavior(self):
        answer = "According to the label [1]: fine.\n\nDisclaimer: x"
        assert score_completion(answer, False, 1, 1, None) == 1.75

    def test_abstention_on_negative_still_wins(self):
        abstain = "The provided evidence does not contain the answer."
        answer = "According to the label [1]: something plausible."
        assert score_completion(abstain, True, None, 1, self.PASSAGES) > \
            score_completion(answer, True, None, 1, self.PASSAGES)
