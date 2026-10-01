import json
import tempfile
import unittest
from pathlib import Path

from core.ablation_report import attribute, compute_summary, evidence_share


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class AttributionRuleTests(unittest.TestCase):
    def test_walks_the_pipeline_backwards(self):
        present, absent = [True, True, False], [False, False, True]
        self.assertEqual(attribute(True, "basic", 1.0, absent), "correct")
        self.assertEqual(attribute(False, "basic", 1.0, present), "generation_error")
        self.assertEqual(attribute(False, "basic", 1.0, absent), "context_loss")
        self.assertEqual(attribute(False, "basic", 0.5, absent), "retrieval_miss")
        self.assertEqual(attribute(False, "high_level", None, absent), "retrieval_miss")
        self.assertEqual(attribute(False, "info_not_found", None, present), "missed_abstention")
        self.assertEqual(attribute(True, "info_not_found", None, None), "correct")
        self.assertEqual(attribute(False, "basic", 1.0, [None, None]), "ungraded")

    def test_evidence_share_ignores_ungraded_facts(self):
        self.assertEqual(evidence_share([True, None, False]), 0.5)
        self.assertIsNone(evidence_share([]))


class SummaryAttributionTests(unittest.TestCase):
    def test_summary_counts_outcomes_per_model(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "manifest.json").write_text(json.dumps({"candidates": {"org/m": "m"}, "judge_model": "j"}))
            write_jsonl(run / "retrieval.jsonl", [{"question_id": "q1", "recall_at_k": 1.0},
                                                  {"question_id": "q2", "recall_at_k": 0.0},
                                                  {"question_id": "q3", "recall_at_k": 1.0}])
            write_jsonl(run / "m.jsonl", [{"question_id": q, "answer": "a", "seconds": 1} for q in ("q1", "q2", "q3")])
            write_jsonl(run / "m_judged.jsonl", [
                {"question_id": "q1", "correct": True, "facts_present": [True], "judge_parse_ok": True},
                {"question_id": "q2", "correct": False, "facts_present": [False], "judge_parse_ok": True},
                {"question_id": "q3", "correct": False, "facts_present": [False], "judge_parse_ok": True},
            ])
            write_jsonl(run / "context_facts.jsonl", [{"question_id": "q1", "facts_in_context": [True]},
                                                      {"question_id": "q2", "facts_in_context": [False]},
                                                      {"question_id": "q3", "facts_in_context": [True]}])
            write_jsonl(run / "context_controls.jsonl", [{"question_id": "q1", "facts_in_context": [False, False]}])

            attribution = compute_summary(run, {"q1": "basic", "q2": "basic", "q3": "basic"})["attribution"]

        outcomes = attribution["models"]["org/m"]["outcomes"]
        self.assertEqual((outcomes["correct"], outcomes["retrieval_miss"], outcomes["generation_error"]), (1, 1, 1))
        self.assertAlmostEqual(attribution["context_fact_recall"], 2 / 3)
        self.assertEqual(attribution["context_judge_false_positive_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
