import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core import results_store


class ResultsStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.patch = patch("core.results_store.config.RESULTS_DIR", self.root)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.directory.cleanup()

    def test_save_writes_result_tables_meta_and_index(self):
        source = self.root / "model.pkl"
        source.write_bytes(b"weights")
        path = results_store.save_result(
            "retrieval_eval",
            "Retrieval, test split",
            {"strategies": {"hybrid": {"recall_at_k": 0.716}}},
            headline={"hybrid": 0.716},
            tables={"per_question": [{"question_id": "q1", "recall": 1.0, "ids": ["a", "b"]},
                                     {"question_id": "q2", "extra": True}]},
            files={"model.pkl": source},
            label="test",
        )

        self.assertTrue(path.name.endswith("_test"))
        self.assertEqual(json.loads((path / "result.json").read_text())["strategies"]["hybrid"]["recall_at_k"], 0.716)
        with (path / "per_question.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["ids"], '["a", "b"]')  # lists are JSON-encoded
        self.assertEqual(rows[1]["extra"], "True")  # columns are the union across rows
        self.assertEqual((path / "model.pkl").read_bytes(), b"weights")
        meta = json.loads((path / "meta.json").read_text())
        self.assertEqual(meta["kind"], "retrieval_eval")
        self.assertIn("TOP_K_DEFAULT", meta["config"])

        index = results_store.list_results()
        self.assertEqual(index[0]["id"], path.name)
        self.assertEqual(index[0]["headline"], {"hybrid": 0.716})
        self.assertIn("Retrieval, test split", (self.root / "README.md").read_text())
        entry, latest_path = results_store.latest("retrieval_eval")
        self.assertEqual(latest_path, path)
        self.assertIsNone(results_store.latest("calibrator_eval"))

    def test_rejects_unknown_kind_and_nan(self):
        with self.assertRaises(ValueError):
            results_store.save_result("bogus", "t", {}, headline={})
        with self.assertRaises(ValueError):
            results_store.save_result("retrieval_eval", "t", {"auc": float("nan")}, headline={}, label="nan")


if __name__ == "__main__":
    unittest.main()
