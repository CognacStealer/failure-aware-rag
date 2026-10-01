import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from core.calibrator import RetrievalCalibrator
from core.crag_corrector import CRAGCorrector
from core.router import AdaptiveRouter
from pipelines.adaptive_pipeline import AdaptivePipeline
from pipelines.crag_pipeline import CRAGPipeline


class FakeHybrid:
    def __init__(self, docs):
        self.docs = docs
        self.calls = []

    def retrieve(self, query, top_k):
        self.calls.append((query, top_k))
        return self.docs[:top_k]


class FakeGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, query, documents):
        self.calls.append((query, documents))
        return "generated answer"


class FakeCalibrator:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def extract_signals(self, query, docs):
        return {"dense_top": 1.0}

    def predict_distribution(self, signals):
        return self.mean, self.std


def keyword_scorer(pairs):
    """Stand-in cross-encoder: fraction of query words present in the passage."""
    scores = []
    for query, passage in pairs:
        words = query.split()
        scores.append(sum(word in passage for word in words) / len(words))
    return scores


class CorrectorTests(unittest.TestCase):
    def setUp(self):
        self.corrector = CRAGCorrector(keyword_scorer, min_relevance=0.75)
        self.documents = [
            {"doc_id": "a", "content": "alpha launch notes", "rrf_score": 0.01},
            {"doc_id": "b", "content": "alpha budget review", "rrf_score": 0.02},
            {"doc_id": "c", "content": "unrelated content", "rrf_score": 0.03},
        ]

    def test_filters_and_ranks_by_passage_relevance(self):
        corrected = self.corrector.correct("alpha launch", self.documents, top_k=3)
        self.assertEqual([doc["doc_id"] for doc in corrected], ["a"])
        self.assertEqual(corrected[0]["relevance_score"], 1.0)
        self.assertEqual(corrected[0]["passage"], "alpha launch notes")

    def test_grades_the_best_passage_of_long_documents(self):
        filler = "filler text " * 300
        documents = [{"doc_id": "long", "content": filler + "alpha launch details", "rrf_score": 0.01}]
        corrected = self.corrector.correct("alpha launch", documents, top_k=1)
        self.assertEqual(corrected[0]["relevance_score"], 1.0)
        self.assertIn("alpha launch", corrected[0]["passage"])

    def test_falls_back_to_best_graded_document_when_all_are_filtered(self):
        corrected = self.corrector.correct("missing terms", self.documents, top_k=2)
        self.assertEqual(len(corrected), 1)
        self.assertEqual(corrected[0]["doc_id"], "c")  # ties broken by rrf_score
        self.assertEqual(corrected[0]["relevance_score"], 0.0)

    def test_empty_candidates_return_empty(self):
        self.assertEqual(self.corrector.correct("alpha", [], top_k=3), [])

    def test_crag_fetches_wider_pool_then_generates_from_corrected_context(self):
        hybrid = FakeHybrid(self.documents)
        generator = FakeGenerator()
        pipeline = CRAGPipeline(hybrid, self.corrector, generator, retrieval_multiplier=3)

        result = pipeline.run("alpha launch", top_k=1)

        self.assertEqual(hybrid.calls, [("alpha launch", 3)])
        self.assertEqual([doc["doc_id"] for doc in result["documents"]], ["a"])
        self.assertEqual(generator.calls[0][1], result["documents"])


class AdaptivePipelineTests(unittest.TestCase):
    def _pipeline(self, mean, std):
        docs = [{"doc_id": "a", "content": "alpha launch", "rrf_score": 0.03}]
        hybrid = FakeHybrid(docs)
        generator = FakeGenerator()
        corrector = CRAGCorrector(keyword_scorer, min_relevance=0.1)
        crag = CRAGPipeline(hybrid, corrector, generator, retrieval_multiplier=2)
        settings = SimpleNamespace(
            ROUTER_ABSTAIN_MEAN_THRESHOLD=0.4,
            ROUTER_ABSTAIN_STD_THRESHOLD=0.3,
            ROUTER_FAST_MEAN_THRESHOLD=0.7,
        )
        pipeline = AdaptivePipeline(
            hybrid, crag, FakeCalibrator(mean, std), AdaptiveRouter(settings), generator
        )
        return pipeline, hybrid, generator

    def test_fast_track_generates_from_hybrid_results(self):
        pipeline, hybrid, generator = self._pipeline(0.8, 0.05)
        result = pipeline.run("alpha", 1)
        self.assertEqual(result["track"], "Fast")
        self.assertEqual(result["answer"], "generated answer")
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(hybrid.calls, [("alpha", 1)])

    def test_corrective_track_uses_crag_candidate_pool(self):
        pipeline, hybrid, generator = self._pipeline(0.55, 0.05)
        result = pipeline.run("alpha", 1)
        self.assertEqual(result["track"], "Corrective")
        self.assertEqual(hybrid.calls, [("alpha", 1), ("alpha", 2)])
        self.assertEqual(len(generator.calls), 1)

    def test_abstain_track_skips_generation(self):
        pipeline, _hybrid, generator = self._pipeline(0.2, 0.05)
        result = pipeline.run("alpha", 1)
        self.assertEqual(result["track"], "Abstain")
        self.assertEqual(result["answer"], "I don't know.")
        self.assertEqual(generator.calls, [])

    def test_high_uncertainty_abstains_even_with_high_mean(self):
        settings = SimpleNamespace(
            ROUTER_ABSTAIN_MEAN_THRESHOLD=0.4,
            ROUTER_ABSTAIN_STD_THRESHOLD=0.3,
            ROUTER_FAST_MEAN_THRESHOLD=0.7,
        )
        decision = AdaptiveRouter(settings).route(0.9, 0.31)
        self.assertEqual(decision.track, "Abstain")


def example(scale, label):
    return {**{name: scale for name in RetrievalCalibrator.FEATURE_NAMES}, "label": label}


class RetrievalCalibratorTests(unittest.TestCase):
    def test_unfitted_model_falls_back_to_the_corrective_track(self):
        calibrator = RetrievalCalibrator("unused.pkl")
        self.assertEqual(calibrator.predict_distribution({}), (0.5, 0.0))

    def test_extract_signals_from_fused_results(self):
        documents = [
            {"doc_id": "a", "bm25_score": 8.0, "dense_score": 0.7},
            {"doc_id": "b", "bm25_score": 4.0},
            {"doc_id": "c", "dense_score": 0.5},
        ]
        signals = RetrievalCalibrator.extract_signals("alpha launch", documents)
        self.assertEqual(set(signals), set(RetrievalCalibrator.FEATURE_NAMES))
        self.assertAlmostEqual(signals["bm25_top"], 4.0)  # 8.0 over two query terms
        self.assertAlmostEqual(signals["bm25_margin"], 0.5)
        self.assertAlmostEqual(signals["dense_margin"], 0.2)
        self.assertAlmostEqual(signals["agreement"], 1 / 3)
        empty = RetrievalCalibrator.extract_signals("", [])
        self.assertTrue(all(value == 0.0 for value in empty.values()))

    def test_fit_predict_and_reload(self):
        examples = [example(0.1 * i, int(i >= 5)) for i in range(10)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibrator.pkl"
            calibrator = RetrievalCalibrator(path, n_models=5)
            calibrator.fit(examples)
            low = calibrator.predict_distribution(examples[0])
            high = calibrator.predict_distribution(examples[-1])
            self.assertLess(low[0], 0.5)
            self.assertGreater(high[0], 0.5)
            self.assertGreaterEqual(high[1], 0.0)
            calibrator.save()

            loaded = RetrievalCalibrator(path)
            self.assertTrue(loaded.load())
            self.assertTrue(loaded.is_fitted)
            self.assertEqual(loaded.example_count, 10)
            self.assertEqual(loaded.predict_distribution(examples[-1]), high)

    def test_fit_rejects_invalid_examples(self):
        calibrator = RetrievalCalibrator("unused.pkl")
        with self.assertRaisesRegex(ValueError, "both"):
            calibrator.fit([example(0.1, 1), example(0.2, 1)])
        with self.assertRaisesRegex(ValueError, "labels"):
            calibrator.fit([example(0.1, 0), example(0.2, 0.4)])
        with self.assertRaisesRegex(ValueError, "missing"):
            calibrator.fit([{"dense_top": 1, "label": 1}, example(0.2, 0)])

    def test_loading_an_incompatible_file_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.pkl"
            import joblib
            joblib.dump({"model": None, "example_count": 3}, path)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                RetrievalCalibrator(path).load()


if __name__ == "__main__":
    unittest.main()
