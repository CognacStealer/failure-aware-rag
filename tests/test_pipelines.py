import unittest
from types import SimpleNamespace

from core.hybrid_rrf import reciprocal_rank_fusion
from pipelines.hybrid_pipeline import HybridPipeline
from pipelines.vanilla_pipeline import VanillaPipeline


class FakeRetriever:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def retrieve(self, query, top_k):
        self.calls.append((query, top_k))
        return self.rows[:top_k]


class FakeGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, query, documents):
        self.calls.append((query, documents))
        return "answer"


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_fuses_shared_documents_and_sorts_by_weighted_score(self):
        dense = [{"doc_id": "a", "content": "dense a"}, {"doc_id": "b"}]
        sparse = [{"doc_id": "b", "content": "sparse b"}, {"doc_id": "a"}]

        result = reciprocal_rank_fusion(
            dense, sparse, alpha=0, dense_weight=1, sparse_weight=2, top_k=2
        )

        self.assertEqual([row["doc_id"] for row in result], ["b", "a"])
        self.assertAlmostEqual(result[0]["rrf_score"], 2 + 1 / 2)
        self.assertAlmostEqual(result[1]["rrf_score"], 1 + 2 / 2)
        self.assertEqual(result[1]["content"], "dense a")

    def test_limits_results_and_handles_empty_inputs(self):
        rows = [{"doc_id": str(i)} for i in range(4)]
        self.assertEqual(len(reciprocal_rank_fusion(rows, [], top_k=2)), 2)
        self.assertEqual(reciprocal_rank_fusion([], []), [])


class PipelineTests(unittest.TestCase):
    def test_vanilla_passes_dense_documents_to_generator(self):
        docs = [{"doc_id": "x", "content": "text"}]
        retriever = FakeRetriever(docs)
        generator = FakeGenerator()

        result = VanillaPipeline(retriever, generator).run("question", 3)

        self.assertEqual(result, {"answer": "answer", "documents": docs})
        self.assertEqual(retriever.calls, [("question", 3)])
        self.assertEqual(generator.calls, [("question", docs)])

    def test_hybrid_fuses_retrieval_before_generation(self):
        dense = FakeRetriever([{"doc_id": "dense", "content": "d"}])
        sparse = FakeRetriever([{"doc_id": "sparse", "content": "s"}])
        generator = FakeGenerator()
        settings = SimpleNamespace(
            RRF_ALPHA=60, RRF_DENSE_WEIGHT=0.8, RRF_BM25_WEIGHT=1.2
        )

        result = HybridPipeline(dense, sparse, generator, settings).run("question", 5)

        self.assertEqual(result["answer"], "answer")
        self.assertEqual([d["doc_id"] for d in result["documents"]], ["sparse", "dense"])
        self.assertEqual(generator.calls[0][1], result["documents"])


if __name__ == "__main__":
    unittest.main()
