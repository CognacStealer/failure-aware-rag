import unittest

from core.retriever_bm25 import BM25Retriever, tokenize
from core.retriever_chroma import ChromaDenseRetriever


class BM25RetrieverTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"doc_id": "a", "content": "alpha project launch", "title": "Launch"},
            {"doc_id": "b", "content": "beta budget review", "title": "Budget"},
            {"doc_id": "c", "content": "gamma planning session", "title": "Planning"},
        ]

    def test_tokenizer_is_case_insensitive_and_splits_punctuation(self):
        self.assertEqual(tokenize("Alpha, BETA-2!"), ["alpha", "beta", "2"])

    def test_bm25_returns_matching_documents_in_rank_order(self):
        retriever = BM25Retriever(self.records)
        rows = retriever.retrieve("alpha launch", top_k=2)
        self.assertEqual(rows[0]["doc_id"], "a")
        self.assertGreater(rows[0]["rrf_score"], 0)
        self.assertEqual(retriever.retrieve("no matching terms"), [])

    def test_empty_bm25_index_is_safe(self):
        self.assertEqual(BM25Retriever([]).retrieve("anything"), [])


class FakeChromaCollection:
    def __init__(self):
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "ids": [["doc-1"]],
            "documents": [["retrieved text"]],
            "metadatas": [[{"title": "A title", "source_type": "wiki"}]],
            "distances": [[0.25]],
        }


class ChromaDenseRetrieverTests(unittest.TestCase):
    def test_collection_embedding_mode_uses_chroma_text_query(self):
        collection = FakeChromaCollection()
        rows = ChromaDenseRetriever(collection, "unused-model").retrieve("my question", 3)

        self.assertEqual(collection.calls[0]["query_texts"], ["my question"])
        self.assertEqual(collection.calls[0]["n_results"], 3)
        self.assertEqual(rows[0]["doc_id"], "doc-1")
        self.assertEqual(rows[0]["content"], "retrieved text")
        self.assertEqual(rows[0]["metadata"]["source_type"], "wiki")
        self.assertAlmostEqual(rows[0]["distance"], 0.25)

    def test_invalid_embedding_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            ChromaDenseRetriever(FakeChromaCollection(), "unused", mode="bad")


if __name__ == "__main__":
    unittest.main()
