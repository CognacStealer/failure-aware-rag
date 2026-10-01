import tempfile
import unittest
from pathlib import Path

from core.retriever_bm25 import BM25Retriever, corpus_signature, tokenize
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
        self.assertEqual(rows[0]["title"], "Launch")
        self.assertGreater(rows[0]["bm25_score"], 0)
        self.assertNotIn("content", rows[0])
        self.assertEqual(retriever.retrieve("no matching terms"), [])

    def test_bm25_matches_rank_bm25_scores(self):
        from rank_bm25 import BM25Okapi

        reference = BM25Okapi([tokenize(r["content"]) for r in self.records])
        retriever = BM25Retriever(self.records)
        for query in ["alpha launch", "beta beta review", "gamma"]:
            expected = reference.get_scores(tokenize(query))
            for got, want in zip(retriever.scores(query), expected):
                self.assertAlmostEqual(float(got), want, places=5)

    def test_rows_are_hydrated_through_fetch_content(self):
        fetched = []

        def fetch(ids):
            fetched.append(ids)
            return {doc_id: f"text of {doc_id}" for doc_id in ids}

        rows = BM25Retriever(self.records, fetch_content=fetch).retrieve("alpha", top_k=1)
        self.assertEqual(fetched, [["a"]])
        self.assertEqual(rows[0]["content"], "text of a")

    def test_cache_round_trip_is_tied_to_the_corpus(self):
        retriever = BM25Retriever(self.records)
        signature = corpus_signature(r["doc_id"] for r in self.records)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bm25.joblib"
            retriever.save(path, signature)
            loaded = BM25Retriever.load(path, signature)
            self.assertEqual(loaded.retrieve("alpha launch"), retriever.retrieve("alpha launch"))
            self.assertIsNone(BM25Retriever.load(path, corpus_signature(["other"])))
            self.assertIsNone(BM25Retriever.load(Path(directory) / "missing", signature))

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
        self.assertAlmostEqual(rows[0]["dense_score"], 0.875)  # squared L2 -> cosine

    def test_invalid_embedding_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            ChromaDenseRetriever(FakeChromaCollection(), "unused", mode="bad")


if __name__ == "__main__":
    unittest.main()
