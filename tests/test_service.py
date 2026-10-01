import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.service import RAGService


class FakeCollection:
    def count(self):
        return 3

    def get(self, include, limit=None, offset=None, ids=None):
        self.offsets = getattr(self, "offsets", []) + [offset]
        docs = {"a": ("alpha", "A"), "b": ("beta", "B"), "c": ("gamma", "C")}
        if ids is not None:
            return {"ids": ids, "documents": [docs[i][0] for i in ids]}
        if offset == 0:
            return {
                "ids": list(docs),
                "documents": [content for content, _ in docs.values()],
                "metadatas": [{"title": title} for _, title in docs.values()],
            }
        return {"ids": [], "documents": [], "metadatas": []}

    def query(self, **kwargs):
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}


class FakeClient:
    def __init__(self, collection):
        self.collection = collection

    def get_collection(self, name):
        self.collection_name = name
        return self.collection


class ServiceInitializationTests(unittest.TestCase):
    def test_initializes_retrieval_components_from_configured_collection(self):
        collection = FakeCollection()
        client = FakeClient(collection)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("core.service.config.BM25_CACHE_PATH", Path(directory) / "bm25.joblib"),
            patch("core.service.chromadb.PersistentClient", return_value=client) as persistent,
            patch("core.service.TextGenerator") as generator_cls,
        ):
            service = RAGService()
            service.initialize()
            # ids listing, then the full scan that builds and caches the index
            self.assertEqual(collection.offsets, [0, 0])
            rows = service.sparse.retrieve("alpha", top_k=1)

            collection.offsets = []
            RAGService().initialize()
            self.assertEqual(collection.offsets, [0])  # cache hit: ids listing only

        persistent.assert_called()
        self.assertEqual(client.collection_name, "docs")
        self.assertEqual(rows[0]["content"], "alpha")
        self.assertTrue(service.ready)
        self.assertTrue(service.status()["bm25_built"])
        generator_cls.assert_called()

    def test_empty_collection_does_not_become_ready(self):
        class EmptyCollection(FakeCollection):
            def count(self):
                return 0

        with patch("core.service.chromadb.PersistentClient", return_value=FakeClient(EmptyCollection())):
            service = RAGService()
            with self.assertRaisesRegex(RuntimeError, "is empty"):
                service.initialize()
        self.assertFalse(service.ready)


if __name__ == "__main__":
    unittest.main()
