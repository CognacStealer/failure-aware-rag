import unittest
from unittest.mock import patch

from core.service import RAGService


class FakeCollection:
    def count(self):
        return 2

    def get(self, limit, offset, include):
        self.offsets = getattr(self, "offsets", []) + [offset]
        if offset == 0:
            return {
                "ids": ["a", "b"],
                "documents": ["alpha", "beta"],
                "metadatas": [{"title": "A"}, {"title": "B"}],
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
            patch("core.service.chromadb.PersistentClient", return_value=client) as persistent,
            patch("core.service.TextGenerator") as generator_cls,
        ):
            service = RAGService()
            service.initialize()

        persistent.assert_called_once()
        self.assertEqual(client.collection_name, "docs")
        self.assertEqual(collection.offsets, [0])
        self.assertTrue(service.ready)
        self.assertTrue(service.status()["bm25_built"])
        generator_cls.assert_called_once()

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
