import unittest
from types import SimpleNamespace

from fastapi import HTTPException, Response
from starlette.requests import Request
from pydantic import ValidationError

from api.schemas import QueryRequest, QueryResponse
from api.routes_health import health, ready
from api.routes_query import query_hybrid, query_vanilla


class QuerySchemaTests(unittest.TestCase):
    def test_request_uses_default_top_k_and_rejects_invalid_values(self):
        self.assertEqual(QueryRequest(query="hello").top_k, 10)
        with self.assertRaises(ValidationError):
            QueryRequest(query="hello", top_k=0)
        with self.assertRaises(ValidationError):
            QueryRequest(query="   ")

    def test_response_accepts_fixed_strategy_and_nullable_confidence(self):
        response = QueryResponse(
            query="hello",
            track="Hybrid",
            answer="world",
            retrieved_docs=[{"doc_id": "doc-1", "title": "Title", "rrf_score": 0.1}],
        )
        self.assertIsNone(response.confidence)
        self.assertEqual(response.retrieved_docs[0].doc_id, "doc-1")
        with self.assertRaises(ValidationError):
            QueryResponse(query="hello", track="unknown", answer="world", retrieved_docs=[])


class FakePipeline:
    def __init__(self, track):
        self.track = track
        self.calls = []

    def run(self, query, top_k):
        self.calls.append((query, top_k))
        return {
            "answer": f"{self.track} answer",
            "documents": [{"doc_id": "d1", "title": "Doc", "rrf_score": 0.02}],
        }


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.rag = SimpleNamespace(
            ready=True,
            vanilla=FakePipeline("vanilla"),
            hybrid=FakePipeline("hybrid"),
            status=lambda: {
                "ready": True,
                "chroma_connected": True,
                "bm25_built": True,
                "generator_loaded": True,
                "error": None,
            },
        )
        app = SimpleNamespace(state=SimpleNamespace(rag=self.rag))
        self.request = Request({
            "type": "http",
            "method": "POST",
            "path": "/query",
            "headers": [],
            "query_string": b"",
            "scheme": "http",
            "server": ("test", 80),
            "client": ("test", 1),
            "root_path": "",
            "app": app,
        })

    def test_fixed_strategy_routes_call_expected_pipeline(self):
        vanilla = query_vanilla(QueryRequest(query="  hello  ", top_k=4), self.request)
        hybrid = query_hybrid(QueryRequest(query="hello"), self.request)

        self.assertEqual(vanilla.track, "Vanilla")
        self.assertEqual(vanilla.query, "hello")
        self.assertEqual(self.rag.vanilla.calls, [("hello", 4)])
        self.assertEqual(hybrid.track, "Hybrid")
        self.assertEqual(self.rag.hybrid.calls, [("hello", 10)])

    def test_query_returns_service_unavailable_until_initialized(self):
        self.rag.ready = False
        with self.assertRaises(HTTPException) as raised:
            query_hybrid(QueryRequest(query="hello"), self.request)
        self.assertEqual(raised.exception.status_code, 503)

    def test_health_and_readiness(self):
        self.assertEqual(health(), {"status": "ok"})
        response = Response()
        self.assertTrue(ready(self.request, response)["ready"])
        self.assertEqual(response.status_code, 200)
        self.rag.status = lambda: {"ready": False, "generator_loaded": False}
        response = Response()
        ready(self.request, response)
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
