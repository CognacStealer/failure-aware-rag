import re
from typing import Any

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


class BM25Retriever:
    """In-memory BM25 index built once from Chroma document records."""

    def __init__(self, records: list[dict[str, Any]]):
        self.records = records
        self.index = (
            BM25Okapi([tokenize(record["content"]) for record in records])
            if records
            else None
        )

    def retrieve(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        if not self.records or self.index is None:
            return []
        scores = self.index.get_scores(tokenize(query))
        indexes = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:top_k]
        return [
            {**self.records[i], "rrf_score": float(scores[i])}
            for i in indexes
            if scores[i] > 0
        ]
