from collections.abc import Callable
from typing import Any

import numpy as np

from core.chunking import best_passages

Scorer = Callable[[list[tuple[str, str]]], list[float]]


class CrossEncoderScorer:
    """Scores (query, passage) pairs with a cross-encoder, loaded on first use."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None

    def __call__(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self.model is None:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name, max_length=512)
        logits = np.asarray(self.model.predict(pairs, batch_size=32, show_progress_bar=False))
        return (1.0 / (1.0 + np.exp(-logits))).tolist()


class CRAGCorrector:
    """Grade candidates by their most relevant passages, drop irrelevant ones, and rerank."""

    def __init__(self, scorer: Scorer, min_relevance: float = 0.0, passages_per_doc: int = 2):
        if not 0.0 <= min_relevance <= 1.0:
            raise ValueError("min_relevance must be between 0 and 1")
        self.scorer = scorer
        self.min_relevance = min_relevance
        self.passages_per_doc = passages_per_doc

    def correct(
        self, query: str, documents: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        if not documents:
            return []
        # Long documents are graded on the passages sharing the most query terms,
        # not on their opening text.
        pairs, owners, passages = [], [], []
        for index, doc in enumerate(documents):
            for passage in best_passages(query, doc.get("content", ""), limit=self.passages_per_doc):
                title = doc.get("title", "")
                pairs.append((query, f"{title}\n{passage}" if title else passage))
                owners.append(index)
                passages.append(passage)
        scores = self.scorer(pairs) if pairs else []

        best: dict[int, tuple[float, str]] = {}
        for owner, score, passage in zip(owners, scores, passages):
            if owner not in best or score > best[owner][0]:
                best[owner] = (float(score), passage)

        graded = [
            {**doc, "relevance_score": best.get(i, (0.0, ""))[0], "passage": best.get(i, (0.0, ""))[1]}
            for i, doc in enumerate(documents)
        ]
        graded.sort(
            key=lambda doc: (-doc["relevance_score"], -doc.get("rrf_score", 0.0), str(doc.get("doc_id", "")))
        )
        kept = [doc for doc in graded if doc["relevance_score"] >= self.min_relevance]
        # If the grader rejects everything, keep the best-graded document rather
        # than generating from empty context.
        return (kept or graded[:1])[:top_k]
