"""Okapi BM25 over a sparse term-weight matrix, sized for a laptop.

Scoring matches rank_bm25.BM25Okapi (k1=1.5, b=0.75, epsilon floor for negative
IDF), but the index is a float32 CSC matrix of precomputed per-term weights
instead of one Python dict per document, and document text is not kept in
memory: callers pass ``fetch_content`` to hydrate the returned rows.
"""

import hashlib
import re
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

TOKEN_PATTERN = r"(?u)\w+"


def tokenize(text: str) -> list[str]:
    return re.findall(TOKEN_PATTERN, text.lower())


def corpus_signature(doc_ids: Iterable[str]) -> str:
    digest = hashlib.sha1()
    for doc_id in sorted(doc_ids):
        digest.update(doc_id.encode())
        digest.update(b"\0")
    return digest.hexdigest()


class BM25Retriever:
    def __init__(
        self,
        records: Iterable[dict[str, Any]] = (),
        *,
        fetch_content: Callable[[list[str]], dict[str, str]] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        self.fetch_content = fetch_content
        self.doc_ids: list[str] = []
        self.titles: list[str] = []
        self.vocabulary: dict[str, int] = {}
        self.weights: sparse.csc_matrix | None = None

        def contents():
            for record in records:
                self.doc_ids.append(str(record["doc_id"]))
                self.titles.append(record.get("title", ""))
                yield record.get("content", "") or ""

        vectorizer = CountVectorizer(token_pattern=TOKEN_PATTERN, lowercase=True, dtype=np.int32)
        try:
            counts = vectorizer.fit_transform(contents()).tocsr()
        except ValueError:  # empty corpus or no tokens at all
            return
        self.vocabulary = vectorizer.vocabulary_
        self.weights = self._weights(counts, k1, b, epsilon)

    @staticmethod
    def _weights(counts: sparse.csr_matrix, k1: float, b: float, epsilon: float) -> sparse.csc_matrix:
        n_docs = counts.shape[0]
        doc_lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
        avgdl = doc_lengths.mean() if n_docs else 0.0
        doc_freq = np.bincount(counts.indices, minlength=counts.shape[1])
        idf = np.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5))
        idf[idf < 0] = epsilon * idf.mean()

        rows = np.repeat(np.arange(n_docs), np.diff(counts.indptr))
        tf = counts.data.astype(np.float64)
        norm = k1 * (1 - b + b * doc_lengths[rows] / max(avgdl, 1e-9))
        data = idf[counts.indices] * tf * (k1 + 1) / (tf + norm)
        return sparse.csr_matrix(
            (data.astype(np.float32), counts.indices, counts.indptr), shape=counts.shape
        ).tocsc()

    @property
    def size(self) -> int:
        return len(self.doc_ids)

    def scores(self, query: str) -> np.ndarray:
        if self.weights is None:
            return np.zeros(self.size, dtype=np.float32)
        term_counts = Counter(t for t in tokenize(query) if t in self.vocabulary)
        if not term_counts:
            return np.zeros(self.size, dtype=np.float32)
        columns = [self.vocabulary[term] for term in term_counts]
        repeats = np.array(list(term_counts.values()), dtype=np.float32)
        return np.asarray(self.weights[:, columns] @ repeats).ravel()

    def retrieve(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        scores = self.scores(query)
        if not scores.size:
            return []
        k = min(top_k, scores.size)
        candidates = np.argpartition(-scores, k - 1)[:k]
        ranked = sorted((i for i in candidates if scores[i] > 0), key=lambda i: (-scores[i], i))
        rows = [
            {"doc_id": self.doc_ids[i], "title": self.titles[i], "bm25_score": float(scores[i])}
            for i in ranked
        ]
        if self.fetch_content and rows:
            contents = self.fetch_content([row["doc_id"] for row in rows])
            for row in rows:
                row["content"] = contents.get(row["doc_id"], "")
        return rows

    def save(self, path: str | Path, signature: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "signature": signature,
            "doc_ids": self.doc_ids,
            "titles": self.titles,
            "vocabulary": self.vocabulary,
            "weights": self.weights,
        }, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        signature: str,
        fetch_content: Callable[[list[str]], dict[str, str]] | None = None,
    ) -> "BM25Retriever | None":
        """Return the cached index, or None when it is missing or built from another corpus."""
        path = Path(path)
        if not path.exists():
            return None
        saved = joblib.load(path)
        if saved.get("signature") != signature:
            return None
        retriever = cls(fetch_content=fetch_content)
        retriever.doc_ids = saved["doc_ids"]
        retriever.titles = saved["titles"]
        retriever.vocabulary = saved["vocabulary"]
        retriever.weights = saved["weights"]
        return retriever
