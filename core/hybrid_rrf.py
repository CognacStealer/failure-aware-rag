from collections import defaultdict
from typing import Any


def reciprocal_rank_fusion(
    dense: list[dict[str, Any]],
    sparse: list[dict[str, Any]],
    *,
    alpha: float = 60.0,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Fuse ranked result lists, merging each document's fields from both lists.

    Fields from the dense list win on conflict; per-retriever scores such as
    ``dense_score`` and ``bm25_score`` are kept side by side.
    """
    scores: dict[str, float] = defaultdict(float)
    docs: dict[str, dict[str, Any]] = {}
    for weight, results in ((dense_weight, dense), (sparse_weight, sparse)):
        for rank, item in enumerate(results):
            doc_id = str(item["doc_id"])
            scores[doc_id] += weight / (alpha + rank + 1)
            docs[doc_id] = {**item, **docs.get(doc_id, {})}
    ranked = sorted(scores, key=lambda key: (-scores[key], key))[:top_k]
    return [{**docs[key], "rrf_score": scores[key]} for key in ranked]
