from collections import defaultdict
from typing import Any


def reciprocal_rank_fusion(
    dense: list[dict[str, Any]],
    sparse: list[dict[str, Any]],
    *,
    alpha: float = 60.0,
    dense_weight: float = 0.8,
    sparse_weight: float = 1.2,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Fuse ranked result lists while retaining each document's metadata."""
    scores: dict[str, float] = defaultdict(float)
    docs: dict[str, dict[str, Any]] = {}
    for weight, results in ((dense_weight, dense), (sparse_weight, sparse)):
        for rank, item in enumerate(results):
            doc_id = str(item["doc_id"])
            scores[doc_id] += weight / (alpha + rank + 1)
            docs.setdefault(doc_id, item)
    ranked = sorted(scores, key=lambda key: (-scores[key], key))[:top_k]
    return [{**docs[key], "rrf_score": scores[key]} for key in ranked]
