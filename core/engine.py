"""Run a RAG strategy step by step, emitting an event as each stage starts and finishes.

The stages and their order match the pipelines in pipelines/ (vanilla, hybrid,
CRAG, adaptive), so the live view shows exactly what the API does:

    dense -> bm25 -> fusion -> calibrator -> route -> rerank -> generate

Each event is a plain dict; routes_engine turns them into server-sent events.
"""

import time
from collections.abc import Iterator
from typing import Any

from core.hybrid_rrf import reciprocal_rank_fusion

STRATEGIES = ("adaptive", "crag", "hybrid", "vanilla")
PREVIEW_DOCS = 5
PASSAGE_PREVIEW_CHARS = 600
ABSTAIN_ANSWER = "I don't know."


def _doc_view(doc: dict[str, Any]) -> dict[str, Any]:
    view = {"doc_id": doc["doc_id"], "title": doc.get("title", "")}
    for key in ("dense_score", "bm25_score", "rrf_score", "relevance_score"):
        if key in doc:
            view[key] = round(float(doc[key]), 4)
    if doc.get("passage"):
        view["passage"] = doc["passage"][:PASSAGE_PREVIEW_CHARS]
    return view


def plan(strategy: str) -> list[str]:
    """Stages a strategy may run, in order (adaptive skips rerank or generation by route)."""
    return {
        "vanilla": ["dense", "generate"],
        "hybrid": ["dense", "bm25", "fusion", "generate"],
        "crag": ["dense", "bm25", "fusion", "rerank", "generate"],
        "adaptive": ["dense", "bm25", "fusion", "calibrator", "route", "rerank", "generate"],
    }[strategy]


def run(service: Any, query: str, top_k: int, strategy: str) -> Iterator[dict[str, Any]]:
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    config = service.hybrid.config
    started = time.perf_counter()
    timings: dict[str, float] = {}
    yield {"type": "start", "strategy": strategy, "query": query, "top_k": top_k, "plan": plan(strategy)}

    def finish(stage: str, t0: float, **details: Any) -> dict[str, Any]:
        timings[stage] = round((time.perf_counter() - t0) * 1000)
        return {"type": "stage", "stage": stage, "status": "done", "ms": timings[stage], **details}

    def skip(stage: str, reason: str) -> dict[str, Any]:
        return {"type": "stage", "stage": stage, "status": "skipped", "reason": reason}

    # Corrective paths rerank a wider candidate pool; RRF output is a sorted list,
    # so the top_k prefix of the wide pool is exactly what plain hybrid returns.
    pool = top_k * service.crag.retrieval_multiplier if strategy in {"crag", "adaptive"} else top_k
    depth = max(pool, getattr(config, "RRF_CANDIDATES", pool))

    yield {"type": "stage", "stage": "dense", "status": "running"}
    t0 = time.perf_counter()
    dense = service.dense.retrieve(query, top_k=top_k if strategy == "vanilla" else depth)
    yield finish("dense", t0, count=len(dense), top=[_doc_view(d) for d in dense[:PREVIEW_DOCS]])

    if strategy == "vanilla":
        documents = dense[:top_k]
        track = "Vanilla"
        confidence = None
    else:
        yield {"type": "stage", "stage": "bm25", "status": "running"}
        t0 = time.perf_counter()
        sparse = service.sparse.retrieve(query, top_k=depth)
        yield finish("bm25", t0, count=len(sparse), top=[_doc_view(d) for d in sparse[:PREVIEW_DOCS]])

        yield {"type": "stage", "stage": "fusion", "status": "running"}
        t0 = time.perf_counter()
        candidates = reciprocal_rank_fusion(
            dense, sparse, alpha=config.RRF_ALPHA, dense_weight=config.RRF_DENSE_WEIGHT,
            sparse_weight=config.RRF_BM25_WEIGHT, top_k=pool,
        )
        fused = candidates[:top_k]
        both = sum(1 for d in fused if "dense_score" in d and "bm25_score" in d)
        yield finish("fusion", t0, count=len(fused), from_both=both, top=[_doc_view(d) for d in fused[:PREVIEW_DOCS]])

        track = {"hybrid": "Hybrid", "crag": "CRAG"}.get(strategy, "Corrective")
        confidence = None
        if strategy == "adaptive":
            yield {"type": "stage", "stage": "calibrator", "status": "running"}
            t0 = time.perf_counter()
            signals = service.calibrator.extract_signals(query, fused)
            mean, std = service.calibrator.predict_distribution(signals)
            confidence = {"mean": mean, "std": std}
            yield finish("calibrator", t0, signals={k: round(v, 4) for k, v in signals.items()},
                         mean=mean, std=std, fitted=service.calibrator.is_fitted)

            t0 = time.perf_counter()
            decision = service.router.route(mean, std)
            track = decision.track
            yield finish("route", t0, track=track, thresholds={
                "abstain_mean": service.router.abstain_mean, "abstain_std": service.router.abstain_std,
                "fast_mean": service.router.fast_mean,
            })

        if strategy == "crag" or track == "Corrective":
            yield {"type": "stage", "stage": "rerank", "status": "running"}
            t0 = time.perf_counter()
            documents = service.crag.corrector.correct(query, candidates, top_k)
            yield finish("rerank", t0, candidates=len(candidates), kept=len(documents),
                         top=[_doc_view(d) for d in documents[:PREVIEW_DOCS]])
        else:
            documents = fused
            if strategy == "adaptive":
                yield skip("rerank", f"{track} track")

    yield {"type": "sources", "documents": [_doc_view(d) for d in documents]}

    if track == "Abstain":
        yield skip("generate", "the calibrator abstained")
        answer = ABSTAIN_ANSWER
        yield {"type": "token", "text": answer}
    else:
        yield {"type": "stage", "stage": "generate", "status": "running", "model": service.generator.model_name}
        t0 = time.perf_counter()
        pieces = []
        for piece in service.generator.stream(query, documents):
            pieces.append(piece)
            yield {"type": "token", "text": piece}
        answer = "".join(pieces).strip()
        yield finish("generate", t0, model=service.generator.model_name, characters=len(answer))

    yield {
        "type": "done",
        "track": track,
        "confidence": confidence,
        "answer": answer,
        "timings_ms": timings,
        "total_ms": round((time.perf_counter() - started) * 1000),
    }
