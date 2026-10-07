"""Run a RAG strategy step by step, emitting an event as each stage starts and finishes.

The stages and their order match the pipelines in pipelines/ (vanilla, hybrid,
CRAG, adaptive), so the live view shows exactly what the API does:

    dense -> bm25 -> fusion -> calibrator -> route -> rerank -> generate

Each event is a plain dict; routes_engine turns them into server-sent events.

Every finished or skipped stage also adds a step to a JSON trace: the function that ran
(module, class and method, read from the live objects), its inputs, every document it
returned with scores, and its timing. The generate step records the exact prompt sent
to the model. The trace arrives with each stage event and in full with the done event.
"""

import time
from collections.abc import Iterator
from typing import Any

from core.generator import build_prompt
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


def _method(obj: Any, name: str) -> str:
    kind = type(obj)
    return f"{kind.__module__}.{kind.__qualname__}.{name}"


def _function(fn: Any) -> str:
    return f"{fn.__module__}.{fn.__qualname__}"


def _doc_refs(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every document a stage returned, in rank order, with its scores."""
    refs = []
    for rank, doc in enumerate(documents, 1):
        ref = {"rank": rank, "doc_id": doc["doc_id"], "title": doc.get("title", "")}
        for key in ("dense_score", "bm25_score", "rrf_score", "relevance_score"):
            if key in doc:
                ref[key] = round(float(doc[key]), 4)
        if doc.get("passage"):
            ref["passage_chars"] = len(doc["passage"])
        refs.append(ref)
    return refs


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
    trace: list[dict[str, Any]] = []
    yield {"type": "start", "strategy": strategy, "query": query, "top_k": top_k, "plan": plan(strategy)}

    def record(stage: str, status: str, **fields: Any) -> dict[str, Any]:
        step = {"step": len(trace) + 1, "stage": stage, "status": status, **fields}
        trace.append(step)
        return step

    def finish(stage: str, t0: float, *, function: str, inputs: dict[str, Any], outputs: dict[str, Any],
               calls: list[str] | None = None, **details: Any) -> dict[str, Any]:
        timings[stage] = round((time.perf_counter() - t0) * 1000)
        step = record(stage, "done", function=function, **({"calls": calls} if calls else {}),
                      inputs=inputs, outputs=outputs, ms=timings[stage])
        return {"type": "stage", "stage": stage, "status": "done", "ms": timings[stage],
                "function": function, "trace_step": step, **details}

    def skip(stage: str, reason: str) -> dict[str, Any]:
        step = record(stage, "skipped", reason=reason)
        return {"type": "stage", "stage": stage, "status": "skipped", "reason": reason, "trace_step": step}

    # Corrective paths rerank a wider candidate pool; RRF output is a sorted list,
    # so the top_k prefix of the wide pool is exactly what plain hybrid returns.
    pool = service.crag.pool_size(top_k) if strategy in {"crag", "adaptive"} else top_k
    depth = max(pool, getattr(config, "RRF_CANDIDATES", pool))

    yield {"type": "stage", "stage": "dense", "status": "running"}
    t0 = time.perf_counter()
    dense_k = top_k if strategy == "vanilla" else depth
    dense = service.dense.retrieve(query, top_k=dense_k)
    yield finish("dense", t0, function=_method(service.dense, "retrieve"),
                 inputs={"query": query, "top_k": dense_k}, outputs={"count": len(dense), "documents": _doc_refs(dense)},
                 count=len(dense), top=[_doc_view(d) for d in dense[:PREVIEW_DOCS]])

    if strategy == "vanilla":
        documents = dense[:top_k]
        track = "Vanilla"
        confidence = None
    else:
        yield {"type": "stage", "stage": "bm25", "status": "running"}
        t0 = time.perf_counter()
        sparse = service.sparse.retrieve(query, top_k=depth)
        yield finish("bm25", t0, function=_method(service.sparse, "retrieve"),
                     inputs={"query": query, "top_k": depth}, outputs={"count": len(sparse), "documents": _doc_refs(sparse)},
                     count=len(sparse), top=[_doc_view(d) for d in sparse[:PREVIEW_DOCS]])

        yield {"type": "stage", "stage": "fusion", "status": "running"}
        t0 = time.perf_counter()
        candidates = reciprocal_rank_fusion(
            dense, sparse, alpha=config.RRF_ALPHA, dense_weight=config.RRF_DENSE_WEIGHT,
            sparse_weight=config.RRF_BM25_WEIGHT, top_k=pool,
        )
        fused = candidates[:top_k]
        both = sum(1 for d in fused if "dense_score" in d and "bm25_score" in d)
        yield finish("fusion", t0, function=_function(reciprocal_rank_fusion),
                     inputs={"dense_documents": len(dense), "bm25_documents": len(sparse), "alpha": config.RRF_ALPHA,
                             "dense_weight": config.RRF_DENSE_WEIGHT, "bm25_weight": config.RRF_BM25_WEIGHT, "top_k": pool},
                     outputs={"count": len(candidates), "kept_top_k": len(fused), "found_by_both": both,
                              "documents": _doc_refs(candidates)},
                     count=len(fused), from_both=both, top=[_doc_view(d) for d in fused[:PREVIEW_DOCS]])

        track = {"hybrid": "Hybrid", "crag": "CRAG"}.get(strategy, "Corrective")
        confidence = None
        if strategy == "adaptive":
            yield {"type": "stage", "stage": "calibrator", "status": "running"}
            t0 = time.perf_counter()
            signals = service.calibrator.extract_signals(query, fused)
            mean, std = service.calibrator.predict_distribution(signals)
            confidence = {"mean": mean, "std": std}
            rounded = {k: round(v, 4) for k, v in signals.items()}
            yield finish("calibrator", t0, function=_method(service.calibrator, "predict_distribution"),
                         calls=[_method(service.calibrator, "extract_signals"),
                                _method(service.calibrator, "predict_distribution")],
                         inputs={"query": query, "documents": len(fused)},
                         outputs={"signals": rounded, "mean": round(mean, 4), "std": round(std, 4),
                                  "ensemble_size": getattr(service.calibrator, "n_models", None),
                                  "fitted": service.calibrator.is_fitted},
                         signals=rounded, mean=mean, std=std, fitted=service.calibrator.is_fitted)

            t0 = time.perf_counter()
            decision = service.router.route(mean, std)
            track = decision.track
            thresholds = {"abstain_mean": service.router.abstain_mean, "abstain_std": service.router.abstain_std,
                          "fast_mean": service.router.fast_mean}
            yield finish("route", t0, function=_method(service.router, "route"),
                         inputs={"mean": round(mean, 4), "std": round(std, 4)},
                         outputs={"track": track, "thresholds": thresholds}, track=track, thresholds=thresholds)

        if strategy == "crag" or track == "Corrective":
            yield {"type": "stage", "stage": "rerank", "status": "running"}
            t0 = time.perf_counter()
            documents = service.crag.rerank(query, candidates, top_k)
            corrector = getattr(service.crag, "corrector", None)
            calls = [_method(corrector, "correct")] if corrector is not None else []
            if hasattr(corrector, "scorer"):
                calls += ["core.chunking.best_passages", _method(corrector.scorer, "__call__")]
            yield finish("rerank", t0, function=_method(service.crag, "rerank"), calls=calls,
                         inputs={"query": query, "candidates": len(candidates), "top_k": top_k},
                         outputs={"count": len(documents), "documents": _doc_refs(documents)},
                         candidates=len(candidates), kept=len(documents),
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
        # The same call the generator makes, so the trace shows exactly what the model read.
        budget = getattr(service.generator, "max_context_chars", None)
        prompt = build_prompt(query, documents, budget) if budget else None
        calls = [_function(build_prompt)]
        if getattr(service.generator, "backend", None) == "ollama":
            calls.append(f"POST {service.generator.host}/api/chat")
        yield finish("generate", t0, function=_method(service.generator, "stream"), calls=calls,
                     inputs={"model": service.generator.model_name, "context_budget_chars": budget,
                             "documents": _doc_refs(documents)},
                     outputs={"answer_chars": len(answer), "prompt_chars": len(prompt) if prompt else None,
                              "prompt": prompt},
                     model=service.generator.model_name, characters=len(answer))

    yield {
        "type": "done",
        "track": track,
        "confidence": confidence,
        "answer": answer,
        "timings_ms": timings,
        "total_ms": round((time.perf_counter() - started) * 1000),
        "trace": trace,
    }
