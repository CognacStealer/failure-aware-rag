from fastapi import APIRouter, HTTPException, Request

from api.schemas import QueryRequest, QueryResponse, RetrievedDocument

router = APIRouter(prefix="/query", tags=["query"])


def _run(request: Request, body: QueryRequest, strategy: str) -> QueryResponse:
    service = request.app.state.rag
    if not service.ready:
        raise HTTPException(status_code=503, detail="RAG service is not ready")
    pipeline = {
        "vanilla": service.vanilla,
        "hybrid": service.hybrid,
        "crag": service.crag,
        "adaptive": service.adaptive,
    }[strategy]
    result = pipeline.run(body.query, body.top_k)
    track = result.get("track", {"crag": "CRAG"}.get(strategy, strategy.title()))
    return QueryResponse(
        query=body.query,
        track=track,
        answer=result["answer"],
        confidence=result.get("confidence"),
        retrieved_docs=[
            RetrievedDocument(
                doc_id=doc["doc_id"],
                title=doc.get("title", ""),
                rrf_score=doc.get("rrf_score"),
                dense_score=doc.get("dense_score"),
                bm25_score=doc.get("bm25_score"),
                relevance_score=doc.get("relevance_score"),
            )
            for doc in result["documents"]
        ],
    )


@router.post("/vanilla", response_model=QueryResponse)
def query_vanilla(body: QueryRequest, request: Request):
    return _run(request, body, "vanilla")


@router.post("/hybrid", response_model=QueryResponse)
def query_hybrid(body: QueryRequest, request: Request):
    return _run(request, body, "hybrid")


@router.post("/crag", response_model=QueryResponse)
def query_crag(body: QueryRequest, request: Request):
    return _run(request, body, "crag")


@router.post("/adaptive", response_model=QueryResponse)
def query_adaptive(body: QueryRequest, request: Request):
    return _run(request, body, "adaptive")
