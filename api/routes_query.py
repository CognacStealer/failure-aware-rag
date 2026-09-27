from fastapi import APIRouter, HTTPException, Request

from api.schemas import QueryRequest, QueryResponse, RetrievedDocument

router = APIRouter(prefix="/query", tags=["query"])


def _run(request: Request, body: QueryRequest, strategy: str) -> QueryResponse:
    service = request.app.state.rag
    if not service.ready:
        raise HTTPException(status_code=503, detail="RAG service is not ready")
    pipeline = service.vanilla if strategy == "vanilla" else service.hybrid
    result = pipeline.run(body.query, body.top_k)
    return QueryResponse(
        query=body.query,
        track="Vanilla" if strategy == "vanilla" else "Hybrid",
        answer=result["answer"],
        retrieved_docs=[
            RetrievedDocument(
                doc_id=doc["doc_id"],
                title=doc.get("title", ""),
                rrf_score=doc.get("rrf_score", 0.0),
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
