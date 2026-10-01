"""Live RAG engine: stream each pipeline stage and the answer as server-sent events."""

import json
import random

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import config
from api.schemas import QueryRequest
from core import engine

router = APIRouter(prefix="/engine", tags=["engine"])


class StreamRequest(QueryRequest):
    strategy: str = Field(default="adaptive", pattern="^(adaptive|crag|hybrid|vanilla)$")


class Example(BaseModel):
    question_id: str
    question: str
    category: str


def _sse(event: dict) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"


@router.post("/stream")
def stream(body: StreamRequest, request: Request):
    service = request.app.state.rag
    if not service.ready:
        raise HTTPException(status_code=503, detail="RAG service is not ready")

    def events():
        try:
            for event in engine.run(service, body.query, body.top_k, body.strategy):
                yield _sse(event)
        except Exception as exc:  # report failures in-stream; the HTTP status is already sent
            yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/examples", response_model=list[Example])
def examples(request: Request, count: int = 4):
    """A few benchmark questions to try, so the page has real queries to start from."""
    service = request.app.state.rag
    if not service.ready:
        return []
    rows = service.client.get_collection(config.CHROMA_QUESTIONS_COLLECTION).get(include=["documents", "metadatas"])
    answerable = [
        (qid, text, (meta or {}).get("question_type", ""))
        for qid, text, meta in zip(rows["ids"], rows["documents"], rows["metadatas"])
        if (meta or {}).get("expected_doc_ids") and len(text) < 220
    ]
    return [Example(question_id=q, question=t, category=c)
            for q, t, c in random.sample(answerable, min(count, len(answerable)))]
