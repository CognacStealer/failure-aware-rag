from typing import Literal

from pydantic import BaseModel, Field, field_validator
from config import TOP_K_DEFAULT


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=20_000)
    top_k: int = Field(default=TOP_K_DEFAULT, ge=1, le=100)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must contain non-whitespace characters")
        return value


class Confidence(BaseModel):
    mean: float
    std: float


class RetrievedDocument(BaseModel):
    doc_id: str
    title: str = ""
    rrf_score: float


class QueryResponse(BaseModel):
    query: str
    track: Literal["Fast", "Corrective", "Abstain", "Vanilla", "Hybrid", "CRAG"]
    answer: str
    confidence: Confidence | None = None
    retrieved_docs: list[RetrievedDocument]
