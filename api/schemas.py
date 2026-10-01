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
    # Each score is present only when the strategy produced it.
    rrf_score: float | None = None
    dense_score: float | None = None
    bm25_score: float | None = None
    relevance_score: float | None = None


class QueryResponse(BaseModel):
    query: str
    track: Literal["Fast", "Corrective", "Abstain", "Vanilla", "Hybrid", "CRAG"]
    answer: str
    confidence: Confidence | None = None
    retrieved_docs: list[RetrievedDocument]


class CalibrationExample(BaseModel):
    # Keys are RetrievalCalibrator.FEATURE_NAMES, as produced by extract_signals.
    signals: dict[str, float]
    # 1 when the retrieved set contained all gold documents, else 0.
    label: int = Field(ge=0, le=1)


class CalibrationRefitRequest(BaseModel):
    examples: list[CalibrationExample] = Field(min_length=2)
