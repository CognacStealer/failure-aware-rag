"""EnterpriseRAG-Bench questions from Chroma, with a fixed train/test split.

Calibrator fitting and threshold tuning use only the train split; reported
numbers come from the test split, so they are not in-sample.
"""

import hashlib
from typing import Any

import config

UNANSWERABLE_TYPES = {"info_not_found"}


def parse_ids(value: str | None) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def split_of(question_id: str) -> str:
    return "test" if int(hashlib.sha1(question_id.encode()).hexdigest(), 16) % 2 else "train"


def load_questions(client: Any, corpus_ids: set[str]) -> list[dict[str, Any]]:
    """Questions whose gold documents are all in the corpus, plus unanswerable ones."""
    rows = client.get_collection(config.CHROMA_QUESTIONS_COLLECTION).get(
        include=["documents", "metadatas"]
    )
    questions = []
    for question_id, text, metadata in zip(rows["ids"], rows["documents"], rows["metadatas"]):
        metadata = metadata or {}
        gold = parse_ids(metadata.get("expected_doc_ids"))
        kind = metadata.get("question_type", "")
        answerable = kind not in UNANSWERABLE_TYPES
        if answerable and (not gold or not set(gold) <= corpus_ids):
            continue
        questions.append({
            "question_id": question_id,
            "question": text,
            "type": kind,
            "gold": gold if answerable else [],
            "split": split_of(question_id),
        })
    return questions


def recall(documents: list[dict[str, Any]], gold: list[str]) -> float | None:
    if not gold:
        return None
    return len({doc["doc_id"] for doc in documents} & set(gold)) / len(set(gold))


def retrieval_succeeded(documents: list[dict[str, Any]], question: dict[str, Any]) -> int:
    """Calibrator label: 1 when every gold document was retrieved; 0 for unanswerable questions."""
    return int(bool(question["gold"]) and recall(documents, question["gold"]) == 1.0)
