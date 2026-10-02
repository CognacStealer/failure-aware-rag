"""Download EnterpriseRAG-Bench and load it into ChromaDB.

Creates (or tops up) two collections in CHROMA_PATH:

* Questions - the 500 benchmark questions; gold document IDs, gold answer, answer
  facts and question type are stored as metadata.
* docs      - a corpus of every gold document plus a seeded random sample of the
  other documents (the full 511,962 documents are far too many to embed on a CPU).

Both are embedded with Chroma's default model (all-MiniLM-L6-v2), which is what
config.EMBEDDING_MODEL expects. The load is resumable and additive: documents and
questions already present are skipped, nothing is deleted or overwritten.

    PYTHONPATH=. python scripts/load_dataset.py                  # ~78k docs, ~45 min on CPU
    PYTHONPATH=. python scripts/load_dataset.py --sample 5000    # quick start
"""

import argparse
import ast
import random
import time

import chromadb
from datasets import load_dataset

import config

DATASET = "onyx-dot-app/EnterpriseRAG-Bench"


def as_list(value) -> list[str]:
    """Dataset list fields, whether they arrive as lists or as their string form."""
    if isinstance(value, str):
        value = ast.literal_eval(value) if value.startswith("[") else [value]
    return [str(item) for item in value or []]


def load_questions(client: chromadb.ClientAPI) -> set[str]:
    rows = load_dataset(DATASET, "questions")["test"]
    collection = client.get_or_create_collection(config.CHROMA_QUESTIONS_COLLECTION)
    present = set(collection.get(include=[])["ids"])
    ids, documents, metadatas, gold = [], [], [], set()
    for row in rows:
        expected = as_list(row["expected_doc_ids"])
        gold.update(expected)
        if row["question_id"] in present:
            continue
        ids.append(row["question_id"])
        documents.append(row["question"])
        metadatas.append({
            "question_type": row["question_type"],
            "source_types": ", ".join(as_list(row["source_types"])),
            # Separators match what scripts/benchmark.py parses.
            "expected_doc_ids": ", ".join(expected),
            "gold_answer": row["gold_answer"] or "",
            "answer_facts": " | ".join(as_list(row["answer_facts"])),
        })
    if ids:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"questions: {len(ids)} added, {len(present)} already present", flush=True)
    return gold


def load_documents(client: chromadb.ClientAPI, gold: set[str], sample: int, seed: int, batch: int) -> None:
    documents = load_dataset(DATASET, "documents")["test"]
    collection = client.get_or_create_collection(config.CHROMA_DOCS_COLLECTION)

    first_row: dict[str, int] = {}
    for index, doc_id in enumerate(documents["doc_id"]):
        first_row.setdefault(doc_id, index)  # the dataset repeats a few IDs; keep the first copy

    distractors = sorted(set(first_row) - gold)
    random.Random(seed).shuffle(distractors)
    wanted = sorted(gold & set(first_row)) + distractors[: max(0, sample - len(gold))]

    present: set[str] = set()
    offset = 0
    while True:
        page = collection.get(limit=20000, offset=offset, include=[])["ids"]
        present.update(page)
        offset += len(page)
        if len(page) < 20000:
            break
    todo = [doc_id for doc_id in wanted if doc_id not in present]
    print(f"documents: {len(wanted)} targeted ({len(gold)} gold), {len(wanted) - len(todo)} already present, "
          f"{len(todo)} to embed", flush=True)

    started = time.time()
    for start in range(0, len(todo), batch):
        rows = documents.select([first_row[doc_id] for doc_id in todo[start:start + batch]])
        collection.add(
            ids=list(rows["doc_id"]),
            documents=[content or "" for content in rows["content"]],
            metadatas=[{"title": title or "", "source_type": source or ""}
                       for title, source in zip(rows["title"], rows["source_type"])],
        )
        done = min(start + batch, len(todo))
        rate = done / max(time.time() - started, 1e-9)
        print(f"  {done}/{len(todo)} embedded, ~{(len(todo) - done) / rate / 60:.0f} min left", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=78053,
                        help="corpus size including every gold document (default matches the reference setup)")
    parser.add_argument("--seed", type=int, default=42, help="seed for sampling distractor documents")
    parser.add_argument("--batch", type=int, default=500)
    parser.add_argument("--questions-only", action="store_true")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    print(f"Chroma store: {config.CHROMA_PATH}", flush=True)
    gold = load_questions(client)
    if not args.questions_only:
        load_documents(client, gold, args.sample, args.seed, args.batch)
    print("done", flush=True)


if __name__ == "__main__":
    main()
