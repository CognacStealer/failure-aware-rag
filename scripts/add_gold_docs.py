"""Add benchmark gold documents that are missing from the Chroma docs collection.

The docs collection holds a 77,668-document sample of EnterpriseRAG-Bench's
511,962 documents, which left most questions' gold documents out of the corpus.
This appends only the missing gold documents, so every question with gold IDs
can be scored against the same distractor pool. Safe to re-run.
"""

import argparse

import chromadb
from datasets import load_dataset

import config

DATASET = "onyx-dot-app/EnterpriseRAG-Bench"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    questions = load_dataset(DATASET, "questions")["test"]
    gold_ids = {doc_id for row in questions for doc_id in row["expected_doc_ids"]}

    collection = chromadb.PersistentClient(path=config.CHROMA_PATH).get_collection(
        config.CHROMA_DOCS_COLLECTION
    )
    present = set(collection.get(ids=sorted(gold_ids), include=[])["ids"])
    missing = gold_ids - present
    print({"gold_ids": len(gold_ids), "present": len(present), "missing": len(missing)})
    if not missing or args.dry_run:
        return

    documents = load_dataset(DATASET, "documents")["test"]
    rows = documents.filter(lambda row: row["doc_id"] in missing, num_proc=4)
    # The dataset repeats a few doc IDs; keep the first copy of each.
    unique = {}
    for doc_id, content, title in zip(rows["doc_id"], rows["content"], rows["title"]):
        unique.setdefault(doc_id, (content, title))
    collection.add(
        ids=list(unique),
        documents=[content for content, _ in unique.values()],
        metadatas=[{"title": title} for _, title in unique.values()],
    )
    print({"added": len(unique), "collection_count": collection.count()})


if __name__ == "__main__":
    main()
