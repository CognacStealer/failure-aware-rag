# Dataset

## What we use

**[EnterpriseRAG-Bench](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench)** (Onyx), two configs:

| Config | Rows | Fields |
|---|---|---|
| `documents` | 511,962 | `doc_id`, `source_type`, `title`, `content` |
| `questions` | 500 | `question_id`, `question_type`, `question`, `expected_doc_ids`, `gold_answer`, `answer_facts` |

Documents imitate a company's internal knowledge: Slack threads, Gmail, Linear and Jira
tickets, Google Drive docs, HubSpot, Fireflies meeting notes, GitHub and Confluence pages.
In our sample a document averages ~7,600 characters.

Each question has gold document IDs (which documents contain the evidence), a gold
answer, and `answer_facts`: the individual facts a complete answer must state. Facts let
us score **completeness** separately from **correctness**.

Question types (500 total):

| Type | n | What it tests |
|---|---|---|
| basic | 175 | single fact from one document |
| semantic | 125 | paraphrased question, little word overlap with the source |
| intra_document_reasoning | 40 | combining facts within one document |
| project_related | 40 | facts tied to a named project |
| constrained | 30 | answer under explicit constraints |
| conflicting_info | 20 | sources disagree |
| completeness | 20 | many facts must all be listed |
| miscellaneous | 20 | other |
| info_not_found | 20 | **unanswerable**: the right answer is to say so |
| high_level | 10 | broad summary questions (no gold documents) |

## Why this dataset

- **Realistic enterprise retrieval.** Long, noisy, multi-source documents with near
  duplicates are what a company RAG system actually faces, unlike Wikipedia-style QA.
- **Gold documents per question** make retrieval measurable (recall@k), which the
  calibrator needs as its training label.
- **Answer facts** make answer completeness measurable without a human grader per answer.
- **Unanswerable questions** (`info_not_found`) test failure awareness, which is the point
  of this project: knowing when retrieval failed and when to abstain.

## How we load it

`./setup.sh` (or `scripts/load_dataset.py` directly) downloads the dataset and builds both
collections described below. `--sample 78053` gives a corpus of the reference size and makeup
(every gold document plus seeded random distractors); the reference store's distractors were
sampled differently, so numbers from a fresh build can differ slightly.

- Documents live in a persistent **ChromaDB** collection `docs`, embedded with Chroma's
  default model (`all-MiniLM-L6-v2`, 384-d). Questions live in the `Questions` collection,
  with `expected_doc_ids`, `gold_answer` and `answer_facts` stored as metadata.
- The full corpus (512k documents, ~2.5B characters) is far too large to embed on a
  CPU-only laptop, so `docs` holds a **77,668-document sample**. That sample contained
  only 337 of the 722 gold documents, so `scripts/add_gold_docs.py` appends the
  **385 missing gold documents** from the cached Hugging Face dataset, giving **78,053
  documents**. Every distractor is still a real benchmark document.
- One document ID appears twice in the source data; the first copy is kept.

## Splits

`scripts/benchmark.py` assigns each question to **train** or **test** by a fixed SHA-1 hash
of its ID (about 50/50). Everything that is tuned (RRF weights, CRAG settings, calibrator,
router thresholds) is tuned on **train**; every reported number comes from **test**.

| Split | Questions | Answerable (gold docs in corpus) | Unanswerable |
|---|---|---|---|
| train | 245 | 234 | 11 |
| test | 245 | 236 | 9 |

10 `high_level` questions have no gold documents and are left out of retrieval metrics;
the generator ablation still answers and grades them.

## Known limitations

- A 15% sample of the distractor pool makes retrieval easier than on the full corpus.
- Embeddings cover only the first ~256 tokens of each document (MiniLM's input limit);
  chunked embeddings are on the roadmap ([ROADMAP.md](ROADMAP.md)).
- Answers are graded by a local LLM judge; see the judge-reliability controls in
  [EVALUATION.md](EVALUATION.md).
