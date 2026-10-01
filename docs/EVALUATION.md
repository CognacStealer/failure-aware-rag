# Evaluation

All numbers are on the **held-out test split** unless marked otherwise. The latest saved
results are always listed in [`results/README.md`](../results/README.md).

## 1. Retrieval

Metric: **recall@10** = share of a question's gold documents in the top 10 (236
answerable test questions). Script: `scripts/evaluate_retrieval.py`.

| Strategy | Before fixes | After fixes |
|---|---|---|
| vanilla (dense) | 0.297 | 0.388 |
| BM25 only | 0.646 | 0.662 |
| hybrid (RRF) | 0.646 - identical to BM25 | **0.716** |
| CRAG (rerank) | 0.522 - worse than hybrid | **0.730** |
| adaptive | abstained on 100% of queries | **0.726** |

"Before" was measured on the original 148 answerable questions whose gold documents
were in the corpus, so the comparison is indicative rather than exact.

## 2. Calibrator

Label: 1 when hybrid retrieval returned every gold document in its top 10, else 0
(unanswerable questions are 0). Scripts: `scripts/train_calibrator.py`,
`scripts/evaluate_calibrator.py`.

| Metric | Test |
|---|---|
| AUC (ranking quality, 0.5 = chance) | 0.725 |
| Brier score (lower is better) | 0.202 vs 0.232 for a constant guess (13% lower) |

The evaluation also records, per question, recall and wall-clock time of both paths, so
the Monitor can show the **cost-vs-quality frontier** over every Fast threshold and a
**reliability diagram**. Joined with judged answers from the ablation it shows
**selective answering**: whether declining the least-confident questions raises
answer accuracy compared with declining at random.

*Finding:* no mean-confidence cutoff justified abstaining - even the lowest-confidence
questions often had their evidence in the reranker's candidate pool - so abstention is
left to high model uncertainty and to the generator's "say you don't know" instruction.
The retrieval signals cannot reliably tell unanswerable questions apart.

## 3. Generator ablation

Question: with retrieval held fixed, how much does the generator matter?

- **Candidates:** Qwen2.5-7B-Instruct, Phi-3-mini-4k-instruct, phi-2 (all 4-bit via Ollama).
- **Context:** identical for every model - CRAG retrieval and the API's prompt, sized to
  fit phi-2's 2,048-token window (4,500 characters); answers up to 200 tokens.
- **Judge:** `llama3.1:8b-instruct-q4_K_M`, deliberately **not** one of the candidates
  (an earlier version used Qwen to judge Qwen). One call grades correctness against the
  gold answer and lists which gold facts the answer states; long fact lists are graded
  in batches of 6.
- **Score:** correctness × share of gold facts stated (0 for incorrect answers).
- **Statistics:** 95% bootstrap intervals per model; paired bootstrap on per-question
  score differences to decide whether a gap is real.
- **Judge controls** on 25 questions: each question's own gold answer (should pass) and
  a different question's gold answer (should fail).

Preliminary, first 25 of 500 questions:

| Model | Correct | Score [95% CI] |
|---|---|---|
| Qwen2.5-7B | 56% | 0.47 [0.29, 0.65] |
| Phi-3-mini | 44% | 0.32 [0.15, 0.51] |
| phi-2 | 32% | 0.22 [0.09, 0.37] |

Judge controls: gold answers judged correct 100% (97% of facts found); another
question's answer judged correct 0% (9% of facts falsely credited). Qwen's lead over
both others is already significant; Phi-3 vs phi-2 is not yet.

### Lessons from building the judge

- A per-fact true/false JSON made the 8B judge count past the last fact and mark nearly
  every fact present; asking for the *numbers of facts present* fixed both.
- Repeating a long reference answer in every batch made calls take minutes; it is now
  sent only with the correctness call.
- Ollama's server keeps each prompt's attention cache in RAM (up to 8 GB). phi-3, which
  has no grouped-query attention, grew to 10.7 GB and was killed by the kernel, so
  candidates are unloaded after every call.
