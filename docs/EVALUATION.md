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

Preliminary, first 100 of 500 questions:

| Model | Correct | Score [95% CI] |
|---|---|---|
| Qwen2.5-7B | 57% | 0.44 [0.35, 0.53] |
| Phi-3-mini | 51% | 0.41 [0.31, 0.50] |
| phi-2 | 40% | 0.31 [0.23, 0.40] |

Judge controls (25 questions): gold answers judged correct 100% (97% of facts found);
another question's answer judged correct 0% (9% of facts falsely credited). Qwen's lead
over phi-2 is significant (+0.13 [0.03, 0.23]); Qwen vs Phi-3 (+0.04) is not - at 25
questions it looked significant, which is why the run covers all 500.

## 4. Error attribution: retrieval or generation?

For every question, which gold facts actually reached the model's context (the
documents in the prompt, not the question) is checked; each wrong answer is then
traced back from the generator:

| Outcome | Meaning | What to fix |
|---|---|---|
| Generation error | ≥ 50% of gold facts were in context, yet the answer was wrong | prompt or model |
| Context loss | every gold document was retrieved, but its facts did not reach the prompt | chunking, context assembly |
| Retrieval miss | a gold document was not retrieved | retrieval, reranking |
| Missed abstention | an unanswerable question got an answer | abstention |

**The fact check is a cross-encoder, not the LLM judge.** The first version asked the
8B judge to list facts present in the context. Its negative control - the same facts
checked against a *different* question's context - exposed it: it found 79% of facts in
the right context and 72% in an unrelated one, i.e. no signal. Scoring each fact against
each document block with the cross-encoder (`ms-marco-MiniLM-L-6-v2`) separated the two
with AUC 0.884 on 142 facts; at a 0.1 threshold it credits 1-3% of facts in unrelated
contexts, and it costs ~3 s per question instead of ~160 s. It is conservative: facts
reworded in the documents can be missed, which shifts some generation errors toward
context loss.

Preliminary, first 100 questions:

| Model | Correct | Generation error | Context loss | Retrieval miss |
|---|---:|---:|---:|---:|
| Qwen2.5-7B | 57% | 22% | 13% | 8% |
| Phi-3-mini | 51% | 20% | 17% | 12% |
| phi-2 | 40% | 28% | 22% | 10% |

Failures split roughly evenly between generation and the stages before it (context loss
plus retrieval miss: 21-32%). Context loss - facts cut by the 4,500-character prompt
budget - stays a large share for every model, and weaker models add more generation
errors. (At 25 questions the split looked tilted toward pre-generation failures; it did
not hold up.)

### Lessons from building the judge

- A per-fact true/false JSON made the 8B judge count past the last fact and mark nearly
  every fact present; asking for the *numbers of facts present* fixed both.
- Repeating a long reference answer in every batch made calls take minutes; it is now
  sent only with the correctness call.
- Ollama's server keeps each prompt's attention cache in RAM (up to 8 GB). phi-3, which
  has no grouped-query attention, grew to 10.7 GB and was killed by the kernel, so
  candidates are unloaded after every call.
