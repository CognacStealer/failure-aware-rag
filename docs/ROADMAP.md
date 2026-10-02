# Roadmap

## Under construction

| Item | Status |
|---|---|
| Generator ablation on all 500 questions | **Running** (crash-safe, ~2-3 days on CPU) with per-question error attribution |
| Context assembly | Next: context loss causes 13-22% of wrong answers; give the prompt the passages that contain the facts rather than an equal budget per document |
| Calibrator evaluation (cost/quality frontier, reliability, selective answering) | Done; selective answering fills in as the ablation grades more held-out questions |
| `core/retriever_faiss.py` | Empty placeholder from the initial scaffold; not used |
| Chunked dense index | Deferred: embedding 78k documents in chunks is a ~7 h CPU job. Today the cross-encoder grades the best passages at query time instead |
| Authentication / deployment | Not started: the website is meant for local use only |

## Next: reinforcement learning for routing

Each question is a single decision followed by a delayed score, so the right framing is a
**contextual bandit**:

- **State:** the calibrator's six retrieval signals.
- **Actions:** Fast, Corrective, Abstain (later: which generator, how many documents).
- **Reward:** judged score minus a latency cost; abstaining earns a small fixed reward,
  larger on genuinely unanswerable questions.
- **Algorithm:** Thompson sampling. The calibrator is already a bootstrap ensemble that
  approximates a posterior, so sampling one member per query explores where it is
  uncertain and exploits where it is confident.
- **Data first:** rewards for every action on the same questions. The ablation answers
  only from reranked context, so one extra pass on fast-path context is needed; then
  compare policies offline before deploying.
- **Online:** 👍/👎 on the Ask page as live reward, with nightly policy updates.
- **Later, on a GPU:** DPO on judged answer pairs (three judged answers per question
  already form preference data) and GRPO with a judge reward for "answer only from
  context; say when it is missing".
- **Main risk:** optimizing for the judge instead of the user (e.g. verbose answers that
  collect fact credit). Guard with the judge controls, a length penalty, and human audits.

## Research directions

| Area | Ideas |
|---|---|
| Uncertainty | Conformal prediction for answering with a guaranteed error rate; answer-level confidence (self-consistency, semantic entropy, token log-probabilities) vs retrieval-only signals |
| Evaluation | Second judge for agreement; 50-item human audit (Cohen's κ); verbosity and position bias tests |
| Learning | Bandit routing; active learning for the calibrator; DPO from judged pairs |
| Faithfulness | Citation accuracy; claim-level groundedness checks; abstention F1 on `info_not_found` |
| Retrieval | Query rewriting (HyDE, multi-query); stronger embedders (e.g. Qwen3-Embedding-0.6B); stronger rerankers; RAG vs long-context |
| Generation | Prompt ablations; context ordering ("lost in the middle"); context compression; q4 vs q8 |
| Multi-step | Iterative retrieval (Self-RAG, ReAct); decomposition for completeness and reasoning questions |
| Systems | Semantic caching; latency and energy per query on CPU |
