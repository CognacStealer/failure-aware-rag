"""Measure retrieval quality of every strategy on the benchmark's held-out test split.

Reports document recall@k against gold IDs for vanilla (dense), BM25, hybrid
and CRAG, and for adaptive: how questions were routed and the recall of the
documents each track actually used. No answers are generated.
"""

import argparse
import time
from collections import defaultdict

import numpy as np

import config
from benchmark import load_questions, ndcg, recall, reciprocal_rank
from core.results_store import save_result
from core.service import RAGService


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=config.TOP_K_DEFAULT)
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--limit", type=int, help="evaluate only the first N questions")
    args = parser.parse_args()

    service = RAGService()
    service.initialize()
    if service.error:
        print("warning:", service.error)
    questions = [
        q for q in load_questions(service.client, set(service.sparse.doc_ids))
        if args.split == "all" or q["split"] == args.split
    ][: args.limit]

    k = args.top_k
    results = defaultdict(list)
    tracks = defaultdict(list)
    seconds = defaultdict(float)
    latencies = defaultdict(list)
    ranks = defaultdict(lambda: defaultdict(list))  # strategy -> metric -> values
    per_question = []
    for question in questions:
        text, gold = question["question"], question["gold"]
        timed = {}
        for name, fetch in (
            ("vanilla", lambda: service.dense.retrieve(text, k)),
            ("bm25", lambda: service.sparse.retrieve(text, k)),
            ("hybrid", lambda: service.hybrid.retrieve(text, k)),
            ("crag", lambda: service.crag.retrieve_corrected(text, k)),
        ):
            started = time.perf_counter()
            timed[name] = fetch()
            elapsed = time.perf_counter() - started
            seconds[name] += elapsed
            latencies[name].append(elapsed)
            if gold:
                results[name].append(recall(timed[name], gold))
                ranks[name]["mrr"].append(reciprocal_rank(timed[name], gold))
                ranks[name]["ndcg"].append(ndcg(timed[name], gold, k))

        signals = service.calibrator.extract_signals(text, timed["hybrid"])
        decision = service.router.route(*service.calibrator.predict_distribution(signals))
        used = {"Fast": timed["hybrid"], "Corrective": timed["crag"], "Abstain": []}[decision.track]
        tracks[decision.track].append(question)
        # Adaptive always runs hybrid; the Corrective track adds the rerank on top of it.
        latencies["adaptive"].append(latencies["hybrid"][-1] + (latencies["crag"][-1] if decision.track == "Corrective" else 0.0))
        if gold:
            results["adaptive"].append(recall(used, gold))
            ranks["adaptive"]["mrr"].append(reciprocal_rank(used, gold))
            ranks["adaptive"]["ndcg"].append(ndcg(used, gold, k))
        per_question.append({
            "question_id": question["question_id"],
            "type": question["type"],
            "answerable": bool(gold),
            "track": decision.track,
            "confidence_mean": round(decision.mean, 4),
            "confidence_std": round(decision.std, 4),
            **{f"{name}_recall": recall(docs, gold) for name, docs in timed.items()},
            "adaptive_recall": recall(used, gold),
            **{f"{name}_mrr": reciprocal_rank(docs, gold) for name, docs in {**timed, "adaptive": used}.items()},
            **{f"{name}_seconds": round(latencies[name][-1], 3) for name in (*timed, "adaptive")},
        })

    answerable = sum(1 for q in questions if q["gold"])
    print(f"split={args.split}  questions={len(questions)}  answerable={answerable}  top_k={k}\n")
    print(f"{'strategy':<10} {'recall@k':>9} {'MRR':>6} {'nDCG':>6} {'hit':>6} {'complete':>9} {'p50 s':>7} {'p95 s':>7}")
    for name in ("vanilla", "bm25", "hybrid", "crag", "adaptive"):
        values = np.array(results[name])
        p50, p95 = np.percentile(latencies[name], [50, 95])
        print(f"{name:<10} {values.mean():>9.3f} {np.mean(ranks[name]['mrr']):>6.3f} {np.mean(ranks[name]['ndcg']):>6.3f} "
              f"{np.mean(values > 0):>6.3f} {np.mean(values == 1):>9.3f} {p50:>7.2f} {p95:>7.2f}")

    print("\nadaptive routing (abstaining counts as recall 0 above):")
    for track in ("Fast", "Corrective", "Abstain"):
        routed = tracks[track]
        unanswerable = sum(1 for q in routed if not q["gold"])
        print(f"  {track:<10} {len(routed):>4}  unanswerable={unanswerable}")

    strategies = {}
    for name in ("vanilla", "bm25", "hybrid", "crag", "adaptive"):
        values = np.array(results[name])
        p50, p95 = np.percentile(latencies[name], [50, 95])
        strategies[name] = {
            "recall_at_k": float(values.mean()),
            "mrr": float(np.mean(ranks[name]["mrr"])),
            "ndcg_at_k": float(np.mean(ranks[name]["ndcg"])),
            "hit_rate": float(np.mean(values > 0)),
            "complete_rate": float(np.mean(values == 1)),
            "seconds_mean": float(np.mean(latencies[name])),
            "seconds_p50": float(p50),
            "seconds_p95": float(p95),
        }
    directory = save_result(
        "retrieval_eval",
        f"Retrieval strategies, {args.split} split, top-{k}",
        {
            "split": args.split,
            "questions": len(questions),
            "answerable": answerable,
            "top_k": k,
            "strategies": strategies,
            "routing": {
                track: {"count": len(tracks[track]), "unanswerable": sum(1 for q in tracks[track] if not q["gold"])}
                for track in ("Fast", "Corrective", "Abstain")
            },
        },
        headline={name: round(strategies[name]["recall_at_k"], 3) for name in ("hybrid", "crag", "adaptive")},
        tables={"per_question": per_question},
        label=args.split,
    )
    print(f"\nsaved to {directory}")


if __name__ == "__main__":
    main()
