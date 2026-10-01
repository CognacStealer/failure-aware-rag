"""Fit the adaptive calibrator on the benchmark train split and report it on the test split.

Label: 1 when hybrid retrieval returned every gold document in its top-k, 0
otherwise (including unanswerable ``info_not_found`` questions). Router
thresholds are suggested from out-of-fold predictions on the train split:

* Fast:    the lowest mean whose train precision (label rate) is >= --fast-precision.
* Abstain: the highest mean below which at most --abstain-recoverable of the
           questions still had all gold documents in the wider candidate pool
           the Corrective track reranks (so abstaining rarely forfeits an answer).
"""

import argparse

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import config
from benchmark import load_questions, recall, retrieval_succeeded
from core.calibrator import RetrievalCalibrator
from core.results_store import save_result
from core.router import AdaptiveRouter
from core.service import RAGService


def collect(service: RAGService, questions: list[dict], top_k: int) -> list[dict]:
    pool = top_k * config.CRAG_RETRIEVAL_MULTIPLIER
    rows = []
    for question in questions:
        candidates = service.hybrid.retrieve(question["question"], pool)
        retrieved = candidates[:top_k]
        rows.append({
            **question,
            "signals": RetrievalCalibrator.extract_signals(question["question"], retrieved),
            "label": retrieval_succeeded(retrieved, question),
            "recoverable": recall(candidates, question["gold"]) == 1.0,
        })
    return rows


def out_of_fold(rows: list[dict], folds: int = 5) -> np.ndarray:
    labels = np.array([row["label"] for row in rows])
    predictions = np.zeros(len(rows))
    for fit_index, predict_index in StratifiedKFold(folds, shuffle=True, random_state=0).split(rows, labels):
        model = RetrievalCalibrator("unused")
        model.fit([rows[i] for i in fit_index])
        predictions[predict_index] = [model.predict_distribution(rows[i]["signals"])[0] for i in predict_index]
    return predictions


def suggest_thresholds(rows: list[dict], means: np.ndarray, fast_precision: float, abstain_recoverable: float) -> tuple[float, float]:
    labels = np.array([row["label"] for row in rows])
    recoverable = np.array([row["recoverable"] for row in rows])
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    fast = next(
        (t for t in grid if (means >= t).sum() >= 5 and labels[means >= t].mean() >= fast_precision),
        1.01,
    )
    abstain = 0.0
    for t in grid:
        below = means < t
        if t < fast and below.sum() >= 3 and recoverable[below].mean() <= abstain_recoverable:
            abstain = float(t)
    return abstain, float(fast)


def report(name: str, rows: list[dict], calibrator: RetrievalCalibrator, router: AdaptiveRouter) -> dict:
    labels = np.array([row["label"] for row in rows])
    predictions = [calibrator.predict_distribution(row["signals"]) for row in rows]
    means = np.array([mean for mean, _ in predictions])
    metrics = {"n": len(rows), "success_rate": float(labels.mean()), "auc": None, "brier": None,
               "brier_constant": float(labels.mean() * (1 - labels.mean())), "routes": {}}
    print(f"\n[{name}] n={len(rows)}  success rate={labels.mean():.3f}")
    if len(set(labels)) == 2:
        metrics["auc"] = float(roc_auc_score(labels, means))
        metrics["brier"] = float(brier_score_loss(labels, means))
        print(f"  AUC={metrics['auc']:.3f}  Brier={metrics['brier']:.3f} (base-rate Brier {metrics['brier_constant']:.3f})")
    tracks = [router.route(mean, std).track for mean, std in predictions]
    for track in ("Fast", "Corrective", "Abstain"):
        chosen = [row for row, t in zip(rows, tracks) if t == track]
        if not chosen:
            print(f"  {track:<10} 0")
            metrics["routes"][track] = {"count": 0}
            continue
        success = float(np.mean([row["label"] for row in chosen]))
        recoverable = float(np.mean([row["recoverable"] for row in chosen]))
        unanswerable = sum(1 for row in chosen if not row["gold"])
        metrics["routes"][track] = {"count": len(chosen), "retrieval_complete": success,
                                    "gold_in_pool": recoverable, "unanswerable": unanswerable}
        print(f"  {track:<10} {len(chosen):>3}  retrieval complete={success:.2f}  "
              f"gold in pool={recoverable:.2f}  unanswerable={unanswerable}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top-k", type=int, default=config.TOP_K_DEFAULT)
    parser.add_argument("--fast-precision", type=float, default=0.9)
    parser.add_argument("--abstain-recoverable", type=float, default=0.1)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    service = RAGService()
    service.initialize()
    service.sparse.fetch_content = None  # signals need scores only
    questions = load_questions(service.client, set(service.sparse.doc_ids))
    rows = collect(service, questions, args.top_k)
    train = [row for row in rows if row["split"] == "train"]
    test = [row for row in rows if row["split"] == "test"]

    abstain, fast = suggest_thresholds(train, out_of_fold(train), args.fast_precision, args.abstain_recoverable)
    calibrator = RetrievalCalibrator(config.CALIBRATOR_MODEL_PATH)
    calibrator.fit(train)
    calibrator.save()

    print({
        "train": len(train),
        "test": len(test),
        "top_k": args.top_k,
        "label": "all gold documents in hybrid top-k",
        "model_path": str(config.CALIBRATOR_MODEL_PATH),
        "last_refit": calibrator.last_refit,
    })
    print(f"\nSuggested: ROUTER_ABSTAIN_MEAN_THRESHOLD={abstain:.2f} ROUTER_FAST_MEAN_THRESHOLD={fast:.2f}")
    print(f"Configured: abstain<{config.ROUTER_ABSTAIN_MEAN_THRESHOLD} (or std>{config.ROUTER_ABSTAIN_STD_THRESHOLD}), "
          f"fast>={config.ROUTER_FAST_MEAN_THRESHOLD}")
    router = AdaptiveRouter(config)
    train_metrics = report("train (in-sample)", train, calibrator, router)
    test_metrics = report("test (held out)", test, calibrator, router)

    directory = save_result(
        "calibrator_training",
        f"Calibrator fit on {len(train)} train questions",
        {
            "label": "all gold documents in hybrid top-k",
            "top_k": args.top_k,
            "features": list(RetrievalCalibrator.FEATURE_NAMES),
            "ensemble_size": calibrator.n_models,
            "suggested_thresholds": {"abstain_mean": abstain, "fast_mean": fast,
                                     "fast_precision_target": args.fast_precision,
                                     "abstain_recoverable_target": args.abstain_recoverable},
            "train": train_metrics,
            "test": test_metrics,
        },
        headline={"test_auc": test_metrics["auc"], "test_brier": test_metrics["brier"],
                  "brier_constant": test_metrics["brier_constant"]},
        tables={"examples": [
            {"question_id": row["question_id"], "split": row["split"], "type": row["type"], "label": row["label"],
             "recoverable": row["recoverable"], **row["signals"],
             "p_success": calibrator.predict_distribution(row["signals"])[0]}
            for row in rows
        ]},
        files={"calibrator.pkl": config.CALIBRATOR_MODEL_PATH},
    )
    print(f"\nsaved to {directory}")


if __name__ == "__main__":
    main()
