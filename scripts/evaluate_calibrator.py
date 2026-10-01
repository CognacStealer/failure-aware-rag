"""Measure what the retrieval calibrator buys: calibration quality and cost-vs-quality of routing.

For every benchmark question this records the calibrator's prediction, the
route it picks, and recall plus wall-clock time of both retrieval paths
(hybrid = Fast track, CRAG = Corrective track). Results are written per
question so the dashboard can also join them with judged generator answers
(selective answering). Headline metrics are computed on the held-out test split.

Crash-safe: every question is appended to a progress file as soon as it is
measured, and a re-run resumes from it (unless the calibrator was refit in
between, which makes the earlier rows stale). The progress file is removed once
the result is saved to results/. Use --fresh to discard it and start over.

    PYTHONPATH=.:scripts python scripts/evaluate_calibrator.py
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

import config
from benchmark import load_questions, recall, retrieval_succeeded
from core.results_store import save_result
from core.service import RAGService

PROGRESS_PATH = Path("data/calibrator_eval_progress.jsonl")


def load_progress(path: Path, calibrator_refit: str | None) -> dict[str, dict]:
    """Rows measured by an interrupted run, if they came from the same fitted calibrator."""
    if not path.exists():
        return {}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines or json.loads(lines[0]).get("calibrator_last_refit") != calibrator_refit:
        print("progress file was made with a different calibrator fit; starting over", flush=True)
        path.unlink()
        return {}
    rows = {}
    for line in lines[1:]:
        try:
            row = json.loads(line)
        except ValueError:  # a line cut off by the interruption
            continue
        rows[row["question_id"]] = row
    return rows


def reliability_bins(probabilities: np.ndarray, labels: np.ndarray, bins: int = 5) -> list[dict]:
    """Equal-count bins (n is small, so equal-width bins would be mostly empty)."""
    order = np.argsort(probabilities)
    return [
        {
            "predicted": float(probabilities[chunk].mean()),
            "observed": float(labels[chunk].mean()),
            "n": int(len(chunk)),
        }
        for chunk in np.array_split(order, bins) if len(chunk)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top-k", type=int, default=config.TOP_K_DEFAULT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress", type=Path, default=PROGRESS_PATH)
    parser.add_argument("--fresh", action="store_true", help="discard progress from an interrupted run")
    args = parser.parse_args()

    service = RAGService()
    service.initialize()
    if not service.calibrator.is_fitted:
        raise SystemExit("calibrator is not fitted; run scripts/train_calibrator.py first")
    questions = load_questions(service.client, set(service.sparse.doc_ids))[: args.limit]

    if args.fresh and args.progress.exists():
        args.progress.unlink()
    done = load_progress(args.progress, service.calibrator.last_refit)
    if done:
        print(f"resuming: {len(done)} questions already measured", flush=True)
    else:
        args.progress.parent.mkdir(parents=True, exist_ok=True)
        args.progress.write_text(json.dumps({"calibrator_last_refit": service.calibrator.last_refit}) + "\n")

    rows = []
    for n, question in enumerate(questions, 1):
        if question["question_id"] in done:
            rows.append(done[question["question_id"]])
            continue
        text = question["question"]
        started = time.perf_counter()
        hybrid = service.hybrid.retrieve(text, args.top_k)
        hybrid_seconds = time.perf_counter() - started
        started = time.perf_counter()
        crag = service.crag.retrieve_corrected(text, args.top_k)
        crag_seconds = time.perf_counter() - started

        signals = service.calibrator.extract_signals(text, hybrid)
        mean, std = service.calibrator.predict_distribution(signals)
        row = {
            "question_id": question["question_id"],
            "split": question["split"],
            "type": question["type"],
            "answerable": bool(question["gold"]),
            "p_success": mean,
            "p_std": std,
            "route": service.router.route(mean, std).track,
            "label": retrieval_succeeded(hybrid, question),
            "hybrid_recall": recall(hybrid, question["gold"]),
            "crag_recall": recall(crag, question["gold"]),
            "hybrid_seconds": round(hybrid_seconds, 3),
            "crag_seconds": round(crag_seconds, 3),
        }
        rows.append(row)
        with args.progress.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(row) + "\n")
        if n % 25 == 0:
            print(f"{n}/{len(questions)}", flush=True)

    test = [row for row in rows if row["split"] == "test"]
    probabilities = np.array([row["p_success"] for row in test])
    labels = np.array([row["label"] for row in test])
    answerable = [row for row in test if row["answerable"]]

    def routed_recall(row: dict, fast_threshold: float) -> tuple[float, float]:
        fast = row["p_success"] >= fast_threshold
        recall_value = row["hybrid_recall"] if fast else row["crag_recall"]
        # The adaptive pipeline retrieves the Corrective pool once, so the Corrective track costs
        # exactly what CRAG costs (crag_seconds already includes its hybrid retrieval).
        seconds = row["hybrid_seconds"] if fast else row["crag_seconds"]
        return recall_value, seconds

    sweep = []
    for threshold in np.round(np.arange(0.0, 1.0001, 0.05), 2):
        values = [routed_recall(row, threshold) for row in answerable]
        sweep.append({
            "fast_threshold": float(threshold),
            "fast_share": float(np.mean([row["p_success"] >= threshold for row in answerable])),
            "recall": float(np.mean([v for v, _ in values])),
            "seconds": float(np.mean([s for _, s in values])),
        })

    configured = config.ROUTER_FAST_MEAN_THRESHOLD
    adaptive = [routed_recall(row, configured) for row in answerable]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "calibrator": {"model_path": str(config.CALIBRATOR_MODEL_PATH), "examples": service.calibrator.example_count,
                       "last_refit": service.calibrator.last_refit},
        "router": {"abstain_mean": config.ROUTER_ABSTAIN_MEAN_THRESHOLD, "abstain_std": config.ROUTER_ABSTAIN_STD_THRESHOLD,
                   "fast_mean": configured},
        "test": {
            "n": len(test),
            "answerable": len(answerable),
            "success_rate": float(labels.mean()),
            "auc": float(roc_auc_score(labels, probabilities)) if len(set(labels.tolist())) == 2 else None,
            "brier": float(brier_score_loss(labels, probabilities)),
            "brier_constant": float(labels.mean() * (1 - labels.mean())),
            "reliability": reliability_bins(probabilities, labels),
            "routes": {track: sum(row["route"] == track for row in test) for track in ("Fast", "Corrective", "Abstain")},
            "strategies": {
                "hybrid": {"recall": float(np.mean([r["hybrid_recall"] for r in answerable])),
                           "seconds": float(np.mean([r["hybrid_seconds"] for r in answerable]))},
                "crag": {"recall": float(np.mean([r["crag_recall"] for r in answerable])),
                         "seconds": float(np.mean([r["crag_seconds"] for r in answerable]))},
                "adaptive": {"recall": float(np.mean([v for v, _ in adaptive])),
                             "seconds": float(np.mean([s for _, s in adaptive]))},
            },
            "threshold_sweep": sweep,
        },
        "questions": rows,
    }
    print(json.dumps({k: v for k, v in report["test"].items() if k not in {"threshold_sweep", "reliability"}}, indent=2))
    directory = save_calibration_report(report)
    args.progress.unlink(missing_ok=True)
    print(f"\nsaved to {directory}")


def save_calibration_report(report: dict) -> Path:
    test = report["test"]
    return save_result(
        "calibrator_eval",
        f"Calibrator routing and calibration, {test['n']} test questions",
        {key: value for key, value in report.items() if key != "questions"},
        headline={
            "auc": test["auc"],
            "adaptive_recall": round(test["strategies"]["adaptive"]["recall"], 3),
            "crag_recall": round(test["strategies"]["crag"]["recall"], 3),
            "time_saved": round(1 - test["strategies"]["adaptive"]["seconds"] / test["strategies"]["crag"]["seconds"], 3),
        },
        tables={"per_question": report["questions"]},
    )


if __name__ == "__main__":
    main()
