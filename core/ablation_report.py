"""Read generator-ablation run directories and aggregate their results.

Shared by scripts/run_local_generator_ablation.py (which writes the runs) and the
live dashboard API (which reads them while a run is still in progress).
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:  # a line still being written by the run
            continue
        rows[row["question_id"]] = row
    return rows


def slug(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1].replace(".", "_")


def completeness(facts: list[bool | None]) -> float:
    graded = [fact for fact in facts if fact is not None]
    return float(np.mean(graded)) if graded else 0.0


def interval(values: list[float], rng: np.random.Generator) -> tuple[float, float]:
    samples = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def read_manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def compute_summary(run_dir: Path, categories: dict[str, str]) -> dict:
    """Metrics over questions judged for every model, so models are compared on the same set."""
    manifest = read_manifest(run_dir)
    models = list(manifest["candidates"])
    retrieval = read_jsonl(run_dir / "retrieval.jsonl")
    judged = {model: read_jsonl(run_dir / f"{slug(model)}_judged.jsonl") for model in models}
    answers = {model: read_jsonl(run_dir / f"{slug(model)}.jsonl") for model in models}
    common = sorted(set.intersection(*(set(rows) for rows in judged.values()))) if models else []
    rng = np.random.default_rng(0)

    recalls = [retrieval[q]["recall_at_k"] for q in common if q in retrieval and retrieval[q]["recall_at_k"] is not None]
    summary = {
        "questions_judged_by_all_models": len(common),
        "judge_model": manifest["judge_model"],
        "retrieval_recall_at_k": float(np.mean(recalls)) if recalls else None,
        "questions_by_category": dict(sorted(Counter(categories.get(q, "unknown") for q in common).items())),
        "models": {},
    }
    scores = {}
    for model in models:
        rows = [judged[model][q] for q in common]
        if not rows:
            continue
        correct = [float(r["correct"]) for r in rows]
        complete = [completeness(r["facts_present"]) for r in rows]
        scores[model] = [c * f for c, f in zip(correct, complete)]
        by_category = defaultdict(list)
        for q, r in zip(common, rows):
            by_category[categories.get(q, "unknown")].append(float(r["correct"]))
        model_answers = [answers[model][q] for q in common if q in answers[model]]
        summary["models"][model] = {
            "correctness": float(np.mean(correct)),
            "correctness_95ci": interval(correct, rng),
            "completeness": float(np.mean(complete)),
            "leaderboard_score": float(np.mean(scores[model])),
            "leaderboard_95ci": interval(scores[model], rng),
            "mean_generation_seconds": float(np.mean([a["seconds"] for a in model_answers])),
            "context_truncated": int(sum(a.get("context_truncated", False) for a in model_answers)),
            "answers_cut_at_token_limit": int(sum(a.get("hit_token_limit", False) for a in model_answers)),
            "judge_parse_failures": int(sum(not r["judge_parse_ok"] for r in rows)),
            "facts_ungraded_by_judge": int(sum(r.get("facts_ungraded", 0) for r in rows)),
            "correctness_by_category": {c: round(float(np.mean(v)), 3) for c, v in sorted(by_category.items())},
        }

    # Paired comparison: bootstrap the per-question leaderboard-score difference.
    scored = list(scores)
    summary["paired_leaderboard_differences"] = {}
    for i, first in enumerate(scored):
        for second in scored[i + 1:]:
            diff = [a - b for a, b in zip(scores[first], scores[second])]
            low, high = interval(diff, rng)
            summary["paired_leaderboard_differences"][f"{slug(first)} - {slug(second)}"] = {
                "first": first, "second": second,
                "mean_difference": float(np.mean(diff)), "95ci": (low, high), "significant": low > 0 or high < 0,
            }

    sanity = list(read_jsonl(run_dir / "judge_sanity.jsonl").values())
    if sanity:
        negatives = [s["negative_control"] for s in sanity if "negative_control" in s]
        summary["judge_sanity"] = {
            "n": len(sanity),
            "gold_answer_marked_correct": float(np.mean([s["correct"] for s in sanity])),
            "gold_answer_facts_present": float(np.mean([completeness(s["facts_present"]) for s in sanity])),
            # Another question's gold answer: ideally 0 and 0. Nonzero values are the judge's
            # false-positive rates; completeness is inflated by roughly the second one.
            "wrong_answer_marked_correct": float(np.mean([n["correct"] for n in negatives])) if negatives else None,
            "wrong_answer_facts_present": float(np.mean([completeness(n["facts_present"]) for n in negatives])) if negatives else None,
        }
    return summary
