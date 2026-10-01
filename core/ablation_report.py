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


# A question's evidence counts as "in context" when the judge finds at least this share
# of its gold facts in the documents the model was given.
EVIDENCE_THRESHOLD = 0.5
OUTCOMES = ("correct", "generation_error", "context_loss", "retrieval_miss", "missed_abstention", "ungraded")


def evidence_share(facts_in_context: list[bool | None] | None) -> float | None:
    graded = [fact for fact in (facts_in_context or []) if fact is not None]
    return sum(graded) / len(graded) if graded else None


def attribute(correct: bool, category: str, recall: float | None, facts_in_context: list[bool | None] | None) -> str:
    """Where an answer failed, walking the pipeline backwards from the generator.

    generation_error  the evidence was in the model's context, yet the answer was wrong
    context_loss      every gold document was retrieved, but its facts did not reach the prompt
    retrieval_miss    a gold document was not retrieved and the evidence was not in context
    missed_abstention an unanswerable question got an answer instead of "not found"
    """
    if category == "info_not_found":
        return "correct" if correct else "missed_abstention"
    if correct:
        return "correct"
    share = evidence_share(facts_in_context)
    if share is None:
        return "ungraded"
    if share >= EVIDENCE_THRESHOLD:
        return "generation_error"
    if recall is not None and recall >= 1.0:
        return "context_loss"
    return "retrieval_miss"


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

    summary["attribution"] = compute_attribution(run_dir, common, categories, judged, retrieval)

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


def compute_attribution(
    run_dir: Path,
    common: list[str],
    categories: dict[str, str],
    judged: dict[str, dict[str, dict]],
    retrieval: dict[str, dict],
) -> dict | None:
    """Split every model's outcomes into correct / generation / context / retrieval failures."""
    context = read_jsonl(run_dir / "context_facts.jsonl")
    questions = [q for q in common if q in context]
    if not questions:
        return None
    shares = [evidence_share(context[q]["facts_in_context"]) for q in questions
              if categories.get(q) != "info_not_found"]
    shares = [s for s in shares if s is not None]
    result = {
        "evidence_threshold": EVIDENCE_THRESHOLD,
        "questions": len(questions),
        "context_fact_recall": float(np.mean(shares)) if shares else None,
        "evidence_in_context_rate": float(np.mean([s >= EVIDENCE_THRESHOLD for s in shares])) if shares else None,
        "models": {},
    }
    for model, verdicts in judged.items():
        counts = Counter(
            attribute(verdicts[q]["correct"], categories.get(q, ""), retrieval.get(q, {}).get("recall_at_k"),
                      context[q]["facts_in_context"])
            for q in questions
        )
        without_evidence = sum(
            1 for q in questions
            if verdicts[q]["correct"] and categories.get(q) != "info_not_found"
            and (evidence_share(context[q]["facts_in_context"]) or 0.0) < EVIDENCE_THRESHOLD
        )
        result["models"][model] = {
            "outcomes": {outcome: counts.get(outcome, 0) for outcome in OUTCOMES},
            "correct_without_evidence": without_evidence,
        }
    controls = list(read_jsonl(run_dir / "context_controls.jsonl").values())
    if controls:
        negatives = [evidence_share(c["facts_in_context"]) for c in controls]
        negatives = [n for n in negatives if n is not None]
        result["context_judge_false_positive_rate"] = float(np.mean(negatives)) if negatives else None
        result["context_controls"] = len(controls)
    return result
