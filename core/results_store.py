"""One place for every experiment result, with enough provenance to reproduce it.

Layout::

    results/
      index.json                 registry of every saved result (newest first)
      README.md                  the same registry as a readable table (regenerated)
      <kind>/<id>/
        result.json              metrics and any structured output
        <table>.csv              per-question tables, for spreadsheets
        meta.json                git commit, config snapshot, command, python
        <extra files>            e.g. the model file a result was produced with

Results are never overwritten: each save gets a new timestamped id.
"""

import csv
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config

KINDS = {
    "retrieval_eval": "Retrieval evaluation",
    "calibrator_training": "Calibrator training",
    "calibrator_eval": "Calibrator evaluation",
    "generator_ablation": "Generator ablation",
}


REPO = Path(__file__).resolve().parent.parent


def _root() -> Path:
    return config.RESULTS_DIR


def portable(value: Any) -> Any:
    """Paths relative to the repository (or ~), so saved results carry no machine-specific paths."""
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        value = value.replace(f"{REPO}/", "").replace(str(REPO), ".")
        return value.replace(str(Path.home()), "~")
    return value


def machine() -> dict[str, Any]:
    """Hardware that produced a result: useful to readers, unlike a host name."""
    info: dict[str, Any] = {"os": platform.system(), "python": platform.python_version()}
    try:
        cpu = next(line for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
        info["cpu"] = " ".join(cpu.split(":", 1)[1].replace("(R)", "").replace("(TM)", "").split())
        kb = next(line for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemTotal"))
        info["ram_gb"] = round(int(kb.split()[1]) / 1024**2)
    except (OSError, StopIteration, ValueError):
        info["cpu"] = platform.processor() or None
    return info


def _git() -> dict[str, Any]:
    repo = Path(__file__).resolve().parent.parent

    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=10).stdout.strip()

    try:
        return {"commit": run("rev-parse", "HEAD") or None, "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def config_snapshot() -> dict[str, Any]:
    return {
        name: (str(value) if isinstance(value, Path) else value)
        for name, value in vars(config).items()
        if name.isupper() and isinstance(value, (str, int, float, bool, Path))
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):  # numpy scalars
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        columns += [key for key in row if key not in columns]
    with path.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, default=_json_default) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })


def save_result(
    kind: str,
    title: str,
    result: dict[str, Any],
    *,
    headline: dict[str, Any],
    tables: dict[str, list[dict[str, Any]]] | None = None,
    files: dict[str, Path] | None = None,
    label: str | None = None,
) -> Path:
    """Persist one result and register it; returns its directory."""
    if kind not in KINDS:
        raise ValueError(f"unknown result kind '{kind}'")
    created = datetime.now(timezone.utc)
    result_id = created.strftime("%Y%m%dT%H%M%SZ") + (f"_{label}" if label else "")
    directory = _root() / kind / result_id
    directory.mkdir(parents=True, exist_ok=False)

    write_json(directory / "result.json", portable(result))
    for name, rows in (tables or {}).items():
        write_csv(directory / f"{name}.csv", rows)
    for name, source in (files or {}).items():
        shutil.copy2(source, directory / name)
    meta = {
        "id": result_id,
        "kind": kind,
        "title": title,
        "created_at": created.isoformat(),
        "headline": headline,
        "git": _git(),
        "command": portable(sys.argv),
        "machine": machine(),
        "config": portable(config_snapshot()),
    }
    write_json(directory / "meta.json", meta)

    index = list_results()
    index.insert(0, {
        "id": result_id,
        "kind": kind,
        "title": title,
        "created_at": meta["created_at"],
        "headline": headline,
        "path": str(directory.relative_to(_root())),
        "files": sorted(p.name for p in directory.iterdir()),
        "git_commit": meta["git"]["commit"],
        "git_dirty": meta["git"]["dirty"],
    })
    write_json(_root() / "index.json", index)
    _write_readme(index)
    return directory


def list_results() -> list[dict[str, Any]]:
    path = _root() / "index.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []


def latest(kind: str) -> tuple[dict[str, Any], Path] | None:
    for entry in list_results():
        if entry["kind"] == kind:
            return entry, _root() / entry["path"]
    return None


def _fmt(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def _load(entry: dict[str, Any]) -> dict[str, Any]:
    return json.loads((_root() / entry["path"] / "result.json").read_text(encoding="utf-8"))


def _link(entry: dict[str, Any], label: str = "folder") -> str:
    return f"[{label}]({entry['path']})"


def _ranked_table(rows: list[tuple[str, dict[str, Any]]]) -> list[str]:
    return [
        "| Strategy | Recall@10 | MRR | nDCG@10 | Found all gold docs | Latency p50 | p95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[f"| {name} | **{v['recall_at_k']:.3f}** | {v['mrr']:.3f} | {v['ndcg_at_k']:.3f} | {v['complete_rate']:.1%} "
          f"| {v['seconds_p50']:.2f} s | {v['seconds_p95']:.2f} s |" for name, v in rows],
        "",
        "MRR and nDCG reward putting the gold documents *first*, which is what the reranker buys; latencies were",
        "measured on a CPU-only laptop while other jobs were running, so compare them relative to each other.",
    ]


def _section_retrieval(entry: dict[str, Any]) -> list[str]:
    r = _load(entry)
    names = {"vanilla": "Vanilla (dense)", "bm25": "BM25", "hybrid": "Hybrid (RRF)", "crag": "CRAG (rerank)",
             "adaptive": "Adaptive (calibrated)"}
    rows = [(names.get(k, k), v) for k, v in r["strategies"].items()]
    recalls = [f"{v['recall_at_k']:.3f}" for _, v in rows]
    routing = r.get("routing", {})
    return [
        "## Retrieval",
        "",
        f"Recall@{r['top_k']} on {r['answerable']} answerable **held-out** questions: the share of each question's",
        "gold documents among the documents handed to the generator.",
        "",
        *(_ranked_table(rows) if all("mrr" in v for _, v in rows) else [
            "| Strategy | Recall@10 | Found ≥1 gold doc | Found all gold docs |",
            "|---|---:|---:|---:|",
            *[f"| {name} | **{v['recall_at_k']:.3f}** | {v['hit_rate']:.1%} | {v['complete_rate']:.1%} |" for name, v in rows],
        ]),
        "",
        "```mermaid",
        "xychart-beta",
        '    title "Recall@10, held-out questions"',
        f"    x-axis [{', '.join(n.split(' ')[0] for n, _ in rows)}]",
        '    y-axis "recall@10" 0 --> 1',
        f"    bar [{', '.join(recalls)}]",
        "```",
        "",
        f"Adaptive routing sent {routing.get('Fast', {}).get('count', 0)} questions down the fast path, "
        f"{routing.get('Corrective', {}).get('count', 0)} to cross-encoder correction and "
        f"{routing.get('Abstain', {}).get('count', 0)} to abstention. {_link(entry, 'Per-question data')}",
        "",
    ]


def _section_calibrator(training: dict[str, Any] | None, evaluation: dict[str, Any] | None) -> list[str]:
    lines = ["## Calibrator", "",
             "Predicts, before answering, whether retrieval found every gold document; the router uses it",
             "to answer fast, rerank first, or abstain.", ""]
    if training:
        t = _load(training)["test"]
        suggested = _load(training).get("suggested_thresholds", {})
        lines += [
            "| Held-out metric | Calibrator | Constant guess |",
            "|---|---:|---:|",
            f"| AUC (0.5 = chance) | **{t['auc']:.3f}** | 0.500 |",
            f"| Brier score (lower is better) | **{t['brier']:.3f}** | {t['brier_constant']:.3f} |",
            "",
            f"Suggested thresholds from out-of-fold training predictions: fast path if p ≥ "
            f"{suggested.get('fast_mean', float('nan')):.2f}; "
            + (f"abstain below {suggested['abstain_mean']:.2f}. " if suggested.get("abstain_mean")
               else "no confidence cutoff justified abstaining, so abstention is left to model uncertainty. ")
            + _link(training, "Training run and model file"),
            "",
        ]
    if evaluation:
        e = _load(evaluation)["test"]
        st = e["strategies"]
        saved = 1 - st["adaptive"]["seconds"] / st["crag"]["seconds"]
        lines += [
            "| Policy | Recall@10 | Retrieval time per query |",
            "|---|---:|---:|",
            f"| Never rerank (hybrid) | {st['hybrid']['recall']:.3f} | {st['hybrid']['seconds']:.2f} s |",
            f"| Always rerank (CRAG) | {st['crag']['recall']:.3f} | {st['crag']['seconds']:.2f} s |",
            f"| **Calibrated routing** | **{st['adaptive']['recall']:.3f}** | **{st['adaptive']['seconds']:.2f} s** |",
            "",
            (f"Routing matches always-rerank recall" if st["adaptive"]["recall"] >= 0.995 * st["crag"]["recall"]
             else f"Routing keeps {st['adaptive']['recall'] / st['crag']['recall']:.1%} of always-rerank recall")
            + f" and saves {saved:.0%} of its retrieval time. {_link(evaluation, 'Evaluation data')}",
            "",
        ]
    elif training:
        lines += ["*Cost-vs-quality evaluation of the routing is running.*", ""]
    return lines


def _section_ablation(entry: dict[str, Any]) -> list[str]:
    r = _load(entry)
    summary, manifest = r["summary"], r["manifest"]
    n, total = summary["questions_judged_by_all_models"], manifest["question_count"]
    models = sorted(summary["models"].items(), key=lambda kv: -kv[1]["leaderboard_score"])
    lines = [
        "## Generator comparison",
        "",
        f"Same retrieved context for every model; answers graded by an independent judge "
        f"(`{summary['judge_model']}`). **{n} of {total} questions graded so far.**",
        "",
        "| Model | Correct | Facts stated | Score | 95% interval | Avg. answer time |",
        "|---|---:|---:|---:|---|---:|",
        *[f"| {m.split('/')[-1]} | {v['correctness']:.0%} | {v['completeness']:.0%} | **{v['leaderboard_score']:.2f}** "
          f"| {v['leaderboard_95ci'][0]:.2f} – {v['leaderboard_95ci'][1]:.2f} | {v['mean_generation_seconds']:.0f} s |"
          for m, v in models],
        "",
        "Score = correct × share of gold facts stated.",
        "",
    ]
    pairs = summary.get("paired_leaderboard_differences", {})
    if pairs:
        lines += ["| Comparison | Score difference | 95% interval | Verdict |", "|---|---:|---|---|"]
        for pair in pairs.values():
            first, second = pair["first"].split("/")[-1], pair["second"].split("/")[-1]
            low, high = pair["95ci"]
            verdict = "real difference" if pair["significant"] else "not yet distinguishable"
            lines.append(f"| {first} vs {second} | {pair['mean_difference']:+.2f} | {low:+.2f} – {high:+.2f} | {verdict} |")
        lines.append("")
    attribution = summary.get("attribution")
    if attribution:
        labels = [("correct", "Correct"), ("generation_error", "Generation error"), ("context_loss", "Context loss"),
                  ("retrieval_miss", "Retrieval miss"), ("missed_abstention", "Missed abstention")]
        lines += [
            "### Why answers fail",
            "",
            f"For {attribution['questions']} questions the judge checked which gold facts reached each model's",
            f"context ({attribution['context_fact_recall']:.0%} on average; "
            f"{attribution['evidence_in_context_rate']:.0%} of questions had at least "
            f"{attribution['evidence_threshold']:.0%} of their facts in context). Each wrong answer is traced to the",
            "stage that lost the evidence: a *generation error* had the evidence and still failed, *context loss*",
            "retrieved the documents but not the facts into the prompt, a *retrieval miss* never found the documents.",
            "",
            "| Model | " + " | ".join(label for _, label in labels) + " |",
            "|---|" + "---:|" * len(labels),
        ]
        for model, data in attribution["models"].items():
            counts = data["outcomes"]
            total = sum(counts.values()) or 1
            lines.append(f"| {model.split('/')[-1]} | "
                         + " | ".join(f"{counts.get(key, 0)} ({counts.get(key, 0) / total:.0%})" for key, _ in labels) + " |")
        if attribution.get("context_judge_false_positive_rate") is not None:
            lines += ["", f"Context check false-positive rate (facts credited to another question's context): "
                          f"{attribution['context_judge_false_positive_rate']:.0%} on {attribution['context_controls']} controls."]
        lines.append("")
    sanity = summary.get("judge_sanity")
    if sanity:
        lines += [
            f"Judge check on {sanity['n']} questions: gold answers judged correct "
            f"{sanity['gold_answer_marked_correct']:.0%}; another question's answer judged correct "
            f"{sanity['wrong_answer_marked_correct']:.0%} ({sanity['wrong_answer_facts_present']:.0%} of its facts "
            f"falsely credited). {_link(entry, 'Answers, judgments and per-question table')}",
            "",
        ]
    return lines


def _optional(render, *args) -> list[str]:
    """A section that cannot be rendered (e.g. an older or partial result) is left out, never fatal."""
    try:
        return render(*args)
    except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError, OSError):
        return []


def rebuild_readme() -> None:
    _write_readme(list_results())


def _write_readme(index: list[dict[str, Any]]) -> None:
    latest_of = {}
    for entry in index:  # newest first
        latest_of.setdefault(entry["kind"], entry)
    lines = [
        "# Results",
        "",
        "Every experiment saved by this repository, newest first. Each folder holds `result.json` (metrics),",
        "per-question CSVs and `meta.json` (git commit, configuration and hardware it was produced on).",
        "This page is generated by `core/results_store.py`; reproduce any result with `scripts/run_pipeline.sh`",
        "(see [docs/WORKFLOW.md](../docs/WORKFLOW.md)).",
        "",
    ]
    if "retrieval_eval" in latest_of:
        lines += _optional(_section_retrieval, latest_of["retrieval_eval"])
    if "calibrator_training" in latest_of or "calibrator_eval" in latest_of:
        lines += _optional(_section_calibrator, latest_of.get("calibrator_training"), latest_of.get("calibrator_eval"))
    if "generator_ablation" in latest_of:
        lines += _optional(_section_ablation, latest_of["generator_ablation"])
    lines += [
        "## All saved results",
        "",
        "| Date (UTC) | Experiment | Headline | Folder | Commit |",
        "|---|---|---|---|---|",
    ]
    for entry in index:
        headline = ", ".join(f"{key.replace('_', ' ')} {_fmt(value)}" for key, value in entry["headline"].items())
        commit = (entry.get("git_commit") or "")[:7] + (" + local changes" if entry.get("git_dirty") else "")
        lines.append(
            f"| {entry['created_at'][:16].replace('T', ' ')} | {KINDS[entry['kind']]}: {entry['title']} "
            f"| {headline} | {_link(entry, entry['id'])} | `{commit}` |"
        )
    (_root() / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
