"""Live view of generator-ablation runs, read from their run directories while they execute."""

import csv
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import chromadb
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from sklearn.metrics import roc_auc_score

import config
from core import results_store
from core.ablation_report import attribute, compute_summary, evidence_share, read_jsonl, read_manifest, slug

router = APIRouter(prefix="/ablation", tags=["ablation"])

RUNNER_SCRIPT = "run_local_generator_ablation.py"
DEFAULT_RUN_DIR = "data/ablation_runs/full_benchmark"
DEFAULT_SANITY = 25
RECENT_LIMIT = 30
SECONDS_PER_RETRIEVAL = 2.5

_questions: dict[str, dict] = {}
_summary_cache: dict[str, tuple[tuple, dict]] = {}


def _runs_root() -> Path:
    return config.ABLATION_RUNS_DIR


def _run_dirs() -> dict[str, Path]:
    root = _runs_root()
    runs = {}
    if root.is_dir():
        for path in sorted(root.iterdir()):
            manifest = path / "manifest.json"
            if manifest.is_file():
                try:
                    if "candidates" in json.loads(manifest.read_text(encoding="utf-8")):
                        runs[path.name] = path
                except ValueError:
                    continue
    return runs


def _questions_by_id(request: Request) -> dict[str, dict]:
    if not _questions:
        client = getattr(request.app.state.rag, "client", None) or chromadb.PersistentClient(path=config.CHROMA_PATH)
        rows = client.get_collection(config.CHROMA_QUESTIONS_COLLECTION).get(include=["documents", "metadatas"])
        for question_id, text, metadata in zip(rows["ids"], rows["documents"], rows["metadatas"]):
            metadata = metadata or {}
            _questions[question_id] = {
                "question": text,
                "category": metadata.get("question_type", ""),
                "gold_answer": metadata.get("gold_answer", ""),
                "facts": len([f for f in metadata.get("answer_facts", "").split("|") if f.strip()]),
            }
    return _questions


def _runner_process(run_dir: Path) -> dict | None:
    """The live runner writing to run_dir, found by scanning /proc (Linux)."""
    proc = Path("/proc")
    if not proc.is_dir():
        return None
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().decode(errors="replace").split("\0")
            if not any(arg.endswith(RUNNER_SCRIPT) for arg in argv) or "--summarize" in argv:
                continue
            cwd = Path(os.readlink(entry / "cwd"))
            started = (entry / "stat").stat().st_mtime
        except (OSError, ValueError):
            continue
        target = argv[argv.index("--run-dir") + 1] if "--run-dir" in argv else DEFAULT_RUN_DIR
        if (cwd / target).resolve() == run_dir.resolve():
            return {"pid": int(entry.name), "started_at": started}
    return None


def _ollama_loaded() -> list[str]:
    try:
        with urllib.request.urlopen(f"{config.OLLAMA_HOST.rstrip('/')}/api/ps", timeout=1.5) as response:
            return [model["name"] for model in json.load(response).get("models", [])]
    except (urllib.error.URLError, OSError, ValueError):
        return []


def _signature(run_dir: Path) -> tuple:
    return tuple(sorted((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in run_dir.glob("*.json*")))


def _mean(values: list[float], default: float) -> float:
    return sum(values) / len(values) if values else default


@router.get("/runs")
def list_runs():
    runs = []
    for name, path in _run_dirs().items():
        manifest = read_manifest(path)
        runs.append({
            "name": name,
            "question_count": manifest.get("question_count"),
            "candidates": manifest["candidates"],
            "judge_model": manifest.get("judge_model"),
            "running": _runner_process(path) is not None,
        })
    return runs


@router.get("/runs/{name}")
def run_status(name: str, request: Request):
    run_dir = _run_dirs().get(name)
    if run_dir is None:
        raise HTTPException(status_code=404, detail=f"unknown run '{name}'")
    manifest = read_manifest(run_dir)
    questions = _questions_by_id(request)
    models = list(manifest["candidates"])
    tags = manifest["candidates"]
    total = manifest["question_count"]
    sanity_target = min(manifest.get("judge_sanity", DEFAULT_SANITY), total)

    retrieval = read_jsonl(run_dir / "retrieval.jsonl")
    answers = {model: read_jsonl(run_dir / f"{slug(model)}.jsonl") for model in models}
    judged = {model: read_jsonl(run_dir / f"{slug(model)}_judged.jsonl") for model in models}
    sanity = read_jsonl(run_dir / "judge_sanity.jsonl")
    context = read_jsonl(run_dir / "context_facts.jsonl")
    order = list(retrieval)

    # What is happening now: the model Ollama has loaded, else inferred from file counts.
    process = _runner_process(run_dir)
    loaded = _ollama_loaded()
    phase, active_model, current_qid = "stopped", None, None
    if process:
        if manifest["judge_model"] in loaded:
            phase = "judging"
            active_model = next(
                (m for m in models if any(q in answers[m] and q not in judged[m] for q in order)), None
            )
        else:
            active_model = next((m for m in models if tags[m] in loaded), None)
            if active_model is None:
                # Between calls (candidates unload after each answer): the first model with pending work.
                active_model = next((m for m in models if any(q not in answers[m] for q in order)), None)
            phase = "generating" if active_model else "retrieving"
        pending = (
            [q for q in order if q in answers.get(active_model, {}) and q not in judged.get(active_model, {})]
            if phase == "judging" else
            [q for q in order if q not in answers.get(active_model, {})]
        ) if active_model else []
        current_qid = pending[0] if pending else None

    # ETA from measured per-call times; each candidate call includes its model load.
    eta = 0.0
    judge_times = [row.get("seconds", 0) for m in models for row in judged[m].values() if row.get("seconds")]
    judge_mean = _mean(judge_times, 40.0)
    for model in models:
        gen_mean = _mean([row["seconds"] for row in answers[model].values()], 90.0)
        eta += (total - len(answers[model])) * gen_mean + (total - len(judged[model])) * judge_mean
    eta += max(0, sanity_target - len(sanity)) * 2 * judge_mean
    eta += (total - len(retrieval)) * SECONDS_PER_RETRIEVAL

    signature = _signature(run_dir)
    cached = _summary_cache.get(name)
    if cached and cached[0] == signature:
        summary = cached[1]
    else:
        summary = compute_summary(run_dir, {q: info["category"] for q, info in questions.items()})
        _summary_cache[name] = (signature, summary)

    # Newest first: follow the file the runner is appending to right now (rows carry no timestamps).
    if phase == "judging" and active_model:
        anchor = list(judged[active_model])
    elif active_model:
        anchor = list(answers[active_model])
    else:
        anchor = [q for q in order if any(q in answers[m] for m in models)]
    recent = []
    for qid in reversed(anchor[-RECENT_LIMIT:]):
        info = questions.get(qid, {})
        entry = {
            "question_id": qid,
            "question": info.get("question", ""),
            "category": info.get("category", ""),
            "gold_answer": info.get("gold_answer", ""),
            "recall_at_k": retrieval[qid].get("recall_at_k"),
            "evidence_share": evidence_share(context[qid]["facts_in_context"]) if qid in context else None,
            "answers": {},
        }
        for model in models:
            if qid not in answers[model]:
                continue
            answer = answers[model][qid]
            verdict = judged[model].get(qid)
            entry["answers"][model] = {
                "answer": answer["answer"],
                "seconds": answer["seconds"],
                "cut_off": answer.get("hit_token_limit", False),
                "verdict": None if verdict is None else {
                    "correct": verdict["correct"],
                    "facts_present": sum(1 for f in verdict["facts_present"] if f),
                    "facts_total": len(verdict["facts_present"]),
                    "outcome": attribute(verdict["correct"], info.get("category", ""),
                                         retrieval[qid].get("recall_at_k"),
                                         context[qid]["facts_in_context"]) if qid in context else None,
                },
            }
        recent.append(entry)

    current = questions.get(current_qid, {}) if current_qid else {}
    return {
        "name": name,
        "manifest": manifest,
        "running": process is not None,
        "process_started_at": process["started_at"] if process else None,
        "ollama_loaded": loaded,
        "activity": {
            "phase": phase,
            "model": active_model,
            "question_id": current_qid,
            "question": current.get("question"),
            "category": current.get("category"),
        },
        "progress": {
            "total": total,
            "retrieved": len(retrieval),
            "fully_judged": summary["questions_judged_by_all_models"],
            "sanity": {"done": len(sanity), "target": sanity_target},
            "models": {m: {"answered": len(answers[m]), "judged": len(judged[m])} for m in models},
        },
        "eta_seconds": round(eta) if process else None,
        "summary": summary,
        "recent": recent,
        "server_time": time.time(),
    }


def _selective_answering(report_rows: list[dict], judged: dict[str, dict[str, dict]]) -> dict:
    """Answer only the questions the calibrator is most confident about: does accuracy rise?

    Held-out (test-split) questions only. Random abstention keeps accuracy flat at the
    overall rate, so any rise above that line is what the calibrator's ranking buys.
    """
    predictions = {row["question_id"]: row["p_success"] for row in report_rows if row["split"] == "test"}
    curves = {}
    for model, verdicts in judged.items():
        scored = sorted(
            ((predictions[q], float(v["correct"])) for q, v in verdicts.items() if q in predictions),
            key=lambda pair: -pair[0],
        )
        if len(scored) < 5:
            curves[model] = {"n": len(scored), "points": [], "overall": None, "auc": None}
            continue
        correct = [c for _, c in scored]
        points = []
        for coverage in [round(c, 1) for c in np.arange(1.0, 0.15, -0.1)]:
            kept = max(1, round(coverage * len(scored)))
            points.append({"coverage": coverage, "accuracy": float(np.mean(correct[:kept])), "answered": kept})
        auc = None
        if 0 < sum(correct) < len(correct):
            auc = float(roc_auc_score(correct, [p for p, _ in scored]))
        curves[model] = {"n": len(scored), "points": points, "overall": float(np.mean(correct)), "auc": auc}
    return curves


def _load_calibration_report() -> tuple[dict, list[dict]] | None:
    """Latest calibrator evaluation from results/, else the legacy single-file report."""
    found = results_store.latest("calibrator_eval")
    if found:
        entry, directory = found
        report = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        with (directory / "per_question.csv").open(encoding="utf-8") as handle:
            rows = [
                {"question_id": row["question_id"], "split": row["split"], "p_success": float(row["p_success"])}
                for row in csv.DictReader(handle)
            ]
        report["result_id"] = entry["id"]
        return report, rows
    legacy = config.CALIBRATION_REPORT_PATH
    if legacy.is_file():
        report = json.loads(legacy.read_text(encoding="utf-8"))
        return report, report.pop("questions", [])
    return None


@router.get("/calibration")
def calibration(run: str = "full_benchmark"):
    loaded = _load_calibration_report()
    if loaded is None:
        return {"available": False, "hint": "run scripts/evaluate_calibrator.py"}
    report, rows = loaded
    selective = {}
    run_dir = _run_dirs().get(run)
    if run_dir is not None:
        models = list(read_manifest(run_dir)["candidates"])
        judged = {m: read_jsonl(run_dir / f"{slug(m)}_judged.jsonl") for m in models}
        selective = _selective_answering(rows, judged)
    return {"available": True, **report, "selective_answering": selective}
