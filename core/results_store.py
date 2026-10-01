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


def _root() -> Path:
    return config.RESULTS_DIR


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

    write_json(directory / "result.json", result)
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
        "command": sys.argv,
        "python": platform.python_version(),
        "host": platform.node(),
        "config": config_snapshot(),
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


def _write_readme(index: list[dict[str, Any]]) -> None:
    lines = [
        "# Results",
        "",
        "Generated by `core/results_store.py` - do not edit by hand. Each row is a folder with",
        "`result.json`, per-question CSVs and `meta.json` (git commit, config snapshot, command).",
        "",
        "| Created (UTC) | Kind | Title | Headline | Folder |",
        "|---|---|---|---|---|",
    ]
    for entry in index:
        headline = ", ".join(f"{key} {_fmt(value)}" for key, value in entry["headline"].items())
        commit = (entry.get("git_commit") or "")[:7] + ("*" if entry.get("git_dirty") else "")
        lines.append(
            f"| {entry['created_at'][:16].replace('T', ' ')} | {KINDS[entry['kind']]} | {entry['title']} "
            f"| {headline} | [{entry['path']}]({entry['path']}) `{commit}` |"
        )
    (_root() / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
