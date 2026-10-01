"""Browse and download saved experiment results (see core/results_store.py)."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

import config
from core import results_store

router = APIRouter(prefix="/results", tags=["results"])


@router.get("")
def list_results():
    return results_store.list_results()


@router.get("/{kind}/{result_id}/{filename}")
def result_file(kind: str, result_id: str, filename: str):
    # Only files the index lists are served, so no path in the URL reaches the filesystem directly.
    entry = next(
        (e for e in results_store.list_results() if e["kind"] == kind and e["id"] == result_id),
        None,
    )
    if entry is None or filename not in entry["files"]:
        raise HTTPException(status_code=404, detail="no such result file")
    return FileResponse(config.RESULTS_DIR / entry["path"] / filename, filename=filename)
