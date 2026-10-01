from fastapi import APIRouter, HTTPException, Request

from api.schemas import CalibrationRefitRequest
from core.calibrator import RetrievalCalibrator

router = APIRouter(prefix="/calibration", tags=["calibration"])


@router.post("/refit")
def refit(body: CalibrationRefitRequest, request: Request):
    service = request.app.state.rag
    if not service.ready or service.calibrator is None:
        raise HTTPException(status_code=503, detail="RAG service is not ready")
    replacement = RetrievalCalibrator(service.calibrator.model_path)
    try:
        replacement.fit([example.model_dump() for example in body.examples])
        replacement.save()
    except (ValueError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    service.calibrator = replacement
    service.adaptive.calibrator = replacement
    return {
        "fitted": True,
        "example_count": replacement.example_count,
        "last_refit": replacement.last_refit,
    }


@router.get("/status")
def status(request: Request):
    calibrator = request.app.state.rag.calibrator
    if calibrator is None:
        return {"fitted": False, "example_count": 0, "last_refit": None}
    return {
        "fitted": calibrator.is_fitted,
        "example_count": calibrator.example_count,
        "last_refit": calibrator.last_refit,
    }
