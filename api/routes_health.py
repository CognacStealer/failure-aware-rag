from fastapi import APIRouter, Request, Response

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/ready")
def ready(request: Request, response: Response):
    status = request.app.state.rag.status()
    if not status["ready"] or not status["generator_loaded"]:
        response.status_code = 503
    return status
