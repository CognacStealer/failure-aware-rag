from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes_ablation import router as ablation_router
from api.routes_health import router as health_router
from api.routes_calibration import router as calibration_router
from api.routes_engine import router as engine_router
from api.routes_query import router as query_router
from api.routes_results import router as results_router
from core.service import service


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = service
    try:
        service.initialize()
    except Exception as exc:
        service.error = f"{type(exc).__name__}: {exc}"
    yield


app = FastAPI(title="Failure-Aware Adaptive RAG API", lifespan=lifespan)
app.state.rag = service
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
app.include_router(health_router)
app.include_router(calibration_router)
app.include_router(query_router)
app.include_router(ablation_router)
app.include_router(results_router)
app.include_router(engine_router)


@app.get("/", include_in_schema=False)
def frontend():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/dashboard", include_in_schema=False)
def ablation_dashboard():
    return FileResponse(Path(__file__).parent / "static" / "ablation.html")
