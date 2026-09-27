from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes_health import router as health_router
from api.routes_query import router as query_router
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
app.include_router(health_router)
app.include_router(query_router)
