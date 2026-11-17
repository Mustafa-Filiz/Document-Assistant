from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.rag.api import router as ask_router
from .config import settings

app = FastAPI(
    title="Construction Documents RAG Assistant",
    version="0.2.0",
    description="Q&A over public German construction regulations, with source pages.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ask_router)


@app.get("/")
def read_root() -> dict:
    return {"name": "construction-rag", "version": app.version, "docs": "/docs"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
