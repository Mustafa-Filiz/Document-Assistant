"""POST /ask"""
from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException

from .llm import LLMError
from .retriever import IndexNotReadyError
from .schemas import AskRequest, AskResponse
from .service import AskService, build_service

router = APIRouter(tags=["ask"])


@lru_cache(maxsize=1)
def _cached_service() -> AskService:
    return build_service()  # loads the embedding model once; exceptions are not cached


def get_service() -> AskService:
    try:
        return _cached_service()
    except IndexNotReadyError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:  # bad LLM_PROVIDER / LLM_MODEL configuration
        raise HTTPException(status_code=500, detail=f"Server configuration error: {e}") from e


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, service: AskService = Depends(get_service)) -> AskResponse:
    try:
        result = service.ask(req.question, k=req.k, doc_id=req.doc_id, generate=req.generate)
    except IndexNotReadyError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return AskResponse(**asdict(result))
