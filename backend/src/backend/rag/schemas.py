"""HTTP request/response models (pydantic). Kept separate so the core has no pydantic dependency."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, examples=["Wann darf im Außenbereich gebaut werden?"])
    k: int = Field(5, ge=1, le=20, description="How many chunks to retrieve")
    doc_id: Optional[str] = Field(None, description="Restrict to one document id from data/sources.csv, e.g. 'baybo'")
    generate: bool = Field(True, description="false = retrieval only (no LLM call), useful for debugging")


class SourceOut(BaseModel):
    n: int
    doc_id: str
    doc_name: str
    section_ref: str
    section_title: str
    hierarchy: str
    absatz: str
    page_start: int
    page_end: int
    page_label: str
    reference: str
    score: float
    text: str
    in_context: bool
    cited: bool


class AskResponse(BaseModel):
    question: str
    status: Literal["answered", "no_answer", "retrieval_only"]
    answer: str
    reason: Optional[str] = None
    top_score: Optional[float] = None
    threshold: float
    language: str
    model: Optional[str] = None
    sources: List[SourceOut]
