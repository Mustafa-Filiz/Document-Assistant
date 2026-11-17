"""The /ask pipeline: retrieve -> similarity gate -> LLM (sources only) -> citation check.

Four independent ways to end with "I don't know" (status="no_answer"):
  1. no_results       nothing retrieved (e.g. unknown doc_id filter)
  2. below_threshold  best chunk is less similar than MIN_SIMILARITY -> the LLM is never called
  3. llm_declined     the LLM answered NO_ANSWER because the sources do not contain the answer
  4. uncited_answer   the LLM answered without any valid [n] citation -> we refuse to show it
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Set

from ..config import settings
from .lang import NO_ANSWER_MESSAGES, detect_language
from .llm import LLMClient, build_llm_client
from .prompts import NO_ANSWER_TOKEN, SYSTEM_PROMPT, build_user_prompt, page_label, reference
from .retriever import Hit, Retriever

log = logging.getLogger("backend.rag")

_CITATION_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


@dataclass
class Source:
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
    in_context: bool  # score >= threshold, i.e. it was given to the LLM
    cited: bool = False  # the final answer cites it as [n]


@dataclass
class AskResult:
    question: str
    status: str  # "answered" | "no_answer" | "retrieval_only"
    answer: str
    reason: Optional[str]
    top_score: Optional[float]
    threshold: float
    language: str
    model: Optional[str]
    sources: List[Source] = field(default_factory=list)


def extract_citations(answer: str, max_n: int) -> Set[int]:
    """Valid source numbers cited in `answer` ('[1]', '[2][3]', '[1, 2]'); out-of-range ones are ignored."""
    found: Set[int] = set()
    for group in _CITATION_RE.findall(answer):
        for num in re.split(r"[,;]", group):
            n = int(num)
            if 1 <= n <= max_n:
                found.add(n)
    return found


def is_no_answer(text: str) -> bool:
    return text.strip().strip(".!\"'` ").upper().startswith(NO_ANSWER_TOKEN)


def _to_source(n: int, hit: Hit, threshold: float) -> Source:
    m = hit.meta
    return Source(
        n=n,
        doc_id=str(m.get("doc_id", "")),
        doc_name=str(m.get("doc_name", "")),
        section_ref=str(m.get("section_ref", "")),
        section_title=str(m.get("section_title", "")),
        hierarchy=str(m.get("hierarchy", "")),
        absatz=str(m.get("absatz", "")),
        page_start=int(m.get("page_start", 0)),
        page_end=int(m.get("page_end", 0)),
        page_label=page_label(m),
        reference=reference(m),
        score=hit.score,
        text=hit.text,
        in_context=hit.score >= threshold,
    )


class AskService:
    def __init__(self, retriever: Retriever, llm: Optional[LLMClient],
                 min_similarity: float, default_k: int = 5):
        self.retriever = retriever
        self.llm = llm
        self.min_similarity = min_similarity
        self.default_k = default_k

    def ask(self, question: str, k: Optional[int] = None, doc_id: Optional[str] = None,
            generate: bool = True) -> AskResult:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        lang = detect_language(question)
        hits = self.retriever.search(question, k or self.default_k, doc_id)
        sources = [_to_source(i, h, self.min_similarity) for i, h in enumerate(hits, start=1)]
        top = hits[0].score if hits else None

        def decline(reason: str) -> AskResult:
            log.info("no_answer reason=%s top_score=%s", reason, top)
            return AskResult(question, "no_answer", NO_ANSWER_MESSAGES[lang], reason, top,
                             self.min_similarity, lang, getattr(self.llm, "model", None), sources)

        if not hits:
            return decline("no_results")
        # Hits are sorted by score, so the in-context hits are the first ones and
        # their position equals the [n] number the LLM cites.
        context = [h for h in hits if h.score >= self.min_similarity]
        if not context:
            return decline("below_threshold")

        if not generate or self.llm is None:
            reason = "generation_disabled" if self.llm is not None else "no_llm_configured"
            return AskResult(question, "retrieval_only", "", reason, top, self.min_similarity,
                             lang, None, sources)

        raw = self.llm.complete(SYSTEM_PROMPT, build_user_prompt(question, context))
        if is_no_answer(raw):
            return decline("llm_declined")
        cited = extract_citations(raw, len(context))
        if not cited:
            # An answer we cannot trace to a source is exactly what this project must not show.
            log.warning("discarding uncited LLM answer: %r", raw[:200])
            return decline("uncited_answer")
        for s in sources:
            s.cited = s.n in cited
        log.info("answered top_score=%s cited=%s", top, sorted(cited))
        return AskResult(question, "answered", raw, None, top, self.min_similarity, lang,
                         getattr(self.llm, "model", None), sources)


def build_service(strategy: Optional[str] = None, min_similarity: Optional[float] = None,
                  top_k: Optional[int] = None) -> AskService:
    retriever = Retriever.from_settings(strategy=strategy)
    return AskService(
        retriever,
        build_llm_client(settings),
        settings.min_similarity if min_similarity is None else min_similarity,
        top_k or settings.top_k,
    )
