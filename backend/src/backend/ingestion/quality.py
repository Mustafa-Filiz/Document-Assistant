"""Cheap heuristics that flag chunks which are probably broken.

They do NOT replace reading chunks by hand (task 5) - they tell you where to look first
and give comparable numbers across chunking strategies.
"""
from __future__ import annotations

import re
import statistics
from typing import Dict, Iterable, List, Optional, Tuple

from .extract import is_tabular
from .models import Chunk

HEADING_LINE_RE = re.compile(r"^(?:§|Art\.)\s*\d+[a-z]?\s+\S", re.M)
FOOTNOTE_RE = re.compile(
    r"\(\+\+\+.*?\+\+\+\)"
    r"|^\s*Fußnote"
    r"|^\s*\d{1,2}\)\s+(?:Geändert|Eingefügt|Aufgehoben|Neugefasst|Früher|Siehe|Art\.|§)"
    r"|Zuletzt geändert durch|Änderungshinweis",
    re.M | re.I,
)

ALL_FLAGS = (
    "table_like",
    "footnote_or_amendment_note",
    "crosses_section",
    "ref_out_of_order",
    "starts_mid_sentence",
    "ends_mid_sentence",
    "too_short",
    "too_long",
    "bad_chars",
)


def ref_number(ref: str) -> Optional[int]:
    m = re.match(r"(?:§|Art\.)\s*(\d+)", ref)
    return int(m.group(1)) if m else None


def ordering_anomalies(chunks: Iterable[Chunk]) -> set:
    """Chunks (in document order) whose § number is smaller than the previous one.

    Statutes number monotonically, so a drop usually means a wrapped in-text reference was
    mistaken for a heading ("heading drift").
    """
    out, prev = set(), None
    for c in chunks:
        n = ref_number(c.section_ref)
        if n is None:
            continue
        if prev is not None and n < prev:
            out.add(c.chunk_id)
        prev = n
    return out


def flags(chunk: Chunk, max_chars: int = 1500, min_chars: int = 150, anomalies: set = frozenset()) -> List[str]:
    t = chunk.text
    out: List[str] = []
    if sum(1 for ln in t.split("\n") if is_tabular(ln)) >= 4:
        out.append("table_like")
    if FOOTNOTE_RE.search(t):
        out.append("footnote_or_amendment_note")
    if any(m.start() > 0 for m in HEADING_LINE_RE.finditer(t)):
        out.append("crosses_section")
    if chunk.chunk_id in anomalies:
        out.append("ref_out_of_order")
    if t[:1].islower():
        out.append("starts_mid_sentence")
    if not re.search(r"[.;:!?)\"“]$", t.rstrip()):
        out.append("ends_mid_sentence")
    if len(t) < min_chars:
        out.append("too_short")
    if len(t) > max_chars * 1.1:
        out.append("too_long")
    if "\ufffd" in t:
        out.append("bad_chars")
    return out


def summarize(chunks: List[Chunk], max_chars: int = 1500) -> Dict[str, float]:
    """Length distribution + share of flagged chunks."""
    if not chunks:
        return {"chunks": 0}
    lens = sorted(c.char_len for c in chunks)
    anomalies = ordering_anomalies(chunks)
    counts = {f: 0 for f in ALL_FLAGS}
    for c in chunks:
        for f in flags(c, max_chars, anomalies=anomalies):
            counts[f] += 1
    n = len(chunks)
    res: Dict[str, float] = {
        "chunks": n,
        "mean_chars": round(statistics.mean(lens)),
        "median_chars": round(statistics.median(lens)),
        "p95_chars": lens[min(n - 1, int(0.95 * n))],
        "max_chars": lens[-1],
        "split_within_clause_%": round(100 * sum(c.split_within_clause for c in chunks) / n, 1),
    }
    for f in ALL_FLAGS:
        res[f"{f}_%"] = round(100 * counts[f] / n, 1)
    return res
