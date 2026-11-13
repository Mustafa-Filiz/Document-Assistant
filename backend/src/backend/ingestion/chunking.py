"""Chunking strategies.

fixed         naive baseline: one long text, sliding window (size/overlap in characters),
              no heading prefix. Section/page metadata is looked up by position.
clause        structure-aware: split at §/Art. boundaries, then pack whole Absätze up to
              max_chars. An over-long Absatz is cut at sentence boundaries. The embedded
              text carries a heading prefix ("Doc | Teil > Abschnitt | § 34 Titel").
clause_plain  same chunks as `clause`, but embedded WITHOUT the heading prefix
              (ablation: does the prefix help retrieval?).

Sizes are in characters; for German ~3.5 chars ~ 1 token, so 1500 chars ~ 430 tokens,
which stays under the 512-token limit of multilingual-e5.
"""
from __future__ import annotations

import re
from bisect import bisect_right
from typing import List, Tuple

from .models import Atom, Chunk, Section

STRATEGIES = ("fixed", "clause", "clause_plain")

_ABBREV = {
    "abs", "nr", "vgl", "bzw", "z", "b", "d", "h", "ggf", "usw", "ca", "art", "anl", "s",
    "buchst", "lit", "u", "a", "i", "v", "m", "mio", "evtl", "tel", "ff", "f", "u.a", "o",
}
_STUB_RE = re.compile(r"weggefallen|aufgehoben|gegenstandslos|nicht belegt", re.I)


def _is_stub(text: str) -> bool:
    return len(text) < 60 and bool(_STUB_RE.search(text))


# --------------------------------------------------------------------------- sentence splitting


def _ends_with_abbrev(s: str) -> bool:
    if not s.endswith("."):
        return False
    last = s.rsplit(None, 1)[-1].rstrip(".;:!?").lower()
    return last in _ABBREV or len(last) == 1


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.;:!?])\s+(?=[A-ZÄÖÜ(„\"])", text)
    out: List[str] = []
    for p in parts:
        if out and _ends_with_abbrev(out[-1]):
            out[-1] += " " + p
        else:
            out.append(p)
    return out


def _hard_split(text: str, max_chars: int) -> List[str]:
    pieces = []
    while len(text) > max_chars:
        cut = max(text.rfind("\n", 0, max_chars), text.rfind(" ", 0, max_chars))
        if cut < max_chars // 2:
            cut = max_chars
        pieces.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        pieces.append(text)
    return pieces


def split_long(text: str, max_chars: int) -> List[str]:
    """Cut an over-long Absatz at sentence boundaries into pieces <= max_chars."""
    pieces: List[str] = []
    cur = ""
    for s in split_sentences(text):
        if len(s) > max_chars:
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.extend(_hard_split(s, max_chars))
        elif cur and len(cur) + 1 + len(s) > max_chars:
            pieces.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}" if cur else s
    if cur:
        pieces.append(cur)
    return pieces


# --------------------------------------------------------------------------- clause strategy


def _header(doc_name: str, sec: Section) -> str:
    parts = [doc_name]
    if sec.hierarchy:
        parts.append(" > ".join(sec.hierarchy))
    if sec.heading:
        parts.append(sec.heading)
    return " | ".join(parts)


def clause_chunks(
    sections: List[Section],
    doc_id: str,
    doc_name: str,
    max_chars: int = 1500,
    prefix: bool = True,
    strategy: str = "clause",
) -> List[Chunk]:
    chunks: List[Chunk] = []

    def add(sec: Section, atoms: List[Atom], text: str, split: bool, part: int = 0) -> None:
        if _is_stub(text):
            return
        labels = [a.label for a in atoms if a.label]
        absatz = ""
        if labels:
            absatz = labels[0] if labels[0] == labels[-1] else f"{labels[0]}-{labels[-1]}"
        if part:
            absatz = f"{absatz} [Teil {part}]".strip()
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}:{strategy}:{len(chunks):05d}",
                doc_id=doc_id,
                doc_name=doc_name,
                strategy=strategy,
                text=text,
                embed_text=f"{_header(doc_name, sec)}\n{text}" if prefix else text,
                page_start=min(a.page_start for a in atoms),
                page_end=max(a.page_end for a in atoms),
                section_ref=sec.ref,
                section_title=sec.title,
                hierarchy=sec.hierarchy_str,
                absatz=absatz,
                refs_covered=sec.ref,
                split_within_clause=split,
            )
        )

    for sec in sections:
        group: List[Atom] = []

        def flush() -> None:
            if group:
                add(sec, list(group), "\n".join(a.text for a in group), False)
                group.clear()

        for atom in sec.atoms:
            if _is_stub(atom.text):  # "(2) weggefallen" carries no content
                continue
            if len(atom.text) > max_chars:
                flush()
                for n, piece in enumerate(split_long(atom.text, max_chars), start=1):
                    add(sec, [atom], piece, True, part=n)
                continue
            size = sum(len(a.text) + 1 for a in group)
            if group and size + len(atom.text) > max_chars:
                flush()
            group.append(atom)
        flush()
    return chunks


# --------------------------------------------------------------------------- fixed strategy


def fixed_chunks(
    sections: List[Section],
    doc_id: str,
    doc_name: str,
    size: int = 1000,
    overlap: int = 150,
) -> List[Chunk]:
    # 1) Flatten the document to one string, remembering which span belongs to which section/page.
    parts: List[str] = []
    spans: List[Tuple[int, int, int, int, int]] = []  # start, end, page0, page1, section idx
    pos = 0
    for si, sec in enumerate(sections):
        pieces: List[Tuple[str, int, int]] = []
        if sec.heading:
            pieces.append((sec.heading, sec.page_start, sec.page_start))
        pieces.extend((a.text, a.page_start, a.page_end) for a in sec.atoms)
        for text, p0, p1 in pieces:
            spans.append((pos, pos + len(text), p0, p1, si))
            parts.append(text + "\n")
            pos += len(text) + 1
    full = "".join(parts)
    starts = [s[0] for s in spans]

    # 2) Slide a window over it, snapping cut points to whitespace.
    chunks: List[Chunk] = []
    start, n = 0, len(full)
    while start < n:
        end = min(start + size, n)
        if end < n:
            cut = full.rfind(" ", start + size // 2, end)
            if cut > start:
                end = cut
        text = full[start:end].strip()
        if text:
            first = max(bisect_right(starts, start) - 1, 0)
            covered = []
            j = first
            while j < len(spans) and spans[j][0] < end:
                if spans[j][1] > start:
                    covered.append(spans[j])
                j += 1
            if covered:
                sec0 = sections[covered[0][4]]
                refs = list(dict.fromkeys(sections[c[4]].ref for c in covered if sections[c[4]].ref))
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}:fixed:{len(chunks):05d}",
                        doc_id=doc_id,
                        doc_name=doc_name,
                        strategy="fixed",
                        text=text,
                        embed_text=text,
                        page_start=min(c[2] for c in covered),
                        page_end=max(c[3] for c in covered),
                        section_ref=sec0.ref,
                        section_title=sec0.title,
                        hierarchy=sec0.hierarchy_str,
                        refs_covered="|".join(refs),
                    )
                )
        if end >= n:
            break
        nxt = full.find(" ", end - overlap, end)
        start = nxt + 1 if nxt != -1 and nxt + 1 > start else end
    return chunks


def build_chunks(
    sections: List[Section],
    doc_id: str,
    doc_name: str,
    strategy: str,
    max_chars: int = 1500,
    fixed_size: int = 1000,
    fixed_overlap: int = 150,
) -> List[Chunk]:
    if strategy == "fixed":
        return fixed_chunks(sections, doc_id, doc_name, fixed_size, fixed_overlap)
    if strategy == "clause":
        return clause_chunks(sections, doc_id, doc_name, max_chars, True, "clause")
    if strategy == "clause_plain":
        return clause_chunks(sections, doc_id, doc_name, max_chars, False, "clause_plain")
    raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
