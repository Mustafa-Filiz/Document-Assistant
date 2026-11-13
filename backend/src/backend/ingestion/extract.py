"""PDF -> clean lines -> sections with page numbers and heading metadata.

Layout assumptions (German statutes from gesetze-im-internet.de / gesetze-bayern.de):
  * headings look like "§ 34 Zulässigkeit von ..." or "Art. 6 Abstandsflächen, Abstände"
  * big structure lines: "Erster Teil ...", "Abschnitt 2 ...", "Kapitel 3 ..."
  * Absätze start with "(1)", "(2)" ...
  * every page has a running header/footer; the first pages hold a table of contents
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .models import Atom, Line, Section

# --------------------------------------------------------------------------- reading

_TRANSLATE = str.maketrans(
    {
        "\u00ad": None,  # soft hyphen
        "\u00a0": " ",
        "\u2009": " ",
        "\u202f": " ",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb00": "ff",
        "\u2011": "-",
        "\x0c": None,
    }
)


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s).translate(_TRANSLATE)
    return re.sub(r"[ \t]+", " ", s).strip()


def read_pdf_lines(path: Union[str, Path]) -> List[Line]:
    """Read every text line with page number. Uses PyMuPDF (bold info), falls back to pypdf."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return _read_with_pypdf(path)
    return _read_with_pymupdf(path, fitz)


def _read_with_pymupdf(path, fitz) -> List[Line]:
    lines: List[Line] = []
    with fitz.open(str(path)) as doc:
        for pno, page in enumerate(doc, start=1):
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:  # 0 = text block
                    continue
                for ln in block["lines"]:
                    spans = [s for s in ln["spans"] if s["text"].strip()]
                    if not spans:
                        continue
                    text = "".join(s["text"] for s in ln["spans"])
                    total = sum(len(s["text"]) for s in spans)
                    bold_chars = sum(
                        len(s["text"])
                        for s in spans
                        if (s["flags"] & 16) or "bold" in s["font"].lower()
                    )
                    lines.append(
                        Line(pno, text, bold_chars >= 0.6 * total, max(s["size"] for s in spans))
                    )
    return lines


def _read_with_pypdf(path) -> List[Line]:
    from pypdf import PdfReader

    lines: List[Line] = []
    for pno, page in enumerate(PdfReader(str(path)).pages, start=1):
        for raw in (page.extract_text() or "").splitlines():
            if raw.strip():
                lines.append(Line(pno, raw))
    return lines


# --------------------------------------------------------------------------- cleaning

PAGE_NUM_RE = re.compile(r"^(?:-\s*)?(?:Seite\s+)?\d{1,4}(?:\s*(?:von|/)\s*\d{1,4})?(?:\s*-)?$", re.I)
AMENDMENT_NOTE_RE = re.compile(r"^\(\+\+\+.*\+\+\+\)$")


def _norm_key(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower())


def drop_headers_footers(lines: List[Line], edge: int = 3, min_ratio: float = 0.3) -> List[Line]:
    """Remove running headers/footers: edge lines whose (digit-normalised) text repeats on many pages."""
    by_page: Dict[int, List[int]] = defaultdict(list)
    for i, ln in enumerate(lines):
        by_page[ln.page].append(i)
    n_pages = len(by_page)

    edge_key: Dict[int, str] = {}
    counts: Counter = Counter()
    for idxs in by_page.values():
        keys = set()
        for i in dict.fromkeys(idxs[:edge] + idxs[-edge:]):
            key = _norm_key(lines[i].text)
            edge_key[i] = key
            keys.add(key)
        counts.update(keys)

    threshold = max(3, int(min_ratio * n_pages))
    drop = {
        i
        for i, key in edge_key.items()
        if counts[key] >= threshold or PAGE_NUM_RE.match(lines[i].text)
    }
    return [ln for i, ln in enumerate(lines) if i not in drop]


# --------------------------------------------------------------------------- heading detection

_REF = r"(?:§|Art\.|Artikel)\s*\d+[a-z]?"
SECTION_RE = re.compile(rf"^(?P<ref>{_REF})\s+(?P<title>\S.*)$")
REF_ONLY_RE = re.compile(rf"^(?P<ref>{_REF})$")
ANLAGE_RE = re.compile(r"^(?P<ref>Anlage\s+\d+[a-z]?)\b\s*(?P<title>.*)$")

# First words that show a line is a wrapped in-text reference ("§ 34 Absatz 1 gilt ..."), not a heading.
_REJECT_FIRST = {
    "absatz", "abs.", "absätze", "absatzes", "satz", "sätze", "nr.", "nummer", "nummern",
    "buchstabe", "buchst.", "halbsatz", "und", "bis", "oder", "sowie", "in", "des", "der", "die",
    "das", "dem", "den", "gilt", "gelten", "bleibt", "bleiben", "findet", "ist", "sind", "wird",
    "werden", "zu", "nach", "mit", "von", "bei", "auf", "an", "i.", "s.", "ff.", "f.",
}


def normalize_ref(ref: str) -> str:
    ref = re.sub(r"\s+", " ", ref.strip())
    ref = re.sub(r"^Artikel\s*", "Art. ", ref)
    return re.sub(r"^(§|Art\.)\s*(?=\d)", r"\1 ", ref)


def _bad_title(title: str) -> bool:
    if not title:
        return False
    return (
        title.split()[0].lower() in _REJECT_FIRST
        or title[0].islower()
        or title.endswith((",", ";"))
    )


def match_section_heading(text: str) -> Optional[Tuple[str, str]]:
    """Return (ref, title) if the line looks like a § / Art. / Anlage heading."""
    t = text.strip()
    if len(t) > 150:
        return None
    m = SECTION_RE.match(t)
    if m:
        title = m.group("title").strip()
        return None if _bad_title(title) else (normalize_ref(m.group("ref")), title)
    m = REF_ONLY_RE.match(t)
    if m:
        return normalize_ref(m.group("ref")), ""
    m = ANLAGE_RE.match(t)
    if m and len(t) <= 120:
        title = m.group("title").strip()
        return None if _bad_title(title) else (normalize_ref(m.group("ref")), title)
    return None


_KIND_LEVEL = {"buch": 0, "teil": 1, "kapitel": 2, "abschnitt": 3, "unterabschnitt": 4, "titel": 4}
_ORD = r"(?:Erste|Zweite|Dritte|Vierte|Fünfte|Sechste|Siebte|Achte|Neunte|Zehnte|Elfte|Zwölfte)[rnms]?"
_KINDS = r"(?:Buch|Teil|Kapitel|Abschnitt|Unterabschnitt|Titel)"
_HIER_RES = [
    re.compile(rf"^(?P<head>{_ORD}\s+(?P<kind>{_KINDS}))\b\.?\s*(?P<rest>.*)$"),
    re.compile(rf"^(?P<head>(?P<kind>{_KINDS})\s+(?:\d+[a-z]?|[IVXLC]+))\b\.?\s*(?P<rest>.*)$"),
    re.compile(rf"^(?P<head>\d+\.\s+(?P<kind>{_KINDS}))\b\.?\s*(?P<rest>.*)$"),
]


def match_hierarchy(text: str) -> Optional[Tuple[int, str, str]]:
    """Return (level, head, rest) for lines like 'Zweiter Teil Besondere Vorschriften'."""
    t = text.strip()
    if len(t) > 100:
        return None
    for rx in _HIER_RES:
        m = rx.match(t)
        if m:
            rest = m.group("rest").strip()
            if rest and (rest[0].islower() or rest.endswith(".")):
                return None
            return _KIND_LEVEL[m.group("kind").lower()], m.group("head"), rest
    return None


def is_tabular(text: str) -> bool:
    """Short line made almost only of digits and separators -> probably a table row."""
    t = text.strip()
    if not t or len(t) > 90 or "§" in t:
        return False
    if sum(c.isdigit() for c in t) < 2:
        return False
    return sum(c.isdigit() or c in " .,%€/-–+" for c in t) / len(t) >= 0.7


def _is_title_line(text: str) -> bool:
    t = text.strip()
    return (
        0 < len(t) <= 90
        and t[0].isupper()
        and not t.endswith(".")
        and not t.startswith("(")
        and match_section_heading(t) is None
        and match_hierarchy(t) is None
        and not is_tabular(t)
    )


def drop_toc_pages(lines: List[Line]) -> List[Line]:
    """Pages that are mostly headings are the table of contents; its entries are not content."""
    by_page: Dict[int, List[Line]] = defaultdict(list)
    for ln in lines:
        by_page[ln.page].append(ln)
    toc = set()
    for page, ls in by_page.items():
        hits = sum(1 for ln in ls if match_section_heading(ln.text) or match_hierarchy(ln.text))
        if hits >= 8 and hits / len(ls) >= 0.35:
            toc.add(page)
    return [ln for ln in lines if ln.page not in toc]


def _bold_is_reliable(lines: List[Line]) -> bool:
    """If (almost) all heading candidates are bold, require bold -> filters wrapped in-text refs."""
    cands = [ln for ln in lines if match_section_heading(ln.text)]
    return len(cands) >= 5 and sum(ln.bold for ln in cands) / len(cands) >= 0.7


# --------------------------------------------------------------------------- body -> atoms

ABSATZ_RE = re.compile(r"^\((\d{1,2}[a-z]?)\)\s*")
LIST_ITEM_RE = re.compile(r"^(?:\d{1,2}\.|\d{1,2}\)|[a-z]\)|[a-z]{2}\)|[-–•])\s+")
_KEEP_HYPHEN_NEXT = {"und", "oder", "bzw.", "sowie", "bis"}


def _join(prev: str, nxt: str) -> str:
    last_line = prev.rsplit("\n", 1)[-1]
    if is_tabular(nxt) or is_tabular(last_line) or LIST_ITEM_RE.match(nxt):
        return prev + "\n" + nxt
    if (
        prev.endswith("-")
        and len(prev) > 1
        and prev[-2].isalpha()
        and nxt[:1].islower()
        and nxt.split()[0].lower() not in _KEEP_HYPHEN_NEXT
    ):
        return prev[:-1] + nxt  # "bau-" + "rechtliche" -> "baurechtliche"
    return prev + " " + nxt


def _make_atoms(buf: List[Tuple[int, str]]) -> List[Atom]:
    atoms: List[Atom] = []
    cur: Optional[Atom] = None
    for page, text in buf:
        m = ABSATZ_RE.match(text)
        if m:
            if cur:
                atoms.append(cur)
            cur = Atom(text, page, page, f"({m.group(1)})")
        elif cur is None:
            cur = Atom(text, page, page)
        else:
            cur.text = _join(cur.text, text)
            cur.page_end = page
    if cur:
        atoms.append(cur)
    return atoms


class _Builder:
    def __init__(self, doc_id: str, first_page: int):
        self.doc_id = doc_id
        self.sections: List[Section] = []
        self.hier: List[Tuple[int, str]] = []
        self._start(first_page, "", "")

    def _start(self, page: int, ref: str, title: str) -> None:
        self.cur = Section(self.doc_id, ref, title, [t for _, t in self.hier], page, page)
        self.buf: List[Tuple[int, str]] = []

    def flush(self) -> None:
        if self.buf:
            self.cur.atoms = _make_atoms(self.buf)
            self.cur.page_end = self.buf[-1][0]
        if self.cur.atoms:  # headings without body (ToC remnants) are dropped
            self.sections.append(self.cur)

    def new_section(self, page: int, ref: str, title: str) -> None:
        self.flush()
        self._start(page, ref, title)

    def push_hierarchy(self, level: int, text: str, page: int) -> None:
        self.flush()
        while self.hier and self.hier[-1][0] >= level:
            self.hier.pop()
        self.hier.append((level, text))
        self._start(page, "", "")


def build_sections(lines: List[Line], doc_id: str) -> List[Section]:
    lines = drop_toc_pages(lines)
    if not lines:
        return []
    require_bold = _bold_is_reliable(lines)
    b = _Builder(doc_id, lines[0].page)
    i = 0
    while i < len(lines):
        ln = lines[i]
        h = match_section_heading(ln.text)
        if h and (ln.bold or not require_bold):
            ref, title = h
            used = 0
            if not title and i + 1 < len(lines) and _is_title_line(lines[i + 1].text):
                title, used = lines[i + 1].text.strip(), 1
            b.new_section(ln.page, ref, title)
            i += 1 + used
            continue
        hh = match_hierarchy(ln.text)
        if hh:
            level, head, rest = hh
            used = 0
            if not rest and i + 1 < len(lines) and _is_title_line(lines[i + 1].text):
                rest, used = lines[i + 1].text.strip(), 1
            b.push_hierarchy(level, f"{head} {rest}".strip(), ln.page)
            i += 1 + used
            continue
        b.buf.append((ln.page, ln.text))
        i += 1
    b.flush()
    return b.sections


# --------------------------------------------------------------------------- entry point


def extract_document(pdf_path: Union[str, Path], doc_id: str) -> Tuple[List[Section], Dict[str, int]]:
    raw = read_pdf_lines(pdf_path)
    lines, dropped_notes = [], 0
    for ln in raw:
        t = normalize_text(ln.text)
        if not t:
            continue
        if AMENDMENT_NOTE_RE.match(t):  # "(+++ Textnachweis ab: ... +++)"
            dropped_notes += 1
            continue
        lines.append(Line(ln.page, t, ln.bold, ln.size))
    n_pages = max((ln.page for ln in lines), default=0)
    lines = drop_headers_footers(lines)
    sections = build_sections(lines, doc_id)
    return sections, {"pages": n_pages, "lines": len(lines), "amendment_notes_dropped": dropped_notes}
