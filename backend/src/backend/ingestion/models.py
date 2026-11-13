"""Data classes shared by the ingestion pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class Line:
    """One physical text line read from a PDF page."""

    page: int
    text: str
    bold: bool = False
    size: float = 0.0


@dataclass
class Atom:
    """Smallest structural unit inside a section: one Absatz, or an unnumbered paragraph."""

    text: str
    page_start: int
    page_end: int
    label: str = ""  # "(1)", "(2a)" ... or "" if the paragraph is unnumbered


@dataclass
class Section:
    """One § / Art. / Anlage with its body split into atoms."""

    doc_id: str
    ref: str  # "§ 34", "Art. 6", "Anlage 1"; "" for text before the first heading
    title: str
    hierarchy: List[str]  # e.g. ["Zweiter Teil Besondere Vorschriften", "Erster Abschnitt ..."]
    page_start: int
    page_end: int = 0
    atoms: List[Atom] = field(default_factory=list)

    @property
    def heading(self) -> str:
        return f"{self.ref} {self.title}".strip()

    @property
    def hierarchy_str(self) -> str:
        return " > ".join(self.hierarchy)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_name: str
    strategy: str
    text: str  # what the user will see / what gets cited
    embed_text: str  # what gets embedded (may carry a heading prefix)
    page_start: int
    page_end: int
    section_ref: str = ""
    section_title: str = ""
    hierarchy: str = ""
    absatz: str = ""
    refs_covered: str = ""  # "|"-joined refs of every section the chunk touches
    split_within_clause: bool = False  # True if one Absatz had to be cut into pieces

    @property
    def char_len(self) -> int:
        return len(self.text)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["char_len"] = self.char_len
        return d

    def to_metadata(self) -> Dict[str, Any]:
        """Flat, None-free metadata (the shape vector DBs accept)."""
        d = self.to_dict()
        d.pop("text")
        d.pop("embed_text")
        return d
