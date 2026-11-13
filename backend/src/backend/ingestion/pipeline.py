"""Glue: sources.csv -> extracted sections -> chunks (per strategy) -> jsonl."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ..config import settings
from .chunking import STRATEGIES, build_chunks
from .extract import extract_document
from .models import Chunk, Section


def load_sources(data_dir: Optional[Path] = None) -> List[dict]:
    data_dir = Path(data_dir or settings.data_dir)
    with open(data_dir / "sources.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def available_sources(only: Optional[Iterable[str]] = None, data_dir: Optional[Path] = None) -> List[Tuple[dict, Path]]:
    """Sources whose PDF exists in data/raw/. Prints what is missing."""
    data_dir = Path(data_dir or settings.data_dir)
    only = set(only) if only else None
    found, missing = [], []
    for row in load_sources(data_dir):
        if only and row["id"] not in only:
            continue
        pdf = data_dir / "raw" / row["filename"]
        (found if pdf.exists() else missing).append((row, pdf))
    for row, _ in missing:
        print(f"  missing PDF for {row['id']}: run `python data/download.py` or save {row['filename']} to data/raw/")
    return found


def process_document(
    row: dict, pdf: Path, strategies: Iterable[str], max_chars: int = 1500
) -> Tuple[List[Section], Dict[str, int], Dict[str, List[Chunk]]]:
    sections, info = extract_document(pdf, row["id"])
    chunks = {s: build_chunks(sections, row["id"], row["name"], s, max_chars) for s in strategies}
    return sections, info, chunks


def write_jsonl(path: Path, chunks: List[Chunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Chunk]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            d.pop("char_len", None)
            out.append(Chunk(**d))
    return out


def all_strategies(arg: str) -> List[str]:
    return list(STRATEGIES) if arg == "all" else [arg]
