"""Build the sheet for hand-reviewing 20 chunks (task 5).

  python -m backend.ingestion.review --strategy clause --n 20 --seed 42

14 chunks are random (stratified over documents) -> use these to estimate the defect rate.
6 chunks are picked because a heuristic flagged them (table, footnote, ...) -> use these to
see HOW things break. Do not mix the two when you quote a defect rate.
Output: docs/chunk-review.md (fill in the checkboxes by reading each chunk against the PDF page).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from ..config import settings
from . import quality
from .chunking import STRATEGIES
from .models import Chunk
from .pipeline import read_jsonl

PRIORITY = ["table_like", "footnote_or_amendment_note", "ref_out_of_order", "starts_mid_sentence", "bad_chars", "too_long"]
DEFECTS = ["table broken", "footnote/amendment note mixed in", "heading drift (wrong § / title)",
           "cut mid-sentence or mid-clause", "page number wrong", "other"]


def sample(chunks_by_doc: Dict[str, List[Chunk]], n_total: int, n_flagged: int, seed: int, max_chars: int):
    rng = random.Random(seed)
    flags_of = {}
    for cs in chunks_by_doc.values():
        an = quality.ordering_anomalies(cs)
        for c in cs:
            flags_of[c.chunk_id] = quality.flags(c, max_chars, anomalies=an)

    chosen, ids = [], set()
    # flagged part: one chunk per flag type, cycling
    pools = {f: [c for cs in chunks_by_doc.values() for c in cs if f in flags_of[c.chunk_id]] for f in PRIORITY}
    for p in pools.values():
        rng.shuffle(p)
    while len([x for x in chosen if x[1] == "flagged"]) < n_flagged and any(pools.values()):
        for f in PRIORITY:
            while pools[f]:
                c = pools[f].pop()
                if c.chunk_id not in ids:
                    chosen.append((c, "flagged"))
                    ids.add(c.chunk_id)
                    break
            if len([x for x in chosen if x[1] == "flagged"]) >= n_flagged:
                break
    # random part: round-robin over documents
    shuffled = {d: rng.sample(cs, len(cs)) for d, cs in chunks_by_doc.items()}
    n_random = n_total - len(chosen)
    got = 0
    while got < n_random and any(shuffled.values()):
        for d in list(shuffled):
            while shuffled[d]:
                c = shuffled[d].pop()
                if c.chunk_id not in ids:
                    chosen.append((c, "random"))
                    ids.add(c.chunk_id)
                    got += 1
                    break
            if got >= n_random:
                break
    return chosen, flags_of


def render(chosen, flags_of, strategy: str, seed: int) -> str:
    out = [f"# Chunk review ({strategy}, seed {seed})\n",
           "Read each chunk next to the PDF page it cites. Tick what is wrong. "
           "`random` chunks estimate the defect rate; `flagged` chunks show failure modes.\n"]
    for i, (c, kind) in enumerate(chosen, 1):
        out.append(f"## {i}. `{c.chunk_id}` ({kind})")
        out.append(f"- {c.doc_name} | {c.section_ref} {c.section_title} | Absatz {c.absatz or '-'} | PDF page {c.page_start}-{c.page_end}")
        out.append(f"- hierarchy: {c.hierarchy or '-'}")
        out.append(f"- auto flags: {', '.join(flags_of[c.chunk_id]) or 'none'}")
        out.append("\n```text\n" + c.text + "\n```\n")
        out.extend(f"- [ ] {d}" for d in DEFECTS)
        out.append("- notes: \n")
    out.append("## Summary (fill in)\n\n| Defect | random chunks (of 14) | flagged chunks (of 6) |\n| --- | --- | --- |")
    out.extend(f"| {d} |  |  |" for d in DEFECTS)
    return "\n".join(out) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", default="clause", choices=STRATEGIES)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--flagged", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-chars", type=int, default=1500)
    ap.add_argument("--processed-dir", type=Path, default=settings.data_dir / "processed")
    ap.add_argument("--out", type=Path, default=settings.data_dir.parent / "docs" / "chunk-review.md")
    args = ap.parse_args(argv)

    files = sorted(args.processed_dir.glob(f"*.{args.strategy}.jsonl"))
    if not files:
        raise SystemExit(f"No {args.strategy} jsonl in {args.processed_dir}. Run `python -m backend.ingestion` first.")
    by_doc = {f.name.split(".")[0]: read_jsonl(f) for f in files}
    chosen, flags_of = sample(by_doc, args.n, args.flagged, args.seed, args.max_chars)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(chosen, flags_of, args.strategy, args.seed), encoding="utf-8")
    (args.processed_dir / "review_sample.json").write_text(json.dumps([c.chunk_id for c, _ in chosen]))
    print(f"Wrote {args.out} ({len(chosen)} chunks)")


if __name__ == "__main__":
    main()
