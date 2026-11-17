"""Pick MIN_SIMILARITY from data instead of guessing.

  python -m backend.rag.calibrate

positives  top-1 score of probe questions (data/probe_questions.csv) whose expected § is in the top-k
negatives  top-1 score of clearly out-of-scope questions (rows with expect=decline_oos in data/try_questions.csv)

Questions that are on-topic but unanswerable (expect=decline, e.g. VOB/B while the file is VOB/A)
are deliberately NOT negatives: no similarity threshold can separate them; the LLM gate must.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import List, Optional, Tuple

from ..config import settings
from .retriever import Retriever


def suggest_threshold(pos: List[float], neg: List[float]) -> Tuple[float, str]:
    """Threshold maximising (share of positives kept) + (share of negatives rejected) - 1.

    Ties go to the higher threshold (fewer wrong answers, more refusals).
    Returns (threshold, note about separability).
    """
    if not pos or not neg:
        raise ValueError("need at least one positive and one negative score")
    vals = sorted(set(pos + neg))
    cands = [(a + b) / 2 for a, b in zip(vals, vals[1:])] or [vals[0]]
    best_t, best_j = cands[0], -2.0
    for t in cands:
        j = sum(p >= t for p in pos) / len(pos) + sum(n < t for n in neg) / len(neg) - 1
        if j >= best_j:
            best_t, best_j = t, j
    note = "clean separation" if min(pos) > max(neg) else "scores OVERLAP: any threshold loses some answers or lets some junk through"
    return round(best_t, 3), note


def _fmt(xs: List[float]) -> str:
    return ", ".join(f"{x:.3f}" for x in sorted(xs))


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--probe", type=Path, default=settings.data_dir / "probe_questions.csv")
    ap.add_argument("--trial", type=Path, default=settings.data_dir / "try_questions.csv")
    args = ap.parse_args(argv)

    retr = Retriever.from_settings()
    pos: List[float] = []
    with open(args.probe, newline="", encoding="utf-8") as f:
        for q in csv.DictReader(f):
            hits_all = retr.search(q["question"], args.k)  # across all docs, like /ask
            if any(h.meta.get("doc_id") == q["doc_id"] and h.meta.get("section_ref") == q["expected_ref"]
                   for h in hits_all):
                pos.append(hits_all[0].score)
    neg: List[float] = []
    with open(args.trial, newline="", encoding="utf-8") as f:
        for q in csv.DictReader(f):
            if q.get("expect") == "decline_oos":
                neg.append(retr.search(q["question"], 1)[0].score)

    print(f"answerable   (n={len(pos)}): {_fmt(pos)}")
    print(f"out-of-scope (n={len(neg)}): {_fmt(neg)}")
    if len(pos) >= 3 and len(neg) >= 3:
        print(f"medians: answerable {statistics.median(pos):.3f}, out-of-scope {statistics.median(neg):.3f}")
        t, note = suggest_threshold(pos, neg)
        print(f"\nsuggested MIN_SIMILARITY={t}  ({note})")
        print("Set it in .env, then re-run the trial. With so few questions treat this as a rough estimate.")
    else:
        print("\nToo few scores to suggest a threshold (need >= 3 of each).")


if __name__ == "__main__":
    main()
