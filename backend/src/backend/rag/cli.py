"""Ask from the terminal (task: try 15-20 questions, record the bad results).

  python -m backend.rag "Wann darf im Außenbereich gebaut werden?"
  python -m backend.rag --no-llm "Was regelt die Veränderungssperre?"      # retrieval only
  python -m backend.rag --batch data/try_questions.csv --out docs/ask-trial-log.md
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from ..config import settings
from .service import AskResult, AskService, build_service


def check_flags(row: dict, res: AskResult) -> List[str]:
    """Objective problems only; whether the answer text is actually right needs a human."""
    flags: List[str] = []
    expect = (row.get("expect") or "answer").strip()
    if expect.startswith("decline") and res.status == "answered":
        flags.append("ANSWERED-BUT-SHOULD-DECLINE (hallucination risk)")
    if expect == "answer" and res.status == "no_answer":
        flags.append(f"OVER-REFUSAL ({res.reason})")
    doc, ref = (row.get("expected_doc") or "").strip(), (row.get("expected_ref") or "").strip()
    if expect == "answer" and doc and ref:
        rank = next((s.n for s in res.sources if s.doc_id == doc and s.section_ref == ref), None)
        if rank is None:
            flags.append(f"RETRIEVAL-MISS (expected {doc} {ref} not in top {len(res.sources)})")
        elif rank > 1:
            flags.append(f"RETRIEVAL-RANK {rank} (expected {doc} {ref})")
        elif res.status == "answered" and not next(s for s in res.sources if s.n == rank).cited:
            flags.append("EXPECTED-SOURCE-NOT-CITED")
    return flags


def print_result(res: AskResult) -> None:
    print(f"\nQ: {res.question}")
    print(f"status={res.status} reason={res.reason} top_score={res.top_score} "
          f"threshold={res.threshold} lang={res.language} model={res.model}")
    if res.answer:
        print(f"\n{res.answer}\n")
    for s in res.sources:
        mark = ("*" if s.cited else " ") + ("" if s.in_context else "~")
        print(f" [{s.n}]{mark} {s.score:.3f}  {s.reference}")
    print(" (* cited in the answer, ~ below threshold: not given to the LLM)")


def render_log(rows: List[dict], results: List[AskResult], svc: AskService) -> str:
    out = [
        "# /ask terminal trial\n",
        f"model: {getattr(svc.llm, 'model', None) or 'none (retrieval only)'}; "
        f"embedding: {settings.embedding_model}; strategy: {svc.retriever.strategy}; "
        f"k={svc.default_k}; MIN_SIMILARITY={svc.min_similarity}\n",
        "Automatic flags only catch objective problems (refusal vs. expectation, expected § not retrieved). "
        "**Read every answer against the PDF page** and fill in the verdict lines; that is the real result.\n",
        "| id | lang | expect | status | top score | auto flags |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    all_flags = [check_flags(r, res) for r, res in zip(rows, results)]
    for r, res, fl in zip(rows, results, all_flags):
        out.append(f"| {r['id']} | {r.get('lang', '')} | {r.get('expect', '')} | {res.status} | "
                   f"{res.top_score} | {'; '.join(fl) or '-'} |")
    out.append("")
    for r, res, fl in zip(rows, results, all_flags):
        out.append(f"## {r['id']}: {res.question}")
        if r.get("note"):
            out.append(f"- note: {r['note']}")
        out.append(f"- expect: {r.get('expect')} {r.get('expected_doc', '')} {r.get('expected_ref', '')}".rstrip())
        out.append(f"- result: **{res.status}** ({res.reason or 'ok'}), top score {res.top_score}")
        out.append(f"- auto flags: {'; '.join(fl) or 'none'}")
        if res.answer:
            out.append("\n```text\n" + res.answer + "\n```")
        out.append("\nSources (`*` cited, `~` below threshold):")
        for s in res.sources:
            mark = ("*" if s.cited else "") + ("" if s.in_context else "~")
            out.append(f"- [{s.n}]{mark} {s.score:.3f} {s.reference}")
        out.append("\n- verdict: [ ] good  [ ] bad    - failure type (retrieval / chunking / LLM / threshold / data): \n- notes: \n")
    return "\n".join(out) + "\n"


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="?")
    ap.add_argument("--batch", type=Path, help="CSV with columns id,lang,question,expect,expected_doc,expected_ref,note")
    ap.add_argument("--out", type=Path, default=settings.data_dir.parent / "docs" / "ask-trial-log.md")
    ap.add_argument("--k", type=int, default=settings.top_k)
    ap.add_argument("--doc", help="restrict to one doc id, e.g. baybo")
    ap.add_argument("--no-llm", action="store_true", help="retrieval only")
    ap.add_argument("--min-similarity", type=float, default=None)
    ap.add_argument("--json", action="store_true", help="print the raw JSON result (single question)")
    args = ap.parse_args(argv)
    if not args.question and not args.batch:
        ap.error("give a question or --batch FILE")

    svc = build_service(min_similarity=args.min_similarity, top_k=args.k)
    gen = not args.no_llm
    if gen and svc.llm is None:
        print("NOTE: LLM_API_KEY is not set -> retrieval only (no answers are generated).")

    if args.question:
        res = svc.ask(args.question, k=args.k, doc_id=args.doc, generate=gen)
        print(json.dumps(asdict(res), ensure_ascii=False, indent=2)) if args.json else print_result(res)
        return

    with open(args.batch, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    results: List[AskResult] = []
    for r in rows:
        res = svc.ask(r["question"], k=args.k, doc_id=args.doc, generate=gen)
        results.append(res)
        flags = check_flags(r, res)
        print(f"{r['id']:<4} {res.status:<14} top={res.top_score}  {'; '.join(flags) or ''}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_log(rows, results, svc), encoding="utf-8")
    print(f"\nWrote {args.out}: now read each answer against the PDF and fill in the verdicts.")


if __name__ == "__main__":
    main()
