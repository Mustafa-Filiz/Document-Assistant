"""Compare chunking strategies (task 2).

  python -m backend.ingestion.compare                 # structural statistics only (no model needed)
  python -m backend.ingestion.compare --probe         # + retrieval probe (hit@k) with the embedding model

The probe embeds data/probe_questions.csv and checks whether the expected § / Art. appears in the
top-k chunks, searching across ALL documents (like the real /ask will). Output: docs/chunking-results.md
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from ..config import settings
from . import quality
from .chunking import STRATEGIES
from .models import Chunk
from .pipeline import available_sources, process_document

KS = (1, 3, 5)


def md_table(headers: List[str], rows: List[List]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def run_probe(chunks_by_strategy: Dict[str, List[Chunk]], questions: List[dict], embedder):
    import numpy as np

    q_emb = np.asarray(embedder.embed_queries([q["question"] for q in questions]))
    res = {}
    for strat, chunks in chunks_by_strategy.items():
        emb = np.asarray(embedder.embed_passages([c.embed_text for c in chunks], show_progress=True))
        sims = q_emb @ emb.T
        ranks = []
        for qi, q in enumerate(questions):
            order = np.argsort(-sims[qi])[:10]
            rank = None
            for r, ci in enumerate(order, 1):
                c = chunks[ci]
                if c.doc_id == q["doc_id"] and q["expected_ref"] in c.refs_covered.split("|"):
                    rank = r
                    break
            ranks.append(rank)
        res[strat] = ranks
    return res


def probe_rows(res, questions):
    rows = []
    for strat, ranks in res.items():
        for label, idx in [("all", range(len(questions)))] + [
            (lang, [i for i, q in enumerate(questions) if q["lang"] == lang])
            for lang in sorted({q["lang"] for q in questions})
        ]:
            idx = list(idx)
            if not idx:
                continue
            hits = [round(100 * sum(1 for i in idx if ranks[i] and ranks[i] <= k) / len(idx)) for k in KS]
            mrr = sum(1 / ranks[i] for i in idx if ranks[i]) / len(idx)
            rows.append([strat, label, len(idx)] + [f"{h}%" for h in hits] + [f"{mrr:.2f}"])
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", nargs="*")
    ap.add_argument("--strategies", nargs="*", default=list(STRATEGIES), choices=STRATEGIES)
    ap.add_argument("--max-chars", type=int, default=1500)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--model", default=settings.embedding_model)
    ap.add_argument("--questions", type=Path, default=settings.data_dir / "probe_questions.csv")
    ap.add_argument("--out", type=Path, default=settings.data_dir.parent / "docs" / "chunking-results.md")
    args = ap.parse_args(argv)

    sources = available_sources(args.docs)
    if not sources:
        raise SystemExit("No PDFs in data/raw/.")
    by_strategy: Dict[str, List[Chunk]] = {s: [] for s in args.strategies}
    per_doc = []
    for row, pdf in sources:
        sections, info, chunks = process_document(row, pdf, args.strategies, args.max_chars)
        for s, cs in chunks.items():
            by_strategy[s].extend(cs)
        per_doc.append([row["id"], info["pages"], len(sections)] + [len(chunks[s]) for s in args.strategies])

    out = [f"# Chunking results (generated)\n\nmax_chars={args.max_chars}; fixed: 1000 chars, 150 overlap.\n"]
    out.append("## Per document\n\n" + md_table(["doc", "pages", "sections"] + [f"{s} chunks" for s in args.strategies], per_doc))

    keys = None
    rows = []
    for s in args.strategies:
        # stats per doc then pooled: chunks are in doc order per doc, so compute on the pooled list
        st = quality.summarize(by_strategy[s], args.max_chars)
        keys = keys or list(st)
        rows.append([s] + [st[k] for k in keys])
    out.append("\n## Structure statistics (lower % is better for every flag)\n\n" + md_table(["strategy"] + keys, rows))

    if args.probe:
        from .embeddings import Embedder

        with open(args.questions, newline="", encoding="utf-8") as f:
            have = {r["id"] for r, _ in sources}
            questions = [q for q in csv.DictReader(f) if q["doc_id"] in have]
        embedder = Embedder(args.model)
        # token-limit check with the real tokenizer
        tok_rows = []
        for s in args.strategies:
            lens = embedder.count_tokens([c.embed_text for c in by_strategy[s]])
            over = round(100 * sum(1 for n in lens if n > embedder.max_seq_length) / len(lens), 1)
            tok_rows.append([s, max(lens), embedder.max_seq_length, f"{over}%"])
        out.append(f"\n## Token lengths ({args.model})\n\n" + md_table(["strategy", "max tokens", "model limit", "chunks over limit (truncated!)"], tok_rows))
        res = run_probe(by_strategy, questions, embedder)
        out.append(f"\n## Retrieval probe ({len(questions)} questions, model {args.model})\n\n"
                   + md_table(["strategy", "lang", "n", "hit@1", "hit@3", "hit@5", "MRR@10"], probe_rows(res, questions)))
        miss = [[q["id"], q["lang"], q["expected_ref"]] + [res[s][i] or ">10" for s in args.strategies]
                for i, q in enumerate(questions)]
        out.append("\n### Rank of the expected § per question\n\n" + md_table(["id", "lang", "expected"] + list(args.strategies), miss))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
