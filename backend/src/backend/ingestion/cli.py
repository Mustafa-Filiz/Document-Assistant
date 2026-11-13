"""Ingestion CLI.

  python -m backend.ingestion                       # extract + chunk (clause) -> data/processed/*.jsonl
  python -m backend.ingestion --strategy all        # all strategies
  python -m backend.ingestion --index               # also embed + write to Chroma
  python -m backend.ingestion --index --smoke-query "Wann ist ein Vorhaben im Außenbereich zulässig?"
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..config import settings
from . import quality
from .chunking import STRATEGIES
from .pipeline import all_strategies, available_sources, process_document, write_jsonl


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", nargs="*", help="source ids from sources.csv (default: all with a PDF)")
    ap.add_argument("--strategy", default="clause", choices=list(STRATEGIES) + ["all"])
    ap.add_argument("--max-chars", type=int, default=1500)
    ap.add_argument("--index", action="store_true", help="embed chunks and write them to Chroma")
    ap.add_argument("--model", default=settings.embedding_model)
    ap.add_argument("--reset", action="store_true", help="drop the Chroma collection(s) first")
    ap.add_argument("--smoke-query", help="after indexing, print the top hits for this question")
    ap.add_argument("--out-dir", type=Path, default=settings.data_dir / "processed")
    args = ap.parse_args(argv)

    strategies = all_strategies(args.strategy)
    sources = available_sources(args.docs)
    if not sources:
        raise SystemExit("No PDFs found in data/raw/. Run `python data/download.py` first.")

    per_strategy = {s: {} for s in strategies}  # strategy -> doc_id -> chunks
    for row, pdf in sources:
        sections, info, chunks = process_document(row, pdf, strategies, args.max_chars)
        print(f"\n{row['id']}: {info['pages']} pages, {len(sections)} sections "
              f"({info['amendment_notes_dropped']} amendment notes dropped)")
        for s, cs in chunks.items():
            write_jsonl(args.out_dir / f"{row['id']}.{s}.jsonl", cs)
            stats = quality.summarize(cs, args.max_chars)
            print(f"  {s:<13} {stats['chunks']:>5} chunks | mean {stats.get('mean_chars')} chars | "
                  f"table {stats.get('table_like_%')}% | footnote {stats.get('footnote_or_amendment_note_%')}% | "
                  f"out-of-order {stats.get('ref_out_of_order_%')}%")
            per_strategy[s][row["id"]] = cs
    print(f"\nWrote jsonl files to {args.out_dir}")

    if not args.index:
        return

    from .embeddings import Embedder
    from .store import get_client, get_collection, query, replace_document

    embedder = Embedder(args.model)
    print(f"\nEmbedding with {args.model} (max_seq_length={embedder.max_seq_length})")
    client = get_client()
    for s in strategies:
        col = get_collection(client, s, args.model, reset=args.reset)
        for doc_id, cs in per_strategy[s].items():
            emb = embedder.embed_passages([c.embed_text for c in cs], show_progress=True)
            replace_document(col, doc_id, cs, emb)
            print(f"  indexed {doc_id} [{s}]: {len(cs)} chunks")
        print(f"  collection legal_{s}: {col.count()} chunks total")

    if args.smoke_query:
        s = strategies[0]
        col = get_collection(client, s, args.model)
        hits = query(col, embedder.embed_queries([args.smoke_query])[0], k=3)
        print(f"\nTop hits for: {args.smoke_query}")
        for h in hits:
            m = h["meta"]
            print(f"  d={h['distance']:.3f}  {m['doc_name']} {m['section_ref']} {m['section_title']} "
                  f"(S. {m['page_start']}-{m['page_end']})\n     {h['text'][:160]!r}")


if __name__ == "__main__":
    main()
