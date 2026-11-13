# Construction Documents RAG Assistant

## Problem

Civil engineers constantly search long German regulations and specifications (building codes, contract terms, fee schedules) for one specific clause. Keyword search misses paraphrases, and general-purpose chatbots invent paragraph numbers. This project is a web app that answers questions about public German construction law, shows the exact source document and page for every answer, and says "I don't know" when the documents do not contain the answer.

## Status

Week 2 of 8: ingestion pipeline (PDF -> sections -> chunks -> embeddings -> Chroma). The API itself is still empty (`/ask` comes in week 3).

## Quick start

```bash
cp .env.example .env
docker compose up --build
curl http://localhost:8000/health   # {"status":"ok"}
```

API docs: http://localhost:8000/docs

## Repository layout

| Folder | Purpose |
| --- | --- |
| `backend/` | FastAPI service (ingestion, retrieval, `/ask`) |
| `frontend/` | Next.js chat UI (week 6) |
| `data/` | Source list (`sources.csv`), downloader, probe questions; PDFs land in `data/raw/` |
| `eval/` | Question set and scoring scripts (week 4) |
| `docs/` | Source table, architecture notes, experiment log |

## Data sources

Only freely available, copyright-free documents are used. German statutes are "amtliche Werke" (§ 5 UrhG). DIN standards are paid and are **not** used. See [`docs/sources.md`](docs/sources.md).

```bash
python data/download.py   # downloads every row of data/sources.csv that has a direct_pdf_url
```

## Week 2: ingestion

```bash
python data/download.py                    # PDFs -> data/raw/
cd backend && uv sync --extra ingest       # PyMuPDF, Chroma, sentence-transformers (kept out of the API image)
uv run pytest                              # pipeline tests on a synthetic PDF
uv run python -m backend.ingestion --strategy all          # extract + chunk -> data/processed/*.jsonl
uv run python -m backend.ingestion.compare --probe         # fixed vs clause chunking -> docs/chunking-results.md
uv run python -m backend.ingestion.review                  # 20-chunk review sheet -> docs/chunk-review.md
uv run python -m backend.ingestion --index --smoke-query "Wann darf im Außenbereich gebaut werden?"
```

| Piece | Where |
| --- | --- |
| Extraction (page + heading metadata) | `backend/src/backend/ingestion/extract.py` |
| Chunking strategies | `.../chunking.py` |
| Embedding model + rationale | `.../embeddings.py`, `docs/embedding-model.md` |
| Vector store (Chroma, one collection per strategy) | `.../store.py` |
| Chunking experiments note | `docs/chunking-experiments.md` |

Before the first download, check `data/sources.csv`: the `geg` URL points to `GModG.pdf` and the `vob_b` URL to a file named `vob-teil-a-abschnitt-1-2019.pdf` (that is Teil A, not Teil B). Open each PDF once to confirm it is the document you expect.
