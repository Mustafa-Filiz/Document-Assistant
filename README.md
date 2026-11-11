# Construction Documents RAG Assistant

## Problem

Civil engineers constantly search long German regulations and specifications (building codes, contract terms, fee schedules) for one specific clause. Keyword search misses paraphrases, and general-purpose chatbots invent paragraph numbers. This project is a web app that answers questions about public German construction law, shows the exact source document and page for every answer, and says "I don't know" when the documents do not contain the answer.

## Status

Week 1 of 8: project skeleton, source list, empty API.

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
| `data/` | Source list (`sources.csv`) and downloader; PDFs land in `data/raw/` |
| `eval/` | Question set and scoring scripts (week 4) |
| `docs/` | Source table, architecture notes, experiment log |

## Data sources

Only freely available, copyright-free documents are used. German statutes are "amtliche Werke" (§ 5 UrhG). DIN standards are paid and are **not** used. See [`docs/sources.md`](docs/sources.md).

```bash
python data/download.py   # downloads every row of data/sources.csv that has a direct_pdf_url
```
