# Construction Documents RAG Assistant

## Problem

Civil engineers constantly search long German regulations and specifications (building codes, contract terms, fee schedules) for one specific clause. Keyword search misses paraphrases, and general-purpose chatbots invent paragraph numbers. This project is a web app that answers questions about public German construction law, shows the exact source document and page for every answer, and says "I don't know" when the documents do not contain the answer.

## Status

Week 3 of 8: `POST /ask` (retrieval + answer generation with sources + "I don't know" gates) on top of the week-2 ingestion pipeline. Details and honest limits: [`docs/ask-api.md`](docs/ask-api.md). The UI comes in week 6.

## Quick start

```bash
cp .env.example .env
docker compose up --build
curl http://localhost:8000/health   # {"status":"ok"}
```

API docs: http://localhost:8000/docs

## Repository layout

| Folder      | Purpose                                                                                    |
| ----------- | ------------------------------------------------------------------------------------------ |
| `backend/`  | FastAPI service (ingestion, retrieval, `/ask`)                                             |
| `frontend/` | Next.js chat UI (week 6)                                                                   |
| `data/`     | Source list (`sources.csv`), downloader, probe + trial questions; PDFs land in `data/raw/` |
| `eval/`     | Question set and scoring scripts (week 4)                                                  |
| `docs/`     | Source table, architecture notes, experiment log                                           |

## Data sources

Only freely available, copyright-free documents are used. German statutes are "amtliche Werke" (§ 5 UrhG). DIN standards are paid and are **not** used. See [`docs/sources.md`](docs/sources.md).

```bash
python data/download.py   # downloads every row of data/sources.csv that has a direct_pdf_url
```

## Week 2: ingestion

```bash
python data/download.py                    # PDFs -> data/raw/
cd backend && uv sync --extra ingest       # PyMuPDF, Chroma, sentence-transformers
uv run pytest                              # pipeline tests on a synthetic PDF
uv run python -m backend.ingestion --strategy all          # extract + chunk -> data/processed/*.jsonl
uv run python -m backend.ingestion.compare --probe         # fixed vs clause chunking -> docs/chunking-results.md
uv run python -m backend.ingestion.review                  # 20-chunk review sheet -> docs/chunk-review.md
uv run python -m backend.ingestion --index --smoke-query "Wann darf im Außenbereich gebaut werden?"
```

| Piece                                              | Where                                          |
| -------------------------------------------------- | ---------------------------------------------- |
| Extraction (page + heading metadata)               | `backend/src/backend/ingestion/extract.py`     |
| Chunking strategies                                | `.../chunking.py`                              |
| Embedding model + rationale                        | `.../embeddings.py`, `docs/embedding-model.md` |
| Vector store (Chroma, one collection per strategy) | `.../store.py`                                 |
| Chunking experiments note                          | `docs/chunking-experiments.md`                 |

Before the first download, check `data/sources.csv`: the `geg` URL points to `GModG.pdf`, and the `vob_b` URL points to a file named `vob-teil-a-abschnitt-1-2019.pdf`. **That is VOB Teil A, not Teil B** (the week-2 chunk review shows § 7 Leistungsbeschreibung and § 15 Aufklärung des Angebotsinhalts, which are VOB/A). Questions about VOB/B (e.g. Mängelansprüche, Abnahme) cannot be answered until a real Teil B is added; trial question `t16` checks that the assistant refuses instead of inventing an answer.

## Week 3: ask

```bash
cp .env.example .env                       # set LLM_API_KEY; without it /ask is retrieval-only
cd backend
uv run python -m backend.ingestion --strategy clause --index      # the index must exist
uv run python -m backend.rag.calibrate                            # pick MIN_SIMILARITY (default 0.80 is a guess)
uv run uvicorn backend.main:app --reload
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "Wann darf im Außenbereich gebaut werden?"}'
uv run python -m backend.rag --batch ../data/try_questions.csv    # 20-question terminal trial -> docs/ask-trial-log.md
uv run pytest
```

| Piece                                                 | Where                                    |
| ----------------------------------------------------- | ---------------------------------------- |
| Retrieval (embed question, top-k from Chroma)         | `backend/src/backend/rag/retriever.py`   |
| Pipeline: similarity gate, LLM, citation check        | `.../rag/service.py`                     |
| Prompt ("sources only", `[n]` citations, `NO_ANSWER`) | `.../rag/prompts.py`                     |
| LLM clients (Anthropic, OpenAI-compatible)            | `.../rag/llm.py`                         |
| HTTP endpoint                                         | `.../rag/api.py`, `.../rag/schemas.py`   |
| Terminal trial + threshold calibration                | `.../rag/cli.py`, `.../rag/calibrate.py` |
| Trial questions                                       | `data/try_questions.csv`                 |
