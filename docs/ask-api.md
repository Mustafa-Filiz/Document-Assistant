# `/ask` API (Week 3)

`POST /ask` takes a question, retrieves the most similar chunks from the Chroma index (week 2),
and either returns an answer **with the sources it used** or says it does not know.

## Request / response

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "Wann darf im Außenbereich gebaut werden?", "k": 5}' | python -m json.tool
```

| Request field | Default | Meaning |
| --- | --- | --- |
| `question` | required (3-1000 chars) | any language; sources are German |
| `k` | 5 | chunks to retrieve (1-20) |
| `doc_id` | none | restrict to one document (`baybo`, `geg`, `hoai`, `baugb`, `baunvo`, `vob_b`) |
| `generate` | true | `false` = retrieval only, no LLM call (debugging) |

Response: `status`, `answer`, `reason`, `top_score`, `threshold`, `language`, `model`, and `sources[]`.
Every source has `reference` (e.g. `Bayerische Bauordnung (BayBO), Art. 6 Abstandsflächen, Abstände, Absatz (1), PDF-Seite 12`),
`doc_id`, `section_ref`, `section_title`, `absatz`, `page_start`/`page_end`, `score`, `text`,
`in_context` (was given to the LLM) and `cited` (the final answer cites it as `[n]`).
Pages are **PDF page numbers** (the page index in the downloaded file), not the printed page numbers.

## `status` and `reason`

| status | reason | what happened |
| --- | --- | --- |
| `answered` | none | LLM answered from the sources and cited at least one valid `[n]` |
| `no_answer` | `below_threshold` | best chunk scored under `MIN_SIMILARITY`; the LLM was **not called** |
| `no_answer` | `no_results` | nothing retrieved (e.g. unknown `doc_id`) |
| `no_answer` | `llm_declined` | LLM judged the sources insufficient and answered `NO_ANSWER` |
| `no_answer` | `uncited_answer` | LLM answered without a valid `[n]`; the text is discarded, not shown |
| `retrieval_only` | `no_llm_configured` / `generation_disabled` | no answer generated; sources returned |

"I don't know" is HTTP 200 (a valid result). 503 = index missing/empty/built with another embedding model;
502 = the LLM provider failed.

## Why two gates

1. **Similarity threshold** catches clearly out-of-scope questions cheaply (no LLM call).
2. **LLM gate + citation check** catches on-topic questions the documents cannot answer
   (trial question `t16`: the indexed `vob_b.pdf` is VOB Teil **A**, so the Teil B limitation period is not in it).
   No similarity threshold can separate those cases.

## Setup

```bash
cp .env.example .env            # set LLM_API_KEY (without it /ask is retrieval-only)
python data/download.py
cd backend && uv sync --extra ingest
uv run python -m backend.ingestion --strategy clause --index     # builds ../.chroma
uv run uvicorn backend.main:app --reload
```

With Docker: build the index on the host first (above), then `docker compose up --build`.
The image installs the `ingest` extra (torch etc.), so it is large; the first `/ask` downloads the embedding model
into the `hf_cache` volume and loads it lazily (first request is slow, later ones are fast).

## `MIN_SIMILARITY` is not calibrated yet

e5 cosine scores are compressed, so the default `0.80` is a **guess**. Measure it on your index:

```bash
uv run python -m backend.rag.calibrate
```

It prints the top-1 scores of answerable probe questions vs. out-of-scope questions and suggests a threshold.
With ~20 questions this is a rough estimate; Week 4's evaluation set should replace it.

## Terminal trial (task: 15-20 questions, record bad results)

```bash
uv run python -m backend.rag "Wann darf im Außenbereich gebaut werden?"
uv run python -m backend.rag --no-llm "Was regelt die Veränderungssperre?"
uv run python -m backend.rag --batch ../data/try_questions.csv --out ../docs/ask-trial-log.md
```

`data/try_questions.csv` has 20 questions: 12 German, 2 Turkish, 1 English in scope; 1 on-topic but unanswerable (`t16`);
4 out of scope. The batch run writes `docs/ask-trial-log.md` with automatic flags (over-refusal, answered-but-should-decline,
expected § not retrieved or not cited) and a verdict line per question. **The automatic flags are not a quality score:**
read each answer against the PDF page and fill in `good`/`bad` and the failure type (retrieval, chunking, LLM, threshold, data).

## Status of this implementation

- Written and unit-tested without network: retrieval ranking and filters, both gates, citation parsing, references,
  language of the refusal, LLM request/response handling, threshold suggestion, trial flags (`tests/test_rag.py`, 22 tests, run with fakes).
- **Not yet run:** the real embedding model, a real Chroma index, a real LLM call, the FastAPI endpoint tests
  (`tests/test_ask_api.py`, skipped where FastAPI is not installed), and the Docker build. No measured answer quality exists until
  you run the trial above.
