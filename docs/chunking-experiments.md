# Chunking experiments (Week 2)

> Status: **method and tooling are done; the numbers must be generated on the real PDFs.**
> The sandbox this was built in had no network, so nothing below is a measured result on BayBO/GEG/etc.
> Run the commands, paste the generated tables into "Results", and write your conclusions under "Decision".

## Question

Does splitting by legal structure (§ / Art. → Absatz) retrieve better than cutting fixed-size windows?
Statutes are written to be cited by § and Absatz, and the project promises to show the exact source, so a chunk that straddles two paragraphs is a worse citation even if the embedding is similar.

## Strategies

| Name | How it cuts | What gets embedded | Metadata |
| --- | --- | --- | --- |
| `fixed` | sliding window, 1000 chars, 150 overlap, snapped to whitespace | raw chunk text | section/page looked up by position; `refs_covered` lists every § the window touches |
| `clause` | split at § / Art. / Anlage, then pack whole Absätze up to 1500 chars; an over-long Absatz is cut at sentence boundaries (`split_within_clause=true`) | `Doc | Teil > Abschnitt | § 34 Titel` + text | exact § , title, hierarchy, Absatz range, pages |
| `clause_plain` | identical chunks to `clause` | raw text, no heading prefix | same as `clause` |

`clause` vs `clause_plain` isolates the effect of the heading prefix. `fixed` vs `clause_plain` isolates the effect of the cut points.

## Extraction choices that affect every strategy

- Running headers/footers: removed when an edge line repeats (digits normalised) on ≥30% of pages; bare page numbers are removed.
- Table of contents pages (≥8 heading-like lines and ≥35% of the page) are dropped, otherwise every § would exist once as an empty "heading" chunk.
- Wrapped in-text references ("§ 34 Absatz 1 gilt ...") are not headings: rejected by first-word list; if the PDF has reliable bold headings, bold is required.
- Hyphenation at line end is repaired ("bau-" + "rechtliche"), except before "und/oder/bis".
- `(+++ Textnachweis ... +++)` amendment notes are dropped; "(weggefallen)" Absätze are dropped.
- Short numeric lines are treated as table rows and kept on separate lines.
- Page range of a chunk = pages of the Absätze it contains (a cut piece inherits its Absatz's pages).

## How to reproduce

```bash
python data/download.py                         # fetch PDFs (check the URL notes in README first)
cd backend && uv sync --extra ingest
uv run python -m backend.ingestion --strategy all
uv run python -m backend.ingestion.compare --probe        # writes docs/chunking-results.md
uv run python -m backend.ingestion.review --strategy clause   # writes docs/chunk-review.md
```

Metrics (all "lower is better" except hit@k / MRR): chunk count and length distribution, `crosses_section`, `starts_mid_sentence`, `ends_mid_sentence`, `table_like`, `footnote_or_amendment_note`, `ref_out_of_order` (heading drift: § numbers must increase), `too_short`/`too_long`, chunks over the model's token limit, and retrieval hit@1/3/5 + MRR@10 on `data/probe_questions.csv`.

Probe questions and expected § were written from my knowledge of these laws, not from the downloaded files. Check `expected_ref` against the PDFs before trusting a miss, and remember a hit means "the right § was retrieved", not "the answer is right".

## Predictions to confirm or reject (written before seeing data)

1. `fixed` will show many `crosses_section` and `starts_mid_sentence` chunks and a worse hit@1 because answers are split across windows.
2. `clause` will have a long tail of very short chunks (one-line §§) and a few over-long ones; very short chunks may rank high on keyword overlap but carry little context.
3. The heading prefix (`clause` vs `clause_plain`) will help most for generic questions ("Wann ist eine Baugenehmigung erforderlich?") and for Turkish queries.
4. Tables (HOAI Anlagen, GEG Anlagen, BauNVO tables) will break under every strategy; that is a separate fix, not a chunking-strategy problem.

## Known failure modes to look for in the manual review

Table rows glued into prose (the synthetic test already shows a table header row merging into the preceding Absatz), footnote/amendment notes inside a clause, heading drift after a wrapped in-text reference, § headings that wrap onto two lines, Anlagen without § headings, and two-column or sidebar layouts in the PDF.

## Results

_Paste `docs/chunking-results.md` here after running._

## Manual review of 20 chunks

_Fill in from `docs/chunk-review.md`: list the broken chunks and the defect type, and quote the defect rate from the 14 random chunks only (the other 6 are chosen because a heuristic flagged them)._

## Decision

_Pending results. Default until then: `clause` with `max_chars=1500`._
