# Embedding model choice

**Decision: `intfloat/multilingual-e5-base` as the default**, configurable through `EMBEDDING_MODEL`.
`BAAI/bge-m3` is the planned upgrade candidate, to be measured in week 5 with the probe set.

## Requirements

1. German legal text (compounds, long sentences, "§ 34 Abs. 2 Satz 1") must embed well.
2. Cross-lingual: I (and other non-native users) may ask in Turkish or English while the sources are German.
3. Runs on a laptop CPU inside Docker without a GPU and without a paid API.
4. Input limit must fit the chunks (~1500 chars, roughly 430 tokens).
5. Permissive license.

## Candidates (figures from the model cards - verify before relying on them)

| Model | Params | Dim | Max tokens | License | Verdict |
| --- | --- | --- | --- | --- | --- |
| `paraphrase-multilingual-MiniLM-L12-v2` | ~118M | 384 | 128 | Apache-2.0 | Rejected: 128-token limit would truncate most chunks silently. |
| `intfloat/multilingual-e5-small` | ~118M | 384 | 512 | MIT | Fallback if CPU speed becomes a problem. |
| **`intfloat/multilingual-e5-base`** | ~278M | 768 | 512 | MIT | **Chosen**: fits 512 tokens, multilingual, light enough for CPU. |
| `BAAI/bge-m3` | ~568M | 1024 | 8192 | MIT | Likely higher quality and allows bigger chunks, but ~2x slower/heavier on CPU. Test in week 5. |
| `jinaai/jina-embeddings-v2-base-de` | ~161M | 768 | 8192 | Apache-2.0 | German/English only (no Turkish); needs `trust_remote_code`. Worth a test if cross-lingual turns out unnecessary. |
| OpenAI `text-embedding-3-small` | API | 1536 | 8191 | proprietary | Rejected for now: costs money, sends text to a third party (fine for public law, but I want a reproducible offline baseline). |

## Why e5-base

- The 512-token limit matches the chunk size I chose (`max_chars=1500`); `compare --probe` reports how many chunks would be truncated, so this is checked, not assumed.
- It covers German, English and Turkish in one model, which keeps cross-lingual questions possible without a translation step.
- Two details that are easy to get wrong are handled in `Embedder`: e5 needs the `query: ` / `passage: ` prefixes, and vectors are L2-normalised so cosine similarity equals the dot product (the Chroma collection uses the cosine space).

## What is NOT proven

I have not benchmarked these models on this corpus. The choice is based on published properties and the constraints above.
The evidence will come from `python -m backend.ingestion.compare --probe --model <name>` (20 questions, hit@1/3/5, MRR), re-run for `multilingual-e5-small` and `bge-m3`. Vectors from different models cannot be mixed, so the store refuses to write into a collection built with another model (use `--reset`).
