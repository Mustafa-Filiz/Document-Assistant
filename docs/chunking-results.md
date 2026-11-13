# Chunking results (generated)

max_chars=1500; fixed: 1000 chars, 150 overlap.

## Per document

| doc | pages | sections | fixed chunks | clause chunks | clause_plain chunks |
| --- | --- | --- | --- | --- | --- |
| baybo | 63 | 97 | 224 | 210 | 210 |
| geg | 79 | 124 | 295 | 273 | 273 |
| hoai | 119 | 75 | 318 | 275 | 275 |
| baugb | 136 | 293 | 587 | 539 | 539 |
| baunvo | 16 | 41 | 52 | 50 | 50 |
| vob_b | 36 | 45 | 86 | 81 | 81 |

## Structure statistics (lower % is better for every flag)

| strategy | chunks | mean_chars | median_chars | p95_chars | max_chars | split_within_clause_% | table_like_% | footnote_or_amendment_note_% | crosses_section_% | ref_out_of_order_% | starts_mid_sentence_% | ends_mid_sentence_% | too_short_% | too_long_% | bad_chars_% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed | 1562 | 993 | 995 | 999 | 999 | 0.0 | 5.4 | 0.5 | 38.3 | 0.3 | 59.2 | 94.7 | 0.0 | 0.0 | 0.0 |
| clause | 1428 | 908 | 958 | 1496 | 1500 | 29.6 | 3.6 | 0.5 | 0.1 | 0.4 | 4.1 | 13.9 | 4.3 | 0.0 | 0.0 |
| clause_plain | 1428 | 908 | 958 | 1496 | 1500 | 29.6 | 3.6 | 0.5 | 0.1 | 0.4 | 4.1 | 13.9 | 4.3 | 0.0 | 0.0 |

## Token lengths (intfloat/multilingual-e5-base)

| strategy | max tokens | model limit | chunks over limit (truncated!) |
| --- | --- | --- | --- |
| fixed | 519 | 512 | 0.1% |
| clause | 775 | 512 | 1.4% |
| clause_plain | 739 | 512 | 0.8% |

## Retrieval probe (20 questions, model intfloat/multilingual-e5-base)

| strategy | lang | n | hit@1 | hit@3 | hit@5 | MRR@10 |
| --- | --- | --- | --- | --- | --- | --- |
| fixed | all | 20 | 50% | 60% | 60% | 0.55 |
| fixed | de | 17 | 53% | 65% | 65% | 0.58 |
| fixed | tr | 3 | 33% | 33% | 33% | 0.37 |
| clause | all | 20 | 55% | 65% | 65% | 0.59 |
| clause | de | 17 | 53% | 65% | 65% | 0.58 |
| clause | tr | 3 | 67% | 67% | 67% | 0.67 |
| clause_plain | all | 20 | 40% | 55% | 55% | 0.47 |
| clause_plain | de | 17 | 41% | 59% | 59% | 0.50 |
| clause_plain | tr | 3 | 33% | 33% | 33% | 0.33 |

### Rank of the expected § per question

| id | lang | expected | fixed | clause | clause_plain |
| --- | --- | --- | --- | --- | --- |
| q01 | de | Art. 6 | 1 | 1 | 1 |
| q02 | de | Art. 2 | 1 | 1 | 1 |
| q03 | de | Art. 57 | 1 | 1 | 2 |
| q04 | de | Art. 31 | 1 | 1 | 1 |
| q05 | de | § 71 | >10 | >10 | >10 |
| q06 | de | § 80 | 1 | 3 | 3 |
| q07 | de | § 34 | 3 | 1 | 1 |
| q08 | de | § 5 | >10 | 6 | 7 |
| q09 | de | § 6 | 8 | >10 | 7 |
| q10 | de | § 34 | 1 | 1 | 1 |
| q11 | de | § 35 | 3 | 1 | >10 |
| q12 | de | § 14 | 1 | 3 | 3 |
| q13 | de | § 4 | 1 | 1 | 1 |
| q14 | de | § 19 | 1 | 1 | 1 |
| q15 | de | § 13 | >10 | >10 | >10 |
| q16 | de | § 12 | >10 | >10 | >10 |
| q17 | de | § 6 | >10 | >10 | >10 |
| q18 | tr | Art. 6 | 1 | 1 | 1 |
| q19 | tr | § 71 | >10 | >10 | >10 |
| q20 | tr | § 35 | 9 | 1 | >10 |
