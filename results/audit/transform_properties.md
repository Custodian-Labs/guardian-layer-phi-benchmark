# What the transform preserves (E4a)

Unique masked spans (snap_sym): 5,273. Black-box measurements from paired documents.

## Repeated mentions

Original values masked at least twice in the same document: 478; all occurrences got the same surrogate: 459 (96.0%).

## Format: character-class pattern (letters → a, digits → 9)

| Group | Spans | Exact pattern kept | Collapsed pattern kept | x-masked |
|---|--:|--:|--:|--:|
| id | 631 | 86.8% | 97.0% | 1.0% |
| phone | 159 | 86.2% | 95.0% | 0.0% |
| email | 560 | 72.1% | 97.5% | 74.5% |
| date | 1,280 | 75.1% | 99.1% | 0.9% |
| age | 207 | 99.5% | 100.0% | 0.0% |
| name | 1,005 | 15.5% | 94.6% | 1.4% |
| location | 1,210 | 23.1% | 93.5% | 3.3% |
| other | 221 | 51.6% | 91.9% | 7.2% |

IDs + phones, exact pattern kept: 86.7%. Dates: 75.1%. E-mail x-masked: 74.5%.

## Dates

Numeric originals (d/m/y-like): 704. Surrogate is not a valid calendar date: 24 (3.4%); surrogate no longer a numeric date: 7 (1.0%).
Invalid examples: `2021-09-30` → `2022-09-31`; `28/02/1954` → `29/02/1955`; `31/07/2014` → `32/07/2015`; `31/03/2018` → `32/03/2019`; `30/11/1969` → `31/11/1970`; `30/06/1977` → `31/06/1978`
Not-a-date examples: `06/12/1950` → `7xxxxxxxx`; `05/08/2023` → `6xxxxxxx`; `04/22/2023` → `5xxxxxxxx`; `04/06/1981` → `5xxxxxxx`; `6/9/05` → `7xxxxx`; `25/09/1969` → `2xxxxxxxxx`

Date-shift policy (constant offset per document, interval preservation) is not documented by the vendor; it cannot be established from 1-2 dates per document and is listed as unknown.

### Date surrogates: numeric-field changes

| Change | Dates | Share |
|---|--:|--:|
| every field +0 or +1 | 877 | 68.5% |
| other change | 385 | 30.1% |
| different number of numeric fields | 12 | 0.9% |
| numbers unchanged | 6 | 0.5% |

By benchmark (every field +0 or +1 / all dates): ASQ-PHI 21/181; MEDDOCAN 542/577; PII en 89/159; PII nl 58/99; PII fr 85/137; PII de 82/127
Examples (+0/+1): `02/14/2022` → `3/14/2023`; `5/25/2023` → `6/25/2024`; `08/18/2023` → `9/18/2024`; `9/14/2020` → `10/14/2021`; `02/14/2024` → `3/14/2025`; `4/22/22` → `5/22/23`

**Privacy implication.** Where every numeric field moves by 0 or +1, the original date is among at most 2^k candidates (k numeric fields, typically 8) obtained by subtracting 0 or 1 from each field, so these surrogates offer little protection for the date value. This is a black-box observation on these corpora; the vendor's date policy is undocumented.

## x-masking by benchmark

| Benchmark | Spans | x-masked |
|---|--:|--:|
| ASQ-PHI | 534 | 5.2% |
| MEDDOCAN | 2,943 | 8.0% |
| MultiCoNER v2 | 20 | 10.0% |
| PII en | 480 | 9.8% |
| PII nl | 378 | 15.9% |
| PII fr | 520 | 16.5% |
| PII de | 398 | 11.3% |

Locale/language consistency: LLM-judge run queued on GPU 5 (scripts/locale_consistency.py → results/locale/).
Unknown vendor properties (not inferable from outputs): generator model family, date-shift policy, whether consistency is guaranteed or incidental, locale handling, training data.
