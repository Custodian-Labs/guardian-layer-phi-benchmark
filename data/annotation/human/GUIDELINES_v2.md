# Surrogate annotation v2: guidelines for the two raters

**Goal.** Validate the surrogate-quality typology by human judgement. Two raters label the same 397 rows **independently**. Do not discuss rows or share files until both are finished.

## Setup

1. Each rater copies `surrogate_quality_v2_blind.csv` to `surrogate_quality_v2_<initials>.csv` and labels only that copy.
2. Do not open `surrogate_quality_v2_key.csv`, `../surrogate_quality_audit.csv` (the AI labels), or any results files.
3. Return your file. The analysis then runs:

```
python scripts/analyze_human_annotation.py data/annotation/human/surrogate_quality_v2_<A>.csv data/annotation/human/surrogate_quality_v2_<B>.csv
```

## What each row shows

| Column | Meaning |
|---|---|
| `original` | The real value in the original document |
| `surrogate` | What the transform put in its place |
| `context_original` | Text around the original value, which is marked `⟦…⟧` |
| `context_transformed` | Text around the surrogate, which is marked `⟦…⟧` |
| `type` | The gold annotation label of the original span |

Judge the **surrogate as a replacement for the original in this context**. You are not judging whether a detector would find it.

## `category` (choose one)

| Value | Use when |
|---|---|
| `well_formed` | A plausible, well-formed value of the same type that fits the document's language and context. |
| `malformed_truncated` | A fragment, a non-word, cut off, or garbled (`Chicago` → `Illino`), or an impossible value (`31/06/1978`, `december 61e`). |
| `x_masked` | Characters replaced by a run of `x` (`nxxxxxx@…`, `2xxxxxxxx`). |
| `type_inconsistent` | Well formed, but a different kind of entity (a date becomes a name; a country becomes a nationality, `Colombia` → `Colombian`). |
| `locale_inconsistent` | Same type and well formed, but implausible for the document's language or region (an Italian given name in a German form; a US-style phone number in a Dutch record). |
| `context_inconsistent` | Well formed and same type, but it contradicts the clinical or document context: wrong sex for the patient (a male name for a pregnant patient), an age that contradicts the stated history, a date before birth, a city that contradicts the stated country. |
| `salience_change` | Well formed and consistent, but a famous or distinctive entity has become an obscure one (`Cedars-Sinai` → `Vidant`). |
| `other_uncertain` | A defect that fits none of the above, or you cannot decide. Explain in `notes`. |

## Other columns

| Column | What to enter |
|---|---|
| `secondary_category` | Optional. Use it when a second category clearly also applies. |
| `boundary_problem` | `Y` if the marked span looks cut or over-extended (it includes or misses neighbouring characters), otherwise `N`. Judge the value itself, then flag the boundary here. |
| `confidence` | `high`, `medium` or `low`. |

## Decision rules

- **Shorter is not truncated.** `Wolverhampton Codsall` → `Walsall` is `well_formed`.
- **When two categories fit,** pick the one that most changes the meaning or plausibility of the document. Put the other in `secondary_category`.
- **Mechanical changes are fine for this rating.** A date that differs only by +1 in some fields (`02/14/2022` → `3/14/2023`) is `well_formed` if it is a valid date. Its privacy weakness is assessed separately.
- **Look at the context columns.** Most `context_inconsistent` and `locale_inconsistent` cases are only visible there.
