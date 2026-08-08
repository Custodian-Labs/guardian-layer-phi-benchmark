# Surrogate-quality annotation guidelines (§7 error typology)

**Goal.** Independently quantify *why* the small masked-span recall residual
happens — i.e. whether the surrogates the transform emits are well-formed,
same-type values, or defective. This is the human-judged counterpart to the
automatic detector scores, and turns the §7 error typology from an assertion
into a measured failure-rate.

**What you're rating.** Each row is one *masked span*: the `original` PHI value,
the `surrogate` the transform put in its place, and a `context` window
(`⟦surrogate⟧` marks the span in the transformed text). You judge the surrogate.

## Workflow

1. Open `surrogate_quality_prefilled.csv`. The `failure`, `valid`, and
   `type_consistent` columns are **pre-filled by a conservative heuristic**
   (`scripts/prefill_annotation.py`) — treat them as *suggestions*, not truth.
2. For every row: read `original`, `surrogate`, `context`, and correct any
   wrong cell. Then set `reviewed = Y`.
3. **The heuristic only reliably catches `x_masked`.** It marks everything else
   `none`/`valid=Y`. Your main job is to find, among the `none` rows, the
   surrogates that are actually **truncated/garbled** or have **salience loss** —
   the heuristic cannot judge these.
4. Two raters should annotate independently; we report agreement (Cohen's κ)
   and the failure-rate. `git diff surrogate_quality_prefilled.csv` shows every
   cell you changed = the heuristic's error set.

## Columns

| column | values | meaning |
|---|---|---|
| `valid` | `Y` / `N` | Is the surrogate a **well-formed value a human reads as real PHI of this type**? `N` if malformed, fragmentary, x-masked, or empty. |
| `type_consistent` | `Y` / `N` | Does it read as the **same PHI type** as the original (a date for a date, a hospital for a hospital)? A well-formed value of the *wrong* type is `valid=Y, type_consistent=N`. |
| `failure` | see below | The single dominant failure mode (or `none`). |
| `reviewed` | `Y` | Set once you've checked the row. |
| `notes` | free text | Anything ambiguous; flag for adjudication. |

## `failure` modes (choose the single dominant one)

- **`none`** — well-formed, same-type, plausible surrogate. (`valid=Y`, `type_consistent=Y`.)
- **`x_masked`** — the generator leaked its own mask into the surrogate: a run of
  `x`s replaces characters. *Auto-detected reliably.*
  Example: `mariaeugenia.palacios@ephpo.es` → `mxxxxxxxxxxx.coronado@ephpo.es`;
  `St. Luke's Hospital` → `Saint. Lxxxxx Hospital`. → `valid=N`.
- **`truncated_garbled`** — the surrogate is a fragment or a non-word: cut off,
  mis-assembled, or not a real value of the type. **Stays superficially
  well-formed, so you must catch it, not the heuristic.**
  Example (paper): `Chicago` → `Illino` (a truncated non-city). → `valid=N`.
- **`salience_loss`** — the surrogate is well-formed and same-type but drops a
  distinctive property that made the original detectable: a famous/specific
  entity becomes a generic one. **Human judgment only.**
  Example (paper): `Cedars-Sinai` → `Vidant` (a real but far less salient
  hospital name). Usually `valid=Y`, `type_consistent=Y`, `failure=salience_loss`.
- **`other`** — a real defect that fits none of the above; explain in `notes`.

## Decision aids

- **Short ≠ truncated.** A shorter valid value is fine:
  `Wolverhampton Codsall` → `Walsall` is a good city surrogate → `none`.
- **Judge the surrogate, not the span boundary.** Some gold spans over-capture
  neighbouring text (e.g. an IP followed by `<br>`); rate the PHI value itself
  and note the boundary issue.
- **Wrong-type but well-formed** → `valid=Y`, `type_consistent=N` (not a
  `failure` unless it also loses salience/garbles).
- When torn between two failure modes, pick the one that best explains why a
  detector would *miss* it, and record the alternative in `notes`.

## Worked gold examples

| original | surrogate | valid | type_consistent | failure |
|---|---|:--:|:--:|---|
| `April 20th, 2024` | `March 19th, 2026` | Y | Y | none |
| `Wolverhampton Codsall` | `Walsall` | Y | Y | none |
| `mariaeugenia.palacios@ephpo.es` | `mxxxxxxxxxxx.coronado@ephpo.es` | N | Y | x_masked |
| `St. Luke's Hospital` | `Saint. Lxxxxx Hospital` | N | Y | x_masked |
| `Chicago` | `Illino` | N | Y | truncated_garbled |
| `Cedars-Sinai` | `Vidant` | Y | Y | salience_loss |

*After both raters finish, run the agreement/summary and fold the failure-rate
into §7; until then the paper states the typology qualitatively.*
