# Human surrogate annotation v2 (E1)

Rows: 197. Category: raw agreement 88.8%, Cohen's κ = 0.786. Boundary problem: raw 99.5%, κ = 0.000.

## Confusion (rater A rows × rater B columns)

| A \ B | well_formed | malformed_truncated | x_masked | type_inconsistent | locale_inconsistent | context_inconsistent | salience_change | other_uncertain |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| well_formed | 128 | 1 | 0 | 1 | 2 | 0 | 0 | 1 |
| malformed_truncated | 1 | 10 | 0 | 1 | 0 | 0 | 0 | 0 |
| x_masked | 0 | 0 | 18 | 0 | 0 | 0 | 0 | 0 |
| type_inconsistent | 1 | 0 | 0 | 4 | 0 | 0 | 1 | 0 |
| locale_inconsistent | 0 | 3 | 0 | 0 | 1 | 2 | 0 | 0 |
| context_inconsistent | 3 | 1 | 0 | 0 | 0 | 12 | 1 | 0 |
| salience_change | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| other_uncertain | 0 | 2 | 0 | 0 | 1 | 0 | 0 | 2 |

Rater A vs AI labels (audit subset, n = 197): raw 75.1%, κ = 0.467.

Rater B vs AI labels (audit subset, n = 197): raw 76.1%, κ = 0.485.

## Consensus categories

| Category | Audit subset | Loss-enriched |
|---|--:|--:|
| well_formed | 128 | 0 |
| malformed_truncated | 10 | 0 |
| x_masked | 18 | 0 |
| type_inconsistent | 4 | 0 |
| locale_inconsistent | 1 | 0 |
| context_inconsistent | 12 | 0 |
| salience_change | 0 | 0 |
| other_uncertain | 2 | 0 |
| disagreement | 22 | 0 |

## Loss given found on the original, by consensus category

| Sample | Category | Spans | Outcomes found on original | Lost | Loss rate | 95% CI (doc cluster) |
|---|---|--:|--:|--:|--:|--|
| audit200 | well_formed | 128 | 1058 | 54 | 5.1% | [3.7, 6.6] |
| audit200 | malformed_truncated | 10 | 69 | 5 | 7.2% | [0.0, 18.5] |
| audit200 | x_masked | 18 | 156 | 13 | 8.3% | [4.2, 13.0] |
| audit200 | type_inconsistent | 4 | 27 | 3 | 11.1% | [0.0, 30.0] |
| audit200 | locale_inconsistent | 1 | 10 | 1 | 10.0% | [10.0, 10.0] |
| audit200 | context_inconsistent | 12 | 75 | 7 | 9.3% | [1.4, 20.3] |
| audit200 | other_uncertain | 2 | 16 | 1 | 6.2% | [0.0, 12.5] |
| audit200 | disagreement | 22 | 169 | 18 | 10.7% | [5.3, 17.0] |

## Attribution of lost outcomes (both samples; loss-enriched rows dominate)

| Bucket | Lost outcomes | Share |
|---|--:|--:|
| detector sensitivity | 54 | 52.9% |
| generator defect | 29 | 28.4% |
| uncertain | 19 | 18.6% |
