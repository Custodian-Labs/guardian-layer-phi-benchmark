# Internal memo — Guardian Layer transform: masking coverage

*Audience: Custodian engineering / product. Not for external sharing — the public report deliberately scopes its claim to utility preservation, which is solid. This memo is about the complementary question (how much gets masked) and where to improve it.*

## TL;DR

- **Utility preservation is solid** (covered in the public report): where the transform substitutes a value, detectors still find the surrogate 97–100% of the time; ranking unchanged.
- **Masking coverage is the gap.** Across the 7 benchmarks the transform altered **39%** of annotated PII overall; **54% on clinical sets** (ASQ-PHI + MEDDOCAN).
- The gap is **uneven by entity type** and — importantly — **mostly a replacement-step issue, not a detection-recall issue**: on a clinical sample, **~77% of the missed identifiers were flagged as sensitive by Custodian's own `analyze_proprietary`, but transform did not replace them.**

## Coverage by entity type (clinical: ASQ-PHI + MEDDOCAN)

| Entity type | Coverage | Read |
|---|--:|---|
| NAME / person | 75.0% | good |
| DATE / age | 73.4% | good |
| ID / contact (MRN, patient ID, phone, email) | **50.7%** | **weak — high-sensitivity identifiers** |
| ORG / facility | 46.6% | weak |
| LOCATION / address | **43.7%** | **weak** |

General-purpose NER/PII corpora (PII-Masking-300k, MultiCoNER) sit lower (~26%), largely because their annotated entities (encyclopedic names, generic places) fall outside Guardian Layer's notion of sensitive content — less of a concern.

## Concrete misses (clinical, left unmasked)

- Patient identifiers: `ID_SUJETO_ASISTENCIA` values like `80926`, `7845693`, `8597254`
- Care-contact IDs: `ID_CONTACTO_ASISTENCIAL` `4387684`, `2389567`
- Facilities / geography: `Hospital de Cruces`, `España`, postal codes `41005`, `28047`, `Ciudad Real`

These are explicit HIPAA Safe-Harbor identifiers — they should be masked.

## Diagnosis: detection vs replacement

On 8 MEDDOCAN docs (62 missed ID/location spans), checking each missed span against Custodian's own `analyze_proprietary` output:

- **Detected but not replaced: 48 / 62 (77%)**
- Not detected: 14 / 62 (23%)

**Interpretation:** the detection layer largely already recognizes these as sensitive; the transform step simply isn't acting on all flagged spans. That is a more tractable fix than improving detection recall — closing it is mostly about the replacement stage honoring everything `analyze` surfaces, especially numeric IDs and geographic/facility entities.

*Caveats:* small sample (8 docs), and "detected" was judged by loose token overlap with `sensitive_words`, so 77% is directional, not exact. A larger, exact-match pass would firm up the number.

## Recommended next steps (eng)

1. **Close the replacement gap on already-detected spans** — prioritize numeric IDs (MRN/patient/contact) and geographic/facility entities, where coverage is lowest and the detector already flags them.
2. **Audit the 23% not-detected** — sample and see whether they are systematic types (e.g., bare numeric IDs without context, postal codes) the detector could be tuned for.
3. **Re-measure coverage by type after the fix** — the per-type map above is the scorecard; target ≥90% on clinical ID/location.

## Method note

Coverage = fraction of gold PII spans whose characters changed between original and transformed text (difflib alignment). Config sweep confirmed `domain="General"` gives the highest coverage; `"Medical"`/`"Healthcare"` were worse or invalid. Utility (97–100%) is measured only on spans that were in fact transformed, so coverage and utility are independent measurements.
