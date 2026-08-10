# ML4H 2026 — submission checklist (Proceedings track)

Our `paper/ml4h/main.tex` compiles with a **vanilla `jmlr.cls` + a stubbed
`\mlhtrack`** so it builds offline. Before submitting, move the content into the
**official ML4H 2026 Overleaf template** and run the checks below.

> ⚠️ **Verify the volatile items against the official ML4H 2026 CFP / OpenReview
> page** — deadline, abstract-vs-paper deadlines, exact page limit, and any
> required statements can change year to year. Items marked **(confirm)** below.

---

## 0. Get the official template
- [ ] Grab the **ML4H 2026 Proceedings** LaTeX template (Overleaf link from the
      ML4H site / OpenReview). It is a PMLR (`jmlr`-based) class that **defines
      `\mlhtrack` and `\finalfalse` / `\finaltrue`** — the macros we stubbed.

## 1. Migrate the content
- [ ] Start from the official template's `main.tex`.
- [ ] **Delete our stub line** `\providecommand{\mlhtrack}[1]{}` (the template
      defines it). **Keep** `\mlhtrack{proceedings}`.
- [ ] Copy our **preamble extras** into the template preamble:
      packages `tikz` (+libraries `arrows.meta,positioning,fit,backgrounds,calc`),
      `pgfplots` (`\pgfplotsset{compat=1.16}`), `booktabs`, `calc`, `microtype`,
      **`graphicx` + `adjustbox`** (the wide tables are wrapped in
      `\begin{adjustbox}{max width=\columnwidth}…` so they shrink-to-fit the
      column instead of overflowing/overlapping — keep this when migrating);
      colors `cblue,caccent,cclay,cpaper,cline,cmuted,ckhaki,hgreen,hyellow,hred`;
      macros `\oldv \newv \hg \hy \hr \card \arr \tp \nk \vc` (drop `\pending`).
- [ ] Paste our body (everything from `\section{Introduction}` to end, incl.
      appendix + `\bibliography{refs}`) into the template's `document`.
- [ ] Copy `refs.bib` alongside. Bibliography = **43 refs**; template uses natbib
      → keep `\citep`/`\citet`. Rebuild: `pdflatex → bibtex → pdflatex ×2`.

## 2. Track / anonymity switches
- [ ] `\mlhtrack{proceedings}` (archival, ~8pp main body). *(confirm the exact
      limit and whether Findings/Proceedings both open this cycle.)*
- [ ] **Submission (anonymous): `\finalfalse`.** Leave authors as the template's
      anonymous default (our file already uses `\author{\Name{Anonymous Author(s)}}`).
- [ ] Camera-ready later: `\finaltrue` + real authors (see §5).

## 3. Page-limit check (redo in the official template)
- [ ] Main body **≤ 8 pages**; references **and** appendix are excluded/unlimited
      *(confirm)*. In our vanilla build the appendix starts on **p9** (main body
      ≤8pp) — the official template's margins differ, so **re-verify** after
      migration and trim if the body spills past p8.
- [ ] Quick check: label the first `\appendix` line and confirm its page ≥ 9.

## 4. Anonymization (submission) — already clean, re-confirm
- [x] No author names, affiliations, emails in `main.tex`.
- [x] **No URLs** (dashboard / GitHub / Colab withheld — abstract says "URLs
      withheld for anonymous review").
- [x] Vendor **not named** — "the commercial transform" throughout; `custodian_labs`
      is **not cited** in this version.
- [x] Self-citations (`bao2024amr`, `bao2024robustness`) are **neutral third-person
      `\citep{}`**, not "our prior work" — double-blind safe. *(If an AC prefers,
      swap to "Anonymous (2024)"; most venues accept neutral cites.)*
- [ ] Re-run the scan after migration:
      `grep -iE 'custodian|qiming|bao et al|sherry|feng|eugenio|meng fon|pages\.dev|github' main.tex`
      (only bibkey `bao2024*` should match).
- [ ] Strip PDF metadata (author/title) from the compiled PDF before upload.

## 5. Camera-ready additions (only if accepted) — do NOT add for submission
- [ ] `\finaltrue`; real authors: **Qiming Bao, Sherry J. H. Feng, Kim Chester
      Eugenio, Meng Fon** — **Custodian Labs**.
- [ ] Restore URLs: dashboard `https://custodianai.pages.dev`, code repo
      `https://github.com/Custodian-Labs/guardian-layer-phi-benchmark`, Colab.
- [ ] Name the commercial transform (Custodian Labs Guardian Layer) + re-add the
      `\citep{custodian_labs}` case-study citation (or the patent once it issues).
- [ ] **Acknowledgments / funding / COI** — note the authors are affiliated with
      the vendor of the evaluated transform (disclose the potential COI plainly).

## 6. Statements ML4H usually wants (confirm which are required)
- [ ] **Data availability**: all 7 benchmarks are public or synthetic
      (ASQ-PHI synthetic, MEDDOCAN shared-task, PII-Masking-300k synthetic,
      MultiCoNER v2 public); **no real patient data**; source licenses respected.
- [ ] **Ethics / broader impact**: dual-use note — the protocol audits whether a
      de-id transform preserves detectability; it does not itself de-identify.
- [ ] **Reproducibility statement**: scripts + 250-doc subsets released;
      C1/C2 use a 120-doc/benchmark subsample (Presidio full corpus); §7 typology
      is an **AI review**, not human-validated.
- [ ] **Limitations** — already a section; ML4H may want it flagged explicitly.

## 7. OpenReview upload
- [ ] Compile final PDF from the **official template** (not our vanilla build).
- [ ] Fill the submission form; attach code/supplement if allowed (anonymized
      repo link or zip). *(confirm supplementary policy.)*
- [ ] **Deadline: 2026-09-10** *(confirm exact date/time-zone; check for an
      earlier abstract-registration deadline.)*

## 8. Final build sanity
- [ ] `pdflatex → bibtex → pdflatex ×2`, zero undefined citations/refs.
- [ ] No overfull `\hbox` into the margin. Wide tables (C1/C2, equivalence,
      whole-doc F1, lost/agree) are `adjustbox`-capped to `\columnwidth`; the
      ml4h C1/C2 show **recall + khaki N only** (per-cell 95% CIs live in the
      arXiv version) so they fit the narrow jmlr column. Re-check visually in
      the official template.
- [ ] All 10 tables + figures render; cross-refs resolve.
