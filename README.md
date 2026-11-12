# Custodian Guardian Layer · PHI/PII Benchmark

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Custodian-Labs/guardian-layer-phi-benchmark/blob/main/notebooks/custodian_guardian_layer_demo.ipynb)

Evaluation harness for **Custodian Labs' Guardian Layer** `transform`, a structure-preserving de-identification
method that replaces each PHI value with a realistic same-type *surrogate*. The harness measures whether the
replaced text stays **detectable** by downstream PHI/PII systems. It is the companion code for the ML4H 2026
paper *"Surrogate Substitution Preserves PHI Detectability: A Multi-Detector Equivalence Study."*

**Detector panel (11):**
- rule-based: Microsoft Presidio;
- encoder tagger: OBI `deid_roberta_i2b2`;
- LLMs: Gemma 4 (31B, E4B), Qwen 3.5 (4B, 9B, 35B-A3B), Llama 3.1-8B, Llama 3.3-70B, DeepSeek V2-Lite and
  GPT-4o-mini.

They are evaluated on **7 benchmarks in 7 languages, 1,750 documents**.

### Links
| | |
|---|---|
| 📊 Interactive dashboard | **https://custodianlabs.pages.dev** |
| 📝 Paper | ML4H 2026, PMLR 346 (preprint: [arXiv:2608.03172](https://arxiv.org/abs/2608.03172)) |
| 💾 Reproducibility package (scripts + data subsets) | https://custodianlabs.pages.dev/code |
| ▶️ Run the demo in Google Colab (one click) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Custodian-Labs/guardian-layer-phi-benchmark/blob/main/notebooks/custodian_guardian_layer_demo.ipynb) |
| 📓 Notebook source | [`notebooks/custodian_guardian_layer_demo.ipynb`](notebooks/custodian_guardian_layer_demo.ipynb) |
| 🌍 Beyond medicine (finance, legal, enterprise) | https://custodianlabs.pages.dev/domains · [`docs/multidomain_results.md`](docs/multidomain_results.md) |
| 🎬 Video de-identification (faces, plates, on-screen text, speech) | https://custodianlabs.pages.dev/video · code and examples in [`custodian_video/`](custodian_video/) |

## Results

### Headline result
On the **5,273 gold PHI spans the transform replaces** (57,035 span × detector outcomes, 11 detectors,
1,142 documents), recall moves **75.9% → 74.7%**: Δ = −1.24 points, 90% document-level cluster-bootstrap CI
[−1.65, −0.84], posterior P(|Δ| < 2) > 0.99. The change is within ±2 points **pooled and under overlap
matching**; it is not uniform:

- Identifier-dense non-English benchmarks exceed ±2 (MEDDOCAN needs ±2.43, PII-nl ±3.08, PII-fr ±2.80).
- Only Gemma 4 31B and Qwen 3.5 9B are equivalent within ±2 on their own.
- With exact-boundary matching the loss is about 2 points.

Two-rater human labels of surrogates (Cohen's κ = 0.79), a same-slot counterfactual with Faker values and
detector test–retest show that the residual has both generator-side and detector-side causes. On ICD-10
diagnosis classification of the MEDDOCAN case reports (CodiEsp labels), accuracy changes by at most 1 point
under substitution, Faker surrogates or redaction.

### Detector comparison
| Detector | Whole-doc F1 orig → transf | Replaced-span recall orig → transf | Δ (pts) | 90% CI |
|---|--:|--:|--:|--:|
| Gemma 4 31B† | .744 → .735 | 91.9 → 91.9 | +0.08 | [−0.51, +0.70] |
| Llama 3.3 70B† | .725 → .719 | 90.6 → 89.2 | −1.40 | [−2.15, −0.64] |
| Qwen 3.5 35B-A3B | .715 → .693 | 86.6 → 85.0 | −1.56 | [−2.56, −0.57] |
| GPT-4o-mini | .706 → .696 | 89.3 → 87.4 | −1.92 | [−2.76, −1.10] |
| Gemma 4 E4B | .737 → .725 | 87.3 → 85.4 | −1.90 | [−2.73, −1.07] |
| Qwen 3.5 9B | .655 → .647 | 79.7 → 79.7 | +0.06 | [−0.88, +1.01] |
| Qwen 3.5 4B | .568 → .554 | 65.9 → 64.8 | −1.10 | [−2.72, +0.48] |
| Presidio | .415 → .414 | 74.9 → 72.9 | −2.01 | [−2.73, −1.30] |
| DeepSeek V2-Lite | .411 → .401 | 51.4 → 47.4 | −4.02 | [−5.69, −2.37] |
| Llama 3.1 8B | .392 → .392 | 47.4 → 50.2 | +2.84 | [+0.51, +5.27] |
| OBI `deid_roberta` | .041 → .042 | 73.3 → 70.6 | −2.69 | [−3.27, −2.10] |

*† incomplete run (Gemma 4 31B: 101/250 transformed ASQ-PHI documents; Llama 3.3 70B: parts of MEDDOCAN and
PII-nl). The panel's OpenAI detector is GPT-4o-mini (OpenAI API, temperature 0). GPT-5 itself was run on a stratified 199-document subset: whole-document F1 .787, the highest
of the panel on those documents (`results/openai_audit/gpt5_comparison.md`). OBI's low whole-document F1 comes from
boundary fragmentation, not missed PHI.*

### Redaction (C1) and open surrogates (C2)
Replaced-span recall (%), first 120 documents per benchmark, mean over the six benchmarks other than MultiCoNER:

| | Presidio | OBI | Qwen 3.5 9B | Gemma 4 31B |
|---|--:|--:|--:|--:|
| **C1, redaction** (`*****`) | 1.6 | 0.7 | 21.5 | 48.2 |
| **C1, Qwen 3.5 9B told that `*` runs mark removed PHI** | | | 71.1 | |
| **C2, Faker surrogates** | 76.9 | 81.6 | 83.3 | 93.6 |

The redaction floor applies only to detectors that are not adapted to the mask token. Faker values are
canonical and often easier to detect than real PHI, so Faker is not a neutral control.

**Beyond medicine.** The same protocol applied to finance, legal (ECHR / TAB) and enterprise text is described in
[`docs/multidomain_results.md`](docs/multidomain_results.md) and at https://custodianlabs.pages.dev/domains.

## Reproducing the paper

All camera-ready numbers are re-analyses of stored predictions, except the runs listed under "new inference".
See [`docs/camera_ready.md`](docs/camera_ready.md) for the full list of commands and outputs.

```bash
python -m unittest discover -s tests -v                          # RUN_DATA_TESTS=1 also checks the paper's denominators
python scripts/camera_ready_numbers.py                           # every number and table -> paper/ml4h/camera_ready/
python scripts/analyze_matching_margins.py                       # matching criteria, per-benchmark/detector margins, margin curve
python scripts/analyze_counterfactual.py --det presidio obi qwen3_5_9b   # same-slot counterfactual
python scripts/analyze_human_annotation.py data/annotation/human/surrogate_quality_audit197_rater1.csv \
       data/annotation/human/surrogate_quality_audit197_rater2.csv       # kappa + outcome linkage
python scripts/summarize_medical_decisions.py                    # ICD-10 downstream comparison
python scripts/audit_transform_properties.py                     # what the transform preserves
```

**Where the data are**
- Per-span predictions for the clinical and general-PII benchmarks (original and transformed, 11 detectors):
  `results/*_250*.jsonl` and `results/t_*_250*.jsonl`.
- Non-medical corpora: `data/domains/`, with per-document predictions in `results/domains/preds/`.
- Known gap: Gemma 4 31B's run on transformed ASQ-PHI covers 101 of the 250 documents.
- One synthetic API-key value in the Gretel finance data matches a SendGrid key pattern. It is replaced everywhere
  by a same-length placeholder `SG.EXAMPLE…`; offsets and results are unchanged.

## Data

**Benchmark data.** ASQ-PHI is from Weatherhead, Golovko & McCaffrey,
*ASQ-PHI: An adversarial synthetic data benchmark for clinical de-identification
and search utility*, **Data in Brief 65:112586, 2026**
([doi:10.1016/j.dib.2026.112586](https://doi.org/10.1016/j.dib.2026.112586)) —
please cite it if you use ASQ-PHI. Other benchmarks: MEDDOCAN, MultiCoNER v2,
PII-Masking-300k (see the paper for full citations).

ICD-10 labels for the MEDDOCAN documents come from CodiEsp (CC BY 4.0); see `data/external/codiesp/README.md`.

## Live dashboard

**https://custodianlabs.pages.dev** has three views:
1. the overall results matrix (systems × benchmarks);
2. a ranking proof: one document rendered by every system;
3. per-benchmark detail views.

## Project layout

```
guardian-layer-phi-benchmark/
├── benchmarks/        # Dataset loaders → Document(text, gold_spans)
│   ├── base.py
│   ├── asq_phi.py            # English clinical-queries synthetic, no DUA
│   ├── meddocan.py           # Spanish clinical synthetic, no DUA
│   ├── pii_masking_300k.py   # Multilingual general PII (HF, open) — EN + NL
│   └── multiconer_v2.py      # Fine-grained noisy NER, 12 langs (HF, open)
├── systems/           # Each model wraps a DeIDSystem.predict(text) → Prediction
│   ├── base.py
│   ├── custodian.py
│   ├── presidio.py
│   ├── obi_deid.py
│   ├── philter.py
│   ├── johnsnow.py
│   ├── llm_openai_compatible.py   # OpenAI / DeepSeek / Kimi / Qwen (hosted)
│   ├── llm_local_hf.py           # Local HF inference (Gemma, Qwen3.5-4B, ...)
│   └── llm_gemma.py
├── evaluation/        # Strict / type / relaxed span scoring + leakage rate
│   └── metrics.py
├── scripts/
│   └── run_benchmark.py
├── data/              # Local datasets (git-ignored, see data/README.md)
├── results/           # Per-doc JSONL + summary JSON
├── custodian_video/   # Local video de-identification (faces, plates, on-screen text, speech) + experiments
├── docs/              # Write-ups, incl. docs/camera_ready.md (paper analyses and commands)
├── paper/ml4h/camera_ready/   # Generated numbers and tables of the ML4H 2026 paper
├── tests/             # Unit tests for re-projection, matching, statistics, annotation
└── web/               # Static dashboard (see web/README.md)
```

## Updating the dashboard

A self-contained static dashboard lives in `web/`. After running benchmarks,
publish the numbers to the dashboard:

```bash
python scripts/publish_results.py
cd web && python -m http.server 8080      # local preview
# or `vercel deploy --prod` for a public URL
```

See [`web/README.md`](web/README.md) for deployment notes and privacy
caveats around publishing per-document samples.

## Quick start

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg
python -m spacy download es_core_news_lg     # MEDDOCAN / CARMEN-I

# 2. Configure secrets (copy template, fill in YOUR keys, NEVER commit)
cp .env.example .env
$EDITOR .env

# 3. Drop ASQ-PHI under data/asq_phi/raw/  (see data/README.md)

# 4. Run a sanity check
python scripts/run_benchmark.py --benchmark asq_phi --systems presidio --limit 20

# 5. Run the real comparison
python scripts/run_benchmark.py --benchmark asq_phi \
    --systems custodian_all presidio obi openai deepseek kimi qwen gemma \
    --include-text

# 6. Publish to the static dashboard
python scripts/publish_results.py
```

`custodian_all` automatically expands into one row per Custodian
configuration (MASKED × redact, MASKED × transform, PROPRIETARY × redact,
PROPRIETARY × transform). Pick a specific variant with
`--systems custodian:PROPRIETARY:transform`.

## Metrics

| Metric | Definition |
| --- | --- |
| Precision / Recall / F1 | Span-level with `--mode` ∈ {strict, type, relaxed} |
| Leakage rate | `1 - recall` (fraction of gold PHI not masked) |
| Char leakage rate | Fraction of gold PHI **characters** not masked (weights long spans heavier) |
| Over-redaction | Predicted spans not overlapping any gold span (= FP). On ASQ-PHI hard negatives this becomes a pure false-positive rate. |

Recall is the regulatory metric for HIPAA — a missed PHI span is a compliance
breach regardless of how clean the rest of the output is. Track it
independently of F1.

## Adding a new system

1. Subclass `systems.base.DeIDSystem`.
2. Implement `predict(text) -> Prediction` returning a list of
   `PredictedSpan(start, end, label, text, score)` and (optionally) a
   transformed text.
3. Register it in `scripts/run_benchmark.py::_build_system`.

## Adding a new benchmark

1. Subclass `benchmarks.base.Benchmark`.
2. Implement `__iter__` yielding `Document(doc_id, text, gold_spans)`.
3. Register in `benchmarks/__init__.py::REGISTRY`.

## Security notes

- `.env` is git-ignored. Never paste API keys into PRs, docs, or chat.
- Never commit DUA-protected text (`data/n2c2_*`, `data/mimic_iv_note/`,
  `data/carmen_i/`) — `.gitignore` covers them by default.
- Results may quote PHI verbatim in the `predictions[].spans[].text` field.
  Keep `results/` local-only when working with non-synthetic corpora.

## Citation

If you use this benchmark, the protocol or the data, please cite:

```bibtex
@InProceedings{bao2026surrogate,
  title     = {Surrogate Substitution Preserves {PHI} Detectability: A Multi-Detector Equivalence Study},
  author    = {Bao, Qiming and Feng, Jia Hui and Eugenio, Kim Chester V. and Foon, Meng},
  booktitle = {Proceedings of the Sixth Machine Learning for Health Symposium},
  series    = {Proceedings of Machine Learning Research},
  volume    = {346},
  publisher = {PMLR},
  year      = {2026}
}
```

Page numbers and the PMLR URL will be added once volume 346 is published. A preprint is available at
[arXiv:2608.03172](https://arxiv.org/abs/2608.03172).
