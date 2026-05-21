# Data directory

Nothing under this folder is checked into git (see top-level `.gitignore`).
Each benchmark has its own subfolder.

## Immediately accessible (no DUA)

### ASQ-PHI

- URL: https://doi.org/10.17632/csz5dzp7nx.1 (Mendeley Data, MIT license)
- Paper: https://www.sciencedirect.com/science/article/pii/S2352340926001393
- Target layout:

```
data/asq_phi/raw/asq_phi.jsonl
```

The Mendeley archive ships a JSON (or CSV) of 1,051 queries; if the field
names diverge from what `benchmarks/asq_phi.py` expects, adjust
`_parse_row()` accordingly.

### MEDDOCAN

- URL: https://zenodo.org/records/4279323 (CC license, public)
- Target layout (BRAT):

```
data/meddocan/raw/train/brat/*.txt   *.ann
data/meddocan/raw/dev/brat/*.txt     *.ann
data/meddocan/raw/test/brat/*.txt    *.ann
```

## DUA-required (submit early — review takes 1–4 weeks)

| Dataset | Application URL |
| --- | --- |
| n2c2 2014 De-ID Track 1 | https://portal.dbmi.hms.harvard.edu/projects/n2c2-2014/ |
| n2c2 2014 Heart Disease Track 2 | same as above |
| n2c2 2006 De-ID + Smoking | https://portal.dbmi.hms.harvard.edu/projects/n2c2-2006/ |
| MIMIC-IV-Note (v2.2) | https://physionet.org/content/mimic-iv-note/2.2/  + CITI training |
| CARMEN-I | https://physionet.org/content/carmen-i/1.0.1/  + Spanish DUA |

Once provisioned, drop files into `data/<benchmark>/raw/` and write a loader
under `benchmarks/`. The base class `Benchmark` only requires `__iter__`
yielding `Document(text, gold_spans=[PHISpan(...)])`.
