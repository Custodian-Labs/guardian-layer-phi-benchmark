# GPT-5 (fresh runs) vs the stored panel, same documents

Documents: 199 (ASQ-PHI 30, MEDDOCAN 30, MultiCoNER v2 19, PII en 30, PII nl 30, PII fr 30, PII de 30). GPT-5 run with the paper's prompt and parser at its default temperature (it rejects temperature 0).

## Whole-document detection on the original documents

| Detector | Docs | P | R | F1 (exact boundary) | F1 (any overlap) |
|---|--:|--:|--:|--:|--:|
| GPT-5 (fresh run) | 199 | 0.840 | 0.740 | 0.787 | 0.853 |
| stored OpenAI detector (gpt-4o-mini) | 199 | 0.807 | 0.698 | 0.749 | 0.824 |
| Gemma 4 31B | 199 | 0.833 | 0.708 | 0.765 | 0.841 |
| Qwen 3.5 35B-A3B | 199 | 0.745 | 0.714 | 0.729 | 0.820 |
| Llama 3.3 70B | 182 | 0.818 | 0.734 | 0.774 | 0.842 |
| Gemma 4 E4B | 199 | 0.813 | 0.696 | 0.750 | 0.811 |
| Qwen 3.5 9B | 199 | 0.727 | 0.602 | 0.659 | 0.752 |
| Presidio | 199 | 0.428 | 0.415 | 0.422 | 0.564 |
| OBI RoBERTa | 199 | 0.034 | 0.057 | 0.043 | 0.471 |

## Masked-span recall, original → transformed (snap_sym, overlap)

Documents covered by GPT-5, the stored OpenAI detector and Gemma 4 31B: 173.

| Detector | Spans | Recall orig → transf. | Diff | 90% CI (doc cluster) |
|---|--:|--|--:|--|
| GPT-5 (fresh run) | 678 | 92.8 → 92.3 | +0.44 | [-1.41, +2.27] |
| stored OpenAI detector (gpt-4o-mini) | 678 | 91.2 → 87.2 | +3.98 | [+1.79, +6.22] |
| Gemma 4 31B | 678 | 91.3 → 92.0 | -0.74 | [-2.83, +1.15] |

GPT-5 calls: 398; estimated cost $29.34.
