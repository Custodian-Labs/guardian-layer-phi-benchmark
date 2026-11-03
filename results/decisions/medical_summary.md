# Clinical decision study: ASQ-PHI and MEDDOCAN (CodiEsp ICD-10)

Paired with the same model on the original text. ICD: macro over 8 chapters (CodiEsp gold, threshold P(yes) > 0.5 for accuracy); 95% CI from 1,000 document bootstraps. Sex / age group: MEDDOCAN gold.

| Model | Arm | ICD acc orig → arm (Δ, 95% CI) | ICD AUROC orig → arm (Δ, 95% CI) | Sex acc Δ | Age-group acc Δ | ASQ-PHI agreement |
|---|---|--|--|--:|--:|--:|
| gemma_4_e4b | custodian | 83.6 → 83.4 (-0.2, [-0.7, +0.3]) | 87.6 → 88.0 (+0.4, [-0.1, +1.0]) | +0.0 | +2.0 | 91.7% |
| gemma_4_e4b | faker | 83.5 → 83.2 (-0.4, [-0.9, +0.2]) | 87.6 → 87.5 (-0.1, [-0.6, +0.3]) | +0.4 | -23.0 | 93.8% |
| gemma_4_e4b | redacted | 83.5 → 83.2 (-0.3, [-0.8, +0.2]) | 87.6 → 87.8 (+0.2, [-0.1, +0.5]) | -0.4 | -6.9 | 90.3% |
| qwen3.5_4b | custodian | 82.2 → 82.5 (+0.3, [-0.1, +0.9]) | 86.9 → 86.7 (-0.2, [-0.7, +0.3]) | +0.8 | -2.4 | 90.3% |
| qwen3.5_4b | faker | 82.1 → 82.2 (+0.2, [-0.3, +0.6]) | 86.8 → 87.0 (+0.1, [-0.2, +0.5]) | -1.2 | -18.1 | 91.5% |
| qwen3.5_4b | redacted | 82.1 → 82.0 (-0.1, [-0.5, +0.3]) | 86.8 → 86.7 (-0.1, [-0.6, +0.3]) | -0.4 | -7.3 | 90.4% |
| qwen3.5_9b | custodian | 83.0 → 83.5 (+0.5, [-0.0, +1.0]) | 87.8 → 87.5 (-0.3, [-0.7, +0.1]) | +0.4 | -1.2 | 92.0% |
| qwen3.5_9b | faker | 83.0 → 82.9 (-0.1, [-0.4, +0.3]) | 87.7 → 87.4 (-0.3, [-0.6, +0.0]) | -0.8 | -14.1 | 91.8% |
| qwen3.5_9b | redacted | 83.0 → 83.0 (+0.0, [-0.4, +0.4]) | 87.7 → 87.6 (-0.1, [-0.3, +0.1]) | -6.5 | -5.2 | 89.7% |
