# Locale consistency of surrogates (LLM judge, P(yes) > 0.5)

| Model | Benchmark | Kind | N | Original (control) | Custodian | Faker |
|---|---|---|--:|--:|--:|--:|
| qwen3_5_9b | all | all | 2114 | 85.4 | 73.6 | 87.8 |
| qwen3_5_9b | asq_phi | all | 268 | 98.1 | 76.1 | 61.2 |
| qwen3_5_9b | asq_phi | loc | 82 | 98.8 | 36.6 | 9.8 |
| qwen3_5_9b | asq_phi | name | 186 | 97.8 | 93.5 | 83.9 |
| qwen3_5_9b | meddocan | all | 1525 | 89.9 | 79.5 | 97.5 |
| qwen3_5_9b | meddocan | loc | 855 | 88.0 | 76.1 | 96.4 |
| qwen3_5_9b | meddocan | name | 670 | 92.4 | 83.7 | 99.0 |
| qwen3_5_9b | pii_dutch | all | 65 | 49.2 | 44.6 | 70.8 |
| qwen3_5_9b | pii_dutch | loc | 39 | 74.4 | 66.7 | 71.8 |
| qwen3_5_9b | pii_dutch | name | 26 | 11.5 | 11.5 | 69.2 |
| qwen3_5_9b | pii_french | all | 85 | 67.1 | 63.5 | 57.6 |
| qwen3_5_9b | pii_french | loc | 67 | 83.6 | 76.1 | 50.7 |
| qwen3_5_9b | pii_french | name | 18 | 5.6 | 16.7 | 83.3 |
| qwen3_5_9b | pii_german | all | 84 | 40.5 | 35.7 | 95.2 |
| qwen3_5_9b | pii_german | loc | 41 | 78.0 | 68.3 | 100.0 |
| qwen3_5_9b | pii_german | name | 43 | 4.7 | 4.7 | 90.7 |
| qwen3_5_9b | pii_masking_300k | all | 87 | 56.3 | 29.9 | 34.5 |
| qwen3_5_9b | pii_masking_300k | loc | 51 | 88.2 | 45.1 | 13.7 |
| qwen3_5_9b | pii_masking_300k | name | 36 | 11.1 | 8.3 | 63.9 |
