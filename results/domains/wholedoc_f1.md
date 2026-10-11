# Whole-document span F1: original vs Custodian transform vs redaction

Mean over detectors; exact = span-exact, label-agnostic (`type`); overlap = any overlap (`relaxed`).

| Domain | Exact: original | Custodian | Δ | Redacted | Overlap: original | Custodian | Δ | Redacted |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Finance | 0.335 | 0.256 | -0.078 | 0.100 | 0.441 | 0.422 | -0.019 | 0.167 |
| Legal | 0.446 | 0.387 | -0.059 | 0.152 | 0.632 | 0.607 | -0.025 | 0.266 |
| Enterprise | 0.645 | 0.577 | -0.068 | 0.251 | 0.801 | 0.781 | -0.020 | 0.412 |

## Per detector (exact-boundary F1; overlap Δ in the last column)

| Domain | Detector | Original | Custodian | Δ exact | Redacted | Δ overlap |
|---|---|--:|--:|--:|--:|--:|
| Finance | Gemma-4-31B | 0.523 | 0.433 | -0.090 | 0.258 | +0.008 |
| Finance | Qwen3.5-35B-A3B | 0.482 | 0.323 | -0.159 | 0.277 | -0.073 |
| Finance | Llama-3.3-70B | 0.479 | 0.365 | -0.113 | 0.096 | -0.034 |
| Finance | Gemma-4-E4B | 0.475 | 0.370 | -0.105 | 0.064 | -0.017 |
| Finance | Qwen3.5-9B | 0.400 | 0.300 | -0.099 | 0.052 | -0.037 |
| Finance | Qwen3.5-4B | 0.379 | 0.274 | -0.104 | 0.157 | -0.046 |
| Finance | DeepSeek-V2-Lite | 0.233 | 0.205 | -0.029 | 0.045 | +0.025 |
| Finance | Presidio | 0.223 | 0.145 | -0.078 | 0.000 | -0.040 |
| Finance | Llama-3.1-8B | 0.138 | 0.135 | -0.003 | 0.044 | +0.030 |
| Finance | OBI deid_roberta | 0.014 | 0.010 | -0.003 | 0.007 | -0.006 |
| Legal | Gemma-4-31B | 0.687 | 0.607 | -0.081 | 0.411 | -0.022 |
| Legal | Llama-3.3-70B | 0.649 | 0.569 | -0.079 | 0.248 | -0.016 |
| Legal | Gemma-4-E4B | 0.605 | 0.561 | -0.044 | 0.119 | -0.013 |
| Legal | Qwen3.5-35B-A3B | 0.583 | 0.438 | -0.145 | 0.426 | -0.093 |
| Legal | Presidio | 0.490 | 0.424 | -0.065 | 0.008 | -0.042 |
| Legal | Qwen3.5-4B | 0.445 | 0.389 | -0.056 | 0.123 | -0.015 |
| Legal | Qwen3.5-9B | 0.411 | 0.344 | -0.067 | 0.034 | -0.015 |
| Legal | DeepSeek-V2-Lite | 0.323 | 0.327 | +0.004 | 0.053 | +0.030 |
| Legal | Llama-3.1-8B | 0.210 | 0.159 | -0.051 | 0.055 | -0.039 |
| Legal | OBI deid_roberta | 0.059 | 0.052 | -0.007 | 0.039 | -0.027 |
| Enterprise | Gemma-4-31B | 0.825 | 0.748 | -0.077 | 0.561 | -0.013 |
| Enterprise | Gemma-4-E4B | 0.813 | 0.708 | -0.106 | 0.131 | -0.028 |
| Enterprise | Qwen3.5-35B-A3B | 0.796 | 0.691 | -0.104 | 0.562 | -0.044 |
| Enterprise | Llama-3.3-70B | 0.789 | 0.721 | -0.068 | 0.385 | -0.014 |
| Enterprise | Qwen3.5-9B | 0.781 | 0.687 | -0.094 | 0.245 | -0.020 |
| Enterprise | Qwen3.5-4B | 0.721 | 0.674 | -0.047 | 0.284 | -0.005 |
| Enterprise | Llama-3.1-8B | 0.661 | 0.551 | -0.110 | 0.269 | -0.065 |
| Enterprise | DeepSeek-V2-Lite | 0.558 | 0.518 | -0.041 | 0.068 | -0.011 |
| Enterprise | Presidio | 0.459 | 0.436 | -0.023 | 0.000 | -0.001 |
| Enterprise | OBI deid_roberta | 0.043 | 0.033 | -0.010 | 0.005 | +0.002 |
