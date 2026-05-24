from benchmarks.base import Benchmark, Document, PHISpan
from benchmarks.asq_phi import ASQPHI
from benchmarks.meddocan import MEDDOCAN
from benchmarks.pii_masking_300k import (
    PIIMasking300k,
    PIIMasking300kDutch,
    PIIMasking300kFrench,
    PIIMasking300kGerman,
)
from benchmarks.multiconer_v2 import MultiCoNERv2

REGISTRY = {
    "asq_phi": ASQPHI,
    "meddocan": MEDDOCAN,
    "pii_masking_300k": PIIMasking300k,
    "pii_masking_300k_dutch": PIIMasking300kDutch,
    "pii_masking_300k_french": PIIMasking300kFrench,
    "pii_masking_300k_german": PIIMasking300kGerman,
    "multiconer_v2": MultiCoNERv2,
}
