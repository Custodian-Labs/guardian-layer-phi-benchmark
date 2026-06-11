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
from benchmarks.transformed import (
    ASQPHITransformed,
    MEDDOCANTransformed,
    PIIMasking300kTransformed,
    PIIMasking300kDutchTransformed,
    PIIMasking300kFrenchTransformed,
    PIIMasking300kGermanTransformed,
    MultiCoNERv2Transformed,
)

REGISTRY = {
    "asq_phi": ASQPHI,
    "meddocan": MEDDOCAN,
    "pii_masking_300k": PIIMasking300k,
    "pii_masking_300k_dutch": PIIMasking300kDutch,
    "pii_masking_300k_french": PIIMasking300kFrench,
    "pii_masking_300k_german": PIIMasking300kGerman,
    "multiconer_v2": MultiCoNERv2,
    # Custodian-transformed variants (same docs, surrogate PHI, remapped gold)
    "asq_phi_transformed": ASQPHITransformed,
    "meddocan_transformed": MEDDOCANTransformed,
    "pii_masking_300k_transformed": PIIMasking300kTransformed,
    "pii_masking_300k_dutch_transformed": PIIMasking300kDutchTransformed,
    "pii_masking_300k_french_transformed": PIIMasking300kFrenchTransformed,
    "pii_masking_300k_german_transformed": PIIMasking300kGermanTransformed,
    "multiconer_v2_transformed": MultiCoNERv2Transformed,
}
