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
from benchmarks.domains import EnterpriseGretel, FinanceGretel, LegalTAB
from benchmarks.transformed import (
    ASQPHITransformed,
    MEDDOCANTransformed,
    PIIMasking300kTransformed,
    PIIMasking300kDutchTransformed,
    PIIMasking300kFrenchTransformed,
    PIIMasking300kGermanTransformed,
    MultiCoNERv2Transformed,
    FinanceGretelTransformed,
    LegalTABTransformed,
    EnterpriseGretelTransformed,
)

REGISTRY = {
    "asq_phi": ASQPHI,
    "meddocan": MEDDOCAN,
    "pii_masking_300k": PIIMasking300k,
    "pii_masking_300k_dutch": PIIMasking300kDutch,
    "pii_masking_300k_french": PIIMasking300kFrench,
    "pii_masking_300k_german": PIIMasking300kGerman,
    "multiconer_v2": MultiCoNERv2,
    # Non-medical domains (finance / legal / multi-domain enterprise)
    "finance_gretel": FinanceGretel,
    "legal_tab": LegalTAB,
    "enterprise_gretel": EnterpriseGretel,
    # Custodian-transformed variants (same docs, surrogate PHI, remapped gold)
    "asq_phi_transformed": ASQPHITransformed,
    "meddocan_transformed": MEDDOCANTransformed,
    "pii_masking_300k_transformed": PIIMasking300kTransformed,
    "pii_masking_300k_dutch_transformed": PIIMasking300kDutchTransformed,
    "pii_masking_300k_french_transformed": PIIMasking300kFrenchTransformed,
    "pii_masking_300k_german_transformed": PIIMasking300kGermanTransformed,
    "multiconer_v2_transformed": MultiCoNERv2Transformed,
    "finance_gretel_transformed": FinanceGretelTransformed,
    "legal_tab_transformed": LegalTABTransformed,
    "enterprise_gretel_transformed": EnterpriseGretelTransformed,
}
