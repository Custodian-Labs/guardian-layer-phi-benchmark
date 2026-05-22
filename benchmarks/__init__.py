from benchmarks.base import Benchmark, Document, PHISpan
from benchmarks.asq_phi import ASQPHI
from benchmarks.meddocan import MEDDOCAN
from benchmarks.pii_masking_300k import PIIMasking300k

REGISTRY = {
    "asq_phi": ASQPHI,
    "meddocan": MEDDOCAN,
    "pii_masking_300k": PIIMasking300k,
}
