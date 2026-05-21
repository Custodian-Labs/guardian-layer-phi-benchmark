from benchmarks.base import Benchmark, Document, PHISpan
from benchmarks.asq_phi import ASQPHI
from benchmarks.meddocan import MEDDOCAN

REGISTRY = {
    "asq_phi": ASQPHI,
    "meddocan": MEDDOCAN,
}
