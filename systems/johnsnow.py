"""John Snow Labs Spark NLP for Healthcare wrapper.

Commercial license required. Set JSL_LICENSE_PATH in `.env` to a JSON license
file containing `secret`, `aws_access_key_id`, `aws_secret_access_key`, etc.

Install (after license is obtained):
    pip install spark-nlp-jsl

This wrapper is a stub — fill in `_load_pipeline()` once the JSL deid model
name is confirmed (e.g. `ner_deid_subentity_augmented`).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from systems.base import DeIDSystem, PredictedSpan, Prediction


class JohnSnowLabs(DeIDSystem):
    name = "john_snow_labs"

    def __init__(self):
        license_path = os.environ.get("JSL_LICENSE_PATH")
        if not license_path or not Path(license_path).exists():
            raise RuntimeError(
                "JSL_LICENSE_PATH unset or invalid. Obtain a license at "
                "https://www.johnsnowlabs.com and point JSL_LICENSE_PATH to "
                "the downloaded JSON."
            )
        self._license = json.loads(Path(license_path).read_text())
        self._pipeline = self._load_pipeline()

    def _load_pipeline(self):
        # TODO: replace stub with actual sparknlp_jsl pipeline construction.
        #
        # Example sketch:
        #   import sparknlp_jsl
        #   spark = sparknlp_jsl.start(self._license["secret"])
        #   pipeline = PretrainedPipeline("ner_deid_subentity_augmented", "en", "clinical/models")
        #   return pipeline
        raise NotImplementedError("Wire up Spark NLP pipeline once license is provisioned.")

    def predict(self, text: str) -> Prediction:
        raise NotImplementedError
