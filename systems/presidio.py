"""Microsoft Presidio baseline.

Install:
    pip install presidio-analyzer presidio-anonymizer
    python -m spacy download en_core_web_lg
    python -m spacy download es_core_news_lg   # for MEDDOCAN/CARMEN-I
"""
from __future__ import annotations

from systems.base import DeIDSystem, PredictedSpan, Prediction


_SPACY_BY_LANG = {
    "en": "en_core_web_lg",
    "es": "es_core_news_lg",
    "ca": "ca_core_news_lg",
    "nl": "nl_core_news_lg",
    "de": "de_core_news_lg",
    "fr": "fr_core_news_lg",
    "it": "it_core_news_lg",
    "pt": "pt_core_news_lg",
}


class Presidio(DeIDSystem):
    name = "presidio"

    def __init__(self, language: str = "en"):
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_anonymizer import AnonymizerEngine

        spacy_model = _SPACY_BY_LANG.get(language)
        if spacy_model is None:
            raise ValueError(f"No spaCy model configured for language {language!r}")

        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": language, "model_name": spacy_model}],
        })
        self._analyzer = AnalyzerEngine(
            nlp_engine=provider.create_engine(),
            supported_languages=[language],
        )
        self._anonymizer = AnonymizerEngine()
        self.language = language

    def predict(self, text: str) -> Prediction:
        results = self._analyzer.analyze(text=text, language=self.language)
        spans = [
            PredictedSpan(
                start=r.start,
                end=r.end,
                label=r.entity_type,
                text=text[r.start:r.end],
                score=float(r.score),
            )
            for r in results
        ]
        anonymized = self._anonymizer.anonymize(text=text, analyzer_results=results)
        return Prediction(spans=spans, transformed_text=anonymized.text, raw=results)
