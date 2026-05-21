"""Custodian Labs Guardian Layer wrapper.

Two compliance modes are GA today:

  * MASKED       — standard NER pipeline. Detects PERSON, PHONE_NUMBER,
                   EMAIL_ADDRESS, LOCATION, DATE_TIME, ID_NUMBER, AGE.
  * PROPRIETARY  — Guardian Layer, domain-specific sensitive content.

Two masking strategies:

  * redact     — replace with `*****`
  * transform  — replace with a plausible alternative

Three relevant endpoints:

  POST /deidentify/text                    -> single de-identified output
  POST /deidentify/text/proprietary/outputs  -> two independent outputs
  POST /analyze/text/proprietary           -> detection-only (no replacement)

We expose one variant per compliance_mode × masking_type combination so a
single benchmark run can grade them side-by-side. Construct via the
`build_variants()` factory.

Docs: https://custodian-docs.vercel.app/guardian-layer/guardian-layer
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from systems.base import DeIDSystem, PredictedSpan, Prediction


ComplianceMode = Literal["MASKED", "PROPRIETARY"]
MaskingType = Literal["redact", "transform"]


@dataclass
class CustodianConfig:
    compliance_mode: ComplianceMode = "MASKED"
    masking_type: MaskingType = "redact"
    domain: str = "General"
    pii_entities: list[str] | None = None  # None or ["ALL"] or e.g. ["PERSON", "ID_NUMBER"]


class Custodian(DeIDSystem):
    """Single Custodian configuration as a benchmark-comparable system."""

    DEIDENTIFY_PATH = "/deidentify/text"
    ANALYZE_PROPRIETARY_PATH = "/analyze/text/proprietary"

    def __init__(
        self,
        config: CustodianConfig | None = None,
        api_key: str | None = None,
        base_url: str = "https://api.custodianai.com",
        timeout: float = 30.0,
    ):
        import httpx

        cfg = config or CustodianConfig()
        key = api_key or os.environ.get("CUSTODIAN_SDK_API_KEY") or os.environ.get("CUSTODIAN_API_KEY")
        if not key:
            raise RuntimeError(
                "CUSTODIAN_SDK_API_KEY not set. Get one from "
                "https://custodian-dashboard.vercel.app/"
            )
        self.cfg = cfg
        self.name = f"custodian_{cfg.compliance_mode.lower()}_{cfg.masking_type}"
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=timeout,
        )

    def predict(self, text: str) -> Prediction:
        payload = {
            "text": text,
            "compliance_mode": self.cfg.compliance_mode,
            "masking_type": self.cfg.masking_type,
            "domain": self.cfg.domain,
            "pii_entities": self.cfg.pii_entities,
        }
        resp = self._client.post(self.DEIDENTIFY_PATH, json=payload)
        resp.raise_for_status()
        body = resp.json()

        entities = (
            body.get("entities")
            or body.get("detections")
            or body.get("phi_entities")
            or []
        )
        spans: list[PredictedSpan] = []
        for e in entities:
            try:
                start = int(e["start"])
                end = int(e["end"])
            except (KeyError, TypeError, ValueError):
                continue
            spans.append(PredictedSpan(
                start=start,
                end=end,
                label=str(e.get("type") or e.get("label") or "PHI"),
                text=e.get("value") or e.get("text") or text[start:end],
                score=float(e.get("score", 1.0)),
            ))

        transformed = (
            body.get("transformed_text")
            or body.get("deidentified_text")
            or body.get("output")
        )
        return Prediction(spans=spans, transformed_text=transformed, raw=body)

    def close(self) -> None:
        self._client.close()


def build_variants(
    modes: list[ComplianceMode] | None = None,
    masking: list[MaskingType] | None = None,
    domain: str = "General",
    pii_entities: list[str] | None = None,
) -> list[Custodian]:
    """Return one Custodian per (mode × masking) combination.

    Use this in the runner when you want to grade every Custodian configuration
    side-by-side in one experiment.
    """
    modes = modes or ["MASKED", "PROPRIETARY"]
    masking = masking or ["redact", "transform"]
    return [
        Custodian(CustodianConfig(
            compliance_mode=m,
            masking_type=k,
            domain=domain,
            pii_entities=pii_entities,
        ))
        for m in modes
        for k in masking
    ]
