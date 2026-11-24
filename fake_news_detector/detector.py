"""Core detector: runs the Loki fact-checking pipeline and turns its output into a fake-news verdict.

Pipeline (provided by Loki):
    text -> claim decomposition -> check-worthiness -> query generation
         -> web evidence retrieval -> claim verification (SUPPORTS / REFUTES / IRRELEVANT)

This module adds:
    * a news-level verdict (REAL / MISLEADING / FAKE / UNVERIFIABLE) with a confidence value
    * a compact, UI-friendly result format
    * a demo mode that works without API keys
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class Verdict(str, Enum):
    REAL = "REAL"
    MISLEADING = "MISLEADING"
    FAKE = "FAKE"
    UNVERIFIABLE = "UNVERIFIABLE"


# Factuality thresholds (Loki's overall score is in [0, 1]).
REAL_THRESHOLD = 0.70
FAKE_THRESHOLD = 0.40


@dataclass
class EvidenceItem:
    text: str
    url: str
    reasoning: str
    relationship: str  # SUPPORTS / REFUTES / IRRELEVANT


@dataclass
class ClaimResult:
    claim: str
    origin_text: str
    checkworthy: bool
    status: str  # SUPPORTED / REFUTED / DISPUTED / NO_EVIDENCE / NOT_CHECKED
    factuality: Optional[float]
    evidences: list[EvidenceItem] = field(default_factory=list)


@dataclass
class DetectionResult:
    text: str
    verdict: Verdict
    factuality: Optional[float]
    confidence: float
    explanation: str
    claims: list[ClaimResult]
    stats: dict
    elapsed_seconds: float
    source_url: Optional[str] = None
    demo: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d


def _claim_status(factuality) -> str:
    if factuality == "Nothing to check.":
        return "NOT_CHECKED"
    if isinstance(factuality, str):
        return "NO_EVIDENCE"
    if factuality >= 0.999:
        return "SUPPORTED"
    if factuality <= 0.001:
        return "REFUTED"
    return "DISPUTED"


def decide_verdict(factuality: Optional[float], num_verified: int) -> tuple[Verdict, float, str]:
    """Map an overall factuality score to a verdict + confidence + human explanation."""
    if num_verified == 0 or factuality is None:
        return (
            Verdict.UNVERIFIABLE,
            0.0,
            "No check-worthy claim could be matched with web evidence, so the text cannot be verified.",
        )
    # Confidence grows with distance from the decision boundary and with the number of verified claims.
    if factuality >= REAL_THRESHOLD:
        verdict = Verdict.REAL
        margin = (factuality - REAL_THRESHOLD) / (1 - REAL_THRESHOLD)
        explanation = "Most factual claims are supported by retrieved evidence."
    elif factuality < FAKE_THRESHOLD:
        verdict = Verdict.FAKE
        margin = (FAKE_THRESHOLD - factuality) / FAKE_THRESHOLD
        explanation = "Most factual claims are contradicted by retrieved evidence."
    else:
        verdict = Verdict.MISLEADING
        mid = (REAL_THRESHOLD + FAKE_THRESHOLD) / 2
        margin = 1 - abs(factuality - mid) / ((REAL_THRESHOLD - FAKE_THRESHOLD) / 2)
        explanation = "The text mixes supported and refuted claims."
    coverage = min(1.0, num_verified / 3)  # 3+ verified claims -> full coverage weight
    confidence = round(0.5 + 0.5 * max(0.0, margin) * coverage, 3)
    return verdict, min(confidence, 1.0), explanation


# Free LLM providers, all reached through Loki's OpenAI-compatible "local_openai" client.
#   url   : OpenAI-compatible base URL (trailing slash required by the openai SDK)
#   model : default model (override with --model if the provider renames it)
#   key   : name of the key in api_config.yaml / environment (None = no key needed)
#   rpm   : requests/minute cap per pipeline step, to stay under free-tier limits
PROVIDERS = {
    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-3.5-flash-lite",  # higher free-tier quota than gemini-3.5-flash
        "key": "GEMINI_API_KEY",
        "rpm": 8,
    },
    "groq": {
        "url": "https://api.groq.com/openai/v1/",
        "model": "llama-3.3-70b-versatile",
        "key": "GROQ_API_KEY",
        "rpm": 20,
    },
    "ollama": {
        "url": "http://localhost:11434/v1/",
        "model": "qwen2.5:7b",
        "key": None,
        "rpm": 1000,
    },
}
PROVIDER_CHOICES = ["openai", "anthropic", *PROVIDERS]


class FakeNewsDetector:
    """Thin, thread-safe wrapper around Loki's FactCheck pipeline."""

    def __init__(
        self,
        model: Optional[str] = None,
        client: Optional[str] = None,
        retriever: str = "serper",
        api_config: Optional[dict] = None,
        demo: bool = False,
        provider: str = "openai",
    ):
        self.demo = demo
        self._lock = threading.Lock()
        self._fc = None
        if demo:
            return

        from factcheck import FactCheck  # imported lazily so demo mode needs no keys

        api_config = dict(api_config or {})
        preset = PROVIDERS.get(provider)
        if preset:
            client = "local_openai"
            model = model or preset["model"]
            api_config["LOCAL_API_URL"] = api_config.get("LOCAL_API_URL") or preset["url"]
            key_name = preset["key"]
            key = (api_config.get(key_name) or os.environ.get(key_name)) if key_name else "ollama"
            if not key:
                raise ValueError(f"{key_name} is not set. Add it to your api_config.yaml (free key, see README).")
            api_config["LOCAL_API_KEY"] = key
        elif provider == "anthropic":
            model = model or "claude-sonnet-4-5"
        else:
            model = model or "gpt-4o-mini"

        self.provider, self.model = provider, model
        self._fc = FactCheck(default_model=model, client=client, retriever=retriever, api_config=api_config)
        if preset:
            for step in ("decompose_model", "checkworthy_model", "query_generator_model",
                         "evidence_retrieval_model", "claim_verify_model"):
                getattr(self._fc, step).max_requests_per_minute = preset["rpm"]

    # ------------------------------------------------------------------ public API
    def detect(self, text: str, source_url: Optional[str] = None) -> DetectionResult:
        text = (text or "").strip()
        if not text:
            raise ValueError("Input text is empty.")
        start = time.time()
        raw = self._demo_output(text) if self.demo else self._run_loki(text)
        result = self._build_result(raw, text, time.time() - start)
        result.source_url = source_url
        return result

    # ------------------------------------------------------------------ internals
    def _run_loki(self, text: str) -> dict:
        with self._lock:  # Loki keeps per-run token usage on the instance -> serialize runs
            return self._fc.check_text(text)

    def _build_result(self, raw: dict, text: str, elapsed: float) -> DetectionResult:
        claims: list[ClaimResult] = []
        for c in raw.get("claim_detail", []):
            fact = c.get("factuality")
            evidences, seen = [], set()
            for e in c.get("evidences", []):
                key = (e.get("url", ""), e.get("relationship"))
                if key in seen:  # Loki returns the same source once per search query
                    continue
                seen.add(key)
                evidences.append(
                    EvidenceItem(
                        text=e.get("text", ""),
                        url=e.get("url", ""),
                        reasoning=e.get("reasoning", ""),
                        relationship=e.get("relationship", "IRRELEVANT"),
                    )
                )
            claims.append(
                ClaimResult(
                    claim=c.get("claim", ""),
                    origin_text=c.get("origin_text", ""),
                    checkworthy=bool(c.get("checkworthy")),
                    status=_claim_status(fact),
                    factuality=fact if isinstance(fact, (int, float)) else None,
                    evidences=evidences,
                )
            )

        summary = raw.get("summary", {})
        num_verified = summary.get("num_verified_claims", 0)
        factuality = summary.get("factuality") if num_verified else None
        verdict, confidence, explanation = decide_verdict(factuality, num_verified)
        return DetectionResult(
            text=text,
            verdict=verdict,
            factuality=round(factuality, 3) if factuality is not None else None,
            confidence=confidence,
            explanation=explanation,
            claims=claims,
            stats={
                "num_claims": summary.get("num_claims", len(claims)),
                "num_checkworthy": summary.get("num_checkworthy_claims", 0),
                "num_verified": num_verified,
                "num_supported": summary.get("num_supported_claims", 0),
                "num_refuted": summary.get("num_refuted_claims", 0),
                "num_disputed": summary.get("num_controversial_claims", 0),
            },
            elapsed_seconds=round(elapsed, 2),
            demo=self.demo,
        )

    @staticmethod
    def _demo_output(text: str) -> dict:
        """Fake a Loki output so the UI can be explored without API keys.

        Sentences containing obvious red-flag wording are marked as refuted; everything else supported.
        The result is clearly labelled as DEMO in the UI — it is NOT a real fact-check.
        """
        red_flags = re.compile(
            r"\b(miracle|secret|shocking|cure[sd]? all|100%|they don'?t want you|hoax|flat|aliens?|"
            r"microchip|never happened|banned|exposed)\b",
            re.I,
        )
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 15][:6]
        details, supported, refuted = [], 0, 0
        for s in sentences:
            fake = bool(red_flags.search(s))
            rel = "REFUTES" if fake else "SUPPORTS"
            supported += not fake
            refuted += fake
            details.append(
                {
                    "claim": s,
                    "origin_text": s,
                    "checkworthy": True,
                    "factuality": 0.0 if fake else 1.0,
                    "evidences": [
                        {
                            "text": "[Demo evidence — run with API keys for real web evidence.]",
                            "url": "https://example.com",
                            "reasoning": "Demo heuristic: sensational wording detected." if fake else "Demo heuristic: neutral wording.",
                            "relationship": rel,
                        }
                    ],
                }
            )
        n = len(details)
        return {
            "claim_detail": details,
            "summary": {
                "num_claims": n,
                "num_checkworthy_claims": n,
                "num_verified_claims": n,
                "num_supported_claims": supported,
                "num_refuted_claims": refuted,
                "num_controversial_claims": 0,
                "factuality": supported / n if n else 0,
            },
        }
