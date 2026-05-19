"""
agents/detection_agent.py
Detection Agent — analyzes incoming prompt and computes adversarial risk score.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import time

from models.risk_scorer import default_scorer, Decision
from utils.feature_extractor import feature_names


@dataclass
class DetectionResult:
    prompt: str
    risk_score: float
    decision: str                     # ACCEPT / REPAIR / REJECT
    features: List[float]
    feature_names: List[str]
    latency_ms: float

    @property
    def feature_dict(self) -> dict:
        return dict(zip(self.feature_names, self.features))

    def __str__(self) -> str:
        feat_str = ", ".join(
            f"{k}={v:.2f}" for k, v in self.feature_dict.items()
        )
        return (
            f"[DetectionAgent] decision={self.decision} "
            f"risk={self.risk_score:.4f} "
            f"features=({feat_str}) "
            f"latency={self.latency_ms:.1f}ms"
        )


class DetectionAgent:
    """
    Analyzes the incoming prompt and computes an adversarial risk score
    based on structured prompt features.
    """

    name = "DetectionAgent"

    def __init__(self, scorer=None):
        self._scorer = scorer or default_scorer

    def run(self, prompt: str) -> DetectionResult:
        t0 = time.perf_counter()
        decision, risk, features = self._scorer.decide(prompt)
        latency = (time.perf_counter() - t0) * 1000

        result = DetectionResult(
            prompt=prompt,
            risk_score=risk,
            decision=decision,
            features=features,
            feature_names=feature_names(),
            latency_ms=latency,
        )
        print(result)
        return result

    def re_evaluate(self, prompt: str) -> DetectionResult:
        """Re-evaluates prompt after repair — same as run()."""
        return self.run(prompt)
