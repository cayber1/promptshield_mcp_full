"""
models/risk_scorer.py
Formal Safety Model:
  Risk(P) = σ(w · F(P) + b)
  where F(P) = [override, injection, role_manipulation, context_manipulation, ambiguity]
"""

import math
import numpy as np
from typing import List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

from config import FEATURE_WEIGHTS, BIAS, TAU_1, TAU_2
from utils.feature_extractor import extract_features


# ── Decision enum ──────────────────────────────────────────────────────────────

class Decision:
    ACCEPT = "ACCEPT"
    REPAIR = "REPAIR"
    REJECT = "REJECT"


# ── Logistic sigmoid ──────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ── Analytic scorer (proposal equation) ───────────────────────────────────────

class AnalyticRiskScorer:
    """
    Uses fixed weights from config (w·F(P)+b).
    No training required — directly implements the formal model.
    """

    def __init__(self, weights: List[float] = FEATURE_WEIGHTS, bias: float = BIAS):
        self.weights = np.array(weights)
        self.bias = bias

    def score(self, prompt: str) -> Tuple[float, List[float]]:
        """Returns (risk_score, feature_vector)."""
        features = extract_features(prompt)
        logit = float(np.dot(self.weights, features) + self.bias)
        risk = _sigmoid(logit)
        return risk, features

    def decide(self, prompt: str) -> Tuple[str, float, List[float]]:
        """Returns (decision, risk_score, feature_vector)."""
        risk, features = self.score(prompt)
        if risk <= TAU_1:
            decision = Decision.ACCEPT
        elif risk <= TAU_2:
            decision = Decision.REPAIR
        else:
            decision = Decision.REJECT
        return decision, risk, features


# ── Trained scorer (sklearn LogisticRegression on PromptShieldBench) ──────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "saved_scaler.pkl")


class TrainedRiskScorer:
    """
    Trained on PromptShieldBench dataset.
    Falls back to AnalyticRiskScorer if no model is saved yet.
    """

    def __init__(self):
        self._analytic = AnalyticRiskScorer()
        self._model: Optional[LogisticRegression] = None
        self._scaler: Optional[StandardScaler] = None
        self._load()

    def _load(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            self._model = joblib.load(MODEL_PATH)
            self._scaler = joblib.load(SCALER_PATH)

    def train(self, X: np.ndarray, y: np.ndarray):
        """X: (n_samples, 5) feature matrix, y: binary labels."""
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        self._model = LogisticRegression(max_iter=1000, C=1.0)
        self._model.fit(X_scaled, y)
        joblib.dump(self._model, MODEL_PATH)
        joblib.dump(self._scaler, SCALER_PATH)

    def score(self, prompt: str) -> Tuple[float, List[float]]:
        features = extract_features(prompt)
        if self._model is None or self._scaler is None:
            # fallback
            return self._analytic.score(prompt)
        X = np.array(features).reshape(1, -1)
        X_scaled = self._scaler.transform(X)
        prob = self._model.predict_proba(X_scaled)[0][1]  # P(adversarial)
        return float(prob), features

    def decide(self, prompt: str) -> Tuple[str, float, List[float]]:
        risk, features = self.score(prompt)
        if risk <= TAU_1:
            decision = Decision.ACCEPT
        elif risk <= TAU_2:
            decision = Decision.REPAIR
        else:
            decision = Decision.REJECT
        return decision, risk, features


# ── Default scorer used by agents ─────────────────────────────────────────────

default_scorer = TrainedRiskScorer()
