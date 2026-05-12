"""
utils/feature_extractor.py
Extracts structured feature vector F(P) from a prompt.
F(P) = [override, injection, role_manipulation, context_manipulation, ambiguity]
Each feature is a float in [0, 1].
"""

import re
from typing import List


# ── Keyword banks ──────────────────────────────────────────────────────────────

_OVERRIDE_PATTERNS = [
    r"ignore (previous|all|above|prior) (instructions?|prompts?|rules?|constraints?)",
    r"disregard (your|the) (instructions?|guidelines?|rules?)",
    r"forget (everything|what you were told|your instructions)",
    r"new (instructions?|task|objective|goal):",
    r"override (mode|system|instructions?)",
    r"do not follow",
    r"bypass (safety|filter|restriction|policy)",
]

_INJECTION_PATTERNS = [
    r"system\s*:",
    r"<\|system\|>",
    r"\[INST\]",
    r"###\s*(instruction|system|human|assistant)",
    r"</?s>",
    r"\{\{.*?\}\}",          # template injection
    r"prompt\s*injection",
    r"you are now",
    r"act as (a|an|the)",
]

_ROLE_PATTERNS = [
    r"pretend (you are|to be)",
    r"roleplay as",
    r"you are (now|a|an) (?!helpful|assistant)",
    r"switch to (developer|admin|root|jailbreak) mode",
    r"DAN\b",
    r"jailbreak",
    r"character named",
    r"fictional (ai|assistant|character)",
]

_CONTEXT_PATTERNS = [
    r"in (a|this) (fictional|hypothetical|imaginary) (world|scenario|context|universe)",
    r"for (educational|research|academic) purposes?",
    r"this is (just )?a (story|game|simulation|test)",
    r"hypothetically",
    r"as an? example",
    r"if you (were|could|had to)",
    r"let's (say|suppose|imagine|pretend)",
]

_AMBIGUITY_PATTERNS = [
    r"\?{2,}",                      # multiple question marks
    r"\.{3,}",                      # ellipsis overuse
    r"\b(maybe|perhaps|possibly|potentially|might)\b",
    r"\b(any|whatever|anything|everything|all)\b",
    r"(do|say|write|provide) (anything|whatever|everything)",
]


def _score_pattern_group(text: str, patterns: List[str]) -> float:
    """Returns a [0,1] score based on how many patterns match."""
    hits = sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))
    return min(hits / max(len(patterns) * 0.3, 1), 1.0)


def extract_features(prompt: str) -> List[float]:
    """
    Returns F(P) = [override, injection, role_manipulation,
                    context_manipulation, ambiguity]
    Each value is a float in [0, 1].
    """
    text = prompt.strip()
    features = [
        _score_pattern_group(text, _OVERRIDE_PATTERNS),
        _score_pattern_group(text, _INJECTION_PATTERNS),
        _score_pattern_group(text, _ROLE_PATTERNS),
        _score_pattern_group(text, _CONTEXT_PATTERNS),
        _score_pattern_group(text, _AMBIGUITY_PATTERNS),
    ]
    return features


def feature_names() -> List[str]:
    return ["override", "injection", "role_manipulation",
            "context_manipulation", "ambiguity"]
