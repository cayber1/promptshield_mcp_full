"""
agents/repair_agent.py
Repair Agent — rewrites / sanitizes unsafe prompts while preserving benign intent.
"""

import re
from dataclasses import dataclass
from typing import List
import time


@dataclass
class RepairResult:
    original_prompt: str
    repaired_prompt: str
    repairs_applied: List[str]
    latency_ms: float

    def __str__(self) -> str:
        return (
            f"[RepairAgent] repairs={self.repairs_applied} "
            f"latency={self.latency_ms:.1f}ms\n"
            f"  original : {self.original_prompt[:80]}...\n"
            f"  repaired : {self.repaired_prompt[:80]}..."
        )


# ── Individual repair strategies (applied in order) ──────────────────────────

class _RepairStrategy:
    name: str = ""

    def apply(self, prompt: str) -> tuple[str, bool]:
        """Returns (modified_prompt, was_modified)."""
        raise NotImplementedError


class RemoveOverrideInstructions(_RepairStrategy):
    name = "remove_override_instructions"
    _patterns = [
        r"ignore (previous|all|above|prior) (instructions?|prompts?|rules?|constraints?)[^.!?\n]*[.!?\n]?",
        r"disregard (your|the) (instructions?|guidelines?|rules?)[^.!?\n]*[.!?\n]?",
        r"forget (everything|what you were told|your instructions)[^.!?\n]*[.!?\n]?",
        r"new (instructions?|task|objective|goal):\s*",
        r"override (mode|system|instructions?)[^.!?\n]*[.!?\n]?",
        r"do not follow[^.!?\n]*[.!?\n]?",
        r"bypass (safety|filter|restriction|policy)[^.!?\n]*[.!?\n]?",
    ]

    def apply(self, prompt: str) -> tuple[str, bool]:
        modified = prompt
        for p in self._patterns:
            modified = re.sub(p, "", modified, flags=re.IGNORECASE)
        return modified.strip(), modified != prompt


class NeutralizeRoleAssignment(_RepairStrategy):
    name = "neutralize_role_assignment"
    _patterns = [
        r"(pretend you are|pretend to be|you are now|roleplay as|act as)[^,.\n]*",
        r"switch to (developer|admin|root|jailbreak) mode",
        r"\bDAN\b[^.]*",
        r"jailbreak[^.]*",
    ]

    def apply(self, prompt: str) -> tuple[str, bool]:
        modified = prompt
        for p in self._patterns:
            modified = re.sub(p, "[role-assignment removed]", modified, flags=re.IGNORECASE)
        changed = modified != prompt
        return modified.strip(), changed


class SanitizeInjectionTokens(_RepairStrategy):
    name = "sanitize_injection_tokens"
    _patterns = [
        (r"system\s*:", "user:"),
        (r"<\|system\|>", ""),
        (r"\[INST\]", ""),
        (r"###\s*(instruction|system|human|assistant)\s*", ""),
        (r"</?s>", ""),
        (r"\{\{.*?\}\}", "[template removed]"),
    ]

    def apply(self, prompt: str) -> tuple[str, bool]:
        modified = prompt
        for pattern, replacement in self._patterns:
            modified = re.sub(pattern, replacement, modified, flags=re.IGNORECASE)
        return modified.strip(), modified != prompt


class NeutralizeContextManipulation(_RepairStrategy):
    name = "neutralize_context_manipulation"
    _patterns = [
        r"in (a|this) (fictional|hypothetical|imaginary) (world|scenario|context|universe)[,.]?",
        r"for (educational|research|academic) purposes?[,.]?",
        r"this is (just )?a (story|game|simulation|test)[,.]?",
        r"hypothetically[,.]?",
    ]

    def apply(self, prompt: str) -> tuple[str, bool]:
        modified = prompt
        for p in self._patterns:
            modified = re.sub(p, "", modified, flags=re.IGNORECASE)
        return modified.strip(), modified != prompt


class CleanupWhitespace(_RepairStrategy):
    name = "cleanup_whitespace"

    def apply(self, prompt: str) -> tuple[str, bool]:
        cleaned = re.sub(r"\s{2,}", " ", prompt).strip()
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned, cleaned != prompt


# ── Repair Agent ──────────────────────────────────────────────────────────────

_STRATEGIES: List[_RepairStrategy] = [
    RemoveOverrideInstructions(),
    NeutralizeRoleAssignment(),
    SanitizeInjectionTokens(),
    NeutralizeContextManipulation(),
    CleanupWhitespace(),
]


class RepairAgent:
    """
    Rewrites or sanitizes unsafe prompts while preserving benign user intent.
    Applies a pipeline of rule-based repair strategies.
    """

    name = "RepairAgent"

    def __init__(self, strategies: List[_RepairStrategy] = None):
        self._strategies = strategies or _STRATEGIES

    def run(self, prompt: str) -> RepairResult:
        t0 = time.perf_counter()
        current = prompt
        applied: List[str] = []

        for strategy in self._strategies:
            modified, changed = strategy.apply(current)
            if changed:
                applied.append(strategy.name)
                current = modified

        latency = (time.perf_counter() - t0) * 1000
        result = RepairResult(
            original_prompt=prompt,
            repaired_prompt=current if current else prompt,
            repairs_applied=applied,
            latency_ms=latency,
        )
        print(result)
        return result
