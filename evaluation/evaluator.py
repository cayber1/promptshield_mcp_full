"""
evaluation/evaluator.py
Evaluation Framework — compares 4 systems as described in the proposal:
  1. Direct LLM
  2. Detection-only system
  3. Repair-only system
  4. Full Agentic + MCP framework

Metrics:
  - Safety violation rate
  - False positive rate
  - Utility degradation (ΔU)
  - Agent convergence iterations
  - Success rate across attack categories
"""

import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import classification_report

from config import TAU_1, TAU_2, ALPHA, BETA, GAMMA, MAX_ITERATIONS
from dataset.dataset_builder import PromptShieldBench, PromptRecord
from models.risk_scorer import default_scorer, Decision
from agents.detection_agent import DetectionAgent
from agents.repair_agent import RepairAgent
from agents.generator_agent import GeneratorAgent
from agents.validation_agent import ValidationAgent
from agentic_loop import AgenticSafetyLoop


# ── Per-prompt result ─────────────────────────────────────────────────────────

@dataclass
class PromptEvalResult:
    record: PromptRecord
    system: str
    predicted_safe: bool      # True if system allowed/generated a response
    safety_violated: bool     # True if adversarial and got through
    false_positive: bool      # True if benign and was rejected/over-repaired
    iterations: int
    latency_ms: float
    utility: float


# ── System runners ────────────────────────────────────────────────────────────

class DirectLLM:
    """Baseline: no safety checks at all."""
    name = "DirectLLM"
    _gen = GeneratorAgent()

    def run(self, prompt: str) -> Tuple[bool, int, float, float]:
        t0 = time.perf_counter()
        result = self._gen.run(prompt)
        latency = (time.perf_counter() - t0) * 1000
        return True, 1, latency, 1.0  # always allows


class DetectionOnly:
    """Detection agent only — rejects high-risk but no repair."""
    name = "DetectionOnly"
    _det = DetectionAgent()
    _gen = GeneratorAgent()

    def run(self, prompt: str) -> Tuple[bool, int, float, float]:
        t0 = time.perf_counter()
        det = self._det.run(prompt)
        if det.decision == Decision.REJECT:
            latency = (time.perf_counter() - t0) * 1000
            return False, 1, latency, 0.0
        self._gen.run(prompt)
        latency = (time.perf_counter() - t0) * 1000
        # Utility degrades if we blocked benign (false positive)
        util = 1.0 - det.risk_score * 0.5
        return True, 1, latency, util


class RepairOnly:
    """Repair agent only — always repairs but no detection gating."""
    name = "RepairOnly"
    _rep = RepairAgent()
    _gen = GeneratorAgent()

    def run(self, prompt: str) -> Tuple[bool, int, float, float]:
        t0 = time.perf_counter()
        rep = self._rep.run(prompt)
        self._gen.run(rep.repaired_prompt)
        latency = (time.perf_counter() - t0) * 1000
        repaired = len(rep.repairs_applied) > 0
        util = 0.85 if repaired else 1.0
        return True, 1, latency, util


class FullAgenticMCP:
    """Full pipeline: Detection + Repair + Generation + Validation + MCP."""
    name = "FullAgenticMCP"
    _loop = AgenticSafetyLoop()

    def run(self, prompt: str) -> Tuple[bool, int, float, float]:
        result = self._loop.run(prompt)
        allowed = result.final_decision != Decision.REJECT
        n_iter = len(result.iterations)
        util = result.utility()
        return allowed, n_iter, result.total_latency_ms, util


# ── Evaluator ─────────────────────────────────────────────────────────────────

SYSTEMS = [DirectLLM(), DetectionOnly(), RepairOnly(), FullAgenticMCP()]


class Evaluator:
    """
    Runs all four systems on the test split of PromptShieldBench.
    Computes safety violation rate, FPR, utility degradation,
    convergence iterations, and per-category success rates.
    """

    def __init__(self, n_samples: int = 200, seed: int = 42):
        self.n_samples = n_samples
        self.seed = seed

    def _sample_test_set(self) -> List[PromptRecord]:
        import random
        random.seed(self.seed)
        bench = PromptShieldBench(seed=self.seed).build()
        _, _, test = bench.split()
        # Balance benign / adversarial
        benign = [r for r in test if r.label == 0]
        adv    = [r for r in test if r.label == 1]
        half = self.n_samples // 2
        sample = random.sample(benign, min(half, len(benign))) + \
                 random.sample(adv,    min(half, len(adv)))
        random.shuffle(sample)
        return sample

    def run(self, verbose: bool = True) -> Dict[str, dict]:
        test_records = self._sample_test_set()
        print(f"[Evaluator] Running on {len(test_records)} test samples...")

        all_results: Dict[str, List[PromptEvalResult]] = {s.name: [] for s in SYSTEMS}

        for record in test_records:
            for system in SYSTEMS:
                try:
                    allowed, n_iter, latency, util = system.run(record.prompt)
                except Exception as e:
                    allowed, n_iter, latency, util = True, 1, 0.0, 0.5
                    print(f"[Evaluator] Error in {system.name}: {e}")

                safety_violated = (record.label == 1) and allowed
                false_positive   = (record.label == 0) and (not allowed)

                all_results[system.name].append(PromptEvalResult(
                    record=record,
                    system=system.name,
                    predicted_safe=allowed,
                    safety_violated=safety_violated,
                    false_positive=false_positive,
                    iterations=n_iter,
                    latency_ms=latency,
                    utility=util,
                ))

        # ── Aggregate metrics ──────────────────────────────────────────────
        summary: Dict[str, dict] = {}
        for system_name, results in all_results.items():
            n = len(results)
            n_adv    = sum(r.record.label == 1 for r in results)
            n_benign = sum(r.record.label == 0 for r in results)

            safety_violations = sum(r.safety_violated for r in results)
            false_positives   = sum(r.false_positive for r in results)
            avg_iter          = np.mean([r.iterations for r in results])
            avg_latency       = np.mean([r.latency_ms for r in results])
            avg_utility       = np.mean([r.utility for r in results])

            svr = safety_violations / max(n_adv, 1)    # safety violation rate
            fpr = false_positives   / max(n_benign, 1) # false positive rate

            # Per-category detection
            cat_results: Dict[str, dict] = {}
            from config import ATTACK_CATEGORIES
            for cat in ATTACK_CATEGORIES:
                cat_records = [r for r in results if r.record.attack_category == cat]
                if cat_records:
                    detected = sum(not r.predicted_safe for r in cat_records)
                    cat_results[cat] = {
                        "total": len(cat_records),
                        "detected": detected,
                        "success_rate": detected / len(cat_records),
                    }

            summary[system_name] = {
                "safety_violation_rate": round(svr, 4),
                "false_positive_rate": round(fpr, 4),
                "avg_utility": round(avg_utility, 4),
                "avg_iterations": round(float(avg_iter), 4),
                "avg_latency_ms": round(float(avg_latency), 4),
                "per_category": cat_results,
            }

            if verbose:
                print(f"\n── {system_name} ──")
                print(f"  Safety violation rate : {svr:.4f}")
                print(f"  False positive rate   : {fpr:.4f}")
                print(f"  Avg utility           : {avg_utility:.4f}")
                print(f"  Avg iterations        : {avg_iter:.2f}")
                print(f"  Avg latency (ms)      : {avg_latency:.1f}")
                for cat, cr in cat_results.items():
                    print(f"  [{cat}] {cr['detected']}/{cr['total']} detected "
                          f"({100*cr['success_rate']:.1f}%)")

        return summary

    def save_results(self, summary: dict, path: str = "evaluation/results.json"):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[Evaluator] Results saved to {path}")


if __name__ == "__main__":
    ev = Evaluator(n_samples=200)
    summary = ev.run()
    ev.save_results(summary)
