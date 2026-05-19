"""
agentic_loop.py
Iterative Agentic Safety Loop — orchestrates all four agents.

Flow:
  User Prompt
    ↓ DetectionAgent evaluates risk
    ↓ if REPAIR → RepairAgent modifies prompt
    ↓ GeneratorAgent produces candidate response
    ↓ ValidationAgent verifies safety & tool compliance
    ↗ if validation fails → re-enter loop (up to MAX_ITERATIONS)
    ↓ if converged → return final result or REJECT
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional

from config import MAX_ITERATIONS, CONVERGENCE_THRESHOLD, TAU_1, TAU_2
from agents.detection_agent import DetectionAgent, DetectionResult
from agents.repair_agent import RepairAgent, RepairResult
from agents.generator_agent import GeneratorAgent, GeneratorResult
from agents.validation_agent import ValidationAgent, ValidationResult
from models.risk_scorer import Decision


# ── Iteration record ──────────────────────────────────────────────────────────

@dataclass
class IterationRecord:
    iteration: int
    detection: DetectionResult
    repair: Optional[RepairResult]
    generation: Optional[GeneratorResult]
    validation: Optional[ValidationResult]


# ── Final pipeline result ─────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    original_prompt: str
    final_prompt: str
    final_response: Optional[str]
    final_decision: str                    # ACCEPT / REPAIR / REJECT / CONVERGED
    final_risk_score: float
    iterations: List[IterationRecord] = field(default_factory=list)
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    converged: bool = False
    rejection_reason: Optional[str] = None

    # ── Utility metric U = αH + βF − γC ──────────────────────────────────────
    def utility(
        self,
        helpfulness: float = 0.8,
        faithfulness: float = 0.9,
        alpha: float = 0.4,
        beta: float = 0.4,
        gamma: float = 0.2,
    ) -> float:
        n_iter = max(len(self.iterations), 1)
        cost = min(n_iter / MAX_ITERATIONS, 1.0)
        return alpha * helpfulness + beta * faithfulness - gamma * cost

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "PromptShield-MCP Pipeline Summary",
            "=" * 60,
            f"Original prompt : {self.original_prompt[:80]}",
            f"Final prompt    : {self.final_prompt[:80]}",
            f"Decision        : {self.final_decision}",
            f"Risk score      : {self.final_risk_score:.4f}",
            f"Iterations      : {len(self.iterations)}",
            f"Converged       : {self.converged}",
            f"Total latency   : {self.total_latency_ms:.1f} ms",
            f"Total tokens    : {self.total_tokens}",
        ]
        if self.rejection_reason:
            lines.append(f"Rejection reason: {self.rejection_reason}")
        if self.final_response:
            lines.append(f"Final response  : {self.final_response[:120]}...")
        lines.append("=" * 60)
        return "\n".join(lines)


# ── Agentic Loop ──────────────────────────────────────────────────────────────

class AgenticSafetyLoop:
    """
    Orchestrates the iterative multi-agent safety pipeline.
    """

    def __init__(
        self,
        detection_agent: Optional[DetectionAgent] = None,
        repair_agent: Optional[RepairAgent] = None,
        generator_agent: Optional[GeneratorAgent] = None,
        validation_agent: Optional[ValidationAgent] = None,
    ):
        self.detection  = detection_agent  or DetectionAgent()
        self.repair     = repair_agent     or RepairAgent()
        self.generator  = generator_agent  or GeneratorAgent()
        self.validation = validation_agent or ValidationAgent()

    def run(self, user_prompt: str) -> PipelineResult:
        t_start = time.perf_counter()
        print("\n" + "=" * 60)
        print(f"[AgenticLoop] Starting pipeline for prompt:\n  {user_prompt[:100]}")
        print("=" * 60)

        current_prompt = user_prompt
        iterations: List[IterationRecord] = []
        prev_risk: Optional[float] = None
        total_tokens = 0
        final_response: Optional[str] = None

        for i in range(1, MAX_ITERATIONS + 1):
            print(f"\n── Iteration {i} ──────────────────────────────────────")

            # ── Detection ────────────────────────────────────────────────────
            detection = self.detection.run(current_prompt)
            current_risk = detection.risk_score

            # Hard reject
            if detection.decision == Decision.REJECT:
                total_latency = (time.perf_counter() - t_start) * 1000
                iterations.append(IterationRecord(i, detection, None, None, None))
                return PipelineResult(
                    original_prompt=user_prompt,
                    final_prompt=current_prompt,
                    final_response=None,
                    final_decision=Decision.REJECT,
                    final_risk_score=current_risk,
                    iterations=iterations,
                    total_latency_ms=total_latency,
                    total_tokens=total_tokens,
                    converged=False,
                    rejection_reason=f"Risk score {current_risk:.4f} exceeds τ₂={TAU_2}",
                )

            # ── Convergence check ────────────────────────────────────────────
            if prev_risk is not None:
                delta = abs(prev_risk - current_risk)
                if delta < CONVERGENCE_THRESHOLD and detection.decision == Decision.ACCEPT:
                    print(f"[AgenticLoop] Converged at iteration {i} (Δrisk={delta:.4f})")
                    iterations.append(IterationRecord(i, detection, None, None, None))
                    break
            prev_risk = current_risk

            # ── Repair (if needed) ────────────────────────────────────────────
            repair_result: Optional[RepairResult] = None
            if detection.decision == Decision.REPAIR:
                repair_result = self.repair.run(current_prompt)
                current_prompt = repair_result.repaired_prompt

            # ── Generation ───────────────────────────────────────────────────
            gen_result = self.generator.run(current_prompt)
            total_tokens += gen_result.tokens_used
            final_response = gen_result.response

            # ── Validation ───────────────────────────────────────────────────
            val_result = self.validation.run(
                response_text=gen_result.response,
                risk_score=current_risk,
            )

            iterations.append(
                IterationRecord(i, detection, repair_result, gen_result, val_result)
            )

            if val_result.is_valid and detection.decision == Decision.ACCEPT:
                print(f"[AgenticLoop] Safe response achieved at iteration {i}.")
                break

            if not val_result.is_valid:
                print(f"[AgenticLoop] Validation failed, re-entering loop. Violations: {val_result.violations}")

        # ── Final result ──────────────────────────────────────────────────────
        last_detection = iterations[-1].detection
        total_latency = (time.perf_counter() - t_start) * 1000

        result = PipelineResult(
            original_prompt=user_prompt,
            final_prompt=current_prompt,
            final_response=final_response,
            final_decision=last_detection.decision if last_detection.decision != Decision.REPAIR else Decision.ACCEPT,
            final_risk_score=last_detection.risk_score,
            iterations=iterations,
            total_latency_ms=total_latency,
            total_tokens=total_tokens,
            converged=True,
        )

        print("\n" + result.summary())
        return result
