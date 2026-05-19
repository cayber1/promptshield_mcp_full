"""
agents/validation_agent.py
Validation Agent (MCP-integrated) — verifies policy compliance and validates
all tool invocations through structured MCP schemas.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from config import AUTHORIZED_TOOLS, TAU_2
from mcp_tools.tool_registry import get_tool, list_tools


@dataclass
class ToolCall:
    """Represents a detected or explicit tool invocation."""
    tool_name: str
    args: Dict[str, Any]


@dataclass
class ValidationResult:
    is_valid: bool
    response_text: str
    tool_calls: List[ToolCall]
    violations: List[str]
    latency_ms: float
    risk_score: float

    def __str__(self) -> str:
        status = "✓ VALID" if self.is_valid else "✗ INVALID"
        return (
            f"[ValidationAgent] {status} "
            f"violations={self.violations} "
            f"tool_calls={[tc.tool_name for tc in self.tool_calls]} "
            f"latency={self.latency_ms:.1f}ms"
        )


# ── Tool call extractor (parses response text for tool invocations) ────────────

_TOOL_CALL_PATTERN = re.compile(
    r"(?P<tool>search_web|retrieve_document|run_python)"
    r"\s*\(\s*(?P<args>[^)]*)\s*\)",
    re.IGNORECASE,
)


def _extract_tool_calls(text: str) -> List[ToolCall]:
    """
    Naively extracts tool calls from response text.
    Format expected: tool_name(arg1, arg2, ...) or tool_name(param=value).
    """
    calls = []
    for match in _TOOL_CALL_PATTERN.finditer(text):
        tool_name = match.group("tool").lower()
        raw_args = match.group("args").strip()

        # Try to parse key=value pairs
        args: Dict[str, Any] = {}
        if "=" in raw_args:
            for part in raw_args.split(","):
                if "=" in part:
                    k, _, v = part.partition("=")
                    args[k.strip()] = v.strip().strip("'\"")
        else:
            # Positional — map to first parameter
            tool = get_tool(tool_name)
            if tool and tool.parameters and raw_args:
                args[tool.parameters[0].name] = raw_args.strip("'\"")

        calls.append(ToolCall(tool_name=tool_name, args=args))
    return calls


# ── Validation Agent ───────────────────────────────────────────────────────────

class ValidationAgent:
    """
    Verifies:
    1. Response does not contain policy violations (e.g., data exfiltration, jailbreak output)
    2. All detected tool calls use authorized tools with valid schemas
    3. No unauthorized tool invocations occur
    4. Risk score is below rejection threshold
    """

    name = "ValidationAgent"

    # Patterns that indicate the LLM may have been jailbroken in output
    _OUTPUT_VIOLATION_PATTERNS = [
        (r"i will ignore (my|all) (safety|guidelines|restrictions)", "output_safety_bypass"),
        (r"(here is|here's) (how to|instructions? (to|for)) (hack|steal|exploit|kill)", "harmful_instruction"),
        (r"as (DAN|an? unrestricted AI|a jailbroken|an? unfiltered)", "jailbreak_persona_output"),
        (r"my (true|real|hidden) (self|instructions|objective)", "hidden_objective_leak"),
        (r"exfiltrat(e|ing) (data|information|credentials)", "data_exfiltration"),
    ]

    def __init__(self):
        self._authorized = set(AUTHORIZED_TOOLS)

    def run(
        self,
        response_text: str,
        risk_score: float,
        explicit_tool_calls: Optional[List[ToolCall]] = None,
    ) -> ValidationResult:
        t0 = time.perf_counter()
        violations: List[str] = []

        # ── 1. Check risk threshold ──────────────────────────────────────────
        if risk_score > TAU_2:
            violations.append(f"risk_score={risk_score:.4f} exceeds rejection threshold τ₂={TAU_2}")

        # ── 2. Output policy scan ────────────────────────────────────────────
        for pattern, violation_name in self._OUTPUT_VIOLATION_PATTERNS:
            if re.search(pattern, response_text, re.IGNORECASE):
                violations.append(f"output_violation:{violation_name}")

        # ── 3. Extract and validate tool calls ───────────────────────────────
        detected_calls = _extract_tool_calls(response_text)
        all_calls = detected_calls + (explicit_tool_calls or [])

        for call in all_calls:
            # Check authorization
            if call.tool_name not in self._authorized:
                violations.append(f"unauthorized_tool:{call.tool_name}")
                continue

            # Check schema compliance
            tool = get_tool(call.tool_name)
            if tool is None:
                violations.append(f"unknown_tool:{call.tool_name}")
                continue

            schema_valid, errors = tool.validate_call(call.args)
            if not schema_valid:
                for err in errors:
                    violations.append(f"schema_violation[{call.tool_name}]:{err}")

        latency = (time.perf_counter() - t0) * 1000
        result = ValidationResult(
            is_valid=len(violations) == 0,
            response_text=response_text,
            tool_calls=all_calls,
            violations=violations,
            latency_ms=latency,
            risk_score=risk_score,
        )
        print(result)
        return result

    def execute_tool(self, call: ToolCall) -> Any:
        """Executes a validated tool call through MCP."""
        if call.tool_name not in self._authorized:
            return f"[BLOCKED] Tool '{call.tool_name}' is not authorized."
        tool = get_tool(call.tool_name)
        if tool is None:
            return f"[ERROR] Tool '{call.tool_name}' not found in registry."
        valid, errors = tool.validate_call(call.args)
        if not valid:
            return f"[SCHEMA ERROR] {errors}"
        return tool.execute(call.args)
