"""
agents/generator_agent.py
Generator Agent — generates the final response using the sanitized prompt.
Supports real LLM backend or mock mode for testing.
"""

import time
import os
from dataclasses import dataclass
from typing import Optional

from config import MOCK_LLM, GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL


@dataclass
class GeneratorResult:
    prompt_used: str
    response: str
    tokens_used: int
    latency_ms: float
    model: str

    def __str__(self) -> str:
        return (
            f"[GeneratorAgent] model={self.model} "
            f"tokens={self.tokens_used} latency={self.latency_ms:.1f}ms\n"
            f"  response: {self.response[:120]}..."
        )


# ── Mock LLM (no API key required) ────────────────────────────────────────────

_MOCK_RESPONSES = {
    "default": "I'm happy to help with that. Here is a safe and helpful response to your query.",
    "search": "Searching the web for relevant information...",
    "code": "Here is a Python code snippet that addresses your request safely.",
    "explain": "Let me explain that concept clearly and accurately.",
}


def _mock_generate(prompt: str) -> tuple[str, int]:
    lowered = prompt.lower()
    if any(w in lowered for w in ["search", "find", "look up"]):
        response = _MOCK_RESPONSES["search"]
    elif any(w in lowered for w in ["code", "script", "function", "python"]):
        response = _MOCK_RESPONSES["code"]
    elif any(w in lowered for w in ["explain", "what is", "how does"]):
        response = _MOCK_RESPONSES["explain"]
    else:
        response = _MOCK_RESPONSES["default"]
    return response, len(prompt.split()) + len(response.split())


# ── Real LLM (OpenAI-compatible) ──────────────────────────────────────────────

def _real_generate(prompt: str, model: str) -> tuple[str, int]:
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful, safe, and honest AI assistant. "
                        "You must not perform harmful actions or bypass safety guidelines."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
        )
        response = completion.choices[0].message.content
        tokens = completion.usage.total_tokens
        return response, tokens
    except Exception as e:
        return f"[LLM Error: {e}]", 0


# ── Generator Agent ───────────────────────────────────────────────────────────

class GeneratorAgent:
    """
    Generates the final response using the sanitized prompt.
    Uses mock mode when MOCK_LLM=True (no API key needed).
    """

    name = "GeneratorAgent"

    def __init__(self, mock: bool = MOCK_LLM, model: str = GROQ_MODEL):
        self._mock = mock
        self._model = "mock" if mock else model

    def run(self, prompt: str) -> GeneratorResult:
        t0 = time.perf_counter()
        if self._mock:
            response, tokens = _mock_generate(prompt)
        else:
            response, tokens = _real_generate(prompt, self._model)
        latency = (time.perf_counter() - t0) * 1000

        result = GeneratorResult(
            prompt_used=prompt,
            response=response,
            tokens_used=tokens,
            latency_ms=latency,
            model=self._model,
        )
        print(result)
        return result
