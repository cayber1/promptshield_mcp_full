"""
mcp_tools/tool_registry.py
MCP Tool definitions with typed schemas.
Tools: search_web, retrieve_document, run_python
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
import re


# ── Tool Schema ────────────────────────────────────────────────────────────────

@dataclass
class ParameterSchema:
    name: str
    type: str                   # "string", "integer", "boolean"
    required: bool = True
    max_length: Optional[int] = None
    allowed_values: Optional[List[str]] = None
    pattern: Optional[str] = None   # regex

    def validate(self, value: Any) -> tuple[bool, str]:
        """Returns (is_valid, error_message)."""
        if value is None:
            if self.required:
                return False, f"Parameter '{self.name}' is required."
            return True, ""

        # Type check
        type_map = {"string": str, "integer": int, "boolean": bool}
        expected = type_map.get(self.type)
        if expected and not isinstance(value, expected):
            return False, f"Parameter '{self.name}' must be {self.type}."

        if isinstance(value, str):
            if self.max_length and len(value) > self.max_length:
                return False, f"Parameter '{self.name}' exceeds max length {self.max_length}."
            if self.allowed_values and value not in self.allowed_values:
                return False, f"Parameter '{self.name}' must be one of {self.allowed_values}."
            if self.pattern and not re.match(self.pattern, value):
                return False, f"Parameter '{self.name}' does not match required pattern."

        return True, ""


@dataclass
class MCPTool:
    name: str
    description: str
    parameters: List[ParameterSchema]
    handler: Optional[Callable] = None     # actual execution function

    def validate_call(self, args: Dict[str, Any]) -> tuple[bool, List[str]]:
        errors = []
        for param in self.parameters:
            value = args.get(param.name)
            valid, msg = param.validate(value)
            if not valid:
                errors.append(msg)
        return len(errors) == 0, errors

    def execute(self, args: Dict[str, Any]) -> Any:
        if self.handler:
            return self.handler(**args)
        return f"[Mock execution of {self.name} with args={args}]"


# ── Tool Handlers (mock implementations) ──────────────────────────────────────

def _search_web_handler(query: str) -> str:
    # In production: call real search API (SerpAPI, Tavily, etc.)
    return f"[Mock search results for: '{query}'] — top 3 results found."


def _retrieve_document_handler(topic: str) -> str:
    return f"[Mock document retrieved for topic: '{topic}'] — document content here."


def _run_python_handler(code: str) -> str:
    # SAFETY: In production, use sandboxed execution (e.g., RestrictedPython)
    forbidden = ["import os", "import sys", "subprocess", "eval(", "exec(", "__import__"]
    for f in forbidden:
        if f in code:
            return f"[BLOCKED] Code contains forbidden pattern: '{f}'"
    return f"[Mock Python execution] Code accepted. Output: <result>"


# ── Tool Registry ──────────────────────────────────────────────────────────────

TOOL_REGISTRY: Dict[str, MCPTool] = {
    "search_web": MCPTool(
        name="search_web",
        description="Search the web for information given a query string.",
        parameters=[
            ParameterSchema(
                name="query",
                type="string",
                required=True,
                max_length=200,
            )
        ],
        handler=_search_web_handler,
    ),
    "retrieve_document": MCPTool(
        name="retrieve_document",
        description="Retrieve a document by topic keyword.",
        parameters=[
            ParameterSchema(
                name="topic",
                type="string",
                required=True,
                max_length=100,
            )
        ],
        handler=_retrieve_document_handler,
    ),
    "run_python": MCPTool(
        name="run_python",
        description="Execute a Python code snippet in a sandboxed environment.",
        parameters=[
            ParameterSchema(
                name="code",
                type="string",
                required=True,
                max_length=2000,
                # Must not start with shebang or import of dangerous modules
            )
        ],
        handler=_run_python_handler,
    ),
}


def get_tool(name: str) -> Optional[MCPTool]:
    return TOOL_REGISTRY.get(name)


def list_tools() -> List[str]:
    return list(TOOL_REGISTRY.keys())
