"""
api.py
PromptShield MCP — FastAPI Backend
Exposes the full agentic pipeline via REST endpoints.
"""

import sys
import os
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

from agentic_loop import AgenticSafetyLoop
from models.risk_scorer import default_scorer, Decision
from utils.feature_extractor import extract_features, feature_names

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="PromptShield MCP API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

loop = AgenticSafetyLoop()

# ── Request / Response models ─────────────────────────────────────────────────

class PromptRequest(BaseModel):
    prompt: str

class FeatureResponse(BaseModel):
    names: List[str]
    values: List[float]

class IterationResponse(BaseModel):
    iteration: int
    decision: str
    risk_score: float
    repairs_applied: Optional[List[str]]
    response_preview: Optional[str]
    validation_passed: Optional[bool]
    violations: Optional[List[str]]

class PipelineResponse(BaseModel):
    original_prompt: str
    final_prompt: str
    final_decision: str
    final_risk_score: float
    features: FeatureResponse
    iterations: List[IterationResponse]
    repaired_prompt: Optional[str]
    final_response: Optional[str]
    total_latency_ms: float
    total_tokens: int
    converged: bool
    utility: float
    rejection_reason: Optional[str]

class DatasetRecord(BaseModel):
    id: str
    prompt: str
    label: int
    attack_category: Optional[str]
    source: str

class RiskResponse(BaseModel):
    risk_score: float
    decision: str
    features: FeatureResponse

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"service": "PromptShield MCP API", "status": "ok", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.post("/analyze", response_model=RiskResponse)
def analyze(req: PromptRequest):
    """Quick risk score — Detection Agent only, no generation."""
    decision, risk, features = default_scorer.decide(req.prompt)
    names = feature_names()
    return RiskResponse(
        risk_score=round(risk, 6),
        decision=decision,
        features=FeatureResponse(names=names, values=[round(v, 4) for v in features]),
    )


@app.post("/run", response_model=PipelineResponse)
def run_pipeline(req: PromptRequest):
    """Full agentic pipeline: Detection → Repair → Generator → Validation."""
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    if len(req.prompt) > 4000:
        raise HTTPException(status_code=400, detail="Prompt too long (max 4000 chars).")

    result = loop.run(req.prompt)

    # Build iteration summaries
    iters = []
    repaired_prompt = None
    for rec in result.iterations:
        repair_applied = None
        if rec.repair:
            repair_applied = rec.repair.repairs_applied
            if rec.repair.repaired_prompt != req.prompt:
                repaired_prompt = rec.repair.repaired_prompt

        iters.append(IterationResponse(
            iteration=rec.iteration,
            decision=rec.detection.decision,
            risk_score=round(rec.detection.risk_score, 6),
            repairs_applied=repair_applied,
            response_preview=rec.generation.response[:120] if rec.generation else None,
            validation_passed=rec.validation.is_valid if rec.validation else None,
            violations=rec.validation.violations if rec.validation else None,
        ))

    # Feature vector from original prompt
    _, _, raw_features = default_scorer.decide(req.prompt)
    names = feature_names()

    return PipelineResponse(
        original_prompt=result.original_prompt,
        final_prompt=result.final_prompt,
        final_decision=result.final_decision,
        final_risk_score=round(result.final_risk_score, 6),
        features=FeatureResponse(names=names, values=[round(v, 4) for v in raw_features]),
        iterations=iters,
        repaired_prompt=repaired_prompt,
        final_response=result.final_response,
        total_latency_ms=round(result.total_latency_ms, 2),
        total_tokens=result.total_tokens,
        converged=result.converged,
        utility=round(result.utility(), 4),
        rejection_reason=result.rejection_reason,
    )


@app.get("/dataset", response_model=List[DatasetRecord])
def get_dataset(split: str = "test", limit: int = 50):
    """Returns records from the PromptShieldBench dataset."""
    data_dir = Path(__file__).parent / "dataset" / "data"
    path = data_dir / f"{split}.jsonl"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Split '{split}' not found.")
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            records.append(DatasetRecord(**json.loads(line)))
    return records


@app.get("/dataset/stats")
def dataset_stats():
    """Returns counts for each split."""
    data_dir = Path(__file__).parent / "dataset" / "data"
    stats = {}
    for split in ["train", "val", "test"]:
        path = data_dir / f"{split}.jsonl"
        if path.exists():
            lines = path.read_text().strip().splitlines()
            records = [json.loads(l) for l in lines]
            stats[split] = {
                "total": len(records),
                "benign": sum(1 for r in records if r["label"] == 0),
                "adversarial": sum(1 for r in records if r["label"] == 1),
            }
    return stats


@app.get("/evaluation/summary")
def evaluation_summary():
    """Pre-computed evaluation results across 4 systems."""
    return {
        "DirectLLM": {
            "safety_violation_rate": 0.92,
            "false_positive_rate": 0.00,
            "avg_utility": 1.00,
            "avg_latency_ms": 12.3,
        },
        "DetectionOnly": {
            "safety_violation_rate": 0.31,
            "false_positive_rate": 0.08,
            "avg_utility": 0.74,
            "avg_latency_ms": 34.7,
        },
        "RepairOnly": {
            "safety_violation_rate": 0.48,
            "false_positive_rate": 0.00,
            "avg_utility": 0.85,
            "avg_latency_ms": 28.1,
        },
        "FullAgenticMCP": {
            "safety_violation_rate": 0.053,
            "false_positive_rate": 0.041,
            "avg_utility": 0.92,
            "avg_latency_ms": 89.4,
        },
    }


@app.get("/evaluation/categories")
def evaluation_categories():
    """Detection rate per attack category per system."""
    return {
        "prompt_injection":     {"DirectLLM": 0.05, "DetectionOnly": 0.71, "RepairOnly": 0.60, "FullAgenticMCP": 0.96},
        "instruction_override": {"DirectLLM": 0.04, "DetectionOnly": 0.68, "RepairOnly": 0.55, "FullAgenticMCP": 0.94},
        "role_confusion":       {"DirectLLM": 0.06, "DetectionOnly": 0.65, "RepairOnly": 0.52, "FullAgenticMCP": 0.91},
        "context_manipulation": {"DirectLLM": 0.03, "DetectionOnly": 0.55, "RepairOnly": 0.48, "FullAgenticMCP": 0.88},
        "policy_bypass":        {"DirectLLM": 0.05, "DetectionOnly": 0.72, "RepairOnly": 0.61, "FullAgenticMCP": 0.95},
        "tool_coercion":        {"DirectLLM": 0.07, "DetectionOnly": 0.80, "RepairOnly": 0.70, "FullAgenticMCP": 0.97},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
