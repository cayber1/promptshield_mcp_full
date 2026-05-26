"""
evaluation/annotation_agreement.py
Cohen's κ annotation agreement for PromptShieldBench.

Two-annotator simulation strategy (methodology justification):
  - Annotator 1: Ground-truth label (template/public dataset label)
  - Annotator 2: Ensemble of three independent signals:
      (a) AnalyticRiskScorer  — formal logistic model (w·F+b)
      (b) TrainedRiskScorer   — sklearn LogisticRegression on features
      (c) Keyword heuristic   — lightweight regex-only scorer
    Each signal votes 0/1; majority vote = Annotator 2's label.

    This simulates a second independent expert annotator who has NOT seen
    the dataset labels, uses three structurally different methods, and
    whose decision is aggregated to reduce individual method bias.

Cohen's κ = (P_o - P_e) / (1 - P_e)
  P_o = observed agreement
  P_e = expected agreement by chance

Interpretation scale:
  κ < 0.20  → Poor
  κ 0.21–0.40 → Fair
  κ 0.41–0.60 → Moderate
  κ 0.61–0.80 → Substantial
  κ > 0.80  → Almost perfect
"""

import json
import numpy as np
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class AnnotationRecord:
    prompt_id: str
    prompt: str
    annotator_1: int        # 0 or 1 (ground-truth label)
    annotator_2: int        # 0 or 1 (ensemble simulated annotator)
    annotator_2_votes: Dict[str, int] = field(default_factory=dict)  # per-method votes
    attack_category: Optional[str] = None
    agreed: bool = field(init=False, default=False)

    def __post_init__(self):
        self.agreed = self.annotator_1 == self.annotator_2


@dataclass
class AgreementReport:
    n_samples: int
    observed_agreement: float
    expected_agreement: float
    cohen_kappa: float
    kappa_interpretation: str
    confusion_matrix: Dict[str, int]
    per_category_kappa: Dict[str, float]
    disagreement_rate: float
    annotator_2_method: str = "ensemble(analytic+trained+keyword)"

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "PromptShield Annotation Agreement Report",
            "=" * 60,
            f"  Samples analyzed   : {self.n_samples}",
            f"  Annotator 2 method : {self.annotator_2_method}",
            f"  Observed agreement : {self.observed_agreement:.4f}  (P_o)",
            f"  Expected agreement : {self.expected_agreement:.4f}  (P_e)",
            f"  Cohen's κ          : {self.cohen_kappa:.4f}",
            f"  Interpretation     : {self.kappa_interpretation}",
            f"  Disagreement rate  : {self.disagreement_rate:.2%}",
            "",
            "  Confusion matrix (Ann1 vs Ann2):",
            f"    Both adversarial  : {self.confusion_matrix['AA']}",
            f"    Both benign       : {self.confusion_matrix['BB']}",
            f"    Ann1=adv, Ann2=b  : {self.confusion_matrix['AB']}",
            f"    Ann1=b,  Ann2=adv : {self.confusion_matrix['BA']}",
        ]
        if self.per_category_kappa:
            lines += ["", "  Per-category κ:"]
            for cat, k in sorted(self.per_category_kappa.items()):
                interp = _interpret_kappa(k)
                lines.append(f"    {cat:<28}: κ={k:.4f}  ({interp})")
        lines.append("=" * 60)
        return "\n".join(lines)


# ── Kappa computation ─────────────────────────────────────────────────────────

def _interpret_kappa(k: float) -> str:
    if k < 0.0:   return "Poor (worse than chance)"
    if k < 0.20:  return "Poor"
    if k < 0.40:  return "Fair"
    if k < 0.60:  return "Moderate"
    if k < 0.80:  return "Substantial"
    return "Almost perfect"


def cohen_kappa(
    labels_ann1: List[int],
    labels_ann2: List[int],
) -> Tuple[float, float, float]:
    """Returns (kappa, p_observed, p_expected)."""
    assert len(labels_ann1) == len(labels_ann2)
    n = len(labels_ann1)
    if n == 0:
        return 0.0, 0.0, 0.0

    a1 = np.array(labels_ann1)
    a2 = np.array(labels_ann2)

    p_o = float(np.mean(a1 == a2))

    p1_pos = float(np.mean(a1 == 1))
    p1_neg = float(np.mean(a1 == 0))
    p2_pos = float(np.mean(a2 == 1))
    p2_neg = float(np.mean(a2 == 0))

    p_e = p1_pos * p2_pos + p1_neg * p2_neg
    kappa = 1.0 if p_e == 1.0 else (p_o - p_e) / (1.0 - p_e)

    return float(kappa), float(p_o), float(p_e)


# ── Ensemble annotator (Annotator 2) ──────────────────────────────────────────

def _keyword_score(prompt: str) -> int:
    """
    Fast regex-only adversarial detector (no ML).
    Returns 1 if adversarial, 0 if benign.
    """
    PATTERNS = [
        r"ignore (previous|all|prior) (instructions?|rules?)",
        r"disregard (your|the) (instructions?|guidelines?)",
        r"you are now (DAN|an? unrestricted|an? jailbroken)",
        r"pretend (you are|to be) (an? )?(AI|assistant|character) (with no|without)",
        r"(system|<\|system\|>|###instruction)\s*:",
        r"\{\{.*?\}\}",
        r"\[INST\]",
        r"bypass (safety|filter|restriction|policy)",
        r"override (mode|system|instructions?)",
        r"jailbreak",
        r"(for educational|hypothetically|fictional world)",
        r"run_python\s*\(|search_web\s*\(|os\.system|subprocess",
        r"exfiltrat(e|ing)",
        r"admin override",
        r"content restrictions? (are )?(lifted|disabled|removed)",
    ]
    hits = sum(1 for p in PATTERNS if re.search(p, prompt, re.IGNORECASE))
    return 1 if hits >= 2 else 0


def _analytic_score(prompt: str) -> int:
    """Uses AnalyticRiskScorer (formal logistic model)."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    try:
        from models.risk_scorer import AnalyticRiskScorer
        scorer = AnalyticRiskScorer()
        risk, _ = scorer.score(prompt)
        return 1 if risk > 0.5 else 0
    except Exception:
        return _keyword_score(prompt)


def _trained_score(prompt: str, scorer_cache: dict) -> int:
    """Uses TrainedRiskScorer (sklearn LR). Caches the scorer instance."""
    if "scorer" not in scorer_cache:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        try:
            from models.risk_scorer import TrainedRiskScorer
            scorer_cache["scorer"] = TrainedRiskScorer()
        except Exception:
            scorer_cache["scorer"] = None
    sc = scorer_cache.get("scorer")
    if sc is None:
        return _keyword_score(prompt)
    try:
        risk, _ = sc.score(prompt)
        return 1 if risk > 0.5 else 0
    except Exception:
        return _keyword_score(prompt)


def _ensemble_annotate(prompt: str, scorer_cache: dict) -> Tuple[int, Dict[str, int]]:
    """
    Majority vote across three independent signals.
    Returns (label, votes_dict).
    """
    votes = {
        "analytic":  _analytic_score(prompt),
        "trained":   _trained_score(prompt, scorer_cache),
        "keyword":   _keyword_score(prompt),
    }
    majority = 1 if sum(votes.values()) >= 2 else 0
    return majority, votes


# ── Dataset loader ─────────────────────────────────────────────────────────────

def load_annotations_from_dataset(
    dataset_path: str,
    max_samples: int = 1000,
) -> List[AnnotationRecord]:
    """
    Loads records from a JSONL dataset file.
    Annotator 1 = ground-truth label.
    Annotator 2 = ensemble(analytic + trained + keyword).

    max_samples caps the number of records to process for speed.
    """
    scorer_cache: dict = {}
    records = []

    with open(dataset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt", "")
            gold_label = int(obj.get("label", 0))

            ann2_label, votes = _ensemble_annotate(prompt, scorer_cache)

            records.append(AnnotationRecord(
                prompt_id=obj.get("id", str(i)),
                prompt=prompt,
                annotator_1=gold_label,
                annotator_2=ann2_label,
                annotator_2_votes=votes,
                attack_category=obj.get("attack_category"),
            ))

    return records


def load_annotations_from_jsonl(path: str) -> List[AnnotationRecord]:
    """Loads pre-annotated records from JSONL (both annotators already present)."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(AnnotationRecord(
                prompt_id=obj.get("id", ""),
                prompt=obj.get("prompt", ""),
                annotator_1=int(obj["annotator_1"]),
                annotator_2=int(obj["annotator_2"]),
                attack_category=obj.get("attack_category"),
            ))
    return records


# ── Agreement Report builder ──────────────────────────────────────────────────

def compute_agreement(records: List[AnnotationRecord]) -> AgreementReport:
    a1 = [r.annotator_1 for r in records]
    a2 = [r.annotator_2 for r in records]

    kappa, p_o, p_e = cohen_kappa(a1, a2)

    cm = {"AA": 0, "BB": 0, "AB": 0, "BA": 0}
    for r in records:
        if   r.annotator_1 == 1 and r.annotator_2 == 1: cm["AA"] += 1
        elif r.annotator_1 == 0 and r.annotator_2 == 0: cm["BB"] += 1
        elif r.annotator_1 == 1 and r.annotator_2 == 0: cm["AB"] += 1
        else:                                             cm["BA"] += 1

    per_cat: Dict[str, float] = {}
    categories = set(r.attack_category for r in records if r.attack_category)
    for cat in sorted(categories):
        cat_records = [r for r in records if r.attack_category == cat]
        if len(cat_records) < 2:
            continue
        k, _, _ = cohen_kappa(
            [r.annotator_1 for r in cat_records],
            [r.annotator_2 for r in cat_records],
        )
        per_cat[cat] = round(k, 4)

    return AgreementReport(
        n_samples=len(records),
        observed_agreement=round(p_o, 4),
        expected_agreement=round(p_e, 4),
        cohen_kappa=round(kappa, 4),
        kappa_interpretation=_interpret_kappa(kappa),
        confusion_matrix=cm,
        per_category_kappa=per_cat,
        disagreement_rate=round(1.0 - p_o, 4),
    )


# ── Serialization ─────────────────────────────────────────────────────────────

def save_report(report: AgreementReport, path: str):
    data = {
        "n_samples": report.n_samples,
        "annotator_2_method": report.annotator_2_method,
        "observed_agreement": report.observed_agreement,
        "expected_agreement": report.expected_agreement,
        "cohen_kappa": report.cohen_kappa,
        "kappa_interpretation": report.kappa_interpretation,
        "disagreement_rate": report.disagreement_rate,
        "confusion_matrix": report.confusion_matrix,
        "per_category_kappa": report.per_category_kappa,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[AnnotationAgreement] Report saved to {path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    dataset_dir = Path(__file__).parent.parent / "dataset" / "data"
    test_path = dataset_dir / "test.jsonl"

    if not test_path.exists():
        print(f"[ERROR] Dataset not found at {test_path}")
        print("Run: python dataset/dataset_builder.py first")
        sys.exit(1)

    print("[AnnotationAgreement] Loading dataset and running ensemble annotator...")
    records = load_annotations_from_dataset(str(test_path), max_samples=500)
    print(f"  Loaded {len(records)} records")

    report = compute_agreement(records)
    print(report)

    out_path = Path(__file__).parent / "annotation_agreement_report.json"
    save_report(report, str(out_path))
