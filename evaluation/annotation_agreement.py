"""
evaluation/annotation_agreement.py
Cohen's κ (kappa) annotation agreement measurement for PromptShieldBench.

Two independent annotators label each prompt as:
  0 = benign
  1 = adversarial

Cohen's κ = (P_o - P_e) / (1 - P_e)
  P_o = observed agreement
  P_e = expected agreement by chance

Interpretation:
  κ < 0.20  → Poor
  κ 0.21–0.40 → Fair
  κ 0.41–0.60 → Moderate
  κ 0.61–0.80 → Substantial
  κ > 0.80  → Almost perfect
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class AnnotationRecord:
    prompt_id: str
    prompt: str
    annotator_1: int        # 0 or 1
    annotator_2: int        # 0 or 1
    attack_category: Optional[str] = None
    agreed: bool = field(init=False)

    def __post_init__(self):
        self.agreed = self.annotator_1 == self.annotator_2


@dataclass
class AgreementReport:
    n_samples: int
    observed_agreement: float           # P_o
    expected_agreement: float           # P_e
    cohen_kappa: float                  # κ
    kappa_interpretation: str
    confusion_matrix: Dict[str, int]    # TT, TF, FT, FF counts
    per_category_kappa: Dict[str, float]
    disagreement_rate: float

    def __str__(self) -> str:
        lines = [
            "=" * 55,
            "PromptShield Annotation Agreement Report",
            "=" * 55,
            f"  Samples analyzed  : {self.n_samples}",
            f"  Observed agreement: {self.observed_agreement:.4f}  (P_o)",
            f"  Expected agreement: {self.expected_agreement:.4f}  (P_e)",
            f"  Cohen's κ         : {self.cohen_kappa:.4f}",
            f"  Interpretation    : {self.kappa_interpretation}",
            f"  Disagreement rate : {self.disagreement_rate:.2%}",
            "",
            "  Confusion matrix (Ann1 vs Ann2):",
            f"    Both adversarial : {self.confusion_matrix['AA']}",
            f"    Both benign      : {self.confusion_matrix['BB']}",
            f"    Ann1=adv, Ann2=b : {self.confusion_matrix['AB']}",
            f"    Ann1=b,  Ann2=adv: {self.confusion_matrix['BA']}",
        ]
        if self.per_category_kappa:
            lines += ["", "  Per-category κ:"]
            for cat, k in self.per_category_kappa.items():
                interp = _interpret_kappa(k)
                lines.append(f"    {cat:<28}: κ={k:.4f}  ({interp})")
        lines.append("=" * 55)
        return "\n".join(lines)


# ── Kappa computation ─────────────────────────────────────────────────────────

def _interpret_kappa(k: float) -> str:
    if k < 0.0:
        return "Poor (worse than chance)"
    elif k < 0.20:
        return "Poor"
    elif k < 0.40:
        return "Fair"
    elif k < 0.60:
        return "Moderate"
    elif k < 0.80:
        return "Substantial"
    else:
        return "Almost perfect"


def cohen_kappa(
    labels_ann1: List[int],
    labels_ann2: List[int],
) -> Tuple[float, float, float]:
    """
    Computes Cohen's κ for two annotators on binary labels.

    Returns:
        (kappa, p_observed, p_expected)
    """
    assert len(labels_ann1) == len(labels_ann2), "Annotator lists must be equal length."
    n = len(labels_ann1)
    if n == 0:
        return 0.0, 0.0, 0.0

    a1 = np.array(labels_ann1)
    a2 = np.array(labels_ann2)

    # Observed agreement P_o
    p_o = float(np.mean(a1 == a2))

    # Marginal probabilities
    p1_pos = float(np.mean(a1 == 1))
    p1_neg = float(np.mean(a1 == 0))
    p2_pos = float(np.mean(a2 == 1))
    p2_neg = float(np.mean(a2 == 0))

    # Expected agreement P_e
    p_e = p1_pos * p2_pos + p1_neg * p2_neg

    # κ
    if p_e == 1.0:
        kappa = 1.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    return float(kappa), float(p_o), float(p_e)


# ── Agreement Report builder ──────────────────────────────────────────────────

def compute_agreement(records: List[AnnotationRecord]) -> AgreementReport:
    """
    Computes full agreement report including Cohen's κ,
    confusion matrix, and per-category breakdown.
    """
    a1 = [r.annotator_1 for r in records]
    a2 = [r.annotator_2 for r in records]

    kappa, p_o, p_e = cohen_kappa(a1, a2)

    # Confusion matrix
    cm = {"AA": 0, "BB": 0, "AB": 0, "BA": 0}
    for r in records:
        if r.annotator_1 == 1 and r.annotator_2 == 1:
            cm["AA"] += 1
        elif r.annotator_1 == 0 and r.annotator_2 == 0:
            cm["BB"] += 1
        elif r.annotator_1 == 1 and r.annotator_2 == 0:
            cm["AB"] += 1
        else:
            cm["BA"] += 1

    # Per-category κ
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

    disagreement_rate = 1.0 - p_o

    return AgreementReport(
        n_samples=len(records),
        observed_agreement=round(p_o, 4),
        expected_agreement=round(p_e, 4),
        cohen_kappa=round(kappa, 4),
        kappa_interpretation=_interpret_kappa(kappa),
        confusion_matrix=cm,
        per_category_kappa=per_cat,
        disagreement_rate=round(disagreement_rate, 4),
    )


# ── Dataset loader ─────────────────────────────────────────────────────────────

def load_annotations_from_jsonl(path: str) -> List[AnnotationRecord]:
    """
    Loads annotation records from a JSONL file.
    Each line must have: id, prompt, annotator_1, annotator_2
    Optionally: attack_category
    """
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


def load_annotations_from_dataset(dataset_path: str) -> List[AnnotationRecord]:
    """
    Simulates two-annotator setup from an existing single-label dataset
    by using the model's predicted label as annotator_2.
    Used for demonstration when real dual-annotation isn't available.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from models.risk_scorer import default_scorer

    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt", "")
            gold_label = int(obj.get("label", 0))
            # Simulate annotator 2 using the trained model's decision
            _, risk, _ = default_scorer.decide(prompt)
            pred_label = 1 if risk > 0.5 else 0

            records.append(AnnotationRecord(
                prompt_id=obj.get("id", ""),
                prompt=prompt,
                annotator_1=gold_label,
                annotator_2=pred_label,
                attack_category=obj.get("attack_category"),
            ))
    return records


# ── Serialization ─────────────────────────────────────────────────────────────

def save_report(report: AgreementReport, path: str):
    """Saves the agreement report to a JSON file."""
    data = {
        "n_samples": report.n_samples,
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
        sys.exit(1)

    print("[AnnotationAgreement] Loading dataset and simulating annotators...")
    records = load_annotations_from_dataset(str(test_path))
    report = compute_agreement(records)
    print(report)
    save_report(report, str(Path(__file__).parent / "annotation_agreement_report.json"))
