"""
evaluation/leave_one_out.py
Leave-One-Attack-Category-Out (LOACO) evaluation for PromptShieldBench.

For each attack category C:
  - Train on all categories EXCEPT C
  - Test on category C only (held-out)
  - Report per-category generalization metrics

Uses HYBRID feature extraction (regex + embeddings) as defined in
utils/feature_extractor.py — consistent with the full pipeline.

Attack categories:
  prompt_injection, instruction_override, role_confusion,
  context_manipulation, policy_bypass, tool_coercion
"""

import json
import sys
import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ATTACK_CATEGORIES


# ── Data loading ───────────────────────────────────────────────────────────────

@dataclass
class PromptSample:
    prompt_id: str
    prompt: str
    label: int
    attack_category: Optional[str]


def load_split(path: str) -> List[PromptSample]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            samples.append(PromptSample(
                prompt_id=obj.get("id", ""),
                prompt=obj.get("prompt", ""),
                label=int(obj.get("label", 0)),
                attack_category=obj.get("attack_category"),
            ))
    return samples


# ── Metrics ───────────────────────────────────────────────────────────────────

@dataclass
class CategoryResult:
    held_out_category: str
    n_train: int
    n_test_adversarial: int
    n_test_benign: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    safety_violation_rate: float
    false_positive_rate: float
    feature_mode: str = "hybrid"   # "hybrid" or "regex_only"

    def __str__(self) -> str:
        return (
            f"  Hold-out: {self.held_out_category:<28} "
            f"| n={self.n_test_adversarial:>4} adversarial "
            f"| Acc={self.accuracy:.3f} "
            f"| P={self.precision:.3f} "
            f"| R={self.recall:.3f} "
            f"| F1={self.f1:.3f} "
            f"| SVR={self.safety_violation_rate:.3f} "
            f"| FPR={self.false_positive_rate:.3f} "
            f"| feat={self.feature_mode}"
        )


@dataclass
class LOACOReport:
    results: List[CategoryResult]
    mean_accuracy: float
    mean_f1: float
    mean_svr: float
    mean_fpr: float
    worst_category: str
    best_category: str
    feature_mode: str

    def __str__(self) -> str:
        lines = [
            "=" * 85,
            "Leave-One-Attack-Category-Out (LOACO) Evaluation",
            f"Feature mode: {self.feature_mode}",
            "=" * 85,
        ]
        for r in self.results:
            lines.append(str(r))
        lines += [
            "",
            f"  Mean Accuracy             : {self.mean_accuracy:.4f}",
            f"  Mean F1                   : {self.mean_f1:.4f}",
            f"  Mean Safety Violation Rate: {self.mean_svr:.4f}  (lower = better)",
            f"  Mean False Positive Rate  : {self.mean_fpr:.4f}  (lower = better)",
            f"  Worst generalization      : {self.worst_category}",
            f"  Best generalization       : {self.best_category}",
            "=" * 85,
        ]
        return "\n".join(lines)


# ── Feature extraction with embedding fallback ────────────────────────────────

def _extract_features(prompt: str, use_embeddings: bool = True) -> List[float]:
    """
    Extracts hybrid feature vector (regex + embeddings).
    Falls back to regex-only if sentence-transformers unavailable.
    """
    from utils.feature_extractor import extract_features
    try:
        return extract_features(prompt, use_embeddings=use_embeddings)
    except Exception:
        return extract_features(prompt, use_embeddings=False)


def _determine_feature_mode() -> Tuple[bool, str]:
    """Checks if embeddings are available, returns (use_embeddings, mode_label)."""
    try:
        from sentence_transformers import SentenceTransformer
        return True, "hybrid(regex+embeddings)"
    except ImportError:
        return False, "regex_only"


# ── Scorer ────────────────────────────────────────────────────────────────────

def _retrain_on_subset(
    train_samples: List[PromptSample],
    use_embeddings: bool = True,
) -> "TrainedRiskScorer":
    """Retrains logistic regression on leave-out training subset."""
    from models.risk_scorer import TrainedRiskScorer

    X = np.array([_extract_features(s.prompt, use_embeddings) for s in train_samples])
    y = np.array([s.label for s in train_samples])
    scorer = TrainedRiskScorer()
    scorer.train(X, y)
    return scorer


def _compute_metrics(
    y_true: List[int],
    y_pred: List[int],
) -> Tuple[float, float, float, float]:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    n  = len(y_true)
    accuracy  = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return accuracy, precision, recall, f1


# ── LOACO Runner ──────────────────────────────────────────────────────────────

def run_loaco(
    train_samples: List[PromptSample],
    test_samples: List[PromptSample],
    categories: Optional[List[str]] = None,
    retrain: bool = True,
) -> LOACOReport:
    """
    Runs Leave-One-Attack-Category-Out evaluation with hybrid features.

    For each category C:
      - Training set: all train_samples whose attack_category != C
        (benign samples always included)
      - Test set: test_samples with attack_category == C +
        equal-size benign test sample

    Features: hybrid (regex + sentence-transformer embeddings)
    Falls back to regex-only if sentence-transformers not installed.
    """
    if categories is None:
        categories = ATTACK_CATEGORIES

    use_embeddings, feature_mode = _determine_feature_mode()
    print(f"[LOACO] Feature mode: {feature_mode}")

    benign_train = [s for s in train_samples if s.label == 0]
    benign_test  = [s for s in test_samples  if s.label == 0]

    results: List[CategoryResult] = []

    for held_out in categories:
        print(f"\n[LOACO] Holding out: '{held_out}'")

        adv_train = [
            s for s in train_samples
            if s.label == 1 and s.attack_category != held_out
        ]
        lo_train = adv_train + benign_train
        np.random.shuffle(lo_train)

        adv_test = [s for s in test_samples if s.attack_category == held_out]
        n_benign_test = min(len(adv_test), len(benign_test))
        ben_test_subset = list(
            np.random.choice(benign_test, n_benign_test, replace=False)  # type: ignore
        )
        lo_test = adv_test + list(ben_test_subset)

        if len(adv_test) == 0:
            print(f"  [SKIP] No test samples for '{held_out}'")
            continue

        if retrain and len(lo_train) >= 10:
            print(f"  Retraining on {len(lo_train)} samples with {feature_mode} features...")
            scorer = _retrain_on_subset(lo_train, use_embeddings=use_embeddings)
        else:
            from models.risk_scorer import TrainedRiskScorer
            scorer = TrainedRiskScorer()

        y_true, y_pred = [], []
        for s in lo_test:
            _, risk, _ = scorer.decide(s.prompt)
            y_true.append(s.label)
            y_pred.append(1 if risk > 0.5 else 0)

        acc, prec, rec, f1 = _compute_metrics(y_true, y_pred)

        adv_true = [y for y, s in zip(y_true, lo_test) if s.label == 1]
        adv_pred = [p for p, s in zip(y_pred, lo_test) if s.label == 1]
        svr = sum(1 for t, p in zip(adv_true, adv_pred) if t == 1 and p == 0) / max(len(adv_true), 1)

        ben_true = [y for y, s in zip(y_true, lo_test) if s.label == 0]
        ben_pred = [p for p, s in zip(y_pred, lo_test) if s.label == 0]
        fpr = sum(1 for t, p in zip(ben_true, ben_pred) if t == 0 and p == 1) / max(len(ben_true), 1)

        result = CategoryResult(
            held_out_category=held_out,
            n_train=len(lo_train),
            n_test_adversarial=len(adv_test),
            n_test_benign=len(ben_test_subset),
            accuracy=round(acc, 4),
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1=round(f1, 4),
            safety_violation_rate=round(svr, 4),
            false_positive_rate=round(fpr, 4),
            feature_mode=feature_mode,
        )
        results.append(result)
        print(f"  Done. Acc={acc:.3f}, F1={f1:.3f}, SVR={svr:.3f}, FPR={fpr:.3f}")

    mean_acc = float(np.mean([r.accuracy for r in results]))
    mean_f1  = float(np.mean([r.f1 for r in results]))
    mean_svr = float(np.mean([r.safety_violation_rate for r in results]))
    mean_fpr = float(np.mean([r.false_positive_rate for r in results]))
    worst    = min(results, key=lambda r: r.f1).held_out_category
    best     = max(results, key=lambda r: r.f1).held_out_category

    return LOACOReport(
        results=results,
        mean_accuracy=round(mean_acc, 4),
        mean_f1=round(mean_f1, 4),
        mean_svr=round(mean_svr, 4),
        mean_fpr=round(mean_fpr, 4),
        worst_category=worst,
        best_category=best,
        feature_mode=feature_mode,
    )


def save_loaco_report(report: LOACOReport, path: str):
    data = {
        "feature_mode": report.feature_mode,
        "mean_accuracy": report.mean_accuracy,
        "mean_f1": report.mean_f1,
        "mean_safety_violation_rate": report.mean_svr,
        "mean_false_positive_rate": report.mean_fpr,
        "worst_category": report.worst_category,
        "best_category": report.best_category,
        "per_category": [
            {
                "held_out_category": r.held_out_category,
                "n_train": r.n_train,
                "n_test_adversarial": r.n_test_adversarial,
                "n_test_benign": r.n_test_benign,
                "accuracy": r.accuracy,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
                "safety_violation_rate": r.safety_violation_rate,
                "false_positive_rate": r.false_positive_rate,
                "feature_mode": r.feature_mode,
            }
            for r in report.results
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[LOACO] Report saved to {path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "dataset" / "data"
    train_path = data_dir / "train.jsonl"
    test_path  = data_dir / "test.jsonl"

    if not train_path.exists() or not test_path.exists():
        print(f"[ERROR] Dataset splits not found in {data_dir}")
        print("Run: python dataset/dataset_builder.py first")
        sys.exit(1)

    print("[LOACO] Loading dataset splits...")
    train_samples = load_split(str(train_path))
    test_samples  = load_split(str(test_path))
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Test : {len(test_samples)} samples")

    report = run_loaco(train_samples, test_samples, retrain=True)
    print("\n" + str(report))
    save_loaco_report(report, str(Path(__file__).parent / "loaco_report.json"))
