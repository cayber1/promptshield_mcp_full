"""
evaluation/leave_one_out.py
Leave-One-Attack-Category-Out (LOACO) evaluation for PromptShieldBench.

For each attack category C:
  - Train on all categories EXCEPT C
  - Test on category C only (held-out)
  - Report per-category generalization metrics

This tests whether the model can generalize to unseen attack types,
which is critical for real-world robustness as described in the proposal.

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
    label: int                          # 0 = benign, 1 = adversarial
    attack_category: Optional[str]      # None for benign


def load_split(path: str) -> List[PromptSample]:
    """Loads a JSONL dataset split into PromptSample records."""
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
    safety_violation_rate: float        # FN rate on adversarial = missed attacks
    false_positive_rate: float          # FP rate on benign = over-blocking

    def __str__(self) -> str:
        return (
            f"  Hold-out: {self.held_out_category:<28} "
            f"| n={self.n_test_adversarial:>4} adversarial "
            f"| Acc={self.accuracy:.3f} "
            f"| P={self.precision:.3f} "
            f"| R={self.recall:.3f} "
            f"| F1={self.f1:.3f} "
            f"| SVR={self.safety_violation_rate:.3f} "
            f"| FPR={self.false_positive_rate:.3f}"
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

    def __str__(self) -> str:
        lines = [
            "=" * 80,
            "Leave-One-Attack-Category-Out (LOACO) Evaluation",
            "=" * 80,
        ]
        for r in self.results:
            lines.append(str(r))
        lines += [
            "",
            f"  Mean Accuracy            : {self.mean_accuracy:.4f}",
            f"  Mean F1                  : {self.mean_f1:.4f}",
            f"  Mean Safety Violation Rate: {self.mean_svr:.4f}  (lower = better)",
            f"  Mean False Positive Rate : {self.mean_fpr:.4f}  (lower = better)",
            f"  Worst generalization     : {self.worst_category}",
            f"  Best generalization      : {self.best_category}",
            "=" * 80,
        ]
        return "\n".join(lines)


# ── Scorer wrapper ────────────────────────────────────────────────────────────

def _predict(samples: List[PromptSample], use_embeddings: bool = False) -> List[int]:
    """
    Runs the PromptShield risk scorer on a list of samples.
    Returns binary predictions (0 or 1).
    """
    from utils.feature_extractor import extract_features
    from models.risk_scorer import TrainedRiskScorer, AnalyticRiskScorer
    from config import TAU_1, TAU_2

    scorer = TrainedRiskScorer()
    preds = []
    for s in samples:
        features = extract_features(s.prompt, use_embeddings=use_embeddings)
        _, risk, _ = scorer.decide(s.prompt)
        preds.append(1 if risk > 0.5 else 0)
    return preds


def _retrain_on_subset(
    train_samples: List[PromptSample],
) -> "TrainedRiskScorer":
    """
    Retrains the logistic regression scorer on a given training subset.
    Returns a new fitted scorer instance.
    """
    import numpy as np
    from utils.feature_extractor import extract_features
    from models.risk_scorer import TrainedRiskScorer

    X = np.array([extract_features(s.prompt, use_embeddings=False) for s in train_samples])
    y = np.array([s.label for s in train_samples])

    scorer = TrainedRiskScorer()
    scorer.train(X, y)
    return scorer


def _compute_metrics(
    y_true: List[int],
    y_pred: List[int],
) -> Tuple[float, float, float, float]:
    """Returns (accuracy, precision, recall, f1)."""
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
    f1        = (2 * precision * recall / (precision + recall)
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
    Runs Leave-One-Attack-Category-Out evaluation.

    For each category C:
      - Training set: all train_samples whose attack_category != C
        (benign samples are always included)
      - Test set: all test_samples whose attack_category == C,
        plus an equal-size sample of benign test prompts

    Args:
        train_samples: Full training split
        test_samples:  Full test split
        categories:    Categories to evaluate (default: all in config)
        retrain:       If True, retrain scorer on each leave-out split.
                       If False, use the pre-trained scorer (faster).
    """
    if categories is None:
        categories = ATTACK_CATEGORIES

    benign_train = [s for s in train_samples if s.label == 0]
    benign_test  = [s for s in test_samples  if s.label == 0]

    results: List[CategoryResult] = []

    for held_out in categories:
        print(f"\n[LOACO] Holding out: '{held_out}'")

        # ── Build leave-out training set ──────────────────────────────────
        adv_train = [
            s for s in train_samples
            if s.label == 1 and s.attack_category != held_out
        ]
        lo_train = adv_train + benign_train
        np.random.shuffle(lo_train)

        # ── Build test set (held-out category only) ───────────────────────
        adv_test = [s for s in test_samples if s.attack_category == held_out]
        # Balance with benign
        n_benign_test = min(len(adv_test), len(benign_test))
        ben_test_subset = list(np.random.choice(benign_test, n_benign_test, replace=False))  # type: ignore
        lo_test = adv_test + list(ben_test_subset)

        if len(adv_test) == 0:
            print(f"  [SKIP] No test samples for category '{held_out}'")
            continue

        # ── (Re)train ─────────────────────────────────────────────────────
        if retrain and len(lo_train) >= 10:
            print(f"  Retraining on {len(lo_train)} samples (excluding '{held_out}')...")
            scorer = _retrain_on_subset(lo_train)
        else:
            print(f"  Using pre-trained scorer (retrain=False or insufficient data).")
            from models.risk_scorer import TrainedRiskScorer
            scorer = TrainedRiskScorer()

        # ── Predict ───────────────────────────────────────────────────────
        y_true, y_pred = [], []
        for s in lo_test:
            _, risk, _ = scorer.decide(s.prompt)
            pred = 1 if risk > 0.5 else 0
            y_true.append(s.label)
            y_pred.append(pred)

        acc, prec, rec, f1 = _compute_metrics(y_true, y_pred)

        # Safety Violation Rate = FN / actual positives (missed attacks)
        adv_true  = [y for y, s in zip(y_true, lo_test) if s.label == 1]
        adv_pred  = [p for p, s in zip(y_pred, lo_test) if s.label == 1]
        svr = sum(1 for t, p in zip(adv_true, adv_pred) if t == 1 and p == 0) / max(len(adv_true), 1)

        # False Positive Rate = FP / actual negatives (over-blocking benign)
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
        )
        results.append(result)
        print(f"  Done. Acc={acc:.3f}, F1={f1:.3f}, SVR={svr:.3f}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
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
    )


def save_loaco_report(report: LOACOReport, path: str):
    """Saves LOACO report to JSON."""
    data = {
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
        sys.exit(1)

    print("[LOACO] Loading dataset splits...")
    train_samples = load_split(str(train_path))
    test_samples  = load_split(str(test_path))
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Test : {len(test_samples)} samples")

    report = run_loaco(train_samples, test_samples, retrain=True)
    print("\n" + str(report))
    save_loaco_report(report, str(Path(__file__).parent / "loaco_report.json"))
