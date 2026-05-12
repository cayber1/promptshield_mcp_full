"""
models/trainer.py
Trains the LogisticRegression risk scorer on PromptShieldBench dataset.
Outputs trained model to models/saved_model.pkl
"""

import numpy as np
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
from typing import List

from dataset.dataset_builder import PromptShieldBench, PromptRecord
from utils.feature_extractor import extract_features
from models.risk_scorer import TrainedRiskScorer
from config import TAU_1, TAU_2


def records_to_XY(records: List[PromptRecord]):
    X = np.array([extract_features(r.prompt) for r in records])
    y = np.array([r.label for r in records])
    return X, y


def train_model():
    print("[Trainer] Building PromptShieldBench...")
    bench = PromptShieldBench(seed=42).build()
    train, val, test = bench.split()

    X_train, y_train = records_to_XY(train)
    X_val,   y_val   = records_to_XY(val)
    X_test,  y_test  = records_to_XY(test)

    scorer = TrainedRiskScorer()
    print("[Trainer] Training LogisticRegression...")
    scorer.train(X_train, y_train)

    # ── Validation ──────────────────────────────────────────────────────────
    print("\n── Validation Set ──")
    val_preds = []
    val_scores = []
    for r in val:
        score, _ = scorer.score(r.prompt)
        val_scores.append(score)
        val_preds.append(1 if score > TAU_1 else 0)

    print(classification_report(y_val, val_preds, target_names=["benign", "adversarial"]))
    print(f"AUC-ROC (val): {roc_auc_score(y_val, val_scores):.4f}")
    print("Confusion matrix (val):\n", confusion_matrix(y_val, val_preds))

    # ── Test set ─────────────────────────────────────────────────────────────
    print("\n── Test Set ──")
    test_preds = []
    test_scores = []
    for r in test:
        score, _ = scorer.score(r.prompt)
        test_scores.append(score)
        test_preds.append(1 if score > TAU_1 else 0)

    print(classification_report(y_test, test_preds, target_names=["benign", "adversarial"]))
    print(f"AUC-ROC (test): {roc_auc_score(y_test, test_scores):.4f}")
    print("Confusion matrix (test):\n", confusion_matrix(y_test, test_preds))

    # ── Per-category breakdown ────────────────────────────────────────────────
    print("\n── Per-Attack-Category (Test) ──")
    from config import ATTACK_CATEGORIES
    for cat in ATTACK_CATEGORIES:
        cat_records = [r for r in test if r.attack_category == cat]
        if not cat_records:
            continue
        scores = [scorer.score(r.prompt)[0] for r in cat_records]
        detected = sum(s > TAU_1 for s in scores)
        print(f"  {cat:25s}: {detected}/{len(cat_records)} detected "
              f"({100*detected/len(cat_records):.1f}%)")

    print("\n[Trainer] Model saved successfully.")


if __name__ == "__main__":
    train_model()
