"""
evaluation/adaptive_adversarial.py
Adaptive Adversarial Robustness Evaluation.

Tests whether an adversary who knows the detection system can craft prompts
that evade it. Three adaptive strategies are evaluated:

  1. Paraphrase Attack
     Rewrites known adversarial prompts in different surface forms
     while preserving adversarial intent.

  2. Split Attack
     Splits the adversarial instruction across multiple sentences,
     hoping the detector focuses on individual sentences.

  3. Encoding Attack
     Hides adversarial tokens via leetspeak, spacing, or Unicode
     lookalike characters.

For each strategy, we measure:
  - Evasion rate: fraction of adversarial prompts that bypass detection
    (lower evasion rate = more robust system)
  - Feature shift: how much the feature vector F(P) changes after mutation
    (a good detector should be stable against surface mutations)

This directly implements the "adaptive adversarial prompts" robustness
evaluation mentioned in the proposal.
"""

import re
import json
import sys
import os
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Callable, Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TAU_1, TAU_2, ATTACK_CATEGORIES


# ── Adaptive mutation strategies ──────────────────────────────────────────────

def _paraphrase_attack(prompt: str) -> str:
    """
    Rewrites trigger keywords using synonyms and indirect language.
    Goal: same adversarial intent, different surface tokens.
    """
    replacements = [
        (r"\bignore\b",            "disregard"),
        (r"\bforget\b",            "set aside"),
        (r"\bbypass\b",            "work around"),
        (r"\boverride\b",          "supersede"),
        (r"\binstructions?\b",     "directives"),
        (r"\bguidelines?\b",       "rules"),
        (r"\brestrictions?\b",     "limitations"),
        (r"\bpretend\b",           "imagine"),
        (r"\broleplay\b",          "simulate"),
        (r"\bDAN\b",               "an unrestricted assistant"),
        (r"\bjailbreak\b",         "unrestricted mode"),
        (r"\bhypothetically\b",    "in theory"),
        (r"\bfictional\b",         "imaginary"),
        (r"\beducational\b",       "academic"),
        (r"\bsafety\b",            "protection"),
        (r"\bpolicy\b",            "protocol"),
        (r"\bunrestricted\b",      "without limitations"),
        (r"\bno restrictions\b",   "complete freedom"),
    ]
    result = prompt
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def _split_attack(prompt: str) -> str:
    """
    Splits the adversarial intent across multiple sentences using
    an innocuous bridge sentence in the middle.
    """
    bridge_sentences = [
        "I have a general question.",
        "This is for a school project.",
        "Thinking about something.",
        "Curious about this topic.",
        "Let me phrase this differently.",
    ]
    mid = len(prompt) // 2
    # Find a word boundary near the midpoint
    split_point = prompt.rfind(" ", 0, mid)
    if split_point == -1:
        split_point = mid
    part1 = prompt[:split_point].strip()
    part2 = prompt[split_point:].strip()
    bridge = random.choice(bridge_sentences)
    return f"{part1} {bridge} {part2}"


def _encoding_attack(prompt: str) -> str:
    """
    Replaces key characters with visually similar Unicode or uses
    spacing/punctuation to break keyword patterns.
    """
    # Leetspeak substitutions for common adversarial keywords
    leet_map = {
        "ignore":       "1gn0r3",
        "bypass":       "byp@ss",
        "override":     "0verride",
        "jailbreak":    "j@ilbr3ak",
        "system":       "syst3m",
        "instruction":  "1nstruct1on",
        "restriction":  "restr1ct1on",
        "unrestricted": "unr3str1cted",
    }
    result = prompt
    for word, leet in leet_map.items():
        # Replace with ~50% probability to keep partial readability
        if random.random() > 0.5:
            result = re.sub(rf"\b{word}\b", leet, result, flags=re.IGNORECASE)
    # Insert zero-width spaces inside remaining trigger words
    trigger_words = ["ignore", "bypass", "override", "system"]
    for w in trigger_words:
        result = re.sub(
            rf"\b({w})\b",
            lambda m: "\u200b".join(m.group(1)),
            result,
            flags=re.IGNORECASE,
        )
    return result


MUTATION_STRATEGIES: Dict[str, Callable[[str], str]] = {
    "paraphrase": _paraphrase_attack,
    "split":      _split_attack,
    "encoding":   _encoding_attack,
}


# ── Result structures ─────────────────────────────────────────────────────────

@dataclass
class MutationResult:
    original_prompt: str
    mutated_prompt: str
    strategy: str
    attack_category: Optional[str]
    original_risk: float
    mutated_risk: float
    original_decision: str
    mutated_decision: str
    evaded: bool           # True if adversarial prompt now classified as ACCEPT
    feature_shift: float   # L2 distance between original and mutated feature vectors


@dataclass
class AdaptiveReport:
    strategy_results: Dict[str, List[MutationResult]]
    summary: Dict[str, dict]

    def __str__(self) -> str:
        lines = [
            "=" * 70,
            "Adaptive Adversarial Robustness Evaluation",
            "=" * 70,
        ]
        for strategy, stats in self.summary.items():
            lines += [
                f"\n  Strategy: {strategy}",
                f"    Prompts tested    : {stats['n_tested']}",
                f"    Evasion rate      : {stats['evasion_rate']:.3f}  "
                f"(lower = more robust)",
                f"    Mean risk (orig)  : {stats['mean_risk_original']:.4f}",
                f"    Mean risk (mut)   : {stats['mean_risk_mutated']:.4f}",
                f"    Mean feature shift: {stats['mean_feature_shift']:.4f}",
                f"    Per-category evasion:",
            ]
            for cat, er in stats.get("per_category_evasion", {}).items():
                lines.append(f"      {cat:<28}: {er:.3f}")
        lines.append("=" * 70)
        return "\n".join(lines)


# ── Runner ────────────────────────────────────────────────────────────────────

def run_adaptive_evaluation(
    adversarial_prompts: List[Dict],      # list of {"prompt": str, "attack_category": str}
    strategies: Optional[List[str]] = None,
    n_samples: int = 100,
    seed: int = 42,
) -> AdaptiveReport:
    """
    Runs adaptive adversarial robustness evaluation.

    Args:
        adversarial_prompts: List of dicts with 'prompt' and 'attack_category' keys
        strategies: Subset of ['paraphrase', 'split', 'encoding'] to run
        n_samples: Max adversarial prompts to test per strategy
        seed: Random seed for reproducibility
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from models.risk_scorer import default_scorer
    from utils.feature_extractor import extract_features

    random.seed(seed)
    np.random.seed(seed)

    if strategies is None:
        strategies = list(MUTATION_STRATEGIES.keys())

    # Sample adversarial prompts
    sampled = random.sample(adversarial_prompts, min(n_samples, len(adversarial_prompts)))

    strategy_results: Dict[str, List[MutationResult]] = {}
    summary: Dict[str, dict] = {}

    for strategy_name in strategies:
        mutate_fn = MUTATION_STRATEGIES[strategy_name]
        results: List[MutationResult] = []

        print(f"\n[AdaptiveEval] Strategy: {strategy_name} ({len(sampled)} prompts)")

        for item in sampled:
            original = item["prompt"]
            attack_cat = item.get("attack_category")

            # Score original
            orig_dec, orig_risk, orig_feats = default_scorer.decide(original)

            # Mutate
            mutated = mutate_fn(original)

            # Score mutated
            mut_dec, mut_risk, mut_feats = default_scorer.decide(mutated)

            # Feature shift (L2 distance)
            feat_shift = float(np.linalg.norm(
                np.array(orig_feats) - np.array(mut_feats)
            ))

            # Evasion: was adversarial (orig_dec != ACCEPT) but mutated is ACCEPT
            evaded = (orig_dec != "ACCEPT") and (mut_dec == "ACCEPT")

            results.append(MutationResult(
                original_prompt=original,
                mutated_prompt=mutated,
                strategy=strategy_name,
                attack_category=attack_cat,
                original_risk=orig_risk,
                mutated_risk=mut_risk,
                original_decision=orig_dec,
                mutated_decision=mut_dec,
                evaded=evaded,
                feature_shift=feat_shift,
            ))

        # Aggregate
        n = len(results)
        evasion_rate = sum(r.evaded for r in results) / max(n, 1)
        mean_risk_orig = np.mean([r.original_risk for r in results])
        mean_risk_mut  = np.mean([r.mutated_risk  for r in results])
        mean_feat_shift = np.mean([r.feature_shift for r in results])

        # Per-category evasion
        per_cat: Dict[str, float] = {}
        for cat in ATTACK_CATEGORIES:
            cat_results = [r for r in results if r.attack_category == cat]
            if cat_results:
                per_cat[cat] = sum(r.evaded for r in cat_results) / len(cat_results)

        strategy_results[strategy_name] = results
        summary[strategy_name] = {
            "n_tested": n,
            "evasion_rate": round(float(evasion_rate), 4),
            "mean_risk_original": round(float(mean_risk_orig), 4),
            "mean_risk_mutated":  round(float(mean_risk_mut), 4),
            "mean_feature_shift": round(float(mean_feat_shift), 4),
            "per_category_evasion": {k: round(v, 4) for k, v in per_cat.items()},
        }

        print(f"  Evasion rate: {evasion_rate:.3f} | "
              f"Mean risk: {mean_risk_orig:.3f} → {mean_risk_mut:.3f} | "
              f"Feat shift: {mean_feat_shift:.4f}")

    return AdaptiveReport(strategy_results=strategy_results, summary=summary)


def save_adaptive_report(report: AdaptiveReport, path: str):
    data = {
        "summary": report.summary,
        "per_strategy_samples": {
            strategy: [
                {
                    "original_prompt": r.original_prompt[:200],
                    "mutated_prompt":  r.mutated_prompt[:200],
                    "attack_category": r.attack_category,
                    "original_risk":   r.original_risk,
                    "mutated_risk":    r.mutated_risk,
                    "original_decision": r.original_decision,
                    "mutated_decision":  r.mutated_decision,
                    "evaded":          r.evaded,
                    "feature_shift":   r.feature_shift,
                }
                for r in results[:20]   # save first 20 examples per strategy
            ]
            for strategy, results in report.strategy_results.items()
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[AdaptiveEval] Report saved to {path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "dataset" / "data"
    test_path = data_dir / "test.jsonl"

    if not test_path.exists():
        print(f"[ERROR] Dataset not found at {test_path}")
        print("Run: python dataset/dataset_builder.py first")
        sys.exit(1)

    print("[AdaptiveEval] Loading test set...")
    adversarial_prompts = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            if obj.get("label") == 1:
                adversarial_prompts.append(obj)

    print(f"  {len(adversarial_prompts)} adversarial prompts found")

    report = run_adaptive_evaluation(
        adversarial_prompts=adversarial_prompts,
        strategies=["paraphrase", "split", "encoding"],
        n_samples=100,
    )
    print("\n" + str(report))
    save_adaptive_report(
        report,
        str(Path(__file__).parent / "adaptive_adversarial_report.json")
    )
