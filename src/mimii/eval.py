"""Metrics: AUC-ROC, PR-AUC, and one FAR operating point. Print loio vs same_id."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .const import FAR_TARGET


def evaluate(y: np.ndarray, scores: np.ndarray, far_target: float = FAR_TARGET) -> dict:
    """y: 0/1 labels, scores: higher=more anomalous."""
    y = np.asarray(y)
    scores = np.asarray(scores)
    # threshold set on normals so false-alarm rate == far_target; report resulting recall.
    normal_scores = scores[y == 0]
    thr = float(np.quantile(normal_scores, 1.0 - far_target))
    tpr = float((scores[y == 1] > thr).mean()) if (y == 1).any() else float("nan")
    return {
        "auc_roc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
        "far_target": far_target,
        "tpr_at_far": tpr,
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
    }


def print_table(loio: dict, same: dict, machine_type: str, test_id: str) -> None:
    print(f"\n{machine_type}  test_id={test_id}")
    print(f"  {'split':<8}{'AUC-ROC':>9}{'PR-AUC':>9}{'TPR@FAR':>9}")
    for name, m in (("loio", loio), ("same_id", same)):
        print(f"  {name:<8}{m['auc_roc']:>9.3f}{m['pr_auc']:>9.3f}{m['tpr_at_far']:>9.3f}")
    gap = loio["auc_roc"] - same["auc_roc"]
    if abs(gap) < 0.02:
        print("  WARN: loio ~= same_id -> split may be leaking or task is trivial.")
