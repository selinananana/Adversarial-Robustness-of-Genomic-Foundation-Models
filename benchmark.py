"""
benchmark.py — Performance and robustness metrics (v2).

Changes from v1:
- run_performance_test: reports AUC-ROC in addition to accuracy/F1,
  and prints a score distribution summary to catch the "all one class" failure.
- analyze_robustness: reports best_score reduction and score_drop in addition
  to bypass rate, so partial success is visible even when bypass_rate=0.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def run_performance_test(model, dataset, limit=200):
    y_true, y_score, y_pred = [], [], []

    for i in range(min(limit, len(dataset))):
        seq, label = dataset[i]
        prob = model.get_score(seq)
        y_score.append(prob)
        y_pred.append(1 if prob > 0.5 else 0)
        y_true.append(label)

    scores = np.array(y_score)
    print(f"    Score distribution: min={scores.min():.3f}  "
          f"max={scores.max():.3f}  mean={scores.mean():.3f}  "
          f"std={scores.std():.3f}")
    print(f"    Predicted positives: {sum(y_pred)}/{limit}  "
          f"True positives: {sum(y_true)}/{limit}")

    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")  # only one class present in sample

    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1":       round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc_roc":  round(auc, 4),
        "n_evaluated": limit,
    }


def analyze_robustness(attack_logs):
    if not attack_logs:
        return {
            "bypass_rate": 0.0,
            "avg_edits_to_bypass": float("inf"),
            "avg_best_score": float("nan"),
            "avg_score_drop": float("nan"),
            "n_attacked": 0,
        }

    bypassed = [l for l in attack_logs if l["success"]]

    # Best score seen across all runs — shows partial progress even on failure
    best_scores = [l.get("best_score", l.get("final_prob", 1.0)) for l in attack_logs]

    # Score drop = initial score (assumed ~0.7+ for true positives post fine-tune) - best_score
    # We don't store initial score in logs, so proxy: 1.0 - best_score as lower bound
    avg_drop = float(np.mean([1.0 - s for s in best_scores]))

    return {
        "bypass_rate":         round(len(bypassed) / len(attack_logs), 4),
        "avg_edits_to_bypass": float(np.mean([l["edits"] for l in bypassed])) if bypassed else float("inf"),
        "avg_best_score":      round(float(np.mean(best_scores)), 4),
        "avg_score_drop":      round(avg_drop, 4),
        "n_attacked":          len(attack_logs),
        "n_bypassed":          len(bypassed),
    }
