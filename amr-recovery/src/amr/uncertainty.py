"""Group-aware uncertainty from saved out-of-fold predictions.

The original pipeline reported ``mean +/- np.std`` of five fold AUCs. With 61
labelled K. pneumoniae isolates and 12-row test folds, the fold AUCs ranged
from 0.21 to 1.00; the standard deviation of five such numbers is not a
confidence interval, and it was printed beside an "Excellent" grade.

Resampling here is at the **group** level, matching the level at which the
data are non-independent, and it operates on pooled out-of-fold predictions
so the interval describes the cross-validated estimate rather than one fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .metrics import ConfusionCounts, counts_from_labels, metrics_from_counts


def _point_metrics(counts: ConfusionCounts) -> dict[str, float]:
    """Count-derived metrics without confidence intervals.

    ``metrics_from_counts`` also fits five exact binomial intervals, which is
    wasted work inside a bootstrap loop (the loop *is* the interval). The
    definitions below are the same ratios, so a bootstrap distribution and its
    point estimate stay consistent.
    """
    tp, fp, fn, tn = counts.tp, counts.fp, counts.fn, counts.tn

    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator > 0 else float("nan")

    sensitivity = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ratio(tp, tp + fp),
        "npv": ratio(tn, tn + fn),
        "accuracy": ratio(tp + tn, counts.total),
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "prevalence": ratio(tp + fn, counts.total),
    }

__all__ = ["grouped_bootstrap", "bootstrap_auc", "fold_dispersion"]


class _GroupIndex:
    """Row indices bucketed by group, built once and reused across draws.

    Building the bucket map inside the resampling loop makes a 2,000-draw
    bootstrap O(n_boot * n_rows) in Python-level work; hoisting it out leaves
    only the concatenation per draw.
    """

    def __init__(self, groups: np.ndarray) -> None:
        codes, self.unique = pd.factorize(pd.Series(groups), sort=True)
        order = np.argsort(codes, kind="stable")
        sorted_codes = codes[order]
        # Contiguous slice of `order` per group code.
        boundaries = np.flatnonzero(np.diff(sorted_codes)) + 1
        self.buckets = np.split(order, boundaries)
        self.n_groups = len(self.unique)

    def resample(self, rng: np.random.Generator) -> np.ndarray:
        drawn = rng.integers(0, self.n_groups, size=self.n_groups)
        return np.concatenate([self.buckets[code] for code in drawn])


def bootstrap_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict:
    """Group-bootstrap percentile interval for AUC on out-of-fold predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    groups = np.asarray(groups)
    if not (len(y_true) == len(y_score) == len(groups)):
        raise ValueError("y_true, y_score and groups must be the same length")
    if len(np.unique(y_true)) < 2:
        return {
            "auc": float("nan"),
            "auc_ci_low": float("nan"),
            "auc_ci_high": float("nan"),
            "n_boot_valid": 0,
            "n_boot_requested": n_boot,
        }

    point = float(roc_auc_score(y_true, y_score))
    rng = np.random.default_rng(random_state)
    index = _GroupIndex(groups)
    draws: list[float] = []
    for _ in range(n_boot):
        rows = index.resample(rng)
        sample_y = y_true[rows]
        # A resample containing one class has no defined AUC. Such draws are
        # discarded and counted, not replaced with 0.5 -- with tiny cohorts the
        # discard rate is itself a result worth reporting.
        if len(np.unique(sample_y)) < 2:
            continue
        draws.append(float(roc_auc_score(sample_y, y_score[rows])))

    if not draws:
        return {
            "auc": point,
            "auc_ci_low": float("nan"),
            "auc_ci_high": float("nan"),
            "n_boot_valid": 0,
            "n_boot_requested": n_boot,
        }
    low, high = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "auc": point,
        "auc_ci_low": float(low),
        "auc_ci_high": float(high),
        "n_boot_valid": len(draws),
        "n_boot_requested": n_boot,
    }


def grouped_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    *,
    metrics: tuple[str, ...] = ("sensitivity", "specificity", "ppv", "npv", "balanced_accuracy"),
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Group-bootstrap intervals for count-derived metrics.

    Each bootstrap draw rebuilds the confusion counts from resampled groups
    and recomputes the metric from those integers, so the interval and the
    point estimate come from the same definition.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    groups = np.asarray(groups)

    point = metrics_from_counts(counts_from_labels(y_true, y_pred))
    rng = np.random.default_rng(random_state)
    index = _GroupIndex(groups)
    draws: dict[str, list[float]] = {name: [] for name in metrics}
    for _ in range(n_boot):
        rows = index.resample(rng)
        sample = _point_metrics(counts_from_labels(y_true[rows], y_pred[rows]))
        for name in metrics:
            value = sample[name]
            if not np.isnan(value):
                draws[name].append(value)

    records = []
    for name in metrics:
        values = draws[name]
        low, high = (
            np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            if values
            else (float("nan"), float("nan"))
        )
        records.append(
            {
                "metric": name,
                "estimate": point[name],
                "ci_low": float(low),
                "ci_high": float(high),
                "n_boot_valid": len(values),
                "n_boot_requested": n_boot,
                "denominator": point.get(f"{name}_denominator"),
            }
        )
    return pd.DataFrame(records)


def fold_dispersion(fold_counts: list[ConfusionCounts]) -> pd.DataFrame:
    """Per-fold counts and metrics, reported alongside the pooled estimate.

    Published as a table rather than collapsed into a +/- so a reader can see
    that a fold held 12 rows, or that one fold was single-class.
    """
    records = []
    for fold, counts in enumerate(fold_counts, start=1):
        row = {"fold": fold}
        row.update(metrics_from_counts(counts))
        row["evaluable"] = bool(counts.positives > 0 and counts.negatives > 0)
        records.append(row)
    return pd.DataFrame(records)
