"""Diagnostic metrics derived exclusively from integer confusion counts.

Design rule (supervisor action report, section 4.2 "Report integer counts"):
no metric in this project may be computed from a rounded rate. Every value
returned here is a function of TP/FP/FN/TN integers, and the counts travel
with the metrics so any downstream table can be re-derived and re-checked.

The one function that *does* take rates -- ``implied_predictive_values`` --
is an audit tool, not a results generator. It answers "are the PPV/NPV in
this published table compatible with the prevalence/sensitivity/specificity
printed beside them?" and is named so it cannot be mistaken for a result.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from math import sqrt
from typing import Iterable, Sequence

import numpy as np
from scipy import stats

__all__ = [
    "ConfusionCounts",
    "counts_from_labels",
    "metrics_from_counts",
    "aggregate_counts",
    "wilson_ci",
    "clopper_pearson_ci",
    "implied_predictive_values",
    "check_predictive_values",
]


@dataclass(frozen=True)
class ConfusionCounts:
    """Integer confusion counts for a binary task (positive class = resistant)."""

    tp: int
    fp: int
    fn: int
    tn: int

    def __post_init__(self) -> None:
        for name in ("tp", "fp", "fn", "tn"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}")

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def positives(self) -> int:
        """Truly resistant isolate-drug rows."""
        return self.tp + self.fn

    @property
    def negatives(self) -> int:
        """Truly susceptible isolate-drug rows."""
        return self.tn + self.fp

    def __add__(self, other: "ConfusionCounts") -> "ConfusionCounts":
        return ConfusionCounts(
            self.tp + other.tp, self.fp + other.fp, self.fn + other.fn, self.tn + other.tn
        )

    def as_dict(self) -> dict:
        return asdict(self)


def counts_from_labels(y_true: Sequence[int], y_pred: Sequence[int]) -> ConfusionCounts:
    """Confusion counts from hard 0/1 labels.

    Unlike ``sklearn.metrics.confusion_matrix(...).ravel()``, this never
    silently mis-unpacks when a fold contains a single class: the counts are
    computed by direct summation, so a degenerate fold yields zeros in the
    missing cells rather than a ``ValueError`` or a transposed matrix.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    bad = set(np.unique(y_true)) | set(np.unique(y_pred))
    if not bad <= {0, 1}:
        raise ValueError(f"labels must be 0/1, found {sorted(bad)}")
    return ConfusionCounts(
        tp=int(np.sum((y_true == 1) & (y_pred == 1))),
        fp=int(np.sum((y_true == 0) & (y_pred == 1))),
        fn=int(np.sum((y_true == 1) & (y_pred == 0))),
        tn=int(np.sum((y_true == 0) & (y_pred == 0))),
    )


def aggregate_counts(counts: Iterable[ConfusionCounts]) -> ConfusionCounts:
    """Micro-aggregate per-fold counts.

    Micro-averaging (pool the counts, then compute the metric) is the declared
    default for this project. Macro-averaging a *ratio* across folds -- what
    the original pipeline did -- weights a 12-row fold equally with a 3,200-row
    fold and yields a "sensitivity" that corresponds to no confusion matrix.
    """
    total = ConfusionCounts(0, 0, 0, 0)
    for item in counts:
        total = total + item
    return total


def wilson_ci(successes: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if trials == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - alpha / 2)
    phat = successes / trials
    denom = 1 + z**2 / trials
    centre = (phat + z**2 / (2 * trials)) / denom
    half = z * sqrt(phat * (1 - phat) / trials + z**2 / (4 * trials**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def clopper_pearson_ci(successes: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Clopper-Pearson) interval; preferred for proportions near 0 or 1."""
    if trials == 0:
        return (float("nan"), float("nan"))
    lower = 0.0 if successes == 0 else stats.beta.ppf(alpha / 2, successes, trials - successes + 1)
    upper = 1.0 if successes == trials else stats.beta.ppf(
        1 - alpha / 2, successes + 1, trials - successes
    )
    return (float(lower), float(upper))


def _ratio(numerator: int, denominator: int) -> float:
    """Undefined ratios are NaN, never 0.0.

    The original pipeline returned 0.0 when a denominator was empty, which
    silently turned "no susceptible isolates were predicted negative, so NPV
    is undefined" into "NPV = 0%" -- a number that then propagated into a
    table as if it had been measured.
    """
    return float(numerator) / denominator if denominator > 0 else float("nan")


def metrics_from_counts(counts: ConfusionCounts, alpha: float = 0.05) -> dict:
    """Every reportable diagnostic quantity, plus the counts it came from.

    Confidence intervals attached here are binomial intervals conditional on
    the relevant denominator. They describe sampling error in that proportion
    only; they are *not* group-aware and must not be reported as the
    uncertainty of a cross-validated estimate. Use
    ``amr.uncertainty.grouped_bootstrap`` for that.
    """
    tp, fp, fn, tn = counts.tp, counts.fp, counts.fn, counts.tn
    sensitivity = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    ppv = _ratio(tp, tp + fp)
    npv = _ratio(tn, tn + fn)
    prevalence = _ratio(tp + fn, counts.total)

    mcc_denom = sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    out = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_total": counts.total,
        "n_resistant": counts.positives,
        "n_susceptible": counts.negatives,
        "prevalence": prevalence,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "accuracy": _ratio(tp + tn, counts.total),
        "balanced_accuracy": (sensitivity + specificity) / 2
        if not (np.isnan(sensitivity) or np.isnan(specificity))
        else float("nan"),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "mcc": (tp * tn - fp * fn) / mcc_denom if mcc_denom > 0 else float("nan"),
    }
    for name, (successes, trials) in {
        "prevalence": (tp + fn, counts.total),
        "sensitivity": (tp, tp + fn),
        "specificity": (tn, tn + fp),
        "ppv": (tp, tp + fp),
        "npv": (tn, tn + fn),
    }.items():
        low, high = clopper_pearson_ci(successes, trials, alpha)
        out[f"{name}_ci_low"] = low
        out[f"{name}_ci_high"] = high
        out[f"{name}_denominator"] = trials
    return out


def implied_predictive_values(
    prevalence: float, sensitivity: float, specificity: float
) -> tuple[float, float]:
    """AUDIT ONLY. Bayes-implied PPV/NPV for a given prevalence/Se/Sp triple.

    Used to test whether a *published* table is internally coherent. The
    result must never be written into a results table: it is derived from
    rounded inputs and is therefore itself rounded. Real metrics come from
    ``metrics_from_counts``.
    """
    for name, value in (
        ("prevalence", prevalence),
        ("sensitivity", sensitivity),
        ("specificity", specificity),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be a proportion in [0, 1], got {value!r}")
    ppv_denom = sensitivity * prevalence + (1 - specificity) * (1 - prevalence)
    npv_denom = specificity * (1 - prevalence) + (1 - sensitivity) * prevalence
    ppv = sensitivity * prevalence / ppv_denom if ppv_denom > 0 else float("nan")
    npv = specificity * (1 - prevalence) / npv_denom if npv_denom > 0 else float("nan")
    return ppv, npv


def check_predictive_values(
    prevalence: float,
    sensitivity: float,
    specificity: float,
    reported_ppv: float,
    reported_npv: float,
    tolerance: float = 0.02,
) -> dict:
    """Flag a published row whose PPV/NPV cannot arise from its own Se/Sp/prevalence.

    ``tolerance`` is generous (2 percentage points by default) so that
    rounding of the inputs to one decimal place cannot by itself trigger a
    flag; anything exceeding it needs a source-data explanation.
    """
    implied_ppv, implied_npv = implied_predictive_values(prevalence, sensitivity, specificity)
    delta_ppv = reported_ppv - implied_ppv
    delta_npv = reported_npv - implied_npv
    return {
        "prevalence": prevalence,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "reported_ppv": reported_ppv,
        "reported_npv": reported_npv,
        "implied_ppv": implied_ppv,
        "implied_npv": implied_npv,
        "delta_ppv": delta_ppv,
        "delta_npv": delta_npv,
        "ppv_consistent": bool(abs(delta_ppv) <= tolerance),
        "npv_consistent": bool(abs(delta_npv) <= tolerance),
        "consistent": bool(abs(delta_ppv) <= tolerance and abs(delta_npv) <= tolerance),
        "tolerance": tolerance,
    }
