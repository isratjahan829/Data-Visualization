"""Simple comparators that every reported model must be measured against.

Section 4.2 of the action report: "Add simple baselines ... This is
reanalysis, not a new experiment." For C. jejuni ciprofloxacin the dominant
determinant (gyrA T86I) is close to necessary and sufficient, so a
single-feature rule is the honest benchmark for an AUC of 0.997. Without it,
tree-ensemble performance reads as a modelling achievement rather than as a
restatement of one well-known mutation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

__all__ = ["MajorityClassBaseline", "SingleFeatureBaseline", "logistic_baseline", "BASELINES"]


class MajorityClassBaseline:
    """Predicts the training majority class for every row."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MajorityClassBaseline":
        y = np.asarray(y).astype(int)
        counts = np.bincount(y, minlength=2)
        self.majority_ = int(np.argmax(counts))
        self.prevalence_ = float(counts[1] / counts.sum()) if counts.sum() else float("nan")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Constant score: AUC is exactly 0.5 by definition, which is the point.
        column = np.full(len(X), self.prevalence_)
        return np.column_stack([1 - column, column])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.majority_, dtype=int)


class SingleFeatureBaseline:
    """Presence of the single most informative training determinant.

    The feature is chosen inside the training fold by absolute point-biserial
    correlation with the outcome, so the choice is part of the fitted model and
    is reported per fold rather than fixed from prior knowledge.
    """

    def __init__(self, feature: str | None = None) -> None:
        self.feature = feature

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "SingleFeatureBaseline":
        y = np.asarray(y).astype(int)
        if self.feature is not None:
            self.feature_ = self.feature
        else:
            values = X.to_numpy(dtype=float)
            # Guard against zero-variance columns, which give NaN correlations.
            std = values.std(axis=0)
            usable = std > 0
            if not usable.any() or y.std() == 0:
                self.feature_ = X.columns[0]
            else:
                centred = values[:, usable] - values[:, usable].mean(axis=0)
                correlations = (centred * (y - y.mean())[:, None]).mean(axis=0) / (
                    std[usable] * y.std()
                )
                self.feature_ = X.columns[np.flatnonzero(usable)[np.argmax(np.abs(correlations))]]
                self.correlation_ = float(correlations[np.argmax(np.abs(correlations))])
        # Orientation: does carriage predict resistance or susceptibility?
        carriers = X[self.feature_].to_numpy(dtype=float) > 0
        self.positive_when_present_ = bool(
            y[carriers].mean() >= y[~carriers].mean() if carriers.any() and (~carriers).any()
            else True
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        present = X[self.feature_].to_numpy(dtype=float) > 0
        score = present.astype(float) if self.positive_when_present_ else 1 - present.astype(float)
        return np.column_stack([1 - score, score])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def logistic_baseline(random_state: int = 42) -> LogisticRegression:
    """L2 logistic regression with balanced class weights."""
    # `penalty` is deprecated from scikit-learn 1.8; L2 is the default
    # regularisation and `C` still controls its strength.
    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=random_state,
    )


BASELINES = {
    "majority_class": lambda seed: MajorityClassBaseline(),
    "single_determinant": lambda seed: SingleFeatureBaseline(),
    "logistic_regression": logistic_baseline,
}
