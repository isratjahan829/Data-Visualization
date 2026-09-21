"""Grouped-CV evaluation of one pathogen-drug prediction task.

What this does differently from the pipeline under review:

* the feature vocabulary and the scaler are fitted **inside** each training
  fold, so no test-fold information reaches preprocessing;
* every fold's integer TP/FP/FN/TN are saved, and pooled metrics are computed
  from those integers -- PPV and NPV included, which the original never
  emitted at all, leaving the manuscript's values with no traceable source;
* out-of-fold predictions are written for every row, so intervals and any
  later re-analysis derive from saved predictions rather than a rerun;
* folds are grouped on a resolved, reported identifier and asserted not to
  share groups;
* baselines run on identical folds, so the comparison is like for like.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from .baselines import BASELINES
from .features import GeneVocabulary
from .grouping import (
    check_no_group_overlap,
    fold_assignment_table,
    grouped_splits,
    resolve_groups,
)
from .metrics import aggregate_counts, counts_from_labels, metrics_from_counts
from .uncertainty import bootstrap_auc, fold_dispersion, grouped_bootstrap

__all__ = ["TaskSpec", "TaskResult", "run_task", "MODELS"]


def _random_forest(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def _gradient_boosting(seed: int) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.05, random_state=seed
    )


MODELS: dict[str, Callable[[int], object]] = {
    "random_forest": _random_forest,
    "gradient_boosting": _gradient_boosting,
    **BASELINES,
}


@dataclass(frozen=True)
class TaskSpec:
    """One prediction task, fully identified for the results table."""

    task_id: str
    pathogen: str
    drug: str
    outcome_definition: str
    n_splits: int = 5
    random_state: int = 42
    min_labels: int = 50
    min_minority: int = 10
    decision_threshold: float = 0.5
    min_feature_count: int = 5

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "pathogen": self.pathogen,
            "drug": self.drug,
            "outcome_definition": self.outcome_definition,
            "n_splits": self.n_splits,
            "random_state": self.random_state,
            "decision_threshold": self.decision_threshold,
            "min_feature_count": self.min_feature_count,
        }


@dataclass
class TaskResult:
    """All saveable artefacts for one task."""

    spec: TaskSpec
    status: str
    cohort: dict = field(default_factory=dict)
    grouping: dict = field(default_factory=dict)
    folds: pd.DataFrame = field(default_factory=pd.DataFrame)
    oof_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    pooled_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    fold_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    intervals: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_dictionary: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: list[str] = field(default_factory=list)

    def write(self, directory: str | Path) -> list[Path]:
        """Write every artefact; returns the paths, for the run manifest."""
        directory = Path(directory) / self.spec.task_id
        directory.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for name, frame in (
            ("fold_assignments.csv", self.folds),
            ("out_of_fold_predictions.csv", self.oof_predictions),
            ("pooled_metrics.csv", self.pooled_metrics),
            ("fold_metrics.csv", self.fold_metrics),
            ("intervals.csv", self.intervals),
            ("feature_dictionary.csv", self.feature_dictionary),
        ):
            if not frame.empty:
                path = directory / name
                frame.to_csv(path, index=False)
                written.append(path)
        summary = pd.DataFrame(
            [{**self.spec.as_dict(), "status": self.status, **self.cohort, **self.grouping,
              "notes": " | ".join(self.notes)}]
        )
        path = directory / "task_summary.csv"
        summary.to_csv(path, index=False)
        written.append(path)
        return written


def run_task(
    frame: pd.DataFrame,
    spec: TaskSpec,
    *,
    amr_column: str,
    label_column: str = "label",
    models: dict[str, Callable[[int], object]] | None = None,
) -> TaskResult:
    """Evaluate every model on identical grouped folds for one task.

    ``frame`` must be the task's analytic rows only: one row per isolate-drug
    outcome for this pathogen and drug, carrying the AMR genotype text, the
    binary label, and whatever identifier columns grouping can use.
    """
    models = models or MODELS
    notes: list[str] = []

    labelled = frame.loc[frame[label_column].notna()].reset_index(drop=True)
    y = labelled[label_column].astype(int)
    n_resistant = int(y.sum())
    n_susceptible = int((y == 0).sum())
    cohort = {
        "n_labels": len(labelled),
        "n_resistant": n_resistant,
        "n_susceptible": n_susceptible,
        "prevalence": n_resistant / len(labelled) if len(labelled) else float("nan"),
    }

    # Stop conditions are explicit and recorded, so an absent task never
    # becomes an unexplained gap in the results table.
    if len(labelled) < spec.min_labels:
        return TaskResult(
            spec=spec,
            status=f"skipped: {len(labelled)} labels < min_labels={spec.min_labels}",
            cohort=cohort,
        )
    if min(n_resistant, n_susceptible) < spec.min_minority:
        return TaskResult(
            spec=spec,
            status=(
                f"skipped: minority class n={min(n_resistant, n_susceptible)} "
                f"< min_minority={spec.min_minority}"
            ),
            cohort=cohort,
        )

    try:
        resolution = resolve_groups(labelled)
    except ValueError as error:
        return TaskResult(spec=spec, status=f"skipped: {error}", cohort=cohort)
    grouping = resolution.describe()

    try:
        splits = grouped_splits(
            y,
            resolution.groups,
            n_splits=spec.n_splits,
            stratified=True,
            random_state=spec.random_state,
        )
    except ValueError as error:
        return TaskResult(
            spec=spec, status=f"skipped: {error}", cohort=cohort, grouping=grouping
        )

    if not check_no_group_overlap(splits, resolution.groups):
        raise AssertionError(
            f"{spec.task_id}: a group appears in both train and test. "
            "Refusing to report leaked folds."
        )

    folds = fold_assignment_table(splits, y, resolution.groups)
    amr_text = labelled[amr_column]

    oof_rows: list[dict] = []
    per_model_fold_counts: dict[str, list] = {name: [] for name in models}
    dictionaries: list[pd.DataFrame] = []

    for fold_number, (train_index, test_index) in enumerate(splits, start=1):
        y_train = y.iloc[train_index].to_numpy()
        y_test = y.iloc[test_index].to_numpy()

        # Vocabulary fitted on training rows only. This is the leakage fix:
        # in the original, the top-2,000 gene list was learned from the whole
        # file before splitting.
        vocabulary = GeneVocabulary(min_count=spec.min_feature_count)
        X_train = vocabulary.fit(amr_text.iloc[train_index]).transform(
            amr_text.iloc[train_index]
        )
        X_test = vocabulary.transform(amr_text.iloc[test_index])
        dictionary = vocabulary.dictionary_.assign(fold=fold_number)
        dictionaries.append(dictionary)

        if X_train.shape[1] == 0:
            notes.append(f"fold {fold_number}: no determinant met min_feature_count")
            continue
        if len(np.unique(y_train)) < 2:
            notes.append(f"fold {fold_number}: single-class training fold, models not fitted")
            continue

        for name, factory in models.items():
            model = factory(spec.random_state)
            model.fit(X_train, y_train)
            scores = model.predict_proba(X_test)[:, 1]
            predictions = (scores >= spec.decision_threshold).astype(int)
            per_model_fold_counts[name].append(counts_from_labels(y_test, predictions))
            for position, row_index in enumerate(test_index):
                oof_rows.append(
                    {
                        "task_id": spec.task_id,
                        "model": name,
                        "fold": fold_number,
                        "row_index": int(row_index),
                        "group": resolution.groups.iloc[row_index],
                        "y_true": int(y_test[position]),
                        "y_score": float(scores[position]),
                        "y_pred": int(predictions[position]),
                    }
                )

    oof = pd.DataFrame(oof_rows)
    if oof.empty:
        return TaskResult(
            spec=spec,
            status="skipped: no fold produced predictions",
            cohort=cohort,
            grouping=grouping,
            folds=folds,
            notes=notes,
        )

    pooled_records: list[dict] = []
    fold_records: list[pd.DataFrame] = []
    interval_records: list[pd.DataFrame] = []

    for name in models:
        fold_counts = per_model_fold_counts[name]
        if not fold_counts:
            continue
        pooled = aggregate_counts(fold_counts)
        record = {
            "task_id": spec.task_id,
            "pathogen": spec.pathogen,
            "drug": spec.drug,
            "model": name,
            "n_folds_evaluated": len(fold_counts),
            "n_folds_requested": spec.n_splits,
            "aggregation": "micro (counts pooled across folds)",
        }
        record.update(metrics_from_counts(pooled))

        subset = oof.loc[oof["model"] == name]
        record.update(
            bootstrap_auc(
                subset["y_true"].to_numpy(),
                subset["y_score"].to_numpy(),
                subset["group"].to_numpy(),
                random_state=spec.random_state,
            )
        )
        pooled_records.append(record)

        fold_records.append(fold_dispersion(fold_counts).assign(model=name, task_id=spec.task_id))
        interval_records.append(
            grouped_bootstrap(
                subset["y_true"].to_numpy(),
                subset["y_pred"].to_numpy(),
                subset["group"].to_numpy(),
                random_state=spec.random_state,
            ).assign(model=name, task_id=spec.task_id)
        )

    return TaskResult(
        spec=spec,
        status="completed",
        cohort=cohort,
        grouping=grouping,
        folds=folds,
        oof_predictions=oof,
        pooled_metrics=pd.DataFrame(pooled_records),
        fold_metrics=pd.concat(fold_records, ignore_index=True),
        intervals=pd.concat(interval_records, ignore_index=True),
        feature_dictionary=pd.concat(dictionaries, ignore_index=True),
        notes=notes,
    )
