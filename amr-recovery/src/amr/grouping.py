"""Group resolution and grouped cross-validation splits.

The manuscript claims patient-level GroupKFold. No NCBI Pathogen Detection
export in this project carries a patient identifier (finding A6), and both
original code paths fell back to a per-row group -- ``pd.Series(range(n))`` in
one, ``df.index.astype(str)`` in the other. GroupKFold with one group per row
is plain KFold, so the leakage control named in the Methods was never active.

This module makes that failure impossible: it resolves the strongest grouping
column actually present, reports what it chose, and refuses a degenerate
grouping unless the caller explicitly accepts it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

__all__ = ["GroupResolution", "resolve_groups", "grouped_splits", "DEFAULT_GROUP_PREFERENCE"]

# Strongest to weakest. 'SNP cluster' groups genomically near-identical
# isolates (outbreak clusters), which is the tightest defensible proxy for
# non-independence available in these exports. 'BioSample' groups sequencing
# runs of one biological sample. 'Assembly'/'Isolate' are effectively
# row-level and give no protection.
DEFAULT_GROUP_PREFERENCE: tuple[str, ...] = (
    "patient_id",
    "SNP cluster",
    "BioSample",
    "Assembly",
    "Isolate",
)


@dataclass(frozen=True)
class GroupResolution:
    """Which column was used for grouping, and how much protection it gives."""

    column: str
    groups: pd.Series
    n_rows: int
    n_groups: int
    n_missing_filled: int
    candidates_considered: tuple[str, ...]

    @property
    def rows_per_group(self) -> float:
        return self.n_rows / self.n_groups if self.n_groups else float("nan")

    @property
    def is_degenerate(self) -> bool:
        """True when grouping provides no leakage protection over plain KFold."""
        return self.n_groups == self.n_rows

    def describe(self) -> dict:
        return {
            "group_column": self.column,
            "n_rows": self.n_rows,
            "n_groups": self.n_groups,
            "rows_per_group": self.rows_per_group,
            "n_missing_filled": self.n_missing_filled,
            "is_degenerate": self.is_degenerate,
            "candidates_considered": ",".join(self.candidates_considered),
        }


def resolve_groups(
    frame: pd.DataFrame,
    *,
    preference: tuple[str, ...] = DEFAULT_GROUP_PREFERENCE,
    allow_degenerate: bool = False,
    min_rows_per_group: float = 1.0,
) -> GroupResolution:
    """Pick the strongest available grouping column.

    A column qualifies if it exists and yields fewer groups than rows. Missing
    values are given their own singleton group (they cannot be pooled without
    asserting an identity that is not in the data) and are counted, so the
    Methods can state the proportion of unprotected rows.

    Raises
    ------
    ValueError
        If no candidate provides any grouping and ``allow_degenerate`` is
        False. Failing loudly is the point: silently degrading to KFold is
        what produced the unsupportable Methods claim.
    """
    considered = tuple(column for column in preference if column in frame.columns)
    n_rows = len(frame)

    for column in considered:
        raw = frame[column]
        missing = raw.isna()
        # Singleton group per missing value, labelled so it is traceable.
        filled = raw.astype("object").where(
            ~missing, pd.Series([f"__unassigned_{i}__" for i in range(n_rows)], index=raw.index)
        )
        groups = filled.astype(str)
        n_groups = groups.nunique()
        if n_groups < n_rows and n_rows / n_groups >= min_rows_per_group:
            return GroupResolution(
                column=column,
                groups=groups,
                n_rows=n_rows,
                n_groups=int(n_groups),
                n_missing_filled=int(missing.sum()),
                candidates_considered=considered,
            )

    if not allow_degenerate:
        raise ValueError(
            "no grouping column provides leakage protection for these rows "
            f"(considered: {considered or 'none present'}). Pass "
            "allow_degenerate=True only if the analysis is reported as "
            "row-level cross-validation, not patient- or cluster-level."
        )

    fallback = considered[-1] if considered else "__row_index__"
    groups = (
        frame[fallback].astype(str)
        if fallback in frame.columns
        else pd.Series(frame.index.astype(str), index=frame.index)
    )
    return GroupResolution(
        column=fallback,
        groups=groups,
        n_rows=n_rows,
        n_groups=int(groups.nunique()),
        n_missing_filled=0,
        candidates_considered=considered,
    )


def grouped_splits(
    y: pd.Series,
    groups: pd.Series,
    *,
    n_splits: int = 5,
    stratified: bool = True,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Grouped CV splits, stratified by outcome where possible.

    Every fold is returned, including a fold that ends up single-class. The
    original pipeline skipped such folds with ``continue`` and then averaged
    over however many survived, so an "AUC over 5 folds" could silently be an
    average over two. Here the caller sees all folds and records which were
    evaluable.
    """
    y = pd.Series(y).reset_index(drop=True)
    groups = pd.Series(groups).reset_index(drop=True)
    if len(y) != len(groups):
        raise ValueError(f"y and groups differ in length: {len(y)} vs {len(groups)}")

    n_groups = groups.nunique()
    if n_groups < n_splits:
        raise ValueError(
            f"{n_groups} groups cannot be split into {n_splits} folds without "
            "sharing a group across folds"
        )

    indices = np.arange(len(y))
    if stratified:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        return [
            (train, test) for train, test in splitter.split(indices.reshape(-1, 1), y, groups)
        ]
    splitter = GroupKFold(n_splits=n_splits)
    return [(train, test) for train, test in splitter.split(indices.reshape(-1, 1), y, groups)]


def fold_assignment_table(
    splits: list[tuple[np.ndarray, np.ndarray]], y: pd.Series, groups: pd.Series
) -> pd.DataFrame:
    """Saveable record of which row and group landed in which test fold."""
    y = pd.Series(y).reset_index(drop=True)
    groups = pd.Series(groups).reset_index(drop=True)
    rows = []
    for fold, (_, test) in enumerate(splits, start=1):
        for index in test:
            rows.append(
                {
                    "row_index": int(index),
                    "test_fold": fold,
                    "group": groups.iloc[index],
                    "label": int(y.iloc[index]),
                }
            )
    return pd.DataFrame(rows).sort_values("row_index").reset_index(drop=True)


def check_no_group_overlap(splits: list[tuple[np.ndarray, np.ndarray]], groups: pd.Series) -> bool:
    """Assert train and test never share a group. Run this every time."""
    groups = pd.Series(groups).reset_index(drop=True)
    for train, test in splits:
        if set(groups.iloc[train]) & set(groups.iloc[test]):
            return False
    return True
