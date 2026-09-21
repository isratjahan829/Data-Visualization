"""Regression tests pinning each correctness fix.

Each test names the audit finding or review item it guards. The point is not
coverage for its own sake: these are the specific behaviours whose absence
produced unreportable numbers, so a future edit that reintroduces one should
fail here rather than in a table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amr.features import GeneVocabulary, tokenise_amr_cell
from amr.grouping import check_no_group_overlap, grouped_splits, resolve_groups
from amr.metrics import (
    ConfusionCounts,
    aggregate_counts,
    check_predictive_values,
    counts_from_labels,
    implied_predictive_values,
    metrics_from_counts,
)
from amr.phenotype import DrugVocabulary, build_phenotype_ledger, drug_denominator_ledger
from amr.pipeline import TaskSpec, run_task

FIXTURE = Path(__file__).parent / "fixtures" / "SYNTHETIC_Campylobacter_jejuni.csv"


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def test_metrics_derive_from_counts_only():
    counts = ConfusionCounts(tp=90, fp=10, fn=5, tn=895)
    out = metrics_from_counts(counts)
    assert out["sensitivity"] == pytest.approx(90 / 95)
    assert out["specificity"] == pytest.approx(895 / 905)
    assert out["ppv"] == pytest.approx(90 / 100)
    assert out["npv"] == pytest.approx(895 / 900)
    # The counts travel with the metrics so any row can be re-derived.
    assert (out["tp"], out["fp"], out["fn"], out["tn"]) == (90, 10, 5, 895)


def test_undefined_ratio_is_nan_not_zero():
    """Finding: 0.0 for an empty denominator became a reported '0%'."""
    out = metrics_from_counts(ConfusionCounts(tp=0, fp=0, fn=3, tn=7))
    assert np.isnan(out["ppv"])  # nothing was predicted positive
    assert out["npv"] == pytest.approx(0.7)


def test_counts_from_labels_handles_single_class_fold():
    """sklearn's confusion_matrix().ravel() raises here; we return zeros."""
    counts = counts_from_labels([0, 0, 0], [0, 0, 1])
    assert (counts.tp, counts.fp, counts.fn, counts.tn) == (0, 1, 0, 2)


def test_micro_aggregation_differs_from_mean_of_ratios():
    """Guards the fold-averaging defect: a mean of ratios matches no matrix."""
    small = ConfusionCounts(tp=1, fp=0, fn=1, tn=10)  # sensitivity 0.5
    large = ConfusionCounts(tp=90, fp=5, fn=10, tn=900)  # sensitivity 0.9
    pooled = metrics_from_counts(aggregate_counts([small, large]))
    mean_of_ratios = (0.5 + 0.9) / 2
    assert pooled["sensitivity"] == pytest.approx(91 / 102)
    assert pooled["sensitivity"] != pytest.approx(mean_of_ratios, abs=1e-3)


def test_implied_predictive_values_reproduce_the_table3_contradiction():
    """Audit finding A8, C. jejuni gentamicin random forest."""
    ppv, npv = implied_predictive_values(0.008, 0.982, 0.996)
    assert ppv == pytest.approx(0.664, abs=0.002)
    assert npv == pytest.approx(1.0, abs=0.001)
    outcome = check_predictive_values(0.008, 0.982, 0.996, 0.996, 0.999)
    assert outcome["consistent"] is False
    assert outcome["delta_ppv"] > 0.3


def test_check_predictive_values_accepts_a_coherent_row():
    counts = ConfusionCounts(tp=90, fp=10, fn=5, tn=895)
    out = metrics_from_counts(counts)
    outcome = check_predictive_values(
        out["prevalence"], out["sensitivity"], out["specificity"], out["ppv"], out["npv"]
    )
    assert outcome["consistent"] is True


def test_implied_predictive_values_rejects_percentages():
    with pytest.raises(ValueError):
        implied_predictive_values(78.7, 94.2, 73.3)


# --------------------------------------------------------------------------
# phenotype parsing
# --------------------------------------------------------------------------

def test_substring_drug_names_do_not_match():
    """Review item: 'tet' matched oxytetracycline; aliases must be full names."""
    frame = pd.DataFrame(
        {"Isolate": ["A"], "AST phenotypes": ["oxytetracycline=S,tetracycline=R"]}
    )
    result = build_phenotype_ledger(
        frame, isolate_column="Isolate", ast_column="AST phenotypes"
    )
    tetracycline = result.ledger.loc[result.ledger["drug"] == "tetracycline"]
    assert len(tetracycline) == 1
    assert tetracycline.iloc[0]["interpretation"] == "R"
    assert (result.rejects["reason"].str.startswith("drug_not_in_vocabulary")).any()


def test_short_aliases_are_rejected_at_construction():
    with pytest.raises(ValueError, match="too short"):
        DrugVocabulary({"tetracycline": ("tet",)})


def test_intermediate_is_recorded_not_silently_dropped():
    frame = pd.DataFrame({"Isolate": ["A"], "AST phenotypes": ["ciprofloxacin=I"]})
    result = build_phenotype_ledger(
        frame, isolate_column="Isolate", ast_column="AST phenotypes"
    )
    assert result.ledger.iloc[0]["interpretation"] == "I"
    assert pd.isna(result.ledger.iloc[0]["label"])  # excluded from the binary task
    assert result.counts.iloc[0]["intermediate_policy"] == "exclude"


def test_ledger_reconciles_parsed_plus_rejected():
    frame = pd.DataFrame(
        {
            "Isolate": ["A", "B"],
            "AST phenotypes": ["ciprofloxacin=R,vancomycin=S", "gentamicin=MIC4"],
        }
    )
    result = build_phenotype_ledger(
        frame, isolate_column="Isolate", ast_column="AST phenotypes"
    )
    assert result.reconciles()


def test_conflicting_duplicates_are_dropped_and_logged():
    frame = pd.DataFrame(
        {"Isolate": ["A", "A"], "AST phenotypes": ["ciprofloxacin=R", "ciprofloxacin=S"]}
    )
    result = build_phenotype_ledger(
        frame, isolate_column="Isolate", ast_column="AST phenotypes"
    )
    assert result.ledger.empty
    assert (result.rejects["reason"] == "conflicting_duplicate_observations").sum() == 2


def test_denominator_ledger_separates_isolates_from_labels():
    """Audit findings A3/A4: the two must never be summed into one 'n'."""
    frame = pd.DataFrame(
        {
            "Isolate": ["A", "B"],
            "AST phenotypes": [
                "ciprofloxacin=R,tetracycline=S",
                "ciprofloxacin=S,tetracycline=R",
            ],
        }
    )
    result = build_phenotype_ledger(
        frame, isolate_column="Isolate", ast_column="AST phenotypes"
    )
    denominators = drug_denominator_ledger(result.ledger)
    assert denominators["n_labels"].sum() == 4
    assert result.counts.iloc[0]["n_unique_isolates_with_label"] == 2


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------

def test_point_mutation_yields_allele_and_gene():
    assert tokenise_amr_cell("gyrA=T86I,tetO") == [
        ("gene", "gyrA"),
        ("gene", "tetO"),
        ("point_mutation", "gyrA_T86I"),
    ]


def test_repeated_determinant_is_presence_not_count():
    assert tokenise_amr_cell("tetO,tetO") == [("gene", "tetO")]


def test_vocabulary_is_fitted_and_does_not_grow_at_transform():
    """Leakage fix: a test fold may not widen the feature space."""
    train = pd.Series(["tetO", "tetO", "tetO"])
    test = pd.Series(["tetO,blaOXA-61"])
    vocabulary = GeneVocabulary(min_count=2).fit(train)
    encoded = vocabulary.transform(test)
    assert list(encoded.columns) == ["gene__tetO"]
    assert vocabulary.oov_report(test).iloc[0]["determinant"] == "blaOXA-61"


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        GeneVocabulary().transform(pd.Series(["tetO"]))


# --------------------------------------------------------------------------
# grouping
# --------------------------------------------------------------------------

def test_degenerate_grouping_raises_instead_of_becoming_kfold():
    """The central leakage defect: per-row groups made GroupKFold a no-op."""
    frame = pd.DataFrame({"Isolate": [f"I{i}" for i in range(10)]})
    with pytest.raises(ValueError, match="no grouping column"):
        resolve_groups(frame)


def test_strongest_available_column_is_chosen():
    frame = pd.DataFrame(
        {
            "SNP cluster": ["c1", "c1", "c2", "c2"],
            "BioSample": ["b1", "b2", "b3", "b4"],
            "Isolate": ["i1", "i2", "i3", "i4"],
        }
    )
    resolution = resolve_groups(frame)
    assert resolution.column == "SNP cluster"
    assert resolution.n_groups == 2
    assert resolution.is_degenerate is False


def test_missing_group_values_become_traceable_singletons():
    frame = pd.DataFrame({"SNP cluster": ["c1", "c1", None, None]})
    resolution = resolve_groups(frame)
    assert resolution.n_missing_filled == 2
    assert resolution.n_groups == 3


def test_splits_never_share_a_group():
    rng = np.random.default_rng(0)
    groups = pd.Series([f"g{i // 4}" for i in range(200)])
    y = pd.Series(rng.integers(0, 2, size=200))
    splits = grouped_splits(y, groups, n_splits=5)
    assert len(splits) == 5
    assert check_no_group_overlap(splits, groups)


def test_too_few_groups_raises():
    y = pd.Series([0, 1, 0, 1])
    groups = pd.Series(["a", "a", "b", "b"])
    with pytest.raises(ValueError, match="cannot be split"):
        grouped_splits(y, groups, n_splits=5)


# --------------------------------------------------------------------------
# end-to-end, on synthetic data
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_task():
    frame = pd.read_csv(FIXTURE)
    ledger = build_phenotype_ledger(
        frame, isolate_column="Isolate", ast_column="AST phenotypes"
    ).ledger
    cipro = ledger.loc[ledger["drug"] == "ciprofloxacin"]
    merged = cipro.merge(frame, left_on="isolate", right_on="Isolate", how="left")
    return merged


def test_end_to_end_produces_traceable_metrics(synthetic_task):
    spec = TaskSpec(
        task_id="SYNTHETIC_cjejuni_ciprofloxacin",
        pathogen="Campylobacter jejuni (SYNTHETIC)",
        drug="ciprofloxacin",
        outcome_definition="R vs S from synthetic AST text; I excluded",
        min_feature_count=3,
    )
    result = run_task(synthetic_task, spec, amr_column="AMR genotypes")
    assert result.status == "completed"
    assert result.grouping["group_column"] == "SNP cluster"
    assert result.grouping["is_degenerate"] is False

    # Out-of-fold predictions exist for every labelled row, per model.
    n_models = result.pooled_metrics["model"].nunique()
    assert len(result.oof_predictions) == len(synthetic_task) * n_models

    # Every pooled metric reconciles with its own saved counts.
    for row in result.pooled_metrics.itertuples():
        recomputed = metrics_from_counts(
            ConfusionCounts(tp=row.tp, fp=row.fp, fn=row.fn, tn=row.tn)
        )
        assert recomputed["ppv"] == pytest.approx(row.ppv, nan_ok=True)
        assert recomputed["sensitivity"] == pytest.approx(row.sensitivity, nan_ok=True)
        assert row.n_total == row.tp + row.fp + row.fn + row.tn


def test_reported_metrics_pass_their_own_consistency_check(synthetic_task):
    """The output of this pipeline cannot reproduce the Table 3 defect."""
    spec = TaskSpec(
        task_id="SYNTHETIC_consistency",
        pathogen="Campylobacter jejuni (SYNTHETIC)",
        drug="ciprofloxacin",
        outcome_definition="R vs S",
        min_feature_count=3,
    )
    result = run_task(synthetic_task, spec, amr_column="AMR genotypes")
    for row in result.pooled_metrics.itertuples():
        if np.isnan(row.ppv) or np.isnan(row.npv):
            continue
        outcome = check_predictive_values(
            row.prevalence, row.sensitivity, row.specificity, row.ppv, row.npv
        )
        assert outcome["consistent"], f"{row.model} is internally inconsistent"


def test_baselines_are_evaluated_on_the_same_folds(synthetic_task):
    """A dominant single determinant must be reported next to the ensemble."""
    spec = TaskSpec(
        task_id="SYNTHETIC_baselines",
        pathogen="Campylobacter jejuni (SYNTHETIC)",
        drug="ciprofloxacin",
        outcome_definition="R vs S",
        min_feature_count=3,
    )
    result = run_task(synthetic_task, spec, amr_column="AMR genotypes")
    models = set(result.pooled_metrics["model"])
    assert {"random_forest", "majority_class", "single_determinant"} <= models
    folds_per_model = result.oof_predictions.groupby("model")["fold"].nunique()
    assert folds_per_model.nunique() == 1


def test_underpowered_task_is_skipped_with_a_reason():
    frame = pd.DataFrame(
        {
            "Isolate": [f"I{i}" for i in range(20)],
            "SNP cluster": [f"c{i // 4}" for i in range(20)],
            "AMR genotypes": ["tetO"] * 20,
            "label": [1] * 18 + [0] * 2,
        }
    )
    spec = TaskSpec(
        task_id="tiny", pathogen="X", drug="y", outcome_definition="R vs S"
    )
    result = run_task(frame, spec, amr_column="AMR genotypes")
    assert result.status.startswith("skipped:")
    assert "min_labels" in result.status
