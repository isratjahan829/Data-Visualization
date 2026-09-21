"""Auditable re-analysis package for the genomic AMR surveillance manuscript.

Import surface is deliberately small; see ``docs/01_ISSUE_REGISTER.md`` for
the mapping from each review finding to the module that resolves it.
"""

from .metrics import (  # noqa: F401
    ConfusionCounts,
    aggregate_counts,
    check_predictive_values,
    counts_from_labels,
    implied_predictive_values,
    metrics_from_counts,
)
from .features import GeneVocabulary, tokenise_amr_cell  # noqa: F401
from .grouping import grouped_splits, resolve_groups  # noqa: F401
from .phenotype import DrugVocabulary, build_phenotype_ledger  # noqa: F401
from .pipeline import TaskSpec, run_task  # noqa: F401
from .provenance import QCLedger, RunManifest, read_csv_strict  # noqa: F401

__version__ = "0.1.0"
