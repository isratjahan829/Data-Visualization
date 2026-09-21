"""Build an auditable isolate-drug phenotype ledger from NCBI AST text.

Deliverable D2 of the action report. The unit of analysis here is one
**isolate-drug row**, made explicit because conflating it with "isolate" is
the single defect that produced the manuscript's 16,030 and 175 denominators
(see docs/00_AUDIT_FINDINGS.md, findings A3/A4).

Every source cell produces either a parsed label or a row in the reject log
with a reason code. Nothing is dropped silently, so the QC ledger can prove
``parsed + rejected == candidate rows``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

__all__ = ["DrugVocabulary", "PhenotypeLedger", "build_phenotype_ledger", "DEFAULT_DRUGS"]

# Controlled drug vocabulary. Keys are canonical names; values are the
# alternative spellings accepted in the source field.
#
# Matching is on whole tokens only. The original pipeline used
# ``if antibiotic in item`` and an abbreviation table containing 'tet',
# 'gen' and 'ery', so 'oxytetracycline=S' was read as a tetracycline result
# and 'gentamicin' shared a prefix with any token containing 'gen'. Aliases
# below are therefore full drug names, never fragments.
DEFAULT_DRUGS: dict[str, tuple[str, ...]] = {
    "ciprofloxacin": ("ciprofloxacin",),
    "tetracycline": ("tetracycline",),
    "gentamicin": ("gentamicin",),
    "erythromycin": ("erythromycin",),
}

# Interpretation codes. 'I' and the non-susceptible variants are kept as
# distinct outcomes rather than folded into R or dropped without trace; the
# analysis config decides how they are handled, and the decision is recorded.
_INTERPRETATIONS = {
    "r": "R",
    "resistant": "R",
    "i": "I",
    "intermediate": "I",
    "s": "S",
    "susceptible": "S",
    "ss": "S",
    "nonsusceptible": "NS",
    "non-susceptible": "NS",
    "sdd": "SDD",
    "susceptible-dose-dependent": "SDD",
}

_SPLIT_ITEMS = re.compile(r"[,;|]")
_TOKEN = re.compile(r"[^a-z0-9()'\-]+")


@dataclass
class DrugVocabulary:
    """Canonical-name lookup with whole-token matching."""

    drugs: dict[str, tuple[str, ...]] = field(default_factory=lambda: dict(DEFAULT_DRUGS))

    def __post_init__(self) -> None:
        self._alias_to_canonical: dict[str, str] = {}
        for canonical, aliases in self.drugs.items():
            for alias in (canonical,) + tuple(aliases):
                alias = alias.strip().lower()
                if len(alias) < 4:
                    raise ValueError(
                        f"alias {alias!r} for {canonical!r} is too short to match safely; "
                        "use full drug names, not abbreviations"
                    )
                existing = self._alias_to_canonical.get(alias)
                if existing not in (None, canonical):
                    raise ValueError(f"alias {alias!r} is ambiguous ({existing} vs {canonical})")
                self._alias_to_canonical[alias] = canonical

    def resolve(self, token: str) -> str | None:
        """Canonical drug name for an exact token, else None."""
        return self._alias_to_canonical.get(token.strip().lower())


def _parse_item(item: str, vocabulary: DrugVocabulary) -> tuple[str | None, str | None, str]:
    """Parse one ``drug=interpretation`` item.

    Returns ``(canonical_drug, interpretation, reason)``. ``reason`` explains
    a None result so it can be logged.
    """
    raw = item.strip()
    if not raw:
        return None, None, "empty_item"
    if "=" in raw:
        drug_part, _, value_part = raw.partition("=")
    else:
        # Fall back to whitespace separation ("ciprofloxacin R").
        parts = raw.rsplit(None, 1)
        if len(parts) != 2:
            return None, None, "no_separator"
        drug_part, value_part = parts

    drug_token = _TOKEN.sub(" ", drug_part.lower()).strip()
    canonical = vocabulary.resolve(drug_token)
    if canonical is None:
        return None, None, f"drug_not_in_vocabulary:{drug_token[:40]}"

    # Interpretation is the first recognised code in the value. MIC text such
    # as "R (MIC>=4)" resolves to R; "MIC=4" alone resolves to nothing, because
    # applying a breakpoint is a separate, versioned step (see note below).
    value_tokens = [t for t in _TOKEN.sub(" ", value_part.lower()).split() if t]
    for token in value_tokens:
        if token in _INTERPRETATIONS:
            return canonical, _INTERPRETATIONS[token], "ok"
    return canonical, None, f"interpretation_unparsed:{value_part.strip()[:40]}"


def _split_items(cell: str) -> list[str]:
    return [part for part in _SPLIT_ITEMS.split(str(cell)) if part.strip()]


@dataclass
class PhenotypeLedger:
    """Long-format phenotype table plus its reject log and count ledger."""

    ledger: pd.DataFrame
    rejects: pd.DataFrame
    counts: pd.DataFrame

    def reconciles(self) -> bool:
        """True when parsed + rejected accounts for every candidate item."""
        return bool(
            (self.counts["n_parsed"] + self.counts["n_rejected"] == self.counts["n_items"]).all()
        )


def build_phenotype_ledger(
    frame: pd.DataFrame,
    *,
    isolate_column: str,
    ast_column: str,
    vocabulary: DrugVocabulary | None = None,
    intermediate_policy: str = "exclude",
    duplicate_policy: str = "drop_conflicting",
) -> PhenotypeLedger:
    """Explode the free-text AST column into one row per isolate-drug outcome.

    Parameters
    ----------
    intermediate_policy
        ``"exclude"`` (default) drops I/SDD/NS rows from the binary label but
        keeps them in the ledger with ``label = NA``; ``"resistant"`` codes
        them as 1. Either way the choice is recorded in the returned counts so
        the manuscript can state it.
    duplicate_policy
        ``"drop_conflicting"`` (default) removes an isolate-drug pair whose
        repeated entries disagree and records it; ``"keep_first"`` keeps the
        first observation. Repeats are never averaged into a fractional label.
    """
    if intermediate_policy not in {"exclude", "resistant"}:
        raise ValueError(f"unknown intermediate_policy {intermediate_policy!r}")
    if duplicate_policy not in {"drop_conflicting", "keep_first"}:
        raise ValueError(f"unknown duplicate_policy {duplicate_policy!r}")

    vocabulary = vocabulary or DrugVocabulary()
    rows: list[dict] = []
    rejects: list[dict] = []
    n_items = 0

    subset = frame[[isolate_column, ast_column]]
    for isolate, cell in subset.itertuples(index=False, name=None):
        if pd.isna(cell):
            continue
        for item in _split_items(cell):
            n_items += 1
            drug, interpretation, reason = _parse_item(item, vocabulary)
            if drug is None or interpretation is None:
                rejects.append(
                    {
                        "isolate": isolate,
                        "raw_item": item.strip(),
                        "drug": drug,
                        "reason": reason,
                    }
                )
                continue
            rows.append(
                {
                    "isolate": isolate,
                    "drug": drug,
                    "interpretation": interpretation,
                    "raw_item": item.strip(),
                }
            )

    ledger = pd.DataFrame(rows, columns=["isolate", "drug", "interpretation", "raw_item"])
    reject_frame = pd.DataFrame(rejects, columns=["isolate", "raw_item", "drug", "reason"])

    if not ledger.empty:
        ledger["label"] = ledger["interpretation"].map({"R": 1, "S": 0}).astype("Int64")
        if intermediate_policy == "resistant":
            ledger.loc[ledger["interpretation"].isin(["I", "NS"]), "label"] = 1
        ledger = _resolve_duplicates(ledger, duplicate_policy, reject_frame)

    counts = pd.DataFrame(
        [
            {
                "n_source_rows": len(frame),
                "n_rows_with_ast": int(frame[ast_column].notna().sum()),
                "n_items": n_items,
                "n_parsed": len(ledger),
                "n_rejected": len(reject_frame),
                "n_unique_isolates_with_label": int(
                    ledger.loc[ledger["label"].notna(), "isolate"].nunique()
                )
                if not ledger.empty
                else 0,
                "n_isolate_drug_labels": int(ledger["label"].notna().sum())
                if not ledger.empty
                else 0,
                "intermediate_policy": intermediate_policy,
                "duplicate_policy": duplicate_policy,
            }
        ]
    )
    return PhenotypeLedger(ledger=ledger, rejects=reject_frame, counts=counts)


def _resolve_duplicates(
    ledger: pd.DataFrame, policy: str, reject_frame: pd.DataFrame
) -> pd.DataFrame:
    """Collapse repeated isolate-drug observations under the declared policy."""
    grouped = ledger.groupby(["isolate", "drug"])["label"]
    n_distinct = grouped.transform("nunique")
    conflicting = n_distinct > 1
    if policy == "drop_conflicting":
        for row in ledger.loc[conflicting].itertuples():
            reject_frame.loc[len(reject_frame)] = {
                "isolate": row.isolate,
                "raw_item": row.raw_item,
                "drug": row.drug,
                "reason": "conflicting_duplicate_observations",
            }
        ledger = ledger.loc[~conflicting]
    return ledger.drop_duplicates(subset=["isolate", "drug"], keep="first").reset_index(drop=True)


def drug_denominator_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    """Per-drug denominators: the table the manuscript needs and never had.

    Reports unique isolates and isolate-drug labels side by side so the two
    can never again be summed into a single "n" (finding A3).
    """
    if ledger.empty:
        return pd.DataFrame(
            columns=["drug", "n_unique_isolates", "n_labels", "n_resistant", "n_susceptible"]
        )
    labelled = ledger.loc[ledger["label"].notna()]
    out = (
        labelled.groupby("drug")
        .apply(
            lambda g: pd.Series(
                {
                    "n_unique_isolates": g["isolate"].nunique(),
                    "n_labels": len(g),
                    "n_resistant": int((g["label"] == 1).sum()),
                    "n_susceptible": int((g["label"] == 0).sum()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    return out
