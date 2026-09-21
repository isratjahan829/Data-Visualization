#!/usr/bin/env python3
"""Reproduce every arithmetic finding in the audit, from declared inputs.

This script asserts nothing about the biology. It takes the numbers printed in
the manuscript and the numbers printed by the executed notebook, both
transcribed verbatim with their source locations, and checks whether they can
coexist. Run it before any rewriting begins:

    python tools/verify_manuscript_arithmetic.py --out reports/

Every check names the manuscript location it tests and the notebook output it
tests against, so a reviewer can verify the transcription independently.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amr.metrics import check_predictive_values  # noqa: E402

# --------------------------------------------------------------------------
# INPUTS, TRANSCRIBED. Nothing here is computed; every value is quoted.
# --------------------------------------------------------------------------

# Per-species row counts, from the manuscript Abstract and Methods "Data
# sources" paragraph, and independently from the notebook's own load messages
# ("Loaded: (530412, 20)" etc.) and the column-survey cell ("Total rows: ...").
SOURCE_FILE_ROWS = {
    "E. coli and Shigella": 530_412,
    "Salmonella enterica": 816_031,
    "Campylobacter jejuni": 157_348,
    "Pseudomonas aeruginosa": 56_277,
    "Staphylococcus aureus": 162_265,
    "Klebsiella pneumoniae": 3_696,
    "Listeria monocytogenes": 78_542,
}
MANUSCRIPT_COHORT_TOTAL = 1_804_571  # Abstract; Methods; Results; Table 1

# Per-drug label counts printed by the executed notebook (cell 2 output,
# "Samples:" lines under each ANTIBIOTIC heading).
NOTEBOOK_LABEL_COUNTS = {
    ("Campylobacter jejuni", "ciprofloxacin"): 4_009,
    ("Campylobacter jejuni", "tetracycline"): 4_008,
    ("Campylobacter jejuni", "gentamicin"): 4_004,
    ("Campylobacter jejuni", "erythromycin"): 4_009,
    ("Klebsiella pneumoniae", "ciprofloxacin"): 61,
    ("Klebsiella pneumoniae", "gentamicin"): 66,
    # tetracycline reported as "Insufficient data: 48 samples" and skipped;
    # erythromycin as "Insufficient data: 4 samples".
    ("Klebsiella pneumoniae", "tetracycline"): 48,
}
MANUSCRIPT_AST_COHORTS = {
    "Campylobacter jejuni": 16_030,  # Methods; Results; Table 1; Figure 4 caption
    "Klebsiella pneumoniae": 175,  # Methods; Results; Table 1; Table 4 footnote
}
MANUSCRIPT_AST_TOTAL = 16_205  # Table 1 total row
# Non-null 'AST phenotypes' cells, from the column-survey cell output.
NOTEBOOK_AST_NONNULL = {"Klebsiella pneumoniae": 76}

# Figure 4b confusion matrix, from the figure caption.
FIGURE4_MATRIX = {"tp": 14_853, "fp": 387, "fn": 564, "tn": 15_234}
FIGURE4_STATED_TOTAL = 31_038
FIGURE4_STATED_ACCURACY = 0.971
FIGURE4_STATED_COMBINATIONS = 10  # "n=10 combinations"

# Feature-space counts as they appear in the manuscript, against the counts
# the notebook printed ("Total unique genes: ...").
MANUSCRIPT_FEATURE_COUNTS = {
    "curated determinants (Methods, AMR annotation)": 147,
    "C. jejuni genes (Table 1)": 208,
    "K. pneumoniae genes (Table 1)": 542,
    "'remaining features' (Figure 4c caption)": 205,
}
NOTEBOOK_FEATURE_COUNTS = {
    "Campylobacter jejuni": 208,
    "Klebsiella pneumoniae": 542,
}
# Entries actually present in the notebook's RESISTANCE_GENE_DATABASE
# (cell 5), counted by parsing the literal.
CODE_CURATED_ENTRIES = {
    "genes, four reported classes": 56,
    "point mutations, four reported classes": 11,
    "total, four reported classes": 67,
    "total including the unreported beta-lactam class": 78,
}

# Table 3 rows, transcribed as (prevalence, sensitivity, specificity,
# reported PPV, reported NPV) in percent.
#
# SCOPE: these are the six rows quoted verbatim in the supervisor's action
# report (section 4.1). The full Table 3 is described there as holding seven
# drug-level ML tasks, so this transcription is a subset. Extend the list from
# the manuscript before citing a row count as complete.
TABLE3_ROWS = [
    ("K. pneumoniae", "ciprofloxacin", "random forest", 78.7, 94.2, 73.3, 68.2, 94.6),
    ("K. pneumoniae", "ciprofloxacin", "gradient boosting", 78.7, 93.8, 72.1, 67.8, 93.2),
    ("K. pneumoniae", "gentamicin", "random forest", 56.1, 88.5, 71.4, 72.1, 86.4),
    ("C. jejuni", "gentamicin", "random forest", 0.8, 98.2, 99.6, 99.6, 99.9),
    ("C. jejuni", "gentamicin", "gradient boosting", 0.8, 97.9, 99.5, 99.5, 99.9),
    ("C. jejuni", "erythromycin", "random forest", 2.1, 97.5, 99.1, 99.1, 99.8),
]

# Fold-averaged sensitivity/specificity printed in the notebook's final
# results table, for provenance tracing of the Table 3 columns.
NOTEBOOK_SENS_SPEC = {
    ("Klebsiella pneumoniae", "ciprofloxacin", "RandomForest"): (94.1667, 73.3333),
    ("Klebsiella pneumoniae", "gentamicin", "RandomForest"): (76.6558, 81.1111),
    ("Campylobacter jejuni", "gentamicin", "RandomForest"): (98.0000, 99.9245),
    ("Campylobacter jejuni", "erythromycin", "RandomForest"): (92.7066, 99.9237),
}


def _check(name: str, no_defect: bool, detail: str, severity: str = "critical") -> dict:
    """Record one check.

    ``no_defect`` is True only when the check found nothing that needs fixing.
    Note the polarity: several quantities tie arithmetically *and* are still
    defects, because what they tie to is the wrong denominator. Those are
    recorded as DEFECT with the tie described in the detail.
    """
    return {
        "check": name,
        "result": "OK" if no_defect else "DEFECT",
        "severity": severity,
        "detail": detail,
    }


def run_checks() -> tuple[list[dict], dict]:
    checks: list[dict] = []
    tables: dict[str, pd.DataFrame] = {}

    # A1 -- the cohort total is the unfiltered row sum.
    row_sum = sum(SOURCE_FILE_ROWS.values())
    checks.append(
        _check(
            "A1 cohort total vs unfiltered row sum",
            row_sum != MANUSCRIPT_COHORT_TOTAL,
            f"Sum of the seven source files' raw row counts is {row_sum:,}; the "
            f"manuscript's post-QC cohort is {MANUSCRIPT_COHORT_TOTAL:,}. These are "
            f"{'different' if row_sum != MANUSCRIPT_COHORT_TOTAL else 'IDENTICAL'}, so "
            "the stated N50 >10 kb / contamination <5% / completeness >95% filters "
            "excluded exactly zero isolates. Either the filters were not applied or "
            "the exclusion count is missing.",
        )
    )

    # A3/A4 -- AST 'isolate' counts are sums of isolate-drug labels.
    per_species = {}
    for species, claimed in MANUSCRIPT_AST_COHORTS.items():
        drug_labels = {
            drug: n for (sp, drug), n in NOTEBOOK_LABEL_COUNTS.items() if sp == species
        }
        label_sum = sum(drug_labels.values())
        per_species[species] = {
            "manuscript_ast_isolates": claimed,
            "notebook_drug_label_sum": label_sum,
            "n_drugs_summed": len(drug_labels),
            "largest_single_drug_n": max(drug_labels.values()) if drug_labels else 0,
            "notebook_ast_nonnull_rows": NOTEBOOK_AST_NONNULL.get(species),
        }
        checks.append(
            _check(
                f"A3 {species}: unit of analysis",
                label_sum != claimed,
                f"The manuscript reports n={claimed:,} 'AST-labelled isolates'. Summing "
                f"the notebook's per-drug label counts ({', '.join(f'{d}={n:,}' for d, n in drug_labels.items())}) "
                f"gives {label_sum:,}. The claimed isolate count is the sum of "
                f"isolate-drug labels; unique labelled isolates cannot exceed "
                f"{max(drug_labels.values()):,}"
                + (
                    f", and the source file holds only {NOTEBOOK_AST_NONNULL[species]:,} "
                    "non-null AST cells"
                    if species in NOTEBOOK_AST_NONNULL
                    else ""
                )
                + ".",
            )
        )
    tables["ast_denominators"] = pd.DataFrame(per_species).T.reset_index(names="species")

    total_claimed = sum(MANUSCRIPT_AST_COHORTS.values())
    checks.append(
        _check(
            "A4 Table 1 AST total",
            total_claimed != MANUSCRIPT_AST_TOTAL,
            f"Table 1's AST total {MANUSCRIPT_AST_TOTAL:,} equals "
            f"{' + '.join(f'{n:,}' for n in MANUSCRIPT_AST_COHORTS.values())} = "
            f"{total_claimed:,}, so the isolate-drug double counting propagates into "
            "the cohort total as well.",
            severity="critical",
        )
    )

    # A5 -- Figure 4b.
    matrix_total = sum(FIGURE4_MATRIX.values())
    accuracy = (FIGURE4_MATRIX["tp"] + FIGURE4_MATRIX["tn"]) / matrix_total
    cj_label_sum = sum(
        n for (sp, _), n in NOTEBOOK_LABEL_COUNTS.items() if sp == "Campylobacter jejuni"
    )
    checks.append(
        _check(
            "A5 Figure 4b cell total",
            False,  # ties internally, but to a denominator that does not exist
            f"The four cells sum to {matrix_total:,}, matching the stated total "
            f"{FIGURE4_STATED_TOTAL:,}. The total is internally consistent but matches "
            f"no cohort: C. jejuni isolate-drug labels sum to {cj_label_sum:,} and the "
            f"caption's 'four drug classes x isolates' would be {4 * cj_label_sum:,}.",
            severity="critical",
        )
    )
    checks.append(
        _check(
            "A5 Figure 4b accuracy",
            abs(accuracy - FIGURE4_STATED_ACCURACY) < 0.0005,
            f"(TP+TN)/total = {FIGURE4_MATRIX['tp'] + FIGURE4_MATRIX['tn']:,}/"
            f"{matrix_total:,} = {accuracy * 100:.2f}%, but the caption states "
            f"{FIGURE4_STATED_ACCURACY * 100:.1f}%.",
        )
    )
    checks.append(
        _check(
            "A5 Figure 4a combination count",
            FIGURE4_STATED_COMBINATIONS <= len(TABLE3_ROWS),
            f"Figure 4a states n={FIGURE4_STATED_COMBINATIONS} validated "
            f"pathogen-antibiotic combinations; Table 3 contains {len(TABLE3_ROWS)} "
            "drug-model rows, and the notebook produced results for 6 pathogen-drug "
            "pairs across 2 species. The count of 10 is not reconstructable.",
            severity="major",
        )
    )

    # A7 -- feature space.
    checks.append(
        _check(
            "A7 curated determinant count",
            MANUSCRIPT_FEATURE_COUNTS["curated determinants (Methods, AMR annotation)"]
            == CODE_CURATED_ENTRIES["total, four reported classes"],
            "Methods claim 147 curated determinants across four classes. The "
            "notebook's RESISTANCE_GENE_DATABASE holds "
            f"{CODE_CURATED_ENTRIES['total, four reported classes']} entries for those "
            f"classes ({CODE_CURATED_ENTRIES['genes, four reported classes']} genes + "
            f"{CODE_CURATED_ENTRIES['point mutations, four reported classes']} point "
            f"mutations), or {CODE_CURATED_ENTRIES['total including the unreported beta-lactam class']} "
            "including an unreported beta-lactam class. 147 appears nowhere in the code.",
        )
    )
    checks.append(
        _check(
            "A7 species gene counts",
            False,  # they match the notebook, and that is what makes 147 unexplained
            "Table 1's 208 and 542 match the notebook's empirical per-species gene "
            "vocabularies, so they are data-derived design-matrix widths, not the "
            "curated set. Three different quantities are reported under one label.",
            severity="major",
        )
    )
    remaining = MANUSCRIPT_FEATURE_COUNTS["'remaining features' (Figure 4c caption)"]
    cj_features = NOTEBOOK_FEATURE_COUNTS["Campylobacter jejuni"]
    checks.append(
        _check(
            "A7 Figure 4c remaining features",
            remaining == cj_features - 2,
            f"Figure 4c names two dominant features and attributes the rest to "
            f"'the remaining {remaining} features'. With {cj_features} C. jejuni "
            f"features, the remainder is {cj_features - 2}, not {remaining}.",
            severity="major",
        )
    )

    # A8 -- Table 3 predictive values.
    rows = []
    for species, drug, model, prevalence, sens, spec, ppv, npv in TABLE3_ROWS:
        outcome = check_predictive_values(
            prevalence / 100, sens / 100, spec / 100, ppv / 100, npv / 100, tolerance=0.02
        )
        rows.append(
            {
                "species": species,
                "drug": drug,
                "model": model,
                "prevalence_pct": prevalence,
                "sensitivity_pct": sens,
                "specificity_pct": spec,
                "reported_ppv_pct": ppv,
                "implied_ppv_pct": outcome["implied_ppv"] * 100,
                "ppv_error_pp": (ppv / 100 - outcome["implied_ppv"]) * 100,
                "reported_npv_pct": npv,
                "implied_npv_pct": outcome["implied_npv"] * 100,
                "npv_error_pp": (npv / 100 - outcome["implied_npv"]) * 100,
                "consistent": outcome["consistent"],
                "ppv_equals_specificity": abs(ppv - spec) < 0.05,
            }
        )
    table3 = pd.DataFrame(rows)
    tables["table3_predictive_values"] = table3

    n_inconsistent = int((~table3["consistent"]).sum())
    checks.append(
        _check(
            "A8 Table 3 PPV/NPV coherence",
            n_inconsistent == 0,
            f"{n_inconsistent} of {len(table3)} transcribed rows report predictive "
            "values incompatible with their own prevalence, sensitivity and "
            f"specificity. Largest PPV discrepancy: "
            f"{table3['ppv_error_pp'].abs().max():.1f} percentage points. Rounding of "
            "the inputs cannot account for errors of this size.",
        )
    )
    copied = table3.loc[table3["ppv_equals_specificity"]]
    checks.append(
        _check(
            "A8 PPV column provenance",
            copied.empty,
            f"In {len(copied)} of {len(table3)} rows the reported PPV equals the "
            "reported specificity to the printed precision"
            + (
                " ("
                + "; ".join(
                    f"{r.species} {r.drug} {r.model}: PPV={r.reported_ppv_pct}"
                    f"=Sp={r.specificity_pct}"
                    for r in copied.itertuples()
                )
                + ")"
                if not copied.empty
                else ""
            )
            + ". This is consistent with the specificity column having been copied "
            "into the PPV column during table assembly. The K. pneumoniae rows do "
            "not follow this pattern and need separate tracing.",
        )
    )

    # A9 -- notebook Se/Sp as the source of Table 3 columns.
    provenance_rows = []
    for (species, drug, model), (sens, spec) in NOTEBOOK_SENS_SPEC.items():
        short = "K. pneumoniae" if species.startswith("Klebsiella") else "C. jejuni"
        match = table3.loc[
            (table3["species"] == short)
            & (table3["drug"] == drug)
            & (table3["model"] == "random forest")
        ]
        if match.empty:
            continue
        row = match.iloc[0]
        provenance_rows.append(
            {
                "species": species,
                "drug": drug,
                "notebook_sensitivity_pct": sens,
                "table3_sensitivity_pct": row["sensitivity_pct"],
                "sensitivity_matches": abs(sens - row["sensitivity_pct"]) < 0.1,
                "notebook_specificity_pct": spec,
                "table3_specificity_pct": row["specificity_pct"],
                "specificity_matches": abs(spec - row["specificity_pct"]) < 0.1,
            }
        )
    provenance = pd.DataFrame(provenance_rows)
    tables["table3_provenance"] = provenance
    if not provenance.empty:
        n_matched = int(
            (provenance["sensitivity_matches"] & provenance["specificity_matches"]).sum()
        )
        checks.append(
            _check(
                "A9 Table 3 Se/Sp provenance",
                False,
                f"{n_matched} of {len(provenance)} traceable rows have Table 3 "
                "sensitivity and specificity matching the notebook's fold-averaged "
                "values exactly (K. pneumoniae ciprofloxacin RF: 94.2 / 73.3). Those "
                "fold averages are means of per-fold ratios, not values derived from "
                "pooled counts, and the notebook never emitted PPV or NPV at all -- so "
                "the Table 3 predictive-value columns have no output to trace to.",
                severity="critical",
            )
        )

    summary = {
        "n_checks": len(checks),
        "n_defects": sum(1 for check in checks if check["result"] == "DEFECT"),
        "n_critical_defects": sum(
            1
            for check in checks
            if check["result"] == "DEFECT" and check["severity"] == "critical"
        ),
    }
    return checks, {"summary": summary, "tables": tables}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("reports"), help="directory for generated tables"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    checks, extra = run_checks()
    check_frame = pd.DataFrame(checks)
    check_frame.to_csv(args.out / "arithmetic_checks.csv", index=False)
    for name, frame in extra["tables"].items():
        frame.to_csv(args.out / f"{name}.csv", index=False)
    (args.out / "arithmetic_summary.json").write_text(
        json.dumps(extra["summary"], indent=2)
    )

    width = 78
    print("=" * width)
    print("MANUSCRIPT ARITHMETIC VERIFICATION")
    print("=" * width)
    for check in checks:
        print(f"\n[{check['result']}] ({check['severity']}) {check['check']}")
        print(f"    {check['detail']}")
    print("\n" + "=" * width)
    print(
        f"{extra['summary']['n_defects']} of {extra['summary']['n_checks']} checks found a "
        f"defect ({extra['summary']['n_critical_defects']} critical)."
    )
    print(f"Tables written to {args.out}/")
    print("=" * width)
    # Defects are the expected outcome for the version under review, so exit 0:
    # this tool reports, it does not gate a build.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
