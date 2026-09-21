# AMR manuscript recovery package

Audit and corrected re-analysis tooling for the genomic antimicrobial-resistance
surveillance manuscript placed on submission hold.

**Status: the manuscript should not be submitted in its current form.**
See [`docs/02_SUPERVISOR_RESPONSE.md`](docs/02_SUPERVISOR_RESPONSE.md).

## Start here

| Document | What it covers |
|---|---|
| [`docs/00_AUDIT_FINDINGS.md`](docs/00_AUDIT_FINDINGS.md) | 13 findings derived from the executed notebook's own output, with the arithmetic |
| [`docs/01_CODE_DEFECTS.md`](docs/01_CODE_DEFECTS.md) | 26 code defects mapped to notebook locations and to their fixes; plus the 5 known gaps |
| [`docs/02_SUPERVISOR_RESPONSE.md`](docs/02_SUPERVISOR_RESPONSE.md) | Register status, deliverables D1–D10, go/no-go recommendation |

## Reproduce the audit

```bash
pip install -r requirements.txt
python tools/verify_manuscript_arithmetic.py --out reports/
```

Every input to that script is transcribed verbatim from the manuscript or the
executed notebook, with its source location in a comment. It computes nothing
about the biology — it only tests whether the published numbers can coexist.
Current result: **13 defects, 10 critical**.

## Run the tests

```bash
python -m pytest tests/ -q      # 26 tests, ~95s
```

Each test pins one defect from `docs/01_CODE_DEFECTS.md`, so reintroducing a
defect fails here rather than in a table.

## The corrected pipeline

```python
from amr import build_phenotype_ledger, TaskSpec, run_task

ledger = build_phenotype_ledger(
    frame, isolate_column="Isolate", ast_column="AST phenotypes"
)
assert ledger.reconciles()          # parsed + rejected == candidate items

spec = TaskSpec(
    task_id="cjejuni_ciprofloxacin",
    pathogen="Campylobacter jejuni",
    drug="ciprofloxacin",
    outcome_definition="R vs S from NCBI AST phenotypes; I and SDD excluded",
)
result = run_task(analytic_rows, spec, amr_column="AMR genotypes")
result.write("outputs/")
```

`result.write` saves fold assignments, out-of-fold predictions, per-fold
integer confusion counts, pooled metrics, group-aware bootstrap intervals, the
versioned feature dictionary and a task summary carrying the seed, the
threshold and the grouping column actually used.

### What it enforces

- Every metric is derived from integer TP/FP/FN/TN. Undefined ratios are `NaN`,
  never `0.0`.
- Vocabulary and scaling are fitted **inside** the training fold.
- Grouping resolves to a real identifier (`SNP cluster` > `BioSample` > …) and
  **raises** if the best available grouping is one group per row, rather than
  silently degrading to KFold.
- Train/test group overlap is asserted on every split.
- Baselines (majority class, single determinant, logistic regression) run on
  identical folds, so a dominant single mutation cannot masquerade as model
  performance.
- Counts are pooled across folds (micro), not averaged as ratios.
- Malformed CSVs raise with a line-count discrepancy instead of being read with
  `on_bad_lines="skip"`.

## Important limits

- **Not run against the real data.** The NCBI exports are not available in this
  environment. End-to-end testing uses `tests/fixtures/SYNTHETIC_*.csv`,
  generated from a fixed seed with a planted genotype–phenotype relationship.
  **No number produced from that fixture may appear in the manuscript.**
- **Five gaps remain open** (G1–G5 in `docs/01_CODE_DEFECTS.md`): the curated
  determinant mapping, the provenance of Table 2, data access, figure
  generation, and the SHAP rework.
- **No manuscript number has been changed.** Per the action report's absolute
  stop condition, numerical consistency created by manual editing is not
  verification. This package identifies what cannot be retained; it does not
  supply replacements.

## Layout

```
src/amr/        library (metrics, phenotype, features, grouping,
                uncertainty, baselines, pipeline, provenance)
tools/          arithmetic verifier; synthetic fixture generator
tests/          regression tests + synthetic fixture
docs/           audit findings, defect register, supervisor response
reports/        generated output of the arithmetic verifier
```
