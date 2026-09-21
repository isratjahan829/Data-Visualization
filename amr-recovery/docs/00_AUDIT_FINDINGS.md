# Audit findings: what the executed notebook proves

**Status:** evidence gathered, no manuscript numbers changed.
**Scope:** the four supplied files only — the executed Kaggle notebook
(`antimicrobial-resistance-prediction (2).ipynb`), the manuscript
(`Manuscript_genomic_antimicrobial.docx`), and the two supervisor reports.

Every finding below is derived from output that the notebook itself printed,
or from arithmetic on numbers printed in the manuscript. Nothing here is
inferred from the underlying NCBI data, which is not available in this
environment. Reproduce the arithmetic with:

```
python tools/verify_manuscript_arithmetic.py --out reports/
```

The action report's absolute stop condition applies: **no number was edited to
make a table agree.** These findings show which numbers cannot be retained, not
what should replace them.

---

## A1 — The "post-QC cohort" is the unfiltered row total. **Critical**

The seven per-species row counts printed by the notebook's loader sum to
exactly **1,804,571** — the figure the manuscript reports as the cohort
*after* applying "N50 >10 kb, contamination <5%, completeness >95%".

```
530,412 + 816,031 + 157,348 + 56,277 + 162,265 + 3,696 + 78,542 = 1,804,571
```

An exact match means the quality filters excluded zero isolates. Two further
observations make it clear they could not have been applied:

* Of the seven exports, **only `Klebsiella pneumoniae.csv` carries `N50`,
  `Length` and `Contigs` columns.** The other six have no assembly-quality
  fields at all, so no N50 threshold could be evaluated for 1,800,875 of the
  1,804,571 isolates.
* No export carries a contamination or completeness field in any form.

The related claim "Median N50 42·3 kb (IQR 28·1–67·4 kb)" is therefore
computable, at most, over the 3,696-row K. pneumoniae file — not over the
cohort it is attributed to.

**Consequence:** the Methods describe a QC step that did not happen. Either
apply it and report the exclusions, or state that no quality filtering was
performed and justify that.

## A2 — The C. jejuni export is malformed and rows were dropped untracked. **Critical**

The column-survey cell fails on this file:

```
⚠️ Could not read Campylobacter jejuni.csv: Error tokenizing data.
   C error: Expected 37 fields in line 5560, saw 41
```

The analysis pipeline hits the same error but catches it and retries with
`pd.read_csv(..., engine="python", on_bad_lines="skip")`. That silently
discards every malformed row, and the count is never recorded.

So `157,348` — which feeds the cohort total in A1 — is the raw line count of a
file that **was never read in full**. The number of isolates actually analysed
for C. jejuni is unknown and is not recoverable from the notebook output.

**Consequence:** the single largest contributor of usable AST data has an
unknown denominator. This must be resolved before any C. jejuni result is
reported.

## A3 — "16,030 AST-labelled isolates" is a sum of isolate–drug labels. **Critical**

The notebook prints a per-drug label count for C. jejuni:

| drug | labelled rows | resistant | prevalence |
|---|---:|---:|---:|
| ciprofloxacin | 4,009 | 830 | 20·7% |
| tetracycline | 4,008 | 1,914 | 47·8% |
| gentamicin | 4,004 | 33 | 0·8% |
| erythromycin | 4,009 | 85 | 2·1% |
| **sum** | **16,030** | | |

The four counts sum to exactly the manuscript's "n=16,030 AST-labelled
isolates". They are four labels drawn from what is evidently close to the same
~4,009 isolates, not 16,030 distinct isolates.

**The unique AST-labelled C. jejuni isolate count cannot exceed 4,009** — a
fourfold overstatement of the unit of analysis, propagating into Methods,
Results, Table 1 and the Figure 4 caption.

The prevalence column of Table 3 confirms the link: 0·8% matches 33/4,004 and
2·1% matches 85/4,009.

## A4 — The same double count applies to K. pneumoniae, against a hard ceiling. **Critical**

| drug | labelled rows | note |
|---|---:|---|
| ciprofloxacin | 61 | analysed (48 R / 13 S) |
| gentamicin | 66 | analysed (37 R / 29 S) |
| tetracycline | 48 | *"Insufficient data: 48 samples"* — skipped |
| erythromycin | 4 | *"Insufficient data: 4 samples"* — skipped |

`61 + 66 + 48 = 175`, exactly the manuscript's "n=175 AST-labelled isolates".

The ceiling is decisive here. The column survey reports **`AST phenotypes`:
76 non-null values** in a 3,696-row file. There are at most **76** AST-labelled
K. pneumoniae isolates in the entire export. A cohort of 175 isolates does not
exist.

Table 1's AST total, `16,205`, is `16,030 + 175` — so the double counting is
carried into the cohort total as well.

## A5 — Figure 4b describes a cohort that cannot be constructed. **Critical**

The caption's four cells are internally consistent:
`14,853 + 387 + 564 + 15,234 = 31,038`. But:

* stated accuracy is **97·1%**, while `(14,853 + 15,234)/31,038 = **96·94%**`;
* the caption explains the total as "four drug classes × C. jejuni isolates
  with AST labels", which would be `4 × 16,030 = 64,120`;
* the isolate–drug labels that do exist total 16,030 (A3).

**31,038 corresponds to no denominator available in this dataset.** The caption
also labels the panel as ciprofloxacin while describing values "aggregated
across four drug classes" — two different analyses in one figure.

Figure 4a separately states "n=10 validated pathogen-antibiotic combinations".
The notebook produced results for **6** pathogen–drug pairs (C. jejuni × 4,
K. pneumoniae × 2); the action report counts 7 drug-level tasks in Table 3.
Neither route reaches 10.

## A6 — There is no patient identifier, so "patient-level CV" never ran. **Critical**

The Methods claim patient-level `GroupKFold` with "singleton assignment for
8%". No export contains a patient field; the available identifier columns are
`Isolate`, `BioSample`, `Assembly` and `SNP cluster`.

Both code paths fall back to a group-per-row:

```python
# cell 2, create_patient_ids() — final fallback
return pd.Series(df.index.astype(str), index=df.index)

# cell 5, supervised_ml_analysis()
patient_ids = pd.Series(range(len(X_filt)), index=X_filt.index)
```

`GroupKFold` with one group per row **is** `KFold`. The leakage control named
in the Methods was inert in every reported run. Because C. jejuni contains
outbreak clusters of near-identical genomes (the `SNP cluster` field exists
precisely to mark them), near-duplicate isolates were free to straddle the
train/test boundary — which is the most likely explanation for AUC 0·997
alongside a single-mutation baseline of 0·994.

## A7 — Four different quantities are reported as "the feature count". **Critical**

| manuscript figure | what it actually is |
|---|---|
| 147 "curated determinants" | **not present in the code.** The notebook's `RESISTANCE_GENE_DATABASE` holds 56 gene symbols + 11 point mutations = **67** entries for the four reported classes, or 78 including an unreported beta-lactam class. |
| 208 (Table 1, C. jejuni) | the notebook's empirical vocabulary: *"Total unique genes: 208"* |
| 542 (Table 1, K. pneumoniae) | likewise: *"Total unique genes: 542"* |
| 205 ("remaining features", Fig 4c) | inconsistent with 208 − 2 = **206** |

208 and 542 are data-derived design-matrix widths that differ per species;
147 is a curation claim with no artefact behind it. They are not
interchangeable and are currently presented under one label.

## A8 — Table 3's predictive values contradict their own inputs. **Critical**

For every row, PPV and NPV are fixed by prevalence, sensitivity and
specificity through Bayes' theorem. Recomputing from the manuscript's own
printed inputs:

| row | p / Se / Sp (%) | reported PPV / NPV | implied PPV / NPV | PPV error |
|---|---|---|---|---:|
| K. pneumoniae cipro — RF | 78·7 / 94·2 / 73·3 | 68·2 / 94·6 | 92·9 / 77·4 | −24·7 pp |
| K. pneumoniae cipro — GB | 78·7 / 93·8 / 72·1 | 67·8 / 93·2 | 92·5 / 75·9 | −24·7 pp |
| K. pneumoniae gent — RF | 56·1 / 88·5 / 71·4 | 72·1 / 86·4 | 79·8 / 82·9 | −7·7 pp |
| C. jejuni gent — RF | 0·8 / 98·2 / 99·6 | 99·6 / 99·9 | 66·4 / ~100 | +33·2 pp |
| C. jejuni gent — GB | 0·8 / 97·9 / 99·5 | 99·5 / 99·9 | 61·2 / ~100 | +38·3 pp |
| C. jejuni ery — RF | 2·1 / 97·5 / 99·1 | 99·1 / 99·8 | 69·9 / 99·9 | +29·2 pp |

Rounding to one decimal cannot produce errors of 25–38 percentage points.

**A likely mechanism is visible in the C. jejuni rows.** In all three, the
reported PPV equals the reported **specificity** to the printed precision
(99·6 = 99·6; 99·5 = 99·5; 99·1 = 99·1). That is the signature of the
specificity column having been copied into the PPV column during table
assembly. The K. pneumoniae rows do not follow this pattern and need separate
tracing.

## A9 — The predictive values have no output to trace to. **Critical**

The pipeline's metric function does compute PPV and NPV per fold — but the
aggregation step keeps only AUC, F1, sensitivity and specificity:

```python
all_results.append({... 'AUC_mean', 'AUC_std', 'F1',
                    'Sensitivity', 'Specificity', 'N_samples', 'N_resistant'})
```

`all_results.csv`, the pipeline's only saved results artefact, **has no PPV or
NPV column**, and the per-fold confusion counts are discarded.

Meanwhile the columns that *were* saved appear in Table 3: the notebook's
fold-averaged K. pneumoniae ciprofloxacin Random Forest values are
sensitivity 0·941667 and specificity 0·733333 — printed in the manuscript as
94·2 and 73·3.

So Table 3 mixes columns with a traceable (if incorrectly aggregated) source
against columns with **no machine-generated origin at all**. Under the action
report's evidence hierarchy, the PPV/NPV columns are Level 3 evidence with no
Level 2 support and cannot be retained.

A second defect compounds this: the saved sensitivity and specificity are
`np.mean` over per-fold *ratios*. A mean of ratios corresponds to no confusion
matrix — with K. pneumoniae folds of ~12 rows, a fold contributing 1/2 weighs
the same as one contributing 900/1000. Pooled (micro) counts are required.

## A10 — Five of seven species have no AST column; the ML claim covers two. **Major**

Only `Klebsiella pneumoniae.csv` and `Campylobacter jejuni.csv` carry an
`AST phenotypes` column. For the other five the pipeline prints
`⚠ Required columns not found` and skips the species entirely.

The manuscript does state this limitation in Methods. But Figure 4a's "n=10
validated combinations" (A5) and the framing of a "tri-partite framework
across seven pathogens" both imply broader validation than two species and six
drug-model tasks.

## A11 — The surveillance prevalence function does not compute a prevalence. **Critical**

In cell 5, `genomic_surveillance_analysis` accumulates:

```python
for gene in resistance_genes:
    count = sum(1 for g in gene_counter if gene in g)   # unique gene NAMES matching
    samples_with_resistance += count
prevalence = (samples_with_resistance / len(df) * 100)
```

`gene_counter` is keyed by gene name, so `count` is **the number of distinct
gene names containing the substring**, not the number of isolates carrying the
gene. The numerator counts vocabulary entries; the denominator counts
isolates. The sum over genes also double-counts any isolate carrying more than
one determinant, so the value is not bounded by 100%.

Whatever produced Table 2's carriage percentages (99·2%, 99·8%, <2%), it was
not this function. Those percentages need their generating code identified
before they can be retained — and the L. monocytogenes "<2%" value carries the
"negative control validates pipeline specificity" claim.

## A12 — Cohort composition contradicts "clinical isolates". **Critical**

Methods describe "1,804,571 clinical bacterial isolates". The exports show
otherwise:

* `Source type` exists in **one** file (K. pneumoniae) with **7 non-null
  values out of 3,696**. For six species there is no source-type field at all.
* `Food origin` is populated for 19,558 L. monocytogenes, 88,153 S. enterica
  and 30,943 E. coli/Shigella records — food isolates are demonstrably in the
  cohort.
* NCBI Pathogen Detection mixes human, animal, food and environmental
  isolates by design.

There is no field supporting a clinical classification for the overwhelming
majority of records. The related claim that data are "de-identified" implies
patient-derived records that cannot be shown to exist.

## A13 — CARD and ResFinder are cited but never invoked. **Major**

Methods cite CARD v3·2·6, ResFinder v4·1 and AMRFinderPlus v3·11. The
notebook reads the `AMR genotypes` column of pre-computed NCBI CSV exports.
It contains no call to any of the three tools, no merge rule, no version
capture and no raw annotation output.

Likewise "Resistance was evaluated based on EUCAST clinical breakpoints": the
code reads R/S letters directly out of the free-text `AST phenotypes` field.
No MIC value, breakpoint table, standard or standard version is involved.

The "complete workflow runs in under 4 hours" claim has no timing log,
hardware description or input size attached.

---

## Where this leaves the go/no-go decision

Against section 10 of the action report:

* **GO — revised analytic paper** is not currently reachable. A1, A2 and A4
  mean the cohort cannot be reproduced as described; A8/A9 mean the principal
  diagnostic results have no auditable origin.
* **GO — descriptive surveillance only** is reachable *if* A11 is resolved —
  that is, if the code that generated Table 2 is located and its output
  reproduced against per-isolate denominators. The determinant-carriage
  analysis does not depend on the AST denominators that fail in A3/A4.
* The two-species genotype–phenotype analysis is **re-runnable from the same
  public data**, and should be, using the corrected pipeline in `src/amr/`.
  Its honest scale is roughly 4,000 C. jejuni isolates across four drugs, and
  **at most 76 K. pneumoniae isolates** — the latter is a case report's worth
  of data and should be reported as exploratory or dropped.

The recommendation is therefore the action report's own: **do not submit this
version**; rerun stages 1–5 before touching the narrative.

## What is NOT established here

Stating this plainly so the record is accurate:

* **No fabrication is demonstrated.** Every discrepancy above is consistent
  with aggregation errors, manual table assembly and unaudited claim drift.
  A3, A4 and A8 in particular look like one conceptual error (isolate vs
  isolate–drug) and one transcription error (a copied column).
* **The source data are genuine.** NCBI Pathogen Detection is authentic; the
  per-species row counts are internally consistent across two independent
  notebook cells.
* **The derived cohort is unverified, not disproven.** It may well reproduce.
  It simply has not been shown to.
* The exact provenance of Table 2 (A11) and of the K. pneumoniae PPV/NPV rows
  (A8) is **unresolved** and needs the original author's working files.
