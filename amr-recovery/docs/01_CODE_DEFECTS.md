# Code defect register

Each defect names the notebook location, the review item it explains, and the
module that replaces it. Cell numbers refer to
`antimicrobial-resistance-prediction (2).ipynb` (0-indexed, as stored).

| # | Defect | Location | Effect on reported results | Fixed in |
|---|---|---|---|---|
| C1 | Gene vocabulary (`top_genes=2000`) learned from the **whole file** before splitting | cell 2 `extract_gene_features`; cell 5 `extract_genes` | Feature-selection leakage; test folds influence the design matrix | `features.GeneVocabulary` is fit/transform; `pipeline.run_task` fits it inside each training fold |
| C2 | `GroupKFold` given one group per row | cell 2 `create_patient_ids` fallback; cell 5 `pd.Series(range(len(X_filt)))` | "Patient-level CV" was plain KFold; near-identical outbreak genomes straddle folds (finding A6) | `grouping.resolve_groups` picks `SNP cluster` > `BioSample` > `Assembly` and **raises** on a degenerate grouping; `check_no_group_overlap` asserts every split |
| C3 | SHAP block refits a scaler on **all** rows then `train_test_split(..., stratify=y)` with no groups | cell 2, SHAP section | The interpretability result reported beside the CV result comes from a leaked, ungrouped split | SHAP to be run on saved out-of-fold folds; scaling is per-fold in `pipeline.run_task` |
| C4 | PPV/NPV computed per fold then **dropped** before saving | cell 2 `all_results.append(...)` | `all_results.csv` has no PPV/NPV column; Table 3's values have no source (finding A9) | `pipeline` saves pooled and per-fold integer TP/FP/FN/TN; `metrics.metrics_from_counts` derives every value from them |
| C5 | Metrics aggregated as `np.mean` over per-fold **ratios** | cell 2 CV summary | A 12-row fold weighs as much as a 900-row fold; the result matches no confusion matrix | `metrics.aggregate_counts` pools counts (micro); `uncertainty.fold_dispersion` publishes per-fold counts separately |
| C6 | Folds with a single class skipped with `continue`, then averaged over survivors | cell 2 and cell 5 CV loops | "AUC over 5 folds" may be an average over 2, undisclosed | `pipeline` records `n_folds_evaluated` next to `n_folds_requested` and notes each skip |
| C7 | `mean ± np.std` of 5 fold AUCs presented as uncertainty | cell 2 CV summary | With n=61 and folds of ~12, fold AUCs ranged 0·21–1·00; the ± is not a CI | `uncertainty.bootstrap_auc` / `grouped_bootstrap` resample **groups** on pooled out-of-fold predictions |
| C8 | Drug matched by substring, with abbreviations `tet`, `gen`, `ery`, `cip` | cell 5 `parse_ast_label`; cell 2 `if antibiotic in it` | `oxytetracycline=S` read as a tetracycline result; any token containing `gen` matches gentamicin | `phenotype.DrugVocabulary` matches whole tokens and **rejects aliases shorter than 4 characters at construction** |
| C9 | `re.search(r"\b(r\|s\|i)\b", item)` scans the whole item, first match wins, `return None` exits the loop early | cell 2 `parse_ast_label` | A stray standalone letter anywhere in the cell sets the label; later matching items unreachable | `phenotype._parse_item` splits `drug=value` and reads the interpretation from the value side only |
| C10 | Intermediate (`I`), `SDD` and non-susceptible silently unparsed → `None` → row dropped | both parsers | Undocumented exclusions; no count | Interpretations retained in the ledger with `label = NA` under a recorded `intermediate_policy` |
| C11 | No duplicate resolution for repeated isolate–drug observations | both parsers | Conflicting repeats enter the model as independent rows | `phenotype._resolve_duplicates` drops conflicting pairs into the reject log under a declared policy |
| C12 | Unit of analysis never defined; per-drug counts summed into one "n" | cell 2 output → manuscript | Findings A3/A4: 16,030 and 175 | `phenotype.drug_denominator_ledger` reports `n_unique_isolates` and `n_labels` as separate columns |
| C13 | `pd.read_csv(..., engine="python", on_bad_lines="skip")` on parser failure | cell 2 `load_csv_robust`; cell 5 `load_csv` bare `except` | Malformed C. jejuni rows discarded untracked (finding A2) | `provenance.read_csv_strict` **raises**, reporting physical line count vs parseable rows, forcing an explicit exclusion rule |
| C14 | "Prevalence" numerator counts matching **gene names**, not isolates | cell 5 `genomic_surveillance_analysis` | Finding A11: the surveillance output is not a prevalence and is unbounded above 100% | Carriage must be computed per isolate over non-null `AMR genotypes`; see `features.GeneVocabulary.dictionary_` (`training_carriage`) |
| C15 | Bidirectional substring matching for gene→drug mapping (`res_gene in gene or gene in res_gene`) | cell 5 `gene_based_label` | `erm` matches any determinant containing "erm"; short symbols match longer database entries. Drives the "gene-based prediction" arm | Requires an explicit curated mapping table with exact determinant IDs — **not yet written**, see gap G1 |
| C16 | Absence of a known gene ⇒ label "susceptible" | cell 5 `gene_based_label` | Manufactures the "universal susceptibility" finding for species with no annotation column | Absence is uninformative; must be reported as "no determinant detected", never as a phenotype |
| C17 | Two different tokenisers for the same field (one splits on `[;,]`, the other also on `=` and `:` and drops tokens ≤2 chars) | cell 2 vs cell 5 | Two feature spaces from one column; contributes to finding A7 | `features.tokenise_amr_cell` is the single definition; point mutations yield allele **and** gene, each labelled by kind |
| C18 | `StandardScaler` applied to a binary presence/absence matrix, then tree models fit on the scaled values | cell 2 and cell 5 | Harmless to trees but obscures that features are binary; SHAP values are reported in scaled units | Trees consume the 0/1 matrix directly in `pipeline.run_task` |
| C19 | Hierarchical-attention model trained on **fold 1 only**, then tabulated beside 5-fold results | cell 2 (`if fold_num == 1:`) | Table rows report `AUC_std = 0·000000` for a single-fold estimate — e.g. K. pneumoniae ciprofloxacin AUC 0·333 ± 0·000 | All models run on identical folds; single-fold estimates are not produced |
| C20 | `except Exception` around each model, printing 100 characters and continuing | cell 2 model blocks | A systematically failing model silently drops out of the table | Failures surface; skips are recorded in `TaskResult.notes` with a reason |
| C21 | `0.0` returned for undefined ratios (`tp+fp == 0` ⇒ PPV `0.0`) | cell 2 `calculate_clinical_metrics` | "Undefined" becomes a reported "0%" | `metrics._ratio` returns `NaN`; pinned by `test_undefined_ratio_is_nan_not_zero` |
| C22 | `min_samples = 50` / `30` thresholds, inconsistent between cells, undocumented | cell 2 vs cell 5 | K. pneumoniae tetracycline (n=48) excluded by one path, admitted by the other | `TaskSpec.min_labels` / `min_minority` are explicit and written into `task_summary.csv` |
| C23 | No baseline comparator | both cells | AUC 0·997 for C. jejuni ciprofloxacin reads as a modelling result; a single-mutation rule reaches 0·994 | `baselines` (majority class, single determinant, logistic regression) run on the **same folds** |
| C24 | Fold assignments, seeds, out-of-fold predictions and environment never saved | both cells | Nothing downstream can be re-derived without a full rerun | `pipeline.TaskResult.write` saves all of them; `provenance.RunManifest` records config, seeds, package versions and input/output hashes |
| C25 | Output paths hard-coded to two different absolute directories (`/home/claude/improved_results/`, `/kaggle/working/`) | cell 2 vs cell 5 | Two runs wrote to different places; unclear which produced the manuscript tables | Output root is a single configured parameter |
| C26 | `gene_based_prediction` iterates `X.iterrows()` scanning 2,000 columns per row | cell 5 | O(rows × 2,000) Python-level work over a 157,348-row species; likely never completed at full scale | Vectorised presence/absence matrix in `features.GeneVocabulary.transform` |

---

## Known gaps in this package

Stated explicitly so nobody treats the package as complete.

* **G1 — curated determinant→drug mapping.** C15/C16 need a versioned table
  with exact determinant identifiers, a resistance mechanism, a citation per
  entry and an explicit "absence is uninformative" rule. That is a curation
  task requiring domain sign-off, not a code change. Until it exists, the
  gene-based prediction arm cannot be re-run, and the 147-determinant claim
  (A7) has nothing to point at.
* **G2 — Table 2 provenance.** The code that produced the carriage
  percentages has not been located (A11). Until it is, the surveillance arm
  cannot be reproduced even though it is the most salvageable part.
* **G3 — no data access.** Nothing in `src/` has been run against the real
  NCBI exports; they are not available in this environment. The pipeline is
  tested end to end only against `tests/fixtures/SYNTHETIC_*.csv`, which is
  generated from a seed and must never contribute a number to the manuscript.
* **G4 — figures.** No figure-generation code is included. Figures must be
  regenerated from the verified tables once stages 1–5 produce them.
* **G5 — SHAP.** C3's fix is specified but not implemented; the interpretability
  step should consume saved out-of-fold folds and use `TreeExplainer` rather
  than the generic `shap.Explainer(model.predict_proba, ...)` permutation path
  used in cell 2, which is intractable over a 2,000-column matrix.
