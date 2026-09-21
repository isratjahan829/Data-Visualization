# Response to the Full Research Assistant Action Report

**To:** Dr. Mohammad Rifat Ahmmad Rashid
**Re:** Large-scale genomic AMR surveillance manuscript — Stage 0 report
**Constraint observed:** no new biological experiment; no number edited to make
a table agree.

---

## 1. Headline

The inconsistency register in your report is confirmed, and in five places the
executed notebook output identifies the **mechanism**, not just the symptom:

* `16,030` and `175` are sums of per-drug label counts, not isolate counts
  (§A3, §A4). For K. pneumoniae there is a hard ceiling: the source file holds
  **76 non-null AST cells**, so a 175-isolate cohort does not exist.
* `1,804,571` is the **unfiltered** row total of the seven exports, so the
  stated QC filters excluded zero isolates — and six of the seven files have no
  assembly-quality columns to filter on (§A1).
* Patient-level `GroupKFold` never ran: no patient field exists and both code
  paths fall back to one group per row, making it plain KFold (§A6).
* The pipeline's saved results file **has no PPV or NPV column**, so Table 3's
  predictive values have no machine-generated origin (§A9). In the three
  C. jejuni rows the reported PPV equals the reported specificity to the
  printed decimal — consistent with a copied column (§A8).
* The surveillance "prevalence" function divides a count of matching **gene
  names** by a count of isolates (§A11).

Full evidence: [`00_AUDIT_FINDINGS.md`](00_AUDIT_FINDINGS.md).
Code-level mapping: [`01_CODE_DEFECTS.md`](01_CODE_DEFECTS.md).

Two things your report left open that I could not close: the code that
generated Table 2 has not been located, and the K. pneumoniae PPV/NPV rows do
not follow the C. jejuni copy pattern, so they need separate tracing. Both need
the original working files.

**Nothing here demonstrates fabrication.** The pattern is consistent with one
conceptual error (isolate vs isolate–drug), one transcription error, and claim
drift beyond what the code does.

## 2. Register status

Your section 3 register, with what the notebook output settles.

| Severity | Issue | Status | Evidence |
|---|---|---|---|
| Critical | Cohort provenance | **Confirmed, worse than stated** — cohort total = unfiltered row sum; QC fields absent from 6/7 files | §A1 |
| Critical | Clinical-isolate claim | **Confirmed** — `Source type` present in 1 file, 7/3,696 non-null; `Food origin` populated for 3 species | §A12 |
| Critical | C. jejuni unit of analysis | **Resolved** — 4,009+4,008+4,004+4,009 = 16,030; unique isolates ≤ 4,009 | §A3 |
| Critical | K. pneumoniae denominators | **Resolved** — 61+66+48 = 175; ≤76 isolates exist | §A4 |
| Critical | Table 3 PPV/NPV | **Confirmed + mechanism** — 6/6 rows incoherent; PPV = specificity in 3 C. jejuni rows | §A8, §A9 |
| Critical | Figure 4 confusion matrix | **Confirmed** — accuracy 96·94% not 97·1%; 31,038 matches no denominator | §A5 |
| Critical | Feature-space definition | **Resolved** — 208/542 are empirical vocabularies; 147 absent from code (67 entries exist) | §A7 |
| Major | Annotation workflow | **Confirmed** — no CARD/ResFinder call anywhere; pre-computed NCBI column only | §A13 |
| Major | Runtime claim | **No evidence found** — no timing log, hardware or input size | §A13 |
| Major | Patient-level grouping | **Resolved** — no patient field; grouping was per-row | §A6 |
| Major | Validation combinations | **Confirmed** — 6 pathogen-drug pairs ran; Figure 4a says 10 | §A5, §A10 |
| Major | Gene carriage vs resistance | **Confirmed** — absence of a gene is coded as "susceptible" in the gene-based arm | C16 |
| Major | Negative-control claim | **Confirmed** — "<2% validates specificity" rests on the unresolved Table 2 code | §A11 |
| Major | Temporal/geographic results | **Not assessable** — no trend model appears anywhere in the notebook | — |
| Major | Confidence intervals | **Confirmed** — `mean ± np.std` over 5 fold ratios, one model fitted on fold 1 only | C5, C7, C19 |
| Major | Overstated scale/novelty | Authors' decision; superlatives unsupported by any artefact | — |
| Major | Authors/contributors | Authors' decision — not a data question | — |
| Major | Funding contradiction | Authors' decision — not a data question | — |
| Major | AI/figure artifacts | Confirmed by inspection of the manuscript file | — |
| Moderate | Data availability | Addressed in principle by this package | §4 |

## 3. What I built

An auditable re-analysis package, in this directory:

```
src/amr/metrics.py      every metric from integer TP/FP/FN/TN; NaN for
                        undefined ratios; + the PPV/NPV coherence checker
src/amr/phenotype.py    isolate-drug ledger with a reject log that reconciles
src/amr/features.py     one tokeniser; fit/transform vocabulary; versioned
                        feature dictionary
src/amr/grouping.py     resolves the strongest real identifier; RAISES on a
                        degenerate grouping; asserts no train/test group overlap
src/amr/uncertainty.py  group-level bootstrap on saved out-of-fold predictions
src/amr/baselines.py    majority class, single determinant, logistic regression
src/amr/pipeline.py     per-fold preprocessing; saves folds, seeds, OOF
                        predictions, integer counts, per-fold dispersion
src/amr/provenance.py   SHA-256 manifest, QC ledger with a reconciliation
                        assertion, strict CSV reader that refuses to skip rows
tools/verify_manuscript_arithmetic.py   reproduces every finding above
tools/make_synthetic_fixture.py         synthetic data for testing only
tests/test_amr.py       26 tests, each pinning one defect
```

`pytest` passes (26/26). The arithmetic verifier runs and reports 13 defects,
10 critical.

**The pipeline has not been run against the real exports** — they are not
available here. It is tested end to end only on synthetic data, and that data
is prefixed `SYNTHETIC_` so no number from it can drift into a table.

## 4. Deliverables D1–D10

| ID | Deliverable | Status | Blocker |
|---|---|---|---|
| D1 | Cohort provenance package | **Blocked** | Needs the NCBI query, retrieval date and export. Tooling ready: `provenance.RunManifest`, `QCLedger`, `read_csv_strict`. The C. jejuni export must be re-pulled — the current one is malformed (§A2). |
| D2 | Phenotype ledger | **Tooling ready, not run** | `phenotype.build_phenotype_ledger` + `drug_denominator_ledger` produce exactly the per-drug denominator table your report asks for. Needs the exports. |
| D3 | Feature package | **Partially blocked (G1)** | `features.GeneVocabulary` versions the empirical dictionary. The *curated* 147-determinant table does not exist and needs domain sign-off before the gene-based arm can be rerun. |
| D4 | Reproducible code | **Delivered for the ML arm** | Surveillance and figure code still to write (G2, G4). |
| D5 | Validation package | **Tooling ready, not run** | Folds, groups, seeds, OOF predictions, integer counts, CIs, baselines all saved by `pipeline.run_task`. |
| D6 | Tables and figures | **Not started** | Depends on D1–D5. |
| D7 | Revised manuscript | **Not started** | Per your instruction, narrative rewriting waits for stages 1–5. |
| D8 | Ethics/reporting package | **Authors' action** | Authorship, funding, oversight, AI disclosure are not mine to decide. I can compile evidence on request. |
| D9 | Citation audit | **Not started** | Your section 5.3 priorities stand. |
| D10 | Verification memo | **This document is the Stage 0 instalment** | |

## 5. Recommendation

**Do not submit this version.** Concretely, against your section 10:

* **NO-GO for the current empirical claims.** A3, A4, A8 and A9 mean the
  principal validation results cannot be traced to any saved output.
* **GO — descriptive surveillance only** is the realistic route, *conditional
  on closing G2*: locating the code behind Table 2 and recomputing carriage
  per isolate over non-null annotation columns. That arm does not depend on
  the AST denominators that fail.
* The two-species genotype–phenotype analysis **should be re-run** from the
  same public data with the corrected pipeline. Its honest scale is ~4,000
  C. jejuni isolates across four drugs and **at most 76 K. pneumoniae
  isolates**. I recommend reporting K. pneumoniae as exploratory with counts
  and intervals, or dropping it — 61 labelled isolates with 12-row folds
  produced fold AUCs from 0·21 to 1·00 in the run under review.
* Expect the C. jejuni AUC to **fall** once grouping is real. The 0·997 was
  obtained with outbreak clusters free to straddle folds, and the manuscript
  itself reports a single-mutation baseline of 0·994 — the headline number is
  close to a restatement of gyrA T86I.

Claims to remove regardless of how the rerun goes: point-of-care readiness,
clinical deployment, patient outcomes, five-species ML validation, universal
susceptibility, "largest to date", the four-hour runtime, EUCAST breakpoint
application, and CARD/ResFinder annotation.

## 6. What I need to proceed

1. The original working files — scripts, intermediate CSVs, the code behind
   Table 2 and Figure 4 (closes G2, and the K. pneumoniae PPV/NPV tracing).
2. The NCBI query used, with its retrieval date, or authorisation to define a
   fresh snapshot and report the new cohort as the cohort.
3. A decision on G1: who curates the determinant→drug mapping, with citations.
4. A decision on K. pneumoniae: report ≤76 isolates as exploratory, or drop it.

## 7. Stop conditions I will honour

Per your section 10: if asked to make numbers agree without source data and
reproducible code, I stop and escalate rather than reconcile by editing. None
of the numbers in the manuscript have been changed by this work, and none will
be except as output of a rerun.
