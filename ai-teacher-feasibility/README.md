# AI as a Substitute for Classroom Teachers in Higher Education — a soft-computing feasibility study

Replication package for the manuscript *"Can Artificial Intelligence Substitute for Classroom Teachers in
Higher Education? A Hierarchical Fuzzy–Neural Feasibility Assessment Based on Global Student Survey Data"*
(Maliha Binta Sumaiya).

## Dataset

**Higher Education Students' Early Perceptions of ChatGPT: Global Survey Data** — Ravšelj, D. et al.,
Mendeley Data, V2, doi:[10.17632/ymg9nsn6kn.2](https://doi.org/10.17632/ymg9nsn6kn.2), licence CC BY 4.0.
Accompanying article: Ravšelj et al. (2025), *PLOS ONE* 20(2): e0315011,
doi:[10.1371/journal.pone.0315011](https://doi.org/10.1371/journal.pone.0315011).

23,218 higher-education students from 109 countries. The analytic sample has 11,920 ChatGPT users with complete
responses. `data/raw/` contains the unmodified data file and questionnaire, redistributed under CC BY 4.0
with attribution to the original authors.

## Layout

```
data/raw/                 final_dataset.xlsx, questionnaire.pdf  (original files)
data/processed/           analytic_sample.csv (written by the pipeline; institution names removed)
src/config.py             item → factor / activity mapping, thresholds
src/data_prep.py          inclusion criteria, reverse coding, construct scores
src/psychometrics.py      Cronbach's α, ML-CFA (CFI/TLI/RMSEA/SRMR, CR, AVE), HTMT
src/fuzzy_system.py       vectorised hierarchical Mamdani FIS (TRFS) + activity FIS
src/models.py             ANN (nested-tuned MLP), hybrid fuzzy-ANN, LR, SVR, RF, GBM, corrected t-test
src/run_pipeline.py       full analysis → results/tables, results/figures, results/summary.json
src/figures.py            publication figures (PDF + 300-dpi PNG)
src/make_figures.py       rebuild figures from saved results
src/make_latex.py         results → manuscript/generated/*.tex (macros + tables)
notebooks/TRFS_analysis.ipynb   Colab-ready walkthrough (executed)
manuscript/main.tex       manuscript (Elsevier elsarticle), refs.bib, main.pdf
```

## Reproduce

```bash
pip install -r requirements.txt
cd src
python run_pipeline.py          # ~45 min on 4 cores (use --fast for a 7-min smoke test)
python make_latex.py            # numbers/tables for the paper
cd ../manuscript && latexmk -pdf main.tex
```

In Google Colab: open `notebooks/TRFS_analysis.ipynb` and choose *Runtime → Run all*. The first cell clones this
repository and installs the requirements.

## Method summary

| Stage | What is done |
|---|---|
| Measurement | 8 factors (TE, SE, P, AC, C, A, R, E) from 26 Likert items; ML-CFA, α, CR, AVE, HTMT, Harman test |
| Fuzzy | FIS₁ PC(TE,P,AC,A) · FIS₂ IQ(SE,C) · FIS₃ TW(R,E) → FIS₄ TRFS(PC,IQ,TW), 126 rules, centroid |
| Activities | 10 teaching activities, FIS(Capability, R, E); thresholds 35 / 65 → Human / Partial / Full |
| Validation | Teacher Substitution Index (Q24b, Q24d) never used as a fuzzy input |
| Learning | ANN (nested grid search) vs hybrid fuzzy-ANN, LR, SVR, RF, GBM, calibrated FIS; 10×3 repeated CV; Nadeau–Bengio test |
| Importance | ANN permutation importance, FIS ±1 SD sensitivity, Spearman ρ |
| Robustness | 500 random MF / t-norm / rounding configurations; Kendall τ and category agreement |
