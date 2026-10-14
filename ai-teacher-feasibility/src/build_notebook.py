"""Build notebooks/TRFS_analysis.ipynb (run with: python build_notebook.py)."""
import nbformat as nbf

from config import ROOT

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# AI as a Substitute for Classroom Teachers in Higher Education — analysis notebook

Hierarchical fuzzy inference system (Teacher Replacement Feasibility Score, TRFS), activity classification,
ANN vs. benchmark models, factor importance and robustness, on the open **Global ChatGPT Student Survey**
(Ravšelj et al., Mendeley Data, doi:10.17632/ymg9nsn6kn.2, CC BY 4.0).

The notebook calls the same functions as `src/run_pipeline.py`, so every number matches the manuscript.
The two slow steps (30-fold cross-validation and 500 Monte Carlo fuzzy designs, ~30 min) are switched off by
default and their saved results are loaded instead; set `RUN_CV` / `RUN_ROBUSTNESS` to `True` to recompute them."""),
    md("## 0. Setup (Google Colab or local)"),
    code("""import os, sys
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB and not os.path.exists("Data-Visualization"):
    !git clone -q https://github.com/isratjahan829/Data-Visualization.git
    %cd Data-Visualization
    !git checkout -q claude/practical-darwin-bcbsl5
    %pip install -q -r ai-teacher-feasibility/requirements.txt
# locate the project folder
for cand in ["ai-teacher-feasibility", "..", "Data-Visualization/ai-teacher-feasibility"]:
    if os.path.exists(os.path.join(cand, "src", "run_pipeline.py")):
        PROJECT = os.path.abspath(cand); break
sys.path.insert(0, os.path.join(PROJECT, "src"))
os.chdir(os.path.join(PROJECT, "src"))
print("project:", PROJECT)

RUN_CV = False          # True -> re-run 10x3 repeated CV of all models (~20 min)
RUN_ROBUSTNESS = False  # True -> re-run 500 Monte Carlo fuzzy designs (~15 min)"""),
    code("""import json, warnings
import numpy as np, pandas as pd
from IPython.display import Image, display
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

import run_pipeline as rp
import figures
from config import *
from data_prep import load_raw, build_sample, save_processed
TABLES.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)"""),
    md("## 1. Data and analytic sample"),
    code("""raw = load_raw()
d, scored, flow = build_sample(raw)
print(raw.shape); print(flow)
demo = rp.describe_sample(raw, d, flow)
demo.round(1)"""),
    md("## 2. Measurement model (reliability, CFA, HTMT)"),
    code("""ct, phi, H = rp.measurement(raw, scored)
print(rp.SUMMARY["cfa_fit"])
display(ct.round(3))
display(H.round(2))
display(pd.read_csv(TABLES / "measurement_alternatives.csv").round(3))"""),
    md("## 3. Hierarchical fuzzy inference system → overall TRFS"),
    code("""d = rp.fuzzy_scores(d)
print({k: round(v, 2) if isinstance(v, float) else v for k, v in rp.SUMMARY["trfs"].items()})
pd.read_csv(TABLES / "table_trfs_validity.csv").round(3)"""),
    code("""pd.read_csv(TABLES / "rules_TRFS.csv")"""),
    md("## 4. Activity-level feasibility and classification (RQ1, RQ4)"),
    code("""act, M = rp.activity_scores(d, scored)
print(rp.SUMMARY["activity_friedman"])
act.drop(columns=["items"]).round(2)"""),
    md("## 5. ANN vs. benchmark models (RQ3)"),
    code("""if RUN_CV:
    cv, summ = rp.predictive(d)
else:
    cv = pd.read_csv(TABLES / "cv_fold_scores.csv")
    summ = pd.read_csv(TABLES / "table_model_comparison.csv", index_col=0)
summ.round(3)"""),
    md("## 6. Factor importance (RQ2)"),
    code("""imp = rp.importance(d)
imp.round(3)"""),
    md("## 7. Robustness of the fuzzy design and subgroups"),
    code("""if RUN_ROBUSTNESS:
    rob, band = rp.robustness(d, act, draws=500)
else:
    rob = pd.read_csv(TABLES / "robustness_draws.csv")
    band = pd.read_csv(TABLES / "robustness_activity_bands.csv", index_col=0)
print("mean Kendall tau:", rob.kendall_tau.mean().round(3),
      "| mean category agreement:", rob.category_agreement.mean().round(3))
sg, hm = rp.subgroups(d)
hm.round(1)"""),
    md("## 8. Figures"),
    code("""figures.make_all(d, ct, phi, act, cv, summ, imp, rob, band, hm)
for f in sorted(FIGURES.glob("fig*.png")):
    print(f.name); display(Image(filename=str(f), width=900))"""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "colab": {"provenance": []}})
out = ROOT / "notebooks" / "TRFS_analysis.ipynb"
nbf.write(nb, out)
print("wrote", out)
