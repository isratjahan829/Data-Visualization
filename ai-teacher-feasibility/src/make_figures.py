"""Rebuild every figure from the saved results (no model re-fitting)."""
import pandas as pd

import figures
from config import PROCESSED_CSV, TABLES

d = pd.read_csv(PROCESSED_CSV)
ct = pd.read_csv(TABLES / "table_reliability_validity.csv", index_col=0)
phi = pd.read_csv(TABLES / "table_latent_correlations.csv", index_col=0)
act = pd.read_csv(TABLES / "table_activity_feasibility.csv")
cv = pd.read_csv(TABLES / "cv_fold_scores.csv")
summ = pd.read_csv(TABLES / "table_model_comparison.csv", index_col=0)
imp = pd.read_csv(TABLES / "table_factor_importance.csv")
rob = pd.read_csv(TABLES / "robustness_draws.csv")
band = pd.read_csv(TABLES / "robustness_activity_bands.csv", index_col=0)
hm = pd.read_csv(TABLES / "table_activity_by_field.csv", index_col=0)
figures.make_all(d, ct, phi, act, cv, summ, imp, rob, band, hm)
print("figures written")
