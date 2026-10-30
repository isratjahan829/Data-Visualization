"""Load the raw survey, apply inclusion criteria and build construct scores."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (ACTIVITIES, FACTORS, PROCESSED_CSV, RAW_XLSX, REVERSED,
                    TARGET, TARGET_ITEMS, all_model_items)


def load_raw():
    return pd.read_excel(RAW_XLSX, sheet_name=0)


def build_sample(raw: pd.DataFrame):
    """Return (analytic sample with scores, flow-chart counts)."""
    flow = {"total_responses": len(raw)}
    d = raw.copy()
    d = d[d["Q13"] == 1]
    flow["chatgpt_users"] = len(d)
    d = d[d["Q3"].between(18, 100)]
    flow["valid_age_18_100"] = len(d)
    items = all_model_items()
    d = d.dropna(subset=items)
    flow["complete_on_model_items"] = len(d)
    # straight-lining across the 60+ model items carries no information
    d = d[d[items].std(axis=1) > 0]
    flow["after_straightlining_removal"] = len(d)
    d = d.reset_index(drop=True)

    scored = d.copy()
    for it in REVERSED:
        scored[it] = 6 - scored[it]
    for f, spec in FACTORS.items():
        d[f] = scored[spec["items"]].mean(axis=1)
    for a, spec in ACTIVITIES.items():
        d[a] = scored[spec["items"]].mean(axis=1)
    d[TARGET] = (d[TARGET_ITEMS].mean(axis=1) - 1) / 4 * 100
    return d, scored, flow


def save_processed(d):
    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    keep = [c for c in d.columns if c not in ("Q5",)]  # drop free-text institution names
    d[keep].to_csv(PROCESSED_CSV, index=False)
