"""Predictive models of the Teacher Substitution Index (TSI) and the
repeated cross-validation protocol used to compare them."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from config import SEED


class FuzzyCalibrated(BaseEstimator, RegressorMixin):
    """The HFIS score itself, linearly calibrated to the TSI scale on the
    training fold only (the FIS has no learned parameters)."""

    def __init__(self, col=-1):
        self.col = col

    def fit(self, X, y):
        self.lr_ = LinearRegression().fit(np.asarray(X)[:, [self.col]], y)
        return self

    def predict(self, X):
        return self.lr_.predict(np.asarray(X)[:, [self.col]])


def ann(seed=SEED):
    """MLP tuned by an inner 3-fold grid search inside every outer fold."""
    mlp = MLPRegressor(activation="relu", solver="adam", learning_rate_init=1e-3,
                       batch_size=128, max_iter=400, early_stopping=True,
                       validation_fraction=0.15, n_iter_no_change=15, random_state=seed)
    pipe = make_pipeline(StandardScaler(), mlp)
    grid = {"mlpregressor__hidden_layer_sizes": [(16,), (32, 16), (64, 32)],
            "mlpregressor__alpha": [1e-4, 1e-2]}
    return GridSearchCV(pipe, grid, cv=KFold(3, shuffle=True, random_state=seed),
                        scoring="neg_mean_squared_error", n_jobs=-1)


def model_zoo(seed=SEED):
    """name -> (estimator, feature set). Feature sets: 'F' = 8 factors,
    'FH' = 8 factors + HFIS intermediate outputs (hybrid), 'T' = TRFS only."""
    return {
        "Fuzzy HFIS (calibrated)": (FuzzyCalibrated(), "T"),
        "Linear regression": (make_pipeline(StandardScaler(), LinearRegression()), "F"),
        "SVR (RBF)": (make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=5.0, gamma="scale",
                                                          cache_size=1000)), "F"),
        "Random forest": (RandomForestRegressor(n_estimators=300, min_samples_leaf=20,
                                                n_jobs=-1, random_state=seed), "F"),
        "Gradient boosting": (HistGradientBoostingRegressor(learning_rate=0.05, max_iter=400,
                                                            min_samples_leaf=40,
                                                            early_stopping=True,
                                                            random_state=seed), "F"),
        "ANN (MLP)": (ann(seed), "F"),
        "Hybrid fuzzy-ANN": (ann(seed), "FH"),
    }


def metrics(y, p):
    mse = mean_squared_error(y, p)
    return {"MSE": mse, "RMSE": np.sqrt(mse), "MAE": mean_absolute_error(y, p),
            "R2": r2_score(y, p)}


def repeated_cv(Xsets, y, zoo, n_splits=10, n_repeats=3, seed=SEED, log=print):
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    rows, chosen = [], []
    for fold, (tr, te) in enumerate(rkf.split(y)):
        for name, (est, fs) in zoo.items():
            X = Xsets[fs]
            t0 = time.time()
            m = clone(est).fit(X[tr], y[tr])
            r = metrics(y[te], m.predict(X[te]))
            r.update(model=name, fold=fold, n_train=len(tr), n_test=len(te),
                     seconds=time.time() - t0)
            rows.append(r)
            if isinstance(m, GridSearchCV):
                chosen.append({"model": name, "fold": fold, **m.best_params_})
        log(f"  fold {fold + 1}/{n_splits * n_repeats} done")
    return pd.DataFrame(rows), pd.DataFrame(chosen)


def corrected_ttest(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test on paired fold scores."""
    d = np.asarray(a) - np.asarray(b)
    J = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / J + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), J - 1)
    return t, p
