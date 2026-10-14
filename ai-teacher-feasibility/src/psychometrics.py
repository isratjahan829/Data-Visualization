"""Reliability and validity statistics: Cronbach's alpha, McDonald's omega,
composite reliability (CR), average variance extracted (AVE), HTMT, and a
maximum-likelihood confirmatory factor analysis (CFA) written with SciPy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize


def cronbach_alpha(X):
    X = np.asarray(X, float)
    k = X.shape[1]
    return k / (k - 1) * (1 - X.var(axis=0, ddof=1).sum() / X.sum(axis=1).var(ddof=1))


def htmt(df, blocks):
    """Heterotrait-monotrait ratio (Henseler et al., 2015)."""
    R = df.corr().abs()
    names = list(blocks)
    M = pd.DataFrame(np.nan, index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ia, ib = blocks[a], blocks[b]
            hetero = R.loc[ia, ib].values.mean()
            mono_a = R.loc[ia, ia].values[np.triu_indices(len(ia), 1)].mean()
            mono_b = R.loc[ib, ib].values[np.triu_indices(len(ib), 1)].mean()
            M.loc[b, a] = hetero / np.sqrt(mono_a * mono_b)
    return M


class CFA:
    """Correlated-factors CFA, ML estimation on the item correlation matrix.

    Factor variances fixed to 1 (standardised latent variables), each item loads
    on exactly one factor, residuals uncorrelated.
    """

    def __init__(self, blocks):
        self.blocks = blocks
        self.items = [it for b in blocks.values() for it in b]
        self.fnames = list(blocks)
        p, m = len(self.items), len(self.fnames)
        self.p, self.m = p, m
        self.pattern = np.zeros((p, m), bool)
        r = 0
        for j, b in enumerate(blocks.values()):
            for _ in b:
                self.pattern[r, j] = True
                r += 1
        self.tril = np.tril_indices(m, -1)

    def _unpack(self, theta):
        p, m = self.p, self.m
        L = np.zeros((p, m))
        L[self.pattern] = theta[:p]
        Phi = np.eye(m)
        Phi[self.tril] = np.tanh(theta[p:p + len(self.tril[0])])
        Phi = Phi + np.tril(Phi, -1).T
        psi = np.exp(theta[p + len(self.tril[0]):])
        return L, Phi, psi

    def _sigma(self, theta):
        L, Phi, psi = self._unpack(theta)
        return L @ Phi @ L.T + np.diag(psi)

    def _fml(self, theta):
        S = self.S
        Sig = self._sigma(theta)
        sign, logdet = np.linalg.slogdet(Sig)
        if sign <= 0:
            return 1e10
        return logdet + np.trace(S @ np.linalg.inv(Sig)) - self.logdetS - self.p

    def fit(self, df):
        X = df[self.items].to_numpy(float)
        self.n = X.shape[0]
        self.S = np.corrcoef(X, rowvar=False)
        self.logdetS = np.linalg.slogdet(self.S)[1]
        p, m = self.p, self.m
        theta0 = np.concatenate([np.full(p, 0.7), np.full(len(self.tril[0]), 0.3),
                                 np.log(np.full(p, 0.5))])
        res = optimize.minimize(self._fml, theta0, method="L-BFGS-B",
                                options={"maxiter": 5000, "maxfun": 200000})
        self.res = res
        self.L, self.Phi, self.psi = self._unpack(res.x)
        # sign convention: positive loadings
        sgn = np.sign(self.L.sum(axis=0))
        sgn[sgn == 0] = 1
        self.L = self.L * sgn
        self.Phi = self.Phi * np.outer(sgn, sgn)
        return self

    def fit_indices(self):
        p, n = self.p, self.n
        q = p + len(self.tril[0]) + p
        df_m = p * (p + 1) / 2 - q
        chi2 = (n - 1) * self.res.fun
        chi2_0 = -(n - 1) * self.logdetS  # independence model on a correlation matrix
        df_0 = p * (p - 1) / 2
        rmsea = np.sqrt(max(chi2 - df_m, 0) / (df_m * (n - 1)))
        cfi = 1 - max(chi2 - df_m, 0) / max(chi2_0 - df_0, chi2 - df_m, 1e-12)
        tli = ((chi2_0 / df_0) - (chi2 / df_m)) / ((chi2_0 / df_0) - 1)
        Sig = self._sigma(self.res.x)
        resid = (self.S - Sig)[np.tril_indices(p)]
        srmr = np.sqrt(np.mean(resid ** 2))
        return {"n": n, "chi2": chi2, "df": int(df_m), "CFI": cfi, "TLI": tli,
                "RMSEA": rmsea, "SRMR": srmr, "converged": bool(self.res.success)}

    def construct_table(self):
        rows = []
        r = 0
        for j, (f, its) in enumerate(self.blocks.items()):
            lam = self.L[r:r + len(its), j]
            th = self.psi[r:r + len(its)]
            r += len(its)
            cr = lam.sum() ** 2 / (lam.sum() ** 2 + th.sum())
            ave = np.mean(lam ** 2)
            rows.append({"factor": f, "loading_min": lam.min(), "loading_max": lam.max(),
                         "CR": cr, "AVE": ave, "sqrtAVE": np.sqrt(ave)})
        return pd.DataFrame(rows).set_index("factor")

    def loadings(self):
        rows = []
        r = 0
        for j, (f, its) in enumerate(self.blocks.items()):
            for it in its:
                rows.append({"factor": f, "item": it, "loading": self.L[r, j]})
                r += 1
        return pd.DataFrame(rows)


def omega_from_loadings(lam, theta):
    lam, theta = np.asarray(lam), np.asarray(theta)
    return lam.sum() ** 2 / (lam.sum() ** 2 + theta.sum())
