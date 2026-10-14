"""Vectorised hierarchical Mamdani fuzzy inference system (HFIS) for the
Teacher Replacement Feasibility Score (TRFS).

Architecture (Fig. 2 of the manuscript)
    PC  = FIS_1(TE, P, AC, A)      pedagogical capability   (81 rules)
    IQ  = FIS_2(SE, C)             interaction quality      ( 9 rules)
    TW  = FIS_3(R, E)              trustworthiness          ( 9 rules)
    TRFS = FIS_4(PC, IQ, TW)       final feasibility score  (27 rules)

Every input factor lives on the Likert universe [1, 5]; intermediate and
output variables live on [0, 100]. Inference uses min (AND), max
(aggregation), min (implication) and centroid defuzzification. The whole
system is written with NumPy so it scores ~10^4 respondents in milliseconds.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

LEVELS = ("Low", "Medium", "High")


def trapmf(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    left = np.where(b > a, (x - a) / np.maximum(b - a, 1e-12), (x >= a).astype(float))
    right = np.where(d > c, (d - x) / np.maximum(d - c, 1e-12), (x <= d).astype(float))
    return np.clip(np.minimum(np.minimum(left, 1.0), right), 0.0, 1.0)


def trimf(x, a, b, c):
    return trapmf(x, a, b, b, c)


@dataclass
class Partition:
    """Three-term fuzzy partition (Low / Medium / High) of one universe."""
    lo: float
    hi: float
    params: dict = field(default_factory=dict)

    @classmethod
    def likert(cls, shift=0.0, spread=0.0):
        # Low = trap(1,1,2,3), Medium = tri(2,3,4), High = trap(3,4,5,5)
        p = {"Low": (1, 1, 2 + shift - spread, 3 + shift + spread),
             "Medium": (2 + shift - spread, 3 + shift, 4 + shift + spread),
             "High": (3 + shift - spread, 4 + shift + spread, 5, 5)}
        return cls(1.0, 5.0, p)

    @classmethod
    def score(cls, shift=0.0, spread=0.0):
        # Low = trap(0,0,20,50), Medium = tri(20,50,80), High = trap(50,80,100,100)
        p = {"Low": (0, 0, 20 + shift - spread, 50 + shift + spread),
             "Medium": (20 + shift - spread, 50 + shift, 80 + shift + spread),
             "High": (50 + shift - spread, 80 + shift + spread, 100, 100)}
        return cls(0.0, 100.0, p)

    def memberships(self, x):
        """Return array (n, 3) of membership degrees for Low/Medium/High."""
        x = np.clip(np.asarray(x, dtype=float), self.lo, self.hi)
        out = []
        for lvl in LEVELS:
            p = self.params[lvl]
            out.append(trimf(x, *p) if len(p) == 3 else trapmf(x, *p))
        return np.stack(out, axis=-1)


def _round_level(v, mode="half_down"):
    """Map an average level (0=Low .. 2=High) to a consequent index."""
    if mode == "half_up":
        return int(np.floor(v + 0.5))
    return int(np.ceil(v - 0.5))  # conservative: ties go to the lower class


def mean_rule_base(n_inputs, weights=None, rounding="half_down"):
    """Rules whose consequent is the (weighted) mean antecedent level."""
    w = np.ones(n_inputs) if weights is None else np.asarray(weights, float)
    rules = []
    for combo in itertools.product(range(3), repeat=n_inputs):
        rules.append((combo, _round_level(np.dot(w, combo) / w.sum(), rounding)))
    return rules


def trust_rule_base():
    """FIS_3(R, E). Low reliability vetoes trust (Rule 3 of the proposal)."""
    table = {  # (R, E) -> TW
        (0, 0): 0, (0, 1): 0, (0, 2): 0,
        (1, 0): 0, (1, 1): 1, (1, 2): 1,
        (2, 0): 1, (2, 1): 2, (2, 2): 2,
    }
    return [(k, v) for k, v in table.items()]


def final_rule_base(rounding="half_down"):
    """FIS_4(PC, IQ, TW): low trust vetoes; otherwise weighted mean (2:1:1)."""
    rules = []
    for pc, iq, tw in itertools.product(range(3), repeat=3):
        if tw == 0:
            out = 0
        else:
            out = _round_level((2 * pc + iq + tw) / 4, rounding)
        rules.append(((pc, iq, tw), out))
    return rules


def mamdani(inputs, partitions, rules, out_part, t_norm="min", grid=201):
    """Vectorised Mamdani inference.

    inputs      list of arrays, each shape (n,)
    partitions  list of Partition, one per input
    rules       list of (antecedent level tuple, consequent level)
    returns     crisp outputs (n,) and firing strengths per consequent (n, 3)
    """
    mus = [p.memberships(x) for p, x in zip(partitions, inputs)]  # each (n,3)
    n = mus[0].shape[0]
    strength = np.zeros((n, 3))
    for ante, cons in rules:
        if t_norm == "prod":
            fire = np.ones(n)
            for m, lvl in zip(mus, ante):
                fire = fire * m[:, lvl]
        else:
            fire = np.min(np.stack([m[:, lvl] for m, lvl in zip(mus, ante)]), axis=0)
        strength[:, cons] = np.maximum(strength[:, cons], fire)
    y = np.linspace(out_part.lo, out_part.hi, grid)
    out_mu = out_part.memberships(y)  # (grid, 3)
    agg = np.max(np.minimum(strength[:, None, :], out_mu[None, :, :]), axis=2)  # (n, grid)
    denom = agg.sum(axis=1)
    crisp = np.where(denom > 0, (agg * y).sum(axis=1) / np.maximum(denom, 1e-12), 50.0)
    return crisp, strength


class HierarchicalFIS:
    def __init__(self, in_shift=0.0, in_spread=0.0, out_shift=0.0, out_spread=0.0,
                 rounding="half_down", t_norm="min"):
        self.in_part = Partition.likert(in_shift, in_spread)
        self.mid_part = Partition.score(out_shift, out_spread)
        self.out_part = Partition.score(out_shift, out_spread)
        self.t_norm = t_norm
        self.rules = {
            "PC": mean_rule_base(4, rounding=rounding),
            "IQ": mean_rule_base(2, rounding=rounding),
            "TW": trust_rule_base(),
            "TRFS": final_rule_base(rounding=rounding),
        }

    def predict(self, F):
        """F: dict-like with arrays TE, SE, P, AC, C, A, R, E (Likert 1..5)."""
        ip, mp = self.in_part, self.mid_part
        pc, _ = mamdani([F["TE"], F["P"], F["AC"], F["A"]], [ip] * 4,
                        self.rules["PC"], mp, self.t_norm)
        iq, _ = mamdani([F["SE"], F["C"]], [ip] * 2, self.rules["IQ"], mp, self.t_norm)
        tw, _ = mamdani([F["R"], F["E"]], [ip] * 2, self.rules["TW"], mp, self.t_norm)
        trfs, strength = mamdani([pc, iq, tw], [mp] * 3, self.rules["TRFS"],
                                 self.out_part, self.t_norm)
        return {"PC": pc, "IQ": iq, "TW": tw, "TRFS": trfs, "strength": strength}


class ActivityFIS:
    """Three-input FIS used to score one teaching activity:
    feasibility_k = f(Capability_k, Reliability, Ethical suitability).
    Rules follow the proposal: capability and reliability both high -> high;
    low capability or low reliability -> low; ethics moderates the result."""

    def __init__(self, in_shift=0.0, in_spread=0.0, out_shift=0.0, out_spread=0.0,
                 rounding="half_down", t_norm="min"):
        self.in_part = Partition.likert(in_shift, in_spread)
        self.out_part = Partition.score(out_shift, out_spread)
        self.t_norm = t_norm
        rules = []
        for cap, rel, eth in itertools.product(range(3), repeat=3):
            if cap == 0 or rel == 0:
                out = 0
            else:
                out = _round_level((2 * cap + rel + eth) / 4, rounding)
            rules.append(((cap, rel, eth), out))
        self.rules = rules

    def predict(self, cap, rel, eth):
        crisp, strength = mamdani([cap, rel, eth], [self.in_part] * 3, self.rules,
                                  self.out_part, self.t_norm)
        return crisp, strength


def rule_table(rules, names, out_name):
    rows = []
    for i, (ante, cons) in enumerate(rules, 1):
        row = {"Rule": i}
        row.update({n: LEVELS[a] for n, a in zip(names, ante)})
        row[out_name] = LEVELS[cons]
        rows.append(row)
    return rows
