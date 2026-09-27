"""Compute AgriHGT evaluation metrics from saved predictions and write LaTeX table rows.

Every number in the paper's result tables is produced here from real model
outputs; nothing is typed by hand. Run it after saving one prediction file per
experiment (see "Saving predictions" below), then re-compile the paper: the
generated files in ../tables/ are \\input by main.tex.

Saving predictions (add to the evaluation cell of each notebook)
----------------------------------------------------------------
    import numpy as np
    np.savez(
        f"predictions/{run_id}.npz",
        y_true=test_labels,          # (N,) int class indices 0..27
        y_pred=test_pred,            # (N,) int argmax predictions
        probs=test_probs,            # (N, 28) softmax probabilities (optional)
        val_acc=best_val_acc,        # float, validation accuracy of the kept checkpoint (optional)
        class_names=np.array(CLASS_NAMES),  # (28,) str (optional, used for per-class table)
    )

Run ids expected by default (rename freely and edit BACKBONES/ABLATIONS below):
    {backbone}_LR, {backbone}_L0, {backbone}_L1, {backbone}_L2   for each backbone
    A0 ... A8                                                    for the DINOv3 ablation
(A1 may simply be a copy of DINOv3_L0, as in the paper.)

Usage
-----
    python agrihgt_metrics.py --pred-dir predictions --out-dir ../tables --best DINOv3_L2
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
    top_k_accuracy_score,
)

BACKBONES = [
    ("ResNet50", "ResNet50"),
    ("DenseNet121", "DenseNet121"),
    ("EfficientNetB3", "EfficientNet-B3"),
    ("ConvNeXtTiny", "ConvNeXt-Tiny"),
    ("SwinTiny", "Swin-Tiny"),
    ("DINOv2", "DINOv2 ViT-B/14"),
    ("DINOv3", "DINOv3 ViT-B/16"),
]
VARIANTS = [("LR", "LogReg"), ("L0", "HGT-0"), ("L1", "HGT-1"), ("L2", "HGT-2")]
ABLATIONS = [
    ("A0", "Full AgriHGT"),
    ("A1", "No HGT message passing"),
    ("A2", "No leaf similarity"),
    ("A3", "No crop knowledge"),
    ("A4", "No disease taxonomy"),
    ("A5", "No visual grounding"),
    ("A6", "No meta-node identity"),
    ("A7", "No residual bypass"),
    ("A8", "No fusion gate"),
]
N_BOOT = 2000
SEED = 42


def load(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    out["y_true"] = out["y_true"].astype(int)
    out["y_pred"] = out["y_pred"].astype(int)
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def bootstrap_macro_f1(y, p, n_classes, rng) -> tuple[float, float]:
    n = len(y)
    vals = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        vals[b] = f1_score(y[idx], p[idx], labels=range(n_classes), average="macro", zero_division=0)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def ece(probs, y, n_bins: int = 15) -> float:
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def mcnemar(y, pa, pb) -> float:
    """Exact (binomial) McNemar p-value for two classifiers on the same test images."""
    a_right, b_right = pa == y, pb == y
    n01 = int(np.sum(a_right & ~b_right))
    n10 = int(np.sum(~a_right & b_right))
    if n01 + n10 == 0:
        return 1.0
    return float(stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue)


def metrics(r: dict, n_classes: int, rng) -> dict:
    y, p = r["y_true"], r["y_pred"]
    n = len(y)
    k = int((y == p).sum())
    lo, hi = wilson(k, n)
    P, R, F, _ = precision_recall_fscore_support(y, p, labels=range(n_classes), average="macro", zero_division=0)
    m = {
        "n": n,
        "acc": k / n,
        "acc_lo": lo,
        "acc_hi": hi,
        "bal_acc": balanced_accuracy_score(y, p),
        "macro_p": P,
        "macro_r": R,
        "macro_f1": F,
        "weighted_f1": f1_score(y, p, labels=range(n_classes), average="weighted", zero_division=0),
        "kappa": cohen_kappa_score(y, p),
        "mcc": matthews_corrcoef(y, p),
        "val_acc": float(r["val_acc"]) if "val_acc" in r else float("nan"),
    }
    m["f1_lo"], m["f1_hi"] = bootstrap_macro_f1(y, p, n_classes, rng)
    if "probs" in r:
        pr = np.asarray(r["probs"], dtype=float)
        m["top3"] = top_k_accuracy_score(y, pr, k=3, labels=range(n_classes))
        m["auc"] = roc_auc_score(y, pr, multi_class="ovr", average="macro", labels=range(n_classes))
        m["nll"] = log_loss(y, pr, labels=range(n_classes))
        m["ece"] = ece(pr, y)
    else:
        m["top3"] = m["auc"] = m["nll"] = m["ece"] = float("nan")
    return m


def f(x, d=4, pct=False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{100 * x:.2f}" if pct else f"{x:.{d}f}"


def backbone_rows(runs, n_classes, rng) -> str:
    lines = []
    for key, label in BACKBONES:
        present = [(v, vl) for v, vl in VARIANTS if f"{key}_{v}" in runs]
        if not present:
            continue
        best_val = max(
            (runs[f"{key}_{v}"].get("val_acc", -1) for v, _ in VARIANTS[1:] if f"{key}_{v}" in runs),
            default=None,
        )
        for i, (v, vl) in enumerate(present):
            m = metrics(runs[f"{key}_{v}"], n_classes, rng)
            name = label if i == 0 else ""
            mark = r"$^{\dagger}$" if v != "LR" and best_val is not None and runs[f"{key}_{v}"].get("val_acc") == best_val else ""
            lines.append(
                f"{name} & {vl}{mark} & {f(m['val_acc'], pct=True)} & {f(m['acc'], pct=True)} & "
                f"[{f(m['acc_lo'], pct=True)}, {f(m['acc_hi'], pct=True)}] & {f(m['bal_acc'], pct=True)} & "
                f"{f(m['macro_p'])} & {f(m['macro_r'])} & {f(m['macro_f1'])} & {f(m['weighted_f1'])} & "
                f"{f(m['kappa'])} & {f(m['mcc'])} & {f(m['top3'], pct=True)} & {f(m['auc'])} \\\\"
            )
        lines.append(r"\midrule")
    if lines and lines[-1] == r"\midrule":
        lines.pop()
    return "\n".join(lines) + "\n"


def ablation_rows(runs, n_classes, rng) -> str:
    if "A0" not in runs:
        return ""
    ref = runs["A0"]
    m0 = metrics(ref, n_classes, rng)
    lines = []
    for key, label in ABLATIONS:
        if key not in runs:
            continue
        m = metrics(runs[key], n_classes, rng)
        p = "--" if key == "A0" else f"{mcnemar(ref['y_true'], ref['y_pred'], runs[key]['y_pred']):.3f}"
        d_acc = "--" if key == "A0" else f"{100 * (m['acc'] - m0['acc']):+.2f}"
        d_f1 = "--" if key == "A0" else f"{m['macro_f1'] - m0['macro_f1']:+.4f}"
        lines.append(
            f"{key} & {label} & {f(m['val_acc'], pct=True)} & {f(m['acc'], pct=True)} & {d_acc} & "
            f"{f(m['macro_f1'])} & [{f(m['f1_lo'])}, {f(m['f1_hi'])}] & {d_f1} & {f(m['bal_acc'], pct=True)} & "
            f"{f(m['kappa'])} & {f(m['mcc'])} & {f(m['nll'])} & {f(m['ece'])} & {p} \\\\"
        )
    return "\n".join(lines) + "\n"


CROP_CODES = {"bottle gourd": "BG", "bottle_gourd": "BG", "papaya": "PA", "tomato": "TO", "zucchini": "ZU"}


def short_name(name: str, width: int = 27) -> str:
    """Compact class label for the two-column per-class table, e.g. 'BG: Downy Mildew'."""
    low = name.replace("___", " ").replace("__", " ").strip()
    for crop, code in CROP_CODES.items():
        if low.lower().startswith(crop):
            low = f"{code}: " + low[len(crop):].lstrip(" _-:")
            break
    low = low.replace("_", " ")
    return low if len(low) <= width else low[: width - 1] + "."


def perclass_rows(r, n_classes) -> str:
    """Per-class one-vs-rest metrics, laid out as two side-by-side halves of the class list."""
    y, p = r["y_true"], r["y_pred"]
    names = [str(s) for s in r["class_names"]] if "class_names" in r else [f"Class {c}" for c in range(n_classes)]
    cm = confusion_matrix(y, p, labels=range(n_classes))
    N = cm.sum()
    cells = []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = N - tp - fp - fn
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        jac = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        den = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        mcc = (tp * tn - fp * fn) / den if den else 0.0
        name = short_name(names[c]).replace("&", r"\&")
        cells.append(f"{name} & {f(prec, 3)} & {f(rec, 3)} & {f(spec, 3)} & {f(f1, 3)} & {f(jac, 3)} & {f(mcc, 3)} & {tp + fn}")
    half = (len(cells) + 1) // 2
    blank = " & ".join([""] * 8)
    lines = []
    for i in range(half):
        right = cells[half + i] if half + i < len(cells) else blank
        lines.append(f"{cells[i]} & {right} \\\\")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default="predictions")
    ap.add_argument("--out-dir", default="../tables")
    ap.add_argument("--best", default="DINOv3_L2", help="run id used for the per-class table")
    ap.add_argument("--n-classes", type=int, default=28)
    args = ap.parse_args()

    runs = {p.stem: load(p) for p in sorted(Path(args.pred_dir).glob("*.npz"))}
    if not runs:
        raise SystemExit(f"No .npz prediction files found in {args.pred_dir}")
    rng = np.random.default_rng(SEED)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "backbone_rows.tex").write_text(backbone_rows(runs, args.n_classes, rng))
    (out / "ablation_rows.tex").write_text(ablation_rows(runs, args.n_classes, rng))
    if args.best in runs:
        (out / "perclass_rows.tex").write_text(perclass_rows(runs[args.best], args.n_classes))
    print(f"Wrote tables for {len(runs)} runs to {out.resolve()}")


if __name__ == "__main__":
    main()
