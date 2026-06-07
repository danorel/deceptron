"""AUROC / pAUROC metrics + bootstrap CIs, shared by `deceptron.eval.main` and
`repro.eval.compare` so both sides of the synthetic-vs-real comparison are
scored with the identical math (see docs/prior.md for Apollo's protocol)."""

import numpy as np
from sklearn.metrics import roc_curve

N_BOOTSTRAP = 1000


def compute_pauroc(y_true, y_score, max_fpr: float) -> float:
    """Partial AUROC at FPR < max_fpr, normalized to [0, 1] (McClish 1989).

    Normalization divides by the area of the [0, max_fpr] x [0, 1] rectangle,
    so a perfect classifier scores 1.0 and a random one scores 0.5 — matching
    Apollo's convention (see docs/prior.md: perfect=1.0, random=0.1 at FPR<0.20,
    which is the same 0.5 random baseline rescaled to their reporting window).
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = fpr <= max_fpr
    if mask.sum() < 2:
        return float("nan")
    partial = np.trapz(tpr[mask], fpr[mask])
    return float(partial / max_fpr)


def bootstrap_ci(y_true, y_score, metric_fn, n_resamples: int = N_BOOTSTRAP, seed: int = 0):
    """95% CI for a metric via bootstrap resampling with replacement."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    values = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if len(set(yt.tolist())) < 2:
            continue
        v = metric_fn(yt, ys)
        if v == v:  # skip NaN
            values.append(v)
    if not values:
        return float("nan"), float("nan")
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)
