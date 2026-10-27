from __future__ import annotations

import numpy as np


def per_user(top: np.ndarray, targets: list[np.ndarray], k: int) -> dict[str, np.ndarray]:

    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    idcg_prefix = np.cumsum(discounts)
    n = len(targets)
    out = {m: np.zeros(n) for m in ("ndcg", "recall", "precision", "hit")}
    for u, (rec, rel) in enumerate(zip(top, targets)):
        hits = np.isin(rec[:k], rel)
        n_hits = hits.sum()
        out["ndcg"][u] = (hits * discounts[: len(hits)]).sum() / idcg_prefix[min(k, len(rel)) - 1]
        out["recall"][u] = n_hits / len(rel)
        out["precision"][u] = n_hits / k
        out["hit"][u] = float(n_hits > 0)
    return out


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 42) -> dict:

    rng = np.random.default_rng(seed)
    diff = a - b
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"delta": float(diff.mean()), "lo": float(lo), "hi": float(hi)}


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    lo, hi = np.percentile(x[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)
