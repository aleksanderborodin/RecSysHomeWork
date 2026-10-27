from __future__ import annotations

import json

import numpy as np
import polars as pl

import config
from core import metrics, tuning
from core.evaluate import evaluate
from models.ease import EASE
from models.fm import FM
from models.ials import IALS
from models.ials_custom import CustomTargetIALS
from models.item_knn import ItemKNN
from models.popularity import Popularity, Random

NAMES = {"random": "Random", "pop": "Popularity", "pop_recent": "Recent popularity", "knn": "ItemKNN",
         "ials": "iALS", "ease": "EASE", "fm": "FM", "custom": "iALS + custom target"}


def tune_popularity(val, force=False):
    df = tuning.run_grid("popularity", Popularity, config.GRID_POP, val, force)
    recent = df.filter(pl.col("window_days").is_not_null())
    return df, tuning.best(recent, list(config.GRID_POP))


def tune_itemknn(val, force=False):
    df = tuning.run_grid("item_knn", ItemKNN, config.GRID_ITEMKNN, val, force)
    return df, tuning.best(df, list(config.GRID_ITEMKNN))


def tune_ease(val, force=False):
    df = tuning.run_grid("ease", EASE, config.GRID_EASE, val, force)
    return df, tuning.best(df, list(config.GRID_EASE))


def tune_ials(val, force=False):
    make = lambda **p: IALS(iters=config.IALS_ITERS, **p)
    df = tuning.run_grid("ials", make, config.GRID_IALS, val, force)
    return df, tuning.best(df, list(config.GRID_IALS))


def tune_signal(val, ials_best: dict, force=False):

    make = lambda **p: CustomTargetIALS(iters=config.IALS_ITERS, **ials_best, **p)
    df = tuning.run_grid("ials_custom", make, config.GRID_SIGNAL, val, force)
    return df, tuning.best(df, list(config.GRID_SIGNAL))


def tune_fm(val, force=False):
    df = tuning.run_grid("fm", FM, config.GRID_FM, val, force, checkpoints=config.FM_CHECKPOINTS)
    keys = list(config.GRID_FM) + ["epochs"]
    best = tuning.best(df, keys)
    space = {**{k: [best[k]] for k in config.GRID_FM}, "feature_set": config.FM_FEATURE_SETS}
    abl = tuning.run_grid("fm_features", FM, space, val, force, checkpoints=config.FM_CHECKPOINTS)
    return df, abl, tuning.best(abl, keys)


def final_models(best: dict) -> dict:

    return {
        NAMES["random"]: (Random, {}),
        NAMES["pop"]: (Popularity, {}),
        NAMES["pop_recent"]: (Popularity, best["pop_recent"]),
        NAMES["knn"]: (ItemKNN, best["knn"]),
        NAMES["ials"]: (lambda **p: IALS(iters=config.IALS_ITERS, **p), best["ials"]),
        NAMES["ease"]: (EASE, best["ease"]),
        NAMES["fm"]: (FM, best["fm"]),
        NAMES["custom"]: (lambda **p: CustomTargetIALS(iters=config.IALS_ITERS, **p),
                          {**best["ials"], **best["custom"]}),
    }


def run_test(test, best: dict, force=False):
    return tuning.final_test(final_models(best), test, force)


def signal_ablation(stage, ials_best: dict, signal_best: dict, per_user_full: np.ndarray | None = None,
                    force=False, name="signal_ablation") -> pl.DataFrame:

    path = config.TEST / f"{name}_{stage.name}_{tuning.fingerprint(ials_best, signal_best, stage.stats)}.parquet"
    if path.exists() and not force:
        return pl.read_parquet(path)
    make = lambda **p: CustomTargetIALS(iters=config.IALS_ITERS, **ials_best, **p).fit(stage.history)
    if per_user_full is None:
        per_user_full = evaluate(make(**signal_best), stage)["per_user"]["ndcg"]
    rows = []
    variants = {"все компоненты выкл. (= модель 2)": dict(config.SIGNAL_OFF)}
    for k, off in config.SIGNAL_OFF.items():
        if signal_best[k] != off:
            variants[f"без {k}"] = {**signal_best, k: off}
    if signal_best.get("half_life") is not None:
        variants["затухание в c, а не в p"] = {**signal_best, "recency": "c"}
    for label, params in variants.items():
        per = evaluate(make(**params), stage)["per_user"]["ndcg"]
        r = metrics.paired_bootstrap(per, per_user_full)
        rows.append({"вариант": label, "ndcg@10": float(per.mean()), "delta": r["delta"], "lo": r["lo"], "hi": r["hi"],
                     "params": json.dumps(params, default=str)})
    df = pl.DataFrame(rows)
    config.TEST.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return df


def save_best(best: dict) -> None:
    config.TEST.mkdir(parents=True, exist_ok=True)
    (config.TEST / "best_params.json").write_text(json.dumps(best, ensure_ascii=False, indent=2, default=str))


def load_best() -> dict:
    return json.loads((config.TEST / "best_params.json").read_text())
