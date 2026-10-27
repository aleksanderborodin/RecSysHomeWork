from __future__ import annotations

import itertools
import json
import time

import numpy as np
import polars as pl

import config
from core import metrics
from core.evaluate import evaluate, summary_row


def fingerprint(*parts) -> str:

    import hashlib
    payload = json.dumps([parts, str(config.VAL_START), str(config.TEST_START), str(config.TEST_END),
                          config.KCORE_USER, config.KCORE_ITEM, config.K, config.POSITIVE_RATING],
                         sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def grid(space: dict) -> list[dict]:
    keys = list(space)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(space[k] for k in keys))]


def _row(params, res, fit_s):
    return {**params, **{k: v for k, v in summary_row("", res).items() if k != "model"}, "fit_s": round(fit_s, 1)}


def run_grid(name: str, make, space, stage, force=False, checkpoints=None, log=True) -> pl.DataFrame:

    path = config.TUNING / f"{name}_{fingerprint(space, checkpoints, stage.stats)}.parquet"
    if path.exists() and not force:
        return pl.read_parquet(path)
    config.TUNING.mkdir(parents=True, exist_ok=True)
    rows = []
    for params in grid(space):
        t0 = time.time()
        if checkpoints:
            model = make(**params, epochs=max(checkpoints))

            def cb(epoch, m, params=params, t0=t0):
                if epoch in checkpoints:
                    rows.append(_row({**params, "epochs": epoch}, evaluate(m, stage), time.time() - t0))
                    if log:
                        print(name, {**params, "epochs": epoch}, f"{rows[-1][config.PRIMARY_METRIC]:.4f}", flush=True)

            model.fit(stage.history, callback=cb)
            continue
        model = make(**params).fit(stage.history)
        fit_s = time.time() - t0
        rows.append(_row(params, evaluate(model, stage), fit_s))
        if log:
            print(name, params, f"{rows[-1][config.PRIMARY_METRIC]:.4f}", flush=True)
    df = pl.DataFrame(rows)
    df.write_parquet(path)
    return df


def best(df: pl.DataFrame, params: list[str], metric: str = config.PRIMARY_METRIC) -> dict:
    row = df.sort(metric, descending=True).row(0, named=True)
    return {k: row[k] for k in params}


def final_test(models: dict, stage, force=False) -> tuple[pl.DataFrame, dict]:

    path = config.TEST / "metrics.csv"
    per_path = config.TEST / "per_user.parquet"
    recs_path = config.TEST / "recs.parquet"
    stamp = config.TEST / "fingerprint.txt"
    fp = fingerprint({n: p for n, (_, p) in models.items()}, stage.stats)
    if path.exists() and per_path.exists() and stamp.exists() and stamp.read_text() == fp and not force:
        return pl.read_csv(path), _load_per_user(per_path)
    config.TEST.mkdir(parents=True, exist_ok=True)
    rows, per_user, recs = [], {}, {}
    for name, (make, params) in models.items():
        t0 = time.time()
        model = make(**params).fit(stage.history)
        res = evaluate(model, stage, return_recs=True)
        lo, hi = metrics.bootstrap_ci(res["per_user"]["ndcg"])
        rows.append({"model": name, **{k: v for k, v in summary_row(name, res).items() if k != "model"},
                     "ndcg_lo": lo, "ndcg_hi": hi, "fit_s": round(time.time() - t0, 1),
                     "params": json.dumps(params, ensure_ascii=False, default=str)})
        per_user[name] = res["per_user"]
        recs[name] = res["recs"]
        print(name, f"{res[config.PRIMARY_METRIC]:.4f}", flush=True)
    table = pl.DataFrame(rows)
    table.write_csv(path)
    pl.DataFrame({"user_idx": stage.users, **{f"{n}|{m}": v for n, d in per_user.items() for m, v in d.items()}}) \
        .write_parquet(per_path)
    pl.DataFrame({"user_idx": stage.users, **{n: r.tolist() for n, r in recs.items()}}).write_parquet(recs_path)
    stamp.write_text(fp)
    return table, per_user


def _load_per_user(path) -> dict:
    df = pl.read_parquet(path)
    out: dict = {}
    for col in df.columns[1:]:
        name, m = col.split("|")
        out.setdefault(name, {})[m] = df[col].to_numpy()
    return out


def load_recs() -> pl.DataFrame:
    return pl.read_parquet(config.TEST / "recs.parquet")
