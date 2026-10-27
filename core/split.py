from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import polars as pl
import scipy.sparse as sp

import config


@dataclass
class History:

    cutoff: datetime
    pairs: pl.DataFrame
    n_users: int
    n_items: int
    items: pl.DataFrame
    _cache: dict = field(default_factory=dict, repr=False)

    def matrix(self, kind: str = "all") -> sp.csr_matrix:

        if kind not in self._cache:
            p = self.pairs
            if kind == "positive":
                p = p.filter(pl.col("rating") >= config.POSITIVE_RATING)
            elif kind == "rated":
                p = p.filter(pl.col("rating").is_not_null())
            values = (p["rating"].to_numpy().astype(np.float32) if kind == "rated"
                      else np.ones(p.height, dtype=np.float32))
            m = sp.csr_matrix((values, (p["user_idx"].to_numpy(), p["item_idx"].to_numpy())),
                              shape=(self.n_users, self.n_items))
            m.sort_indices()
            self._cache[kind] = m
        return self._cache[kind]

    @property
    def seen(self) -> sp.csr_matrix:
        return self.matrix("all")

    @property
    def catalog(self) -> np.ndarray:

        return np.asarray(self.seen.sum(axis=0)).ravel() > 0

    @property
    def popularity(self) -> np.ndarray:
        return np.asarray(self.seen.sum(axis=0)).ravel()


@dataclass
class Stage:

    name: str
    history: History
    end: datetime
    targets: sp.csr_matrix
    users: np.ndarray
    stats: dict


def rating_at(t: datetime) -> tuple[pl.Expr, pl.Expr]:

    last = pl.col("t_rating").is_not_null() & (pl.col("t_rating") < t)
    first = pl.col("t_rating_first").is_not_null() & (pl.col("t_rating_first") < t)
    rating = pl.when(last).then(pl.col("rating")).when(first).then(pl.col("rating_first"))
    when = pl.when(last).then(pl.col("t_rating")).when(first).then(pl.col("t_rating_first"))
    return rating, when


def history_at(events: pl.DataFrame, items: pl.DataFrame, cutoff: datetime, n_users: int, n_items: int) -> History:
    before = lambda c: pl.col(c).is_not_null() & (pl.col(c) < cutoff)
    rating, t_rating = rating_at(cutoff)
    pairs = (events.filter(pl.col("t_shelf") < cutoff)
             .with_columns(
                 rating.alias("rating"),
                 t_rating.alias("t_rating"),
                 before("t_read").alias("read"),
                 pl.when(before("t_read")).then(pl.col("t_read")).alias("t_read"),
                 before("t_review").alias("review"),
                 pl.when(before("t_review")).then(pl.col("t_review")).alias("t_review"))
             .select("user_idx", "item_idx", "t_shelf", "rating", "t_rating", "read", "t_read", "review", "t_review"))
    return History(cutoff, pairs, n_users, n_items, items)


def targets_in(events: pl.DataFrame, history: History, start: datetime, end: datetime):

    window = events.filter(pl.col("t_shelf").is_between(start, end, closed="left"))
    rating_end, _ = rating_at(end)
    relevant = window.filter(rating_end >= config.POSITIVE_RATING)
    catalog = history.catalog
    seen_users = np.asarray(history.seen.sum(axis=1)).ravel() > 0
    in_catalog = relevant.filter(pl.col("item_idx").is_in(np.flatnonzero(catalog).tolist()))
    warm = in_catalog.filter(pl.col("user_idx").is_in(np.flatnonzero(seen_users).tolist()))
    t = sp.csr_matrix((np.ones(warm.height, dtype=np.float32),
                       (warm["user_idx"].to_numpy(), warm["item_idx"].to_numpy())),
                      shape=(history.n_users, history.n_items))
    users = np.flatnonzero(np.diff(t.indptr) > 0)
    stats = {
        "history_pairs": history.pairs.height,
        "history_users": int(seen_users.sum()),
        "catalog_items": int(catalog.sum()),
        "window_pairs": window.height,
        "relevant_pairs": relevant.height,
        "relevant_out_of_catalog": relevant.height - in_catalog.height,
        "relevant_cold_user": in_catalog.height - warm.height,
        "eval_users": int(len(users)),
        "eval_targets": int(t.nnz),
    }
    return t, users, stats


def make_stage(name: str, events: pl.DataFrame, items: pl.DataFrame, cutoff: datetime, end: datetime) -> Stage:
    n_users = int(events["user_idx"].max()) + 1
    n_items = int(events["item_idx"].max()) + 1
    hist = history_at(events, items, cutoff, n_users, n_items)
    t, users, stats = targets_in(events, hist, cutoff, end)
    return Stage(name, hist, end, t, users, stats)


def make_stages(events: pl.DataFrame, items: pl.DataFrame) -> dict[str, Stage]:

    return {
        "val": make_stage("val", events, items, config.VAL_START, config.TEST_START),
        "test": make_stage("test", events, items, config.TEST_START, config.TEST_END),
    }
