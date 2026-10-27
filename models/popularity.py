from datetime import timedelta

import numpy as np
import polars as pl

from models.base import Recommender


class Popularity(Recommender):

    name = "Popularity"

    def __init__(self, window_days: int | None = None):
        self.window_days = window_days

    def fit(self, history):
        if self.window_days is None:
            self.pop = history.popularity.astype(np.float32)
            return self
        recent = history.pairs.filter(pl.col("t_shelf") >= history.cutoff - timedelta(days=self.window_days))
        cnt = np.bincount(recent["item_idx"].to_numpy(), minlength=history.n_items).astype(np.float32)

        self.pop = cnt + history.popularity.astype(np.float32) / (history.popularity.max() + 1.0)
        return self

    def score(self, users):
        return np.broadcast_to(self.pop, (len(users), len(self.pop)))


class Random(Recommender):
    name = "Random"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, history):
        self.n_items = history.n_items
        return self

    def score(self, users):
        rng = np.random.default_rng(self.seed)
        return rng.random((len(users), self.n_items), dtype=np.float32)
