from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

from core.split import History


@dataclass(frozen=True)
class SignalConfig:
    alpha: float = 10.0
    tau: float = 0.75
    offset: float = 0.5
    kappa: float = 5.0
    alpha_neg: float | None = 10.0
    w_shelf: float = 0.3
    alpha_rev: float = 0.0
    half_life: float | None = 365.0
    recency: str = "p"

    def as_dict(self):
        return asdict(self)


def user_means(pairs: pl.DataFrame, kappa: float) -> pl.DataFrame:
    rated = pairs.filter(pl.col("rating").is_not_null())
    mu = float(rated["rating"].mean())
    return rated.group_by("user_idx").agg(
        ((pl.col("rating").cast(pl.Float64).sum() + kappa * mu) / (pl.len() + kappa)).alias("m_u"))


def build(history: History, cfg: SignalConfig):

    pairs = history.pairs
    mu = float(pairs.filter(pl.col("rating").is_not_null())["rating"].mean())
    df = (pairs.join(user_means(pairs, cfg.kappa), on="user_idx", how="left")
          .with_columns(pl.col("m_u").fill_null(mu)))
    last = pl.max_horizontal("t_shelf", "t_rating", "t_read")
    df = df.with_columns(((pl.lit(history.cutoff) - last).dt.total_seconds() / 86400.0).alias("age_days"))

    rating = df["rating"].to_numpy().astype(float)
    rated = ~np.isnan(rating)
    read = df["read"].to_numpy()
    review = df["review"].to_numpy()
    m_u = df["m_u"].to_numpy()
    age = df["age_days"].to_numpy()

    p = np.ones(len(df))
    if cfg.tau is not None:
        p[rated] = 1.0 / (1.0 + np.exp(-(rating[rated] - m_u[rated] + cfg.offset) / cfg.tau))
    w = np.where(rated | read, 1.0, cfg.w_shelf)
    decay = np.ones(len(df)) if cfg.half_life is None else np.power(2.0, -age / cfg.half_life)
    if cfg.recency == "p":
        p = p * decay
        decay = np.ones(len(df))
    c = 1.0 + cfg.alpha * w * decay + cfg.alpha_rev * review
    if cfg.alpha_neg is not None:
        neg = rated & (rating <= 2)
        p[neg] = 0.0
        c[neg] = 1.0 + cfg.alpha_neg
    diag = df.with_columns(pl.Series("p", p), pl.Series("c", c),
                           pl.Series("kind", np.select([rated & (rating <= 2), rated, read],
                                                       ["негатив r≤2", "оценённые", "прочитано"], "только полка")))
    return (df["user_idx"].to_numpy(), df["item_idx"].to_numpy(),
            p.astype(np.float32), c.astype(np.float32), diag)
