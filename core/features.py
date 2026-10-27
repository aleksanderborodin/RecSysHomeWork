from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import polars as pl
import scipy.sparse as sp

from core.split import History

ITEM_GROUPS = ("item_id", "author", "year", "pages", "format", "language", "publisher", "tfidf", "popularity")
USER_GROUPS = ("user_id", "activity", "mean_rating", "toread_share", "author_profile")
FEATURE_SETS = {
    "ids": ("user_id", "item_id"),
    "ids+item": ("user_id",) + ITEM_GROUPS,
    "ids+item+user": USER_GROUPS + ITEM_GROUPS,
}
STOP = set("the and for with this that from his her are was were has have had its into who what which their "
           "they them she you your our not but all one two can will been more most when where book books poems "
           "poem poetry collection new first edition also over about after than these those such other each".split())


@dataclass
class FeatureSpace:
    user: sp.csr_matrix
    item: sp.csr_matrix
    n_features: int
    groups: dict = field(default_factory=dict)
    vocab: dict = field(default_factory=dict)


def _bucket(values: np.ndarray, edges, labels) -> np.ndarray:
    out = np.full(len(values), len(labels), dtype=np.int64)
    ok = ~np.isnan(values)
    out[ok] = np.digitize(values[ok], edges)
    return out


def _onehot(codes: np.ndarray, n_values: int) -> sp.csr_matrix:
    return sp.csr_matrix((np.ones(len(codes)), (np.arange(len(codes)), codes)), shape=(len(codes), n_values))


def _row_normalize(m: sp.csr_matrix) -> sp.csr_matrix:

    m = sp.csr_matrix(m, dtype=np.float64)
    s = np.asarray(abs(m).sum(axis=1)).ravel()
    s[s == 0] = 1
    return (sp.diags(1 / s) @ m).tocsr()


def _top_codes(values: list, top: int, min_count: int = 1):
    counts = pl.Series(values).value_counts().sort("count", descending=True)
    counts = counts.filter(pl.col("count") >= min_count).head(top)
    vocab = [v for v in counts[counts.columns[0]].to_list() if v is not None]
    index = {v: i for i, v in enumerate(vocab)}
    other = len(vocab)
    return np.array([index.get(v, other) for v in values]), vocab + ["<другое>"]


def tfidf(texts: list[str], catalog: np.ndarray, max_features=3000, min_df=5, max_df=0.3):

    tokens = [[t for t in re.findall(r"[^\W\d_]{3,}", (s or "").lower()) if t not in STOP] for s in texts]
    df_count: dict[str, int] = {}
    for i in np.flatnonzero(catalog):
        for t in set(tokens[i]):
            df_count[t] = df_count.get(t, 0) + 1
    n_docs = int(catalog.sum())
    cand = [(t, c) for t, c in df_count.items() if min_df <= c <= max_df * n_docs]
    vocab = [t for t, _ in sorted(cand, key=lambda x: -x[1])[:max_features]]
    index = {t: j for j, t in enumerate(vocab)}
    idf = np.log((1 + n_docs) / (1 + np.array([df_count[t] for t in vocab]))) + 1
    rows, cols, vals = [], [], []
    for i, toks in enumerate(tokens):
        tf: dict[int, int] = {}
        for t in toks:
            j = index.get(t)
            if j is not None:
                tf[j] = tf.get(j, 0) + 1
        for j, n in tf.items():
            rows.append(i); cols.append(j); vals.append(n * idf[j])
    m = sp.csr_matrix((vals, (rows, cols)), shape=(len(texts), len(vocab)))
    norm = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norm[norm == 0] = 1
    return sp.diags(1 / norm) @ m, vocab


def build(history: History, groups=FEATURE_SETS["ids+item+user"], top_authors=3000, top_publishers=200,
          tfidf_features=3000) -> FeatureSpace:
    items = history.items.sort("item_idx")
    n_u, n_i = history.n_users, history.n_items
    catalog = history.catalog
    pairs = history.pairs
    blocks_u, blocks_i, spans, vocab = [], [], {}, {}
    offset = 0

    def add(name, user_block=None, item_block=None, names=None):
        nonlocal offset
        width = (user_block if user_block is not None else item_block).shape[1]
        blocks_u.append(_row_normalize(user_block) if user_block is not None else sp.csr_matrix((n_u, width)))
        blocks_i.append(_row_normalize(item_block) if item_block is not None else sp.csr_matrix((n_i, width)))
        spans[name] = (offset, offset + width)
        vocab[name] = names
        offset += width

    first_author = [a[0] if len(a) else None for a in items["author_ids"].to_list()]
    author_codes, author_vocab = _top_codes([first_author[i] if catalog[i] else None for i in range(n_i)],
                                            top_authors, min_count=2)


    for g in groups:
        if g == "user_id":
            add(g, user_block=sp.identity(n_u, format="csr"))
        elif g == "item_id":
            add(g, item_block=sp.identity(n_i, format="csr"))
        elif g == "author":
            add(g, item_block=_onehot(author_codes, len(author_vocab)), names=author_vocab)
        elif g == "year":
            labels = ["<1900", "1900-49", "1950-79", "1980-99", "2000-09", "2010-14", "2015+", "нет"]
            codes = _bucket(items["publication_year"].to_numpy().astype(float), [1900, 1950, 1980, 2000, 2010, 2015], labels[:-1])
            add(g, item_block=_onehot(codes, len(labels)), names=labels)
        elif g == "pages":
            labels = ["<50", "50-99", "100-149", "150-249", "250-499", "500+", "нет"]
            codes = _bucket(items["num_pages"].to_numpy().astype(float), [50, 100, 150, 250, 500], labels[:-1])
            add(g, item_block=_onehot(codes, len(labels)), names=labels)
        elif g in ("format", "language", "publisher"):
            col = {"format": "format", "language": "language_code", "publisher": "publisher"}[g]
            top = {"format": 8, "language": 10, "publisher": top_publishers}[g]
            codes, names = _top_codes(items[col].to_list(), top, min_count=3)
            add(g, item_block=_onehot(codes, len(names)), names=names)
        elif g == "tfidf":
            m, names = tfidf(items["description"].to_list(), catalog, max_features=tfidf_features)
            add(g, item_block=m.tocsr(), names=names)
        elif g == "popularity":
            labels = [f"2^{k}" for k in range(16)]
            codes = np.minimum(np.floor(np.log2(history.popularity + 1)).astype(int), 15)
            add(g, item_block=_onehot(codes, 16), names=labels)
        elif g == "activity":
            n = np.asarray(history.seen.sum(axis=1)).ravel()
            codes = np.minimum(np.floor(np.log2(n + 1)).astype(int), 15)
            add(g, user_block=_onehot(codes, 16), names=[f"2^{k}" for k in range(16)])
        elif g == "mean_rating":
            labels = ["<3.5", "3.5-4", "4-4.5", "4.5+", "нет оценок"]
            means = pairs.filter(pl.col("rating").is_not_null()).group_by("user_idx").agg(pl.col("rating").mean())
            v = np.full(n_u, np.nan)
            v[means["user_idx"].to_numpy()] = means["rating"].to_numpy()
            add(g, user_block=_onehot(_bucket(v, [3.5, 4.0, 4.5], labels[:-1]), len(labels)), names=labels)
        elif g == "toread_share":
            labels = ["0-10%", "10-30%", "30-60%", "60-90%", "90%+"]
            only_shelf = pairs.with_columns((pl.col("rating").is_null() & ~pl.col("read")).alias("s"))
            sh = only_shelf.group_by("user_idx").agg(pl.col("s").mean())
            v = np.zeros(n_u)
            v[sh["user_idx"].to_numpy()] = sh["s"].to_numpy()
            add(g, user_block=_onehot(np.digitize(v, [0.1, 0.3, 0.6, 0.9]), len(labels)), names=labels)
        elif g == "author_profile":

            seen = history.seen.tocoo()
            codes = author_codes[seen.col]
            keep = codes != len(author_vocab) - 1
            m = sp.csr_matrix((np.ones(keep.sum()), (seen.row[keep], codes[keep])), shape=(n_u, len(author_vocab)))
            m.data[:] = 1.0
            add(g, user_block=m, names=author_vocab)
        else:
            raise ValueError(g)

    user = sp.hstack(blocks_u, format="csr").astype(np.float32)
    item = sp.hstack(blocks_i, format="csr").astype(np.float32)
    return FeatureSpace(user, item, offset, spans, vocab)
