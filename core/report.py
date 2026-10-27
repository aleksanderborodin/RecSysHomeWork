from __future__ import annotations

import numpy as np
import polars as pl

import config
from core import metrics, plots

M = config.PRIMARY_METRIC


def _slice(df: pl.DataFrame, best: dict, free: list[str]) -> pl.DataFrame:

    cond = pl.lit(True)
    for k, v in best.items():
        if k in free or k not in df.columns:
            continue
        cond = cond & (pl.col(k).is_null() if v is None else (pl.col(k) == v))
    return df.filter(cond)


def sweep_panel(ax, df, best, x, line, xlog=False, xlabel=None, line_fmt="{}", title=None):

    s = _slice(df, best, [x, line]).sort(x)
    xs = sorted(s[x].unique().to_list())
    lines = {}
    for v in sorted(s[line].unique().to_list(), key=lambda t: (t is None, t)):
        part = s.filter(pl.col(line) == v).sort(x)
        lines[line_fmt.format(v)] = part[M].to_list()
    plots.sweep(ax, xs, lines, best=(best[x], float(_slice(df, best, [])[M].max())), xlog=xlog,
                xlabel=xlabel or x, ylabel=f"val {M}", legend_title=line)
    ax.set_title(title or f"{x} × {line}")


def fig_itemknn(df, best):
    fig, (a, b) = plots.figure(12, 4, ncols=2)
    sweep_panel(a, df, best, "shrinkage", "k", xlabel="shrinkage λ", title="Сжатие сходства и число соседей")
    s = df.group_by("source").agg(pl.col(M).max()).sort("source")
    plots.barh(b, s["source"].to_list(), s[M].to_numpy(), highlight=best["source"], fmt="{:.4f}", xlabel=f"лучший val {M}")
    b.set_title("Матрица: все полки (all) или только оценки ≥ 4 (positive)")
    return fig


def fig_ease(df, best):
    fig, ax = plots.figure(12, 4)
    sweep_panel(ax, df, best, "reg", "source", xlog=True, xlabel="λ (лог. шкала)",
                title="EASE: регуляризация и состав матрицы")
    return fig


def fig_ials(df, best):
    fig, (a, b) = plots.figure(12, 4, ncols=2)
    sweep_panel(a, df, best, "alpha", "factors", xlog=True, xlabel="α (лог. шкала)", title="iALS: уверенность α и размерность d")
    sweep_panel(b, df, best, "reg", "factors", xlog=True, xlabel="λ (лог. шкала)", title="iALS: регуляризация λ и размерность d")
    return fig


def fig_fm(df, abl, best):
    fig, (a, b) = plots.figure(12, 4, ncols=2)
    lines, xs = {}, None
    for (lr, reg), part in df.group_by(["lr", "reg"], maintain_order=True):
        part = part.sort("epochs")
        lines[f"lr={lr:g}, λ={reg:g}"] = part[M].to_list()
        xs = part["epochs"].to_list()
    plots.sweep(a, xs, lines, best=(best["epochs"], float(df[M].max())), xlabel="эпоха", ylabel=f"val {M}")
    a.set_title("FM (все признаки): качество по эпохам")
    s = abl.group_by("feature_set").agg(pl.col(M).max())
    plots.barh(b, s["feature_set"].to_list(), s[M].to_numpy(), highlight=best["feature_set"], fmt="{:.4f}",
               xlabel=f"лучший val {M} (лучшая эпоха, те же lr и λ)")
    b.set_title("Вклад метапризнаков")
    return fig


def fig_signal(abl: pl.DataFrame, stage_name: str):
    fig, ax = plots.figure(12, 0.6 + 0.55 * abl.height)
    plots.ablation(ax, abl["вариант"].to_list(), abl["delta"].to_numpy(), abl["lo"].to_numpy(), abl["hi"].to_numpy(),
                   xlabel="Δ NDCG@10 относительно полной модели 5 (paired bootstrap, 95% CI); < 0 - компонента полезна")
    ax.set_title(f"Абляция сигнала модели 5 ({stage_name})")
    return fig


AGE_BINS = [(0, 90, "≤3 мес"), (90, 365, "3-12 мес"), (365, 730, "1-2 года"), (730, 1460, "2-4 года"),
            (1460, 10**6, ">4 лет")]


def fig_p_by_age(diag: pl.DataFrame):
    fig, ax = plots.figure(12, 3.6)
    d = diag.filter(pl.col("kind") != "негатив r≤2")
    groups = [d.filter((pl.col("age_days") >= lo) & (pl.col("age_days") < hi))["p"].to_numpy() for lo, hi, _ in AGE_BINS]
    ax.boxplot(groups, tick_labels=[lab for *_, lab in AGE_BINS], showfliers=False)
    ax.set(xlabel="давность последнего события пары до cutoff", ylabel="цель p",
           title="Цель модели 5: p убывает с давностью (в модели 2 p = 1 для всех пар)")
    return fig


def fig_test_rank(table: pl.DataFrame, highlight: str):
    fig, (a, b) = plots.figure(12, 4, ncols=2)
    plots.barh(a, table["model"].to_list(), table[M].to_numpy(), table["ndcg_lo"].to_numpy(), table["ndcg_hi"].to_numpy(),
               highlight=highlight, fmt="{:.4f}", xlabel="NDCG@10 (усы - bootstrap 95% CI)")
    a.set_title(f"Test NDCG@10 (n = {int(table['users'][0])} пользователей)")
    r = f"recall@{config.K}"
    plots.barh(b, table["model"].to_list(), table[r].to_numpy(), highlight=highlight, fmt="{:.4f}", xlabel="Recall@10")
    b.set_title("Test Recall@10")
    return fig


def pairwise_vs(per_user: dict, ref: str) -> pl.DataFrame:
    rows = []
    for name, d in per_user.items():
        if name == ref:
            continue
        r = metrics.paired_bootstrap(d["ndcg"], per_user[ref]["ndcg"])
        rows.append({"model": name, "Δ NDCG@10 vs " + ref: r["delta"], "lo": r["lo"], "hi": r["hi"]})
    return pl.DataFrame(rows)


def fig_by_activity(per_user: dict, stage, highlight: str):
    hist_len = np.diff(stage.history.seen.indptr)[stage.users]
    bins = [(0, 25, "≤25"), (25, 50, "26-50"), (50, 100, "51-100"), (100, 250, "101-250"), (250, 10**9, ">250")]
    names = list(per_user)
    mat = np.array([[per_user[n]["ndcg"][(hist_len > lo) & (hist_len <= hi)].mean() for lo, hi, _ in bins]
                    for n in names])
    counts = [int(((hist_len > lo) & (hist_len <= hi)).sum()) for lo, hi, _ in bins]
    fig, ax = plots.figure(12, 0.8 + 0.45 * len(names))
    plots.heat(ax, mat, names, [f"{lab}\nn={c}" for (_, _, lab), c in zip(bins, counts)], cbar_label="NDCG@10")
    ax.get_yticklabels()[names.index(highlight)].set_fontweight("bold")
    ax.set(xlabel="пар в истории пользователя до cutoff (n - пользователей в корзине)",
           title="Test NDCG@10 по длине истории пользователя")
    return fig


def fig_popularity_of_recs(recs: pl.DataFrame, stage, highlight: str):
    pop = stage.history.popularity
    pct = np.argsort(np.argsort(-pop)) / len(pop) * 100
    names = [n for n in recs.columns[1:] if n != "Random"]
    med = [float(np.median(np.median(pct[np.array(recs[n].to_list())], axis=1))) for n in names]
    fig, ax = plots.figure(12, 3.8)
    plots.barh(ax, names, med, highlight=highlight, fmt="{:.2f}%",
               xlabel="медианный перцентиль популярности top-10, % (0 - самая популярная книга; Random ≈ 50%)")
    ax.set_title("Насколько популярны рекомендации (больше - менее очевидные книги)")
    return fig


def example_user(recs: pl.DataFrame, stage, items: pl.DataFrame, models: list[str], user: int | None = None,
                 n_hist: int = 6) -> tuple[pl.DataFrame, pl.DataFrame, int]:

    t = stage.targets
    if user is None:
        lens = np.diff(stage.history.seen.indptr)[stage.users]
        cand = stage.users[(lens > 20) & (lens < 80)]
        user = int(cand[len(cand) // 2])
    title = dict(zip(items["item_idx"].to_list(),
                     (items["title"].fill_null("?").str.slice(0, 38) + " - " + items["author"].fill_null("?")
                      .str.slice(0, 18)).to_list()))
    hist = (stage.history.pairs.filter(pl.col("user_idx") == user).sort("t_shelf", descending=True).head(n_hist)
            .select(pl.col("t_shelf").dt.date().alias("полка"), pl.col("rating").alias("оценка"),
                    pl.col("item_idx").replace_strict(title, return_dtype=pl.Utf8).alias("книга")))
    rel = set(t.indices[t.indptr[user]:t.indptr[user + 1]].tolist())
    row = recs.filter(pl.col("user_idx") == user)
    out = {"#": list(range(1, config.K + 1))}
    for m in models:
        out[m] = [("● " if i in rel else "") + title[i] for i in row[m].item()]
    return hist, pl.DataFrame(out), user
