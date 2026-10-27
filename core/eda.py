from __future__ import annotations

import numpy as np
import polars as pl
from matplotlib.ticker import PercentFormatter

import config
from core import plots

MONTH = pl.col("t").dt.truncate("1mo")
N = plots.fmt_int


def quality_table(inter: pl.DataFrame, books: pl.DataFrame) -> pl.DataFrame:

    n = inter.height
    rows = [
        ("взаимодействия", "записей", n),
        ("взаимодействия", "дубли пар (user, book)", n - inter.unique(["user_id", "book_id"]).height),
        ("взаимодействия", "без оценки (rating = 0)", inter.filter(pl.col("rating") == 0).height),
        ("взаимодействия", "без read_at", inter["t_read_at"].null_count()),
        ("взаимодействия", "date_added не распознана", inter["t_added"].null_count()),
        ("взаимодействия", "date_added < 2007", inter.filter(pl.col("t_added") < config.MIN_DATE).height),
        ("книги", "записей", books.height),
        ("книги", "без work_id", books["work_id"].null_count()),
    ]
    for c in ("description", "publication_year", "num_pages", "publisher", "format", "language_code"):
        rows.append(("книги", f"без {c}", books[c].null_count()))
    rows.append(("книги", "без авторов", books.filter(pl.col("author_ids").list.len() == 0).height))
    df = pl.DataFrame(rows, schema=["таблица", "проверка", "строк"], orient="row")
    total = {"взаимодействия": n, "книги": books.height}
    return df.with_columns((pl.col("строк") / pl.col("таблица").replace_strict(total) * 100).round(2).alias("%"))


def kinds(inter: pl.DataFrame) -> pl.DataFrame:
    return (inter.with_columns(pl.when(pl.col("rating") > 0).then(pl.lit("оценка 1-5"))
                               .when(pl.col("is_read")).then(pl.lit("прочитано без оценки"))
                               .otherwise(pl.lit("только полка")).alias("тип"))
            .group_by("тип").len().with_columns((pl.col("len") / inter.height * 100).round(1).alias("%"))
            .sort("len", descending=True))


def sparsity(events: pl.DataFrame) -> dict:
    nu, ni = events["user_idx"].n_unique(), events["item_idx"].n_unique()
    return {"пары": events.height, "пользователи": nu, "произведения": ni,
            "плотность, %": round(100 * events.height / nu / ni, 3),
            "медиана пар на пользователя": float(events.group_by("user_idx").len()["len"].median()),
            "медиана пар на произведение": float(events.group_by("item_idx").len()["len"].median())}


def fig_ratings_and_kinds(inter: pl.DataFrame):
    fig, (a, b) = plots.figure(12, 3.8, ncols=2)
    r = inter.filter(pl.col("rating") > 0).group_by("rating").len().sort("rating")
    vals = r["len"].to_numpy()
    shares = vals / vals.sum()
    colors = [plots.ACCENT if x >= config.POSITIVE_RATING else plots.BASE for x in r["rating"]]
    bars = a.bar([str(x) for x in r["rating"]], shares, color=colors)
    a.bar_label(bars, labels=[f"{v:.0%}" for v in shares], padding=2)
    a.set(xlabel="оценка, звёзды", ylabel="доля оценок",
          title=f"Оценки (n = {N(vals.sum())}); оранжевым - релевантные (≥ 4)")
    a.yaxis.set_major_formatter(PercentFormatter(1.0))
    k = kinds(inter)
    bars = b.bar(k["тип"].to_list(), k["%"].to_numpy())
    b.bar_label(bars, labels=[f"{v:.0f}%" for v in k["%"]], padding=2)
    b.set(ylabel="% записей", title=f"Типы записей (n = {N(inter.height)})")
    return fig


def fig_long_tail(events: pl.DataFrame):
    fig, (a, b) = plots.figure(12, 4, ncols=2)
    u = np.sort(events.group_by("user_idx").len()["len"].to_numpy())[::-1]
    i = np.sort(events.group_by("item_idx").len()["len"].to_numpy())[::-1]
    a.plot(np.arange(1, len(u) + 1), u, label=f"пользователи ({N(len(u))})")
    a.plot(np.arange(1, len(i) + 1), i, label=f"произведения ({N(len(i))})")
    a.set(xscale="log", yscale="log", xlabel="ранг", ylabel="число пар", title="Длинный хвост активности и популярности")
    a.legend()
    for arr, name in ((u, "пользователи"), (i, "произведения")):
        b.plot(np.arange(1, len(arr) + 1) / len(arr), np.cumsum(arr) / arr.sum(), label=name)
    b.plot([0, 1], [0, 1], "k--", lw=1, label="равномерно")
    top10 = np.cumsum(i)[int(0.1 * len(i)) - 1] / i.sum()
    b.plot([0.1], [top10], "ro")
    b.annotate(f"10% произведений = {top10:.0%} пар", (0.1, top10), xytext=(10, -4), textcoords="offset points")
    b.set(xlabel="доля самых активных / популярных", ylabel="доля пар", xlim=(0, 1), ylim=(0, 1.02),
          title="Концентрация пар (кривая Лоренца)")
    b.legend(loc="lower right")
    return fig


def fig_timeline_volume(inter: pl.DataFrame):
    fig, ax = plots.figure(12, 4)
    m_shelf = (inter.select(pl.col("t_added").alias("t"))
               .filter(pl.col("t").is_between(config.MIN_DATE, config.TEST_END, closed="left"))
               .group_by(MONTH).len().sort("t"))
    m_rate = (inter.filter(pl.col("rating") > 0).select(pl.col("t_updated").alias("t"))
              .filter(pl.col("t").is_between(config.MIN_DATE, config.TEST_END, closed="left"))
              .group_by(MONTH).len().sort("t"))
    ax.plot(m_shelf["t"], m_shelf["len"], label="полки (date_added)")
    ax.plot(m_rate["t"], m_rate["len"], label="оценки (date_updated)")
    ax.axvspan(config.VAL_START, config.TEST_START, color="tab:green", alpha=0.2, label="validation")
    ax.axvspan(config.TEST_START, config.TEST_END, color="tab:red", alpha=0.2, label="test")
    ax.set(ylabel="записей в месяц", title="Поток событий по месяцам и границы периодов (краулинг - до ноября 2017)")
    plots.thousands(ax)
    ax.legend(loc="upper left")
    return fig


def fig_lag(inter: pl.DataFrame):
    fig, ax = plots.figure(12, 3.6)
    r = inter.filter((pl.col("rating") > 0)).with_columns(
        ((pl.col("t_updated") - pl.col("t_added")).dt.total_seconds() / 86400).alias("lag"))
    lag = r["lag"].to_numpy()
    lag = lag[np.isfinite(lag)]
    pos = lag[lag > 1 / 24]
    bins = np.logspace(np.log10(1 / 24), np.log10(max(pos.max(), 10)), 50)
    ax.hist(pos, bins=bins)
    ax.axvline(30, color="red", ls="--", label="30 дней")
    ax.set_xscale("log")
    ax.set_xticks([1 / 24, 1, 7, 30, 365, 3650], ["1 ч", "1 д", "1 нед", "30 д", "1 год", "10 лет"])
    plots.thousands(ax)
    share30 = (lag > 30).mean()
    ax.set(xlabel="date_updated - date_added", ylabel="оценённых записей",
           title=f"У {share30:.0%} оценок date_updated позже полки больше чем на 30 дней "
                 f"(показаны {N(len(pos))} из {N(len(lag))} с разницей > 1 ч)")
    ax.legend()
    return fig


def fig_metadata(items: pl.DataFrame):
    fig, (a, b, c) = plots.figure(13, 3.6, ncols=3)
    cols = {"описание": "description", "год": "publication_year", "страницы": "num_pages",
            "издатель": "publisher", "формат": "format", "язык": "language_code"}
    fill = [1 - items[c].null_count() / items.height for c in cols.values()]
    plots.barh(a, list(cols), np.array(fill), fmt="{:.0%}")
    a.set_xlim(0, 1.25)
    a.set_title(f"Заполненность полей (n = {N(items.height)})")
    y = items["publication_year"].drop_nulls().to_numpy()
    y = y[(y > 1500) & (y < 2018)]
    b.hist(np.clip(y, 1900, 2017), bins=np.arange(1900, 2019, 4))
    b.set(xlabel="год (до 1900 - в первом столбце)", ylabel="произведений", title="Год издания")
    p = items["num_pages"].drop_nulls().to_numpy()
    c.hist(np.clip(p, 0, 600), bins=np.arange(0, 610, 20))
    med = float(np.median(p))
    c.axvline(med, color="red", ls="--", label=f"медиана {med:.0f}")
    c.set(xlabel="страниц (> 600 - в последнем столбце)", title="Число страниц")
    c.legend()
    return fig


def fig_funnel(summary: dict):
    fig, ax = plots.figure(12, 3.4)
    steps = summary["funnel"]
    names = [s["step"] for s in steps]
    vals = np.array([s["interactions"] for s in steps])
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals)
    for yi, s in zip(y, steps):
        ax.annotate(f"{N(s['interactions'])} пар · {N(s['users'])} польз. · {N(s['items'])} объектов",
                    (s["interactions"], yi), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)
    ax.set_yticks(y, names)
    ax.set_xlim(0, vals.max() * 1.65)
    plots.thousands(ax, "x")
    ax.set(xlabel="взаимодействий",
           title="Очистка и подвыборка (задание: 100 тыс. - 2 млн взаимодействий, ≥ 1000 пользователей и объектов)")
    return fig
