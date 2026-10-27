from __future__ import annotations

import json

import polars as pl

import config

GR_DATE = "%a %b %d %H:%M:%S %z %Y"
TS = pl.Datetime("us", "UTC")


def _date(col: str) -> pl.Expr:
    return pl.col(col).str.strptime(TS, GR_DATE, strict=False)


def _raw(name: str):
    return config.RAW / config.FILES[name].rsplit("/", 1)[1]


def read_interactions_raw() -> pl.DataFrame:
    schema = {"user_id": pl.Utf8, "book_id": pl.Utf8, "is_read": pl.Boolean, "rating": pl.Int64,
              "date_added": pl.Utf8, "date_updated": pl.Utf8, "read_at": pl.Utf8}
    df = pl.read_ndjson(_raw("interactions"), schema=schema)
    return df.select(
        "user_id", "book_id", "is_read", pl.col("rating").cast(pl.Int8),
        _date("date_added").alias("t_added"),
        _date("date_updated").alias("t_updated"),
        _date("read_at").alias("t_read_at"),
    )


def read_reviews_raw() -> pl.DataFrame:
    schema = {"user_id": pl.Utf8, "book_id": pl.Utf8, "date_added": pl.Utf8, "review_text": pl.Utf8}
    df = pl.read_ndjson(_raw("reviews"), schema=schema)
    return df.select("user_id", "book_id", _date("date_added").alias("t_review"),
                     pl.col("review_text").str.len_chars().alias("review_len"))


def read_books_raw() -> pl.DataFrame:

    schema = {
        "book_id": pl.Utf8, "work_id": pl.Utf8, "title": pl.Utf8, "title_without_series": pl.Utf8,
        "authors": pl.List(pl.Struct({"author_id": pl.Utf8, "role": pl.Utf8})),
        "publisher": pl.Utf8, "num_pages": pl.Utf8, "publication_year": pl.Utf8, "format": pl.Utf8,
        "language_code": pl.Utf8, "is_ebook": pl.Utf8, "description": pl.Utf8, "country_code": pl.Utf8,
    }
    df = pl.read_ndjson(_raw("books"), schema=schema)
    empty_to_null = lambda c: pl.when(pl.col(c).str.strip_chars() == "").then(None).otherwise(pl.col(c).str.strip_chars())
    return df.select(
        "book_id",
        empty_to_null("work_id").alias("work_id"),
        "title", "title_without_series",
        pl.col("authors").list.eval(pl.element().struct.field("author_id")).alias("author_ids"),
        empty_to_null("publisher").alias("publisher"),
        empty_to_null("num_pages").cast(pl.Int32, strict=False).alias("num_pages"),
        empty_to_null("publication_year").cast(pl.Int32, strict=False).alias("publication_year"),
        empty_to_null("format").str.to_lowercase().alias("format"),
        empty_to_null("language_code").alias("language_code"),
        (pl.col("is_ebook") == "true").alias("is_ebook"),
        empty_to_null("description").alias("description"),
    )


def read_authors_raw() -> pl.DataFrame:
    schema = {"author_id": pl.Utf8, "name": pl.Utf8}
    return pl.read_ndjson(_raw("authors"), schema=schema)


def build_interim() -> None:
    config.INTERIM.mkdir(parents=True, exist_ok=True)
    read_interactions_raw().write_parquet(config.INTERIM / "interactions.parquet")
    read_reviews_raw().write_parquet(config.INTERIM / "reviews.parquet")
    books = read_books_raw()
    books.write_parquet(config.INTERIM / "books.parquet")
    used = books.select(pl.col("author_ids").explode()).unique().to_series()
    read_authors_raw().filter(pl.col("author_id").is_in(used.implode())).write_parquet(config.INTERIM / "authors.parquet")


def load_interim(name: str) -> pl.DataFrame:
    return pl.read_parquet(config.INTERIM / f"{name}.parquet")


def to_works(inter: pl.DataFrame, books: pl.DataFrame, reviews: pl.DataFrame) -> pl.DataFrame:

    work = books.select("book_id", pl.coalesce("work_id", pl.lit("b") + pl.col("book_id")).alias("work_id"))
    rev = reviews.filter(pl.col("review_len") > 0).group_by("user_id", "book_id").agg(pl.col("t_review").min())
    df = (inter.join(work, on="book_id", how="left")
          .with_columns(pl.coalesce("work_id", pl.lit("b") + pl.col("book_id")).alias("work_id"))
          .join(rev, on=["user_id", "book_id"], how="left"))
    df = df.with_columns(
        pl.when(pl.col("rating") > 0).then(pl.max_horizontal("t_added", "t_updated")).alias("t_rating"),
        pl.when(pl.col("is_read"))
        .then(pl.max_horizontal("t_added", pl.coalesce("t_read_at", "t_updated"))).alias("t_read"),
    )
    return (df.sort("t_rating", nulls_last=False)
            .group_by("user_id", "work_id")
            .agg(pl.col("t_added").min().alias("t_shelf"),
                 pl.col("rating").filter(pl.col("rating") > 0).last().alias("rating"),
                 pl.col("t_rating").drop_nulls().last().alias("t_rating"),
                 pl.col("rating").filter(pl.col("rating") > 0).first().alias("rating_first"),
                 pl.col("t_rating").drop_nulls().first().alias("t_rating_first"),
                 pl.col("t_read").min().alias("t_read"),
                 pl.col("t_review").min().alias("t_review"),
                 pl.len().alias("n_editions")))


def kcore(pairs: pl.DataFrame, k_user: int, k_item: int, user="user_id", item="work_id") -> pl.DataFrame:
    while True:
        n = pairs.height
        pairs = (pairs.filter(pl.len().over(item) >= k_item)
                      .filter(pl.len().over(user) >= k_user))
        if pairs.height == n:
            return pairs


def works_table(books: pl.DataFrame, authors: pl.DataFrame) -> pl.DataFrame:

    b = books.with_columns(pl.coalesce("work_id", pl.lit("b") + pl.col("book_id")).alias("work_id"))
    known = pl.col("publication_year").is_null() | (pl.col("publication_year") < config.VAL_START.year)
    has_known = b.filter(known).select("work_id").unique()
    b = pl.concat([b.filter(known), b.join(has_known, on="work_id", how="anti")])
    b = b.sort(["work_id", "publication_year"], nulls_last=True)
    first_nonnull = lambda c: pl.col(c).drop_nulls().first().alias(c)
    w = b.group_by("work_id").agg(
        pl.col("book_id").first().alias("book_id"),
        first_nonnull("title_without_series").alias("title"),
        pl.col("author_ids").first(),
        first_nonnull("publisher"),
        pl.col("num_pages").median().cast(pl.Int32),
        pl.col("publication_year").min(),
        pl.col("format").drop_nulls().mode().first().alias("format"),
        pl.col("language_code").drop_nulls().mode().first().alias("language_code"),
        pl.col("is_ebook").all(),
        pl.col("description").sort_by(pl.col("description").str.len_chars(), descending=True).first(),
        pl.len().alias("n_editions"),
    )
    names = dict(authors.iter_rows())
    return w.with_columns(
        pl.col("author_ids").map_elements(lambda ids: names.get(ids[0], "") if len(ids) else "",
                                          return_dtype=pl.Utf8).alias("author"))


def prepare() -> dict:

    inter, books = load_interim("interactions"), load_interim("books")
    reviews, authors = load_interim("reviews"), load_interim("authors")
    funnel = [("исходные записи", inter.height, inter["user_id"].n_unique(), inter["book_id"].n_unique())]

    inter = inter.unique(["user_id", "book_id"], keep="first")
    funnel.append(("без дублей пар", inter.height, inter["user_id"].n_unique(), inter["book_id"].n_unique()))

    inter = inter.filter(pl.col("t_added").is_between(config.MIN_DATE, config.TEST_END, closed="left"))
    funnel.append(("даты в [2007-01, 2017-11)", inter.height, inter["user_id"].n_unique(), inter["book_id"].n_unique()))

    pairs = to_works(inter, books, reviews)
    funnel.append(("издания → произведения", pairs.height, pairs["user_id"].n_unique(), pairs["work_id"].n_unique()))

    train = pairs.filter(pl.col("t_shelf") < config.VAL_START)
    core = kcore(train, config.KCORE_USER, config.KCORE_ITEM)
    users = core["user_id"].unique().sort()
    items = core["work_id"].unique().sort()
    final = pairs.filter(pl.col("user_id").is_in(users.implode()) & pl.col("work_id").is_in(items.implode()))
    funnel.append((f"{config.KCORE_USER}-core по train", final.height, final["user_id"].n_unique(), final["work_id"].n_unique()))


    future = pairs.filter(pl.col("t_shelf") >= config.VAL_START)
    cold = {
        "future_pairs": future.height,
        "future_pairs_cold_user": future.filter(~pl.col("user_id").is_in(users.implode())).height,
        "future_pairs_cold_item": future.filter(pl.col("user_id").is_in(users.implode())
                                                & ~pl.col("work_id").is_in(items.implode())).height,
    }

    user_map = pl.DataFrame({"user_id": users, "user_idx": pl.Series(range(len(users)), dtype=pl.Int32)})
    item_map = pl.DataFrame({"work_id": items, "item_idx": pl.Series(range(len(items)), dtype=pl.Int32)})
    events = (final.join(user_map, on="user_id").join(item_map, on="work_id")
              .select("user_idx", "item_idx", "t_shelf", "rating", "t_rating", "rating_first", "t_rating_first",
                      "t_read", "t_review", "n_editions")
              .sort("user_idx", "t_shelf"))
    works = works_table(books, authors).join(item_map, on="work_id").sort("item_idx")

    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    events.write_parquet(config.PROCESSED / "events.parquet")
    works.write_parquet(config.PROCESSED / "items.parquet")
    user_map.write_parquet(config.PROCESSED / "users.parquet")
    summary = {
        "funnel": [dict(zip(("step", "interactions", "users", "items"), row)) for row in funnel],
        "cold": cold,
        "n_users": len(users), "n_items": len(items), "n_interactions": events.height,
        "cutoffs": {"val_start": str(config.VAL_START.date()), "test_start": str(config.TEST_START.date()),
                    "test_end": str(config.TEST_END.date())},
        "kcore": [config.KCORE_USER, config.KCORE_ITEM],
    }
    (config.PROCESSED / "split.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def load_processed():
    events = pl.read_parquet(config.PROCESSED / "events.parquet")
    items = pl.read_parquet(config.PROCESSED / "items.parquet")
    summary = json.loads((config.PROCESSED / "split.json").read_text())
    return events, items, summary
