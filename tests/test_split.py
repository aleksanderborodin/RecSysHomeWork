from datetime import datetime, timezone

import polars as pl

from core.split import history_at, targets_in

T = lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _events():


    return pl.DataFrame({
        "user_idx": [0, 0, 0, 1, 1],
        "item_idx": [0, 1, 2, 2, 1],
        "t_shelf": [T("2017-01-01"), T("2017-03-05"), T("2017-03-06"), T("2016-12-01"), T("2017-03-10")],
        "rating": [5, 5, 3, 4, 5],
        "t_rating": [T("2017-04-01"), T("2017-03-07"), T("2017-03-08"), T("2016-12-02"), T("2017-07-01")],
        "rating_first": [5, 5, 3, 4, 5],
        "t_rating_first": [T("2017-04-01"), T("2017-03-07"), T("2017-03-08"), T("2016-12-02"), T("2017-07-01")],
        "t_read": [None, None, None, None, None],
        "t_review": [None, None, None, None, None],
        "n_editions": [1, 1, 1, 1, 1],
    }, schema_overrides={"t_read": pl.Datetime("us", "UTC"), "t_review": pl.Datetime("us", "UTC")})


def test_no_future_information_in_history():
    ev = _events()
    h = history_at(ev, None, T("2017-03-01"), 2, 3)
    assert h.pairs.height == 2
    assert h.pairs.filter(pl.col("item_idx") == 0)["rating"].item() is None
    assert (h.pairs["t_shelf"] < T("2017-03-01")).all()


def test_targets_rule():
    ev = _events()
    h = history_at(ev, None, T("2017-03-01"), 2, 3)
    t, users, stats = targets_in(ev, h, T("2017-03-01"), T("2017-06-01"))

    assert stats["relevant_pairs"] == 1 and stats["relevant_out_of_catalog"] == 1
    assert t.nnz == 0 and len(users) == 0


def test_earlier_edition_rating_is_kept():

    ev = _events().with_columns(
        pl.when(pl.col("item_idx") == 0).then(pl.lit(4)).otherwise(pl.col("rating_first")).alias("rating_first"),
        pl.when(pl.col("item_idx") == 0).then(pl.lit(T("2017-02-01"))).otherwise(pl.col("t_rating_first")).alias("t_rating_first"),
        pl.when(pl.col("item_idx") == 0).then(pl.lit(2)).otherwise(pl.col("rating")).cast(pl.Int64).alias("rating"))
    h = history_at(ev, None, T("2017-03-01"), 2, 3)
    assert h.pairs.filter(pl.col("item_idx") == 0)["rating"].item() == 4
