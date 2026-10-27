import numpy as np

from core import metrics


def test_ndcg_recall_by_hand():
    top = np.array([[5, 1, 7], [2, 3, 4]])
    targets = [np.array([1, 9]), np.array([8])]
    m = metrics.per_user(top, targets, k=3)

    dcg = 1 / np.log2(3)
    idcg = 1 + 1 / np.log2(3)
    assert np.isclose(m["ndcg"][0], dcg / idcg)
    assert np.isclose(m["recall"][0], 0.5)
    assert np.isclose(m["precision"][0], 1 / 3)
    assert m["hit"][0] == 1 and m["hit"][1] == 0 and m["ndcg"][1] == 0


def test_ndcg_perfect_is_one():
    top = np.array([[3, 4, 0, 1]])
    m = metrics.per_user(top, [np.array([3, 4])], k=4)
    assert np.isclose(m["ndcg"][0], 1.0)


def test_paired_bootstrap_zero_for_identical():
    x = np.random.default_rng(0).random(500)
    r = metrics.paired_bootstrap(x, x)
    assert r["delta"] == 0 and r["lo"] == 0 and r["hi"] == 0
