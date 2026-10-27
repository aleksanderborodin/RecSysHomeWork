import numpy as np

from core import signals
from core.split import history_at
from tests.test_split import T, _events


def _history():

    return history_at(_events(), None, T("2017-03-01"), 2, 3)


def test_recency_goes_into_target():
    h = _history()
    rows, cols, p, c, diag = signals.build(h, signals.SignalConfig(alpha=10, tau=None, alpha_neg=None, w_shelf=1.0,
                                                                   half_life=30, recency="p"))
    age = diag["age_days"].to_numpy()
    np.testing.assert_allclose(p, 2.0 ** (-age / 30), rtol=1e-6)
    np.testing.assert_allclose(c, 11.0)


def test_recency_in_confidence_is_control():
    h = _history()
    _, _, p, c, diag = signals.build(h, signals.SignalConfig(alpha=10, tau=None, alpha_neg=None, w_shelf=1.0,
                                                             half_life=30, recency="c"))
    age = diag["age_days"].to_numpy()
    np.testing.assert_allclose(p, 1.0)
    np.testing.assert_allclose(c, 1 + 10 * 2.0 ** (-age / 30), rtol=1e-6)


def test_negative_has_zero_preference():
    ev = _events().with_columns(rating=np.array([5, 5, 3, 1, 5]), rating_first=np.array([5, 5, 3, 1, 5]))
    h = history_at(ev, None, T("2017-03-01"), 2, 3)
    _, cols, p, c, _ = signals.build(h, signals.SignalConfig(alpha=10, tau=0.5, alpha_neg=40, half_life=365))
    neg = cols == 2
    assert p[neg].item() == 0.0 and c[neg].item() == 41.0
