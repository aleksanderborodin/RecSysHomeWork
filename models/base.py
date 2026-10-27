from __future__ import annotations

import numpy as np

from core.split import History


class Recommender:
    name = "base"

    def fit(self, history: History) -> "Recommender":
        raise NotImplementedError

    def score(self, users: np.ndarray) -> np.ndarray:

        raise NotImplementedError
