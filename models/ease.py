import numpy as np
import scipy.linalg
import torch

from models.base import Recommender
from models.torch_utils import DEVICE, dense_rows


def ease_weights(X, reg: float) -> np.ndarray:
    G = (X.T @ X).toarray().astype(np.float64)
    G[np.diag_indices_from(G)] += reg
    c, low = scipy.linalg.cho_factor(G, overwrite_a=True, check_finite=False)
    P = scipy.linalg.cho_solve((c, low), np.eye(G.shape[0]), overwrite_b=True, check_finite=False)
    B = -P / np.diag(P)[None, :]
    B[np.diag_indices_from(B)] = 0.0
    return B


class EASE(Recommender):
    name = "EASE"

    def __init__(self, reg: float = 500.0, source: str = "all"):
        self.reg, self.source = reg, source

    def fit(self, history):
        self.X = history.matrix(self.source)
        self.B = torch.as_tensor(ease_weights(self.X, self.reg), dtype=torch.float32, device=DEVICE)
        return self

    def score(self, users):
        return dense_rows(self.X, users) @ self.B
