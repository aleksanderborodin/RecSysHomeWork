import numpy as np
import torch

from models.base import Recommender
from models.torch_utils import DEVICE, dense_rows


def cosine_shrunk(X, shrinkage: float) -> torch.Tensor:
    co = torch.as_tensor((X.T @ X).toarray(), dtype=torch.float32, device=DEVICE)
    n = torch.diagonal(co).clone()
    norm = torch.sqrt(torch.clamp(n, min=1.0))
    sim = co / norm[:, None] / norm[None, :] * (co / (co + shrinkage) if shrinkage > 0 else 1.0)
    sim.fill_diagonal_(0.0)
    return sim


def keep_topk_per_column(sim: torch.Tensor, k: int) -> torch.Tensor:
    if k >= sim.shape[0]:
        return sim
    top = torch.topk(sim, k, dim=0)
    pruned = torch.zeros_like(sim)
    pruned.scatter_(0, top.indices, top.values)
    return pruned


class ItemKNN(Recommender):
    name = "ItemKNN"

    def __init__(self, k: int = 100, shrinkage: float = 10.0, source: str = "all"):
        self.k, self.shrinkage, self.source = k, shrinkage, source

    def fit(self, history):
        self.X = history.matrix(self.source)
        self.S = keep_topk_per_column(cosine_shrunk(self.X, self.shrinkage), self.k)
        return self

    def score(self, users):
        return dense_rows(self.X, users) @ self.S

    def neighbors(self, item: int, n: int = 10):
        col = self.S[:, item]
        top = torch.topk(col, n)
        return top.indices.cpu().numpy(), top.values.cpu().numpy()
