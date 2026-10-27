from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from core import features
from models.base import Recommender
from models.torch_utils import DEVICE


def _sparse(m: sp.csr_matrix, square=False) -> torch.Tensor:
    m = m.tocoo()
    data = m.data ** 2 if square else m.data
    return torch.sparse_coo_tensor(np.vstack([m.row, m.col]), data.astype(np.float32), m.shape,
                                   device=DEVICE, check_invariants=False).coalesce()


class FMCore(torch.nn.Module):
    def __init__(self, n_features: int, factors: int, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.V = torch.nn.Parameter(torch.randn(n_features, factors, generator=g) * 0.01)
        self.w = torch.nn.Parameter(torch.zeros(n_features))

    def side(self, X: torch.Tensor, X2: torch.Tensor):

        e = torch.sparse.mm(X, self.V)
        lin = torch.sparse.mm(X, self.w[:, None]).squeeze(1)
        self_sq = torch.sparse.mm(X2, (self.V ** 2).sum(1, keepdim=True)).squeeze(1)
        return e, lin + 0.5 * ((e ** 2).sum(1) - self_sq)


class FM(Recommender):
    name = "FM"

    def __init__(self, feature_set="ids+item+user", factors=64, lr=3e-3, reg=1e-5, reg_bias=1e-6,
                 epochs=10, batch=8192, seed=42, source="all"):
        self.feature_set, self.factors, self.lr = feature_set, factors, lr
        self.reg, self.reg_bias, self.epochs, self.batch = reg, reg_bias, epochs, batch
        self.seed, self.source = seed, source

    def fit(self, history, callback=None):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.space = features.build(history, features.FEATURE_SETS[self.feature_set])
        self.Xu, self.Xu2 = _sparse(self.space.user), _sparse(self.space.user, True)
        self.Xi, self.Xi2 = _sparse(self.space.item), _sparse(self.space.item, True)
        self.model = FMCore(self.space.n_features, self.factors, self.seed).to(DEVICE)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        pos = history.matrix(self.source).tocoo()
        users = torch.as_tensor(pos.row, dtype=torch.long, device=DEVICE)
        items = torch.as_tensor(pos.col, dtype=torch.long, device=DEVICE)
        seen = history.seen.tocoo()
        n_items = history.n_items
        seen_keys = torch.sort(torch.as_tensor(seen.row.astype(np.int64) * n_items + seen.col, device=DEVICE)).values
        catalog = torch.as_tensor(np.flatnonzero(history.catalog), device=DEVICE)
        self.history_ = []
        for epoch in range(1, self.epochs + 1):
            perm = torch.as_tensor(rng.permutation(len(users)), device=DEVICE)
            total = 0.0
            for s in range(0, len(perm), self.batch):
                b = perm[s:s + self.batch]
                u, i = users[b], items[b]
                j = self._negatives(u, catalog, seen_keys, n_items)
                eu, _ = self.model.side(self.Xu, self.Xu2)
                ei_all, ti_all = self.model.side(self.Xi, self.Xi2)
                e_u, e_i, e_j = eu[u], ei_all[i], ei_all[j]
                delta = ti_all[i] - ti_all[j] + (e_u * (e_i - e_j)).sum(1)
                loss = (F.softplus(-delta).mean()
                        + self.reg * ((e_u ** 2).sum(1) + (e_i ** 2).sum(1) + (e_j ** 2).sum(1)).mean()
                        + self.reg_bias * (ti_all[i] ** 2 + ti_all[j] ** 2).mean())
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(b)
            self._cache_scores()
            self.history_.append({"epoch": epoch, "loss": total / len(perm)})
            if callback is not None:
                callback(epoch, self)
        return self

    @staticmethod
    def _negatives(u, catalog, seen_keys, n_items, max_tries=100):

        j = catalog[torch.randint(len(catalog), (len(u),), device=DEVICE)]
        for _ in range(max_tries):
            keys = u * n_items + j
            pos = torch.searchsorted(seen_keys, keys).clamp(max=len(seen_keys) - 1)
            bad = seen_keys[pos] == keys
            if not bad.any():
                return j
            j = torch.where(bad, catalog[torch.randint(len(catalog), (len(u),), device=DEVICE)], j)
        raise RuntimeError("не удалось выбрать невиденные негативы")

    @torch.no_grad()
    def _cache_scores(self):
        self.e_u, _ = self.model.side(self.Xu, self.Xu2)
        self.e_i, self.t_i = self.model.side(self.Xi, self.Xi2)

    @torch.no_grad()
    def score(self, users):
        u = torch.as_tensor(users, device=DEVICE)
        return self.e_u[u] @ self.e_i.T + self.t_i[None, :]
