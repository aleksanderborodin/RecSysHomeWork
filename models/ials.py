from __future__ import annotations

import numpy as np
import torch

from models.base import Recommender
from models.torch_utils import DEVICE


class ImplicitALS:


    def __init__(self, factors=64, reg=1.0, iters=15, cg_steps=4, seed=42):
        self.factors, self.reg, self.iters, self.cg_steps, self.seed = factors, reg, iters, cg_steps, seed

    def fit(self, rows, cols, p, c, n_users, n_items):
        dev = DEVICE
        rows = torch.tensor(rows, dtype=torch.long, device=dev)
        cols = torch.tensor(cols, dtype=torch.long, device=dev)
        p = torch.tensor(p, dtype=torch.float32, device=dev)
        c = torch.tensor(c, dtype=torch.float32, device=dev)
        g = torch.Generator(device=dev).manual_seed(self.seed)
        scale = 0.01
        self.X = torch.randn(n_users, self.factors, generator=g, device=dev) * scale
        self.Y = torch.randn(n_items, self.factors, generator=g, device=dev) * scale
        self.history_ = []
        for it in range(self.iters):
            self.X = self._half_step(self.X, self.Y, rows, cols, p, c)
            self.Y = self._half_step(self.Y, self.X, cols, rows, p, c)
        return self

    def _half_step(self, Z, F, own, other, p, c):

        d = Z.shape[1]
        G = F.T @ F
        lam = self.reg
        Fo = F[other]
        b = torch.zeros_like(Z).index_add_(0, own, (c * p)[:, None] * Fo)

        def A(z):
            out = z @ G + lam * z
            proj = (z[own] * Fo).sum(1) * (c - 1.0)
            return out.index_add_(0, own, proj[:, None] * Fo)

        x = Z.clone()
        r = b - A(x)
        q = r.clone()
        rs = (r * r).sum(1)
        for _ in range(self.cg_steps):
            Aq = A(q)
            alpha = rs / torch.clamp((q * Aq).sum(1), min=1e-20)
            x = x + alpha[:, None] * q
            r = r - alpha[:, None] * Aq
            rs_new = (r * r).sum(1)
            q = r + (rs_new / torch.clamp(rs, min=1e-20))[:, None] * q
            rs = rs_new
        return x

    def loss(self, rows, cols, p, c) -> float:

        X, Y = self.X, self.Y
        rows = torch.as_tensor(rows, device=DEVICE)
        cols = torch.as_tensor(cols, device=DEVICE)
        p = torch.as_tensor(p, dtype=torch.float32, device=DEVICE)
        c = torch.as_tensor(c, dtype=torch.float32, device=DEVICE)
        pred = (X[rows] * Y[cols]).sum(1)
        background = torch.sum((X.T @ X) * (Y.T @ Y))
        observed = (c * (p - pred) ** 2 - pred ** 2).sum()
        return float(background + observed + self.reg * ((X ** 2).sum() + (Y ** 2).sum()))


class IALS(Recommender):

    name = "iALS"

    def __init__(self, factors=64, reg=1.0, alpha=10.0, iters=15, cg_steps=4, seed=42):
        self.alpha = alpha
        self.solver = ImplicitALS(factors, reg, iters, cg_steps, seed)

    def signal(self, history):
        pairs = history.pairs
        n = pairs.height
        return (pairs["user_idx"].to_numpy(), pairs["item_idx"].to_numpy(),
                np.ones(n, np.float32), np.full(n, 1.0 + self.alpha, np.float32))

    def fit(self, history):
        rows, cols, p, c = self.signal(history)
        self.solver.fit(rows, cols, p, c, history.n_users, history.n_items)
        return self

    def score(self, users):
        u = torch.as_tensor(users, device=DEVICE)
        return self.solver.X[u] @ self.solver.Y.T
