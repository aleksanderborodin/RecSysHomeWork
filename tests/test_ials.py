import numpy as np
import torch

from models.ials import ImplicitALS


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    n_u, n_i = 30, 20
    mask = rng.random((n_u, n_i)) < 0.25
    rows, cols = np.nonzero(mask)
    p = np.where(rng.random(len(rows)) < 0.8, 1.0, 0.0).astype(np.float32)
    c = (1 + rng.uniform(1, 20, len(rows))).astype(np.float32)
    return rows, cols, p, c, n_u, n_i


def test_cg_half_step_matches_exact_solve():
    rows, cols, p, c, n_u, n_i = _toy()
    m = ImplicitALS(factors=4, reg=0.5, iters=1, cg_steps=4)
    m.fit(rows, cols, p, c, n_u, n_i)
    Y = m.Y.double().cpu().numpy()

    m.cg_steps = 50
    X_cg = m._half_step(m.X, m.Y, torch.as_tensor(rows, device=m.X.device), torch.as_tensor(cols, device=m.X.device),
                        torch.as_tensor(p, device=m.X.device), torch.as_tensor(c, device=m.X.device)).double().cpu().numpy()
    for u in range(n_u):
        sel = rows == u
        Yi, ci, pi = Y[cols[sel]], c[sel], p[sel]
        A = Y.T @ Y + (Yi.T * (ci - 1)) @ Yi + 0.5 * np.eye(4)
        b = Yi.T @ (ci * pi)
        assert np.allclose(X_cg[u], np.linalg.solve(A, b), atol=1e-4)


def test_loss_decreases():
    rows, cols, p, c, n_u, n_i = _toy(1)
    losses = []
    for iters in (1, 3, 8):
        m = ImplicitALS(factors=4, reg=0.5, iters=iters, cg_steps=8).fit(rows, cols, p, c, n_u, n_i)
        losses.append(m.loss(rows, cols, p, c))
    assert losses[0] > losses[1] > losses[2]
