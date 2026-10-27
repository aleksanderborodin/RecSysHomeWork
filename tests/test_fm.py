import itertools

import numpy as np
import scipy.sparse as sp
import torch

from models.fm import FMCore, _sparse


def _brute_fm(x, w, V):
    s = float(w @ x)
    for a, b in itertools.combinations(range(len(x)), 2):
        s += float(V[a] @ V[b]) * x[a] * x[b]
    return s


def test_split_scoring_equals_full_fm_up_to_user_constant():
    rng = np.random.default_rng(0)
    n_f, d = 9, 3
    Xu = sp.csr_matrix(np.where(rng.random((2, n_f)) < 0.3, rng.random((2, n_f)), 0) * (np.arange(n_f) < 4))
    Xi = sp.csr_matrix(np.where(rng.random((5, n_f)) < 0.5, rng.random((5, n_f)), 0) * (np.arange(n_f) >= 4))
    m = FMCore(n_f, d, seed=0).to(torch.device("cpu"))
    with torch.no_grad():
        m.w.copy_(torch.randn(n_f))
    W, V = m.w.detach().double().numpy(), m.V.detach().double().numpy()
    dev = lambda t: t.to("cpu")
    import models.fm as fm_mod
    old = fm_mod.DEVICE
    fm_mod.DEVICE = torch.device("cpu")
    try:
        eu, tu = m.side(_sparse(Xu), _sparse(Xu, True))
        ei, ti = m.side(_sparse(Xi), _sparse(Xi, True))
    finally:
        fm_mod.DEVICE = old
    fast = (eu @ ei.T + ti[None, :] + tu[:, None]).detach().double().numpy()
    for u in range(2):
        for i in range(5):
            x = (Xu[u] + Xi[i]).toarray().ravel()
            assert np.isclose(fast[u, i], _brute_fm(x, W, V), atol=1e-5)


def test_bpr_gradient_matches_finite_differences():
    torch.manual_seed(0)
    V = torch.randn(6, 3, dtype=torch.double, requires_grad=True)
    xu, xi, xj = torch.tensor([1., 0, 0, 0, 0, 0]), torch.tensor([0, 1., .5, 0, 0, 0]), torch.tensor([0, 0, 0, 1., 0, .7])

    def loss(V):
        e = lambda x: x.double() @ V
        t = lambda x: 0.5 * ((x.double() @ V) ** 2).sum() - 0.5 * ((x.double() ** 2) @ (V ** 2).sum(1))
        delta = t(xi) - t(xj) + e(xu) @ (e(xi) - e(xj))
        return torch.nn.functional.softplus(-delta)

    loss(V).backward()
    eps = 1e-6
    num = torch.zeros_like(V)
    for a in range(6):
        for f in range(3):
            Vp, Vm = V.detach().clone(), V.detach().clone()
            Vp[a, f] += eps
            Vm[a, f] -= eps
            num[a, f] = (loss(Vp) - loss(Vm)) / (2 * eps)
    assert torch.allclose(V.grad, num, atol=1e-6)
