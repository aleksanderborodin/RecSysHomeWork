import numpy as np
import scipy.sparse as sp

from models.ease import ease_weights


def test_ease_matches_constrained_least_squares():
    rng = np.random.default_rng(0)
    X = (rng.random((60, 8)) < 0.3).astype(float)
    reg = 3.0
    B = ease_weights(sp.csr_matrix(X), reg)
    assert np.allclose(np.diag(B), 0)

    for j in range(X.shape[1]):
        others = [i for i in range(X.shape[1]) if i != j]
        A = X[:, others]
        w = np.linalg.solve(A.T @ A + reg * np.eye(len(others)), A.T @ X[:, j])
        assert np.allclose(B[others, j], w, atol=1e-8)
