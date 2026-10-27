import numpy as np
import scipy.sparse as sp
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dense_rows(m: sp.csr_matrix, rows: np.ndarray) -> torch.Tensor:

    block = m[rows].tocoo()
    out = torch.zeros((len(rows), m.shape[1]), device=DEVICE)
    out[torch.as_tensor(block.row, device=DEVICE), torch.as_tensor(block.col, device=DEVICE)] = \
        torch.as_tensor(block.data, dtype=torch.float32, device=DEVICE)
    return out
