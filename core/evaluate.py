from __future__ import annotations

import numpy as np
import torch

import config
from core import metrics
from core.split import Stage

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def recommend(model, stage: Stage, users: np.ndarray, k: int = config.K, batch: int = 2048) -> np.ndarray:

    hist = stage.history
    seen = hist.seen
    banned = torch.as_tensor(~hist.catalog, device=DEVICE)
    out = np.empty((len(users), k), dtype=np.int64)
    for start in range(0, len(users), batch):
        u = users[start:start + batch]
        s = model.score(u)
        scores = (s.to(DEVICE, torch.float32).clone() if isinstance(s, torch.Tensor)
                  else torch.as_tensor(np.array(s, dtype=np.float32), device=DEVICE))
        scores[:, banned] = -torch.inf
        block = seen[u].tocoo()
        scores[torch.as_tensor(block.row, device=DEVICE), torch.as_tensor(block.col, device=DEVICE)] = -torch.inf
        top = torch.topk(scores, k, dim=1)
        assert torch.isfinite(top.values).all(), "у пользователя меньше K допустимых кандидатов"
        out[start:start + len(u)] = top.indices.cpu().numpy()
    return out


def evaluate(model, stage: Stage, k: int = config.K, return_recs: bool = False) -> dict:
    users = stage.users
    top = recommend(model, stage, users, k)
    t = stage.targets
    targets = [t.indices[t.indptr[u]:t.indptr[u + 1]] for u in users]
    per = metrics.per_user(top, targets, k)
    pop = stage.history.popularity
    pop_rank = np.argsort(np.argsort(-pop)) / len(pop)
    result = {f"{m}@{k}": float(v.mean()) for m, v in per.items()}
    result[f"coverage@{k}"] = len(np.unique(top)) / int(stage.history.catalog.sum())
    result["pop_rank_mean"] = float(pop_rank[top].mean())
    result["users"] = len(users)
    result["per_user"] = per
    if return_recs:
        result["recs"] = top
    return result


def summary_row(name: str, res: dict) -> dict:
    return {"model": name, **{k: v for k, v in res.items() if k not in ("per_user", "recs")}}
