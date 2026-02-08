from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

import torch


@dataclass
class GraphEvalResult:
    logrmse: float
    mape: float
    n_points: int


@torch.no_grad()
def evaluate_graph_forecaster(
    model: torch.nn.Module,
    dataloader,
    *,
    A_norm: Optional[Any] = None,
    adj: Optional[Any] = None,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
    eps: float = 1e-8,
) -> GraphEvalResult:
    A = A_norm if A_norm is not None else adj

    model.eval()

    se_sum = 0.0
    ape_sum = 0.0
    n = 0

    for bidx, batch in enumerate(dataloader):
        if max_batches is not None and bidx >= max_batches:
            break

        x = batch["x"]
        y = batch["y"]

        if device is not None:
            x = x.to(device)
            y = y.to(device)

        if A is None:
            y_hat = model(x)
        else:
            y_hat = model(x, A)

        if y_hat.shape != y.shape:
            raise ValueError(f"pred shape {tuple(y_hat.shape)} != true shape {tuple(y.shape)}")

        diff = (y_hat - y).float()
        se_sum += float(diff.pow(2).sum().detach().cpu().item())

        # MAPE in original space 
        y_raw = torch.expm1(y.float())
        yhat_raw = torch.expm1(y_hat.float())
        denom = y_raw.abs().clamp_min(eps)
        ape_sum += float(((yhat_raw - y_raw).abs() / denom).sum().detach().cpu().item())

        n += int(y.numel())

    if n == 0:
        return GraphEvalResult(logrmse=float("nan"), mape=float("nan"), n_points=0)

    mse = se_sum / n
    return GraphEvalResult(logrmse=float(mse**0.5), mape=float(ape_sum / n), n_points=int(n))
