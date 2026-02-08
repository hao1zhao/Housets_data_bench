from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from housets_bench.bundles.datatypes import ProcBundle
from housets_bench.metrics.regression import logrmse as _logrmse, mape as _mape
from housets_bench.models.base import BaseForecaster


def _to_numpy(x: Any) -> np.ndarray:
    try:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


@dataclass
class EvalResult:
    logrmse: float
    mape: float
    n_points: int


class StreamingEvaluator:
    def __init__(self, bundle: ProcBundle, *, eps: float = 1e-8) -> None:
        self.bundle = bundle
        self.eps = float(eps)

        # mapping from y_cols (model output space) -> processed full feature indices
        proc_names = list(bundle.aligned_proc.schema.continuous_cols)
        name_to_idx = {n: i for i, n in enumerate(proc_names)}
        self.y_idx_full = [name_to_idx[n] for n in bundle.y_cols]

        self.d_proc = int(bundle.aligned_proc.values.shape[-1])
        self.raw_target_index = int(bundle.raw_target_index)

        self._sse_log = 0.0
        self._sum_ape = 0.0
        self._n = 0

    def update(self, y_true_proc: Any, y_pred_proc: Any) -> None:
        yt = _to_numpy(y_true_proc).astype(np.float32)
        yp = _to_numpy(y_pred_proc).astype(np.float32)

        if yt.shape != yp.shape:
            raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
        if yt.ndim != 3:
            raise ValueError(f"Expected [B,H,Dy], got {yt.shape}")

        B, H, Dy = yt.shape

        # Embed into full processed feature dimension
        full_t = np.zeros((B, H, self.d_proc), dtype=np.float32)
        full_p = np.zeros((B, H, self.d_proc), dtype=np.float32)
        full_t[:, :, self.y_idx_full] = yt
        full_p[:, :, self.y_idx_full] = yp

        # 1) Log space (invert everything EXCEPT log stage)
        t_log_full = self.bundle.pipeline.inverse(full_t, keep_log=True)
        p_log_full = self.bundle.pipeline.inverse(full_p, keep_log=True)
        t_log = t_log_full[:, :, self.raw_target_index]
        p_log = p_log_full[:, :, self.raw_target_index]

        diff = (p_log.astype(np.float64) - t_log.astype(np.float64))
        self._sse_log += float(np.sum(diff * diff))
        self._n += int(diff.size)

        # 2) Original space for MAPE
        t_raw_full = self.bundle.pipeline.inverse(full_t, keep_log=False)
        p_raw_full = self.bundle.pipeline.inverse(full_p, keep_log=False)
        t_raw = t_raw_full[:, :, self.raw_target_index].astype(np.float64)
        p_raw = p_raw_full[:, :, self.raw_target_index].astype(np.float64)

        denom = np.maximum(np.abs(t_raw), self.eps)
        self._sum_ape += float(np.sum(np.abs(p_raw - t_raw) / denom))

    def compute(self) -> EvalResult:
        if self._n == 0:
            return EvalResult(logrmse=float("nan"), mape=float("nan"), n_points=0)
        logrmse = float(np.sqrt(self._sse_log / self._n))
        mape = float(self._sum_ape / self._n)
        return EvalResult(logrmse=logrmse, mape=mape, n_points=self._n)


@torch.no_grad()
def evaluate_forecaster(
    model: BaseForecaster,
    bundle: ProcBundle,
    *,
    split: str = "test",
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
) -> EvalResult:
    split = split.lower()
    if split not in ("train", "val", "test"):
        raise ValueError("split must be one of: train/val/test")

    dl = bundle.dataloaders[split]
    evaluator = StreamingEvaluator(bundle)

    pred_len = int(bundle.raw.spec.pred_len)

    for bi, batch in enumerate(dl):
        if max_batches is not None and bi >= max_batches:
            break

        # batch['y'] includes [label_len+pred_len]; we evaluate only the forecast horizon
        y_true = batch["y"][:, -pred_len:, :]

        y_pred = model.predict_batch(batch, bundle=bundle, device=device)

        evaluator.update(y_true, y_pred)

    return evaluator.compute()
