from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TimeRange:
    start: int
    end: int


class GraphWindowDataset(Dataset):

    def __init__(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        split: Tuple[int, int],
        spec,
        allow_history: bool,
    ) -> None:
        assert x_values.ndim == 3, f"x_values must be [N,T,F], got {x_values.shape}"
        assert y_values.ndim == 3 and y_values.shape[-1] == 1, f"y_values must be [N,T,1], got {y_values.shape}"
        self.x = x_values
        self.y = y_values
        self.N, self.T, self.F = x_values.shape
        self.seq_len = int(spec.seq_len)
        self.pred_len = int(spec.pred_len)
        self.split = (int(split[0]), int(split[1]))
        self.allow_history = bool(allow_history)

        self._t0_list = self._make_t0_list()

    def _make_t0_list(self):
        start, end = self.split
        L, H = self.seq_len, self.pred_len

        t0_list = []
        t_pred_min = start
        t_pred_max = end - H 

        if t_pred_max < t_pred_min:
            return []
        if self.allow_history:
            t_pred_min = max(t_pred_min, L)
        else:
            t_pred_min = max(t_pred_min, start + L)
            t_pred_min = max(t_pred_min, L)

        for t_pred in range(t_pred_min, t_pred_max + 1):
            t0 = t_pred - L
            if t0 < 0:
                continue
            # encoder end t0+L == t_pred
            # horizon: [t_pred, t_pred+H)
            if t_pred + H > self.T:
                continue
            t0_list.append(t0)

        return t0_list

    def __len__(self) -> int:
        return len(self._t0_list)

    def __getitem__(self, idx: int):
        t0 = self._t0_list[idx]
        L, H = self.seq_len, self.pred_len

        x_win = self.x[:, t0 : t0 + L, :]            # [N,L,F]
        y_win = self.y[:, t0 + L : t0 + L + H, :]    # [N,H,1]

        # Return as torch tensors with layout [L,N,F] and [H,N,1]
        x_t = torch.from_numpy(np.ascontiguousarray(x_win.transpose(1, 0, 2))).float()
        y_t = torch.from_numpy(np.ascontiguousarray(y_win.transpose(1, 0, 2))).float()

        meta = {"t0": int(t0), "t_pred_start": int(t0 + L)}
        return {"x": x_t, "y": y_t, "meta": meta}


class GraphWindowCollate:
    """Collate: stack dicts into a batch.

    x: [B,L,N,F]
    y: [B,H,N,1]
    meta: list[dict]
    """

    def __call__(self, batch):
        xs = torch.stack([b["x"] for b in batch], dim=0)
        ys = torch.stack([b["y"] for b in batch], dim=0)
        meta = [b["meta"] for b in batch]
        return {"x": xs, "y": ys, "meta": meta}
