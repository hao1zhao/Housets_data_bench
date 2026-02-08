from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch


def read_zip_latlon_csv(path: str | Path) -> Dict[str, Tuple[float, float]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"zip_latlon csv not found: {p}")

    df = pd.read_csv(p)
    if "zipcode" not in df.columns and "zip" in df.columns:
        df = df.rename(columns={"zip": "zipcode"})
    need = {"zipcode", "lat", "lon"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"zip_latlon csv missing columns: {missing}. Need {need}")

    df["zipcode"] = df["zipcode"].astype(str).str.zfill(5)
    df = df.dropna(subset=["lat", "lon"])

    return {z: (float(lat), float(lon)) for z, lat, lon in zip(df["zipcode"], df["lat"], df["lon"])}


def edges_from_graph(graph) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if hasattr(graph, "src") and hasattr(graph, "dst"):
        src = np.asarray(getattr(graph, "src"), dtype=np.int64)
        dst = np.asarray(getattr(graph, "dst"), dtype=np.int64)
        w = None
        for name in ("weight", "weights", "w"):
            if hasattr(graph, name):
                w = np.asarray(getattr(graph, name), dtype=np.float32)
                break
        return src, dst, w

    if hasattr(graph, "edge_index"):
        ei = np.asarray(getattr(graph, "edge_index"))
        if ei.ndim == 2 and ei.shape[0] == 2:
            src, dst = ei[0], ei[1]
        else:
            src, dst = ei[:, 0], ei[:, 1]
        return np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64), None

    if hasattr(graph, "edges"):
        edges = getattr(graph, "edges")
        src = np.fromiter((e[0] for e in edges), dtype=np.int64, count=len(edges))
        dst = np.fromiter((e[1] for e in edges), dtype=np.int64, count=len(edges))
        return src, dst, None

    raise AttributeError("GeoGraph has no (src,dst) nor edge_index nor edges fields.")


def sparse_adj(
    src: np.ndarray,
    dst: np.ndarray,
    n_nodes: int,
    *,
    weight: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    dev = device if device is not None else torch.device("cpu")
    src_t = torch.as_tensor(src, dtype=torch.long, device=dev)
    dst_t = torch.as_tensor(dst, dtype=torch.long, device=dev)
    idx = torch.stack([src_t, dst_t], dim=0)

    if weight is None:
        val = torch.ones(src_t.numel(), dtype=dtype, device=dev)
    else:
        val = torch.as_tensor(weight, dtype=dtype, device=dev)

    A = torch.sparse_coo_tensor(idx, val, size=(n_nodes, n_nodes)).coalesce()
    return A


def normalize_adj_sym(A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    A = A.coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense()
    deg_inv_sqrt = deg.clamp_min(eps).pow(-0.5)
    idx = A.indices()
    val = A.values()
    val = val * deg_inv_sqrt[idx[0]] * deg_inv_sqrt[idx[1]]
    return torch.sparse_coo_tensor(idx, val, size=A.shape).coalesce()


def normalize_adj_rw(A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    A = A.coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense()
    deg_inv = deg.clamp_min(eps).pow(-1.0)
    idx = A.indices()
    val = A.values()
    val = val * deg_inv[idx[0]]
    return torch.sparse_coo_tensor(idx, val, size=A.shape).coalesce()


def spmm_nt(A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    # X_flat: [N, B*T*C]
    B, T, N, C = X.shape
    X_flat = X.permute(2, 0, 1, 3).reshape(N, B * T * C)
    Y_flat = torch.sparse.mm(A, X_flat)
    Y = Y_flat.reshape(N, B, T, C).permute(1, 2, 0, 3).contiguous()
    return Y


def spmm_nct(A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    B, C, N, T = X.shape
    X_flat = X.permute(2, 0, 1, 3).reshape(N, B * C * T)
    Y_flat = torch.sparse.mm(A, X_flat)
    Y = Y_flat.reshape(N, B, C, T).permute(1, 2, 0, 3).contiguous()
    return Y
