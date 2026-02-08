from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from housets_bench.data.io import load_aligned
from housets_bench.data.split import make_ratio_split
from housets_bench.data.windowing import make_window_spec
from housets_bench.data.graph_dataset import GraphWindowDataset, GraphWindowCollate
from housets_bench.graph.geo_knn import build_knn_geo_graph
from housets_bench.graph.torch_adj import (
    edges_from_graph,
    read_zip_latlon_csv,
    sparse_adj,
    normalize_adj_sym,
)
from housets_bench.models.gnn.stgcn import STGCN


@dataclass
class Metrics:
    logrmse: float
    mape: float
    n_points: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--data", type=str, default=str(REPO_ROOT / "data" / "raw" / "HouseTS.csv"))
    p.add_argument("--zip-latlon", type=str, default=str(REPO_ROOT / "data" / "processed" / "zip_latlon.csv"))
    p.add_argument("--target-col", type=str, default="price")
    p.add_argument("--features-mode", type=str, default="S", choices=["S", "MS"], help="S: x=price; MS: x=all cont")

    p.add_argument("--seq-len", type=int, default=6)
    p.add_argument("--label-len", type=int, default=0, help="unused by STGCN (keep 0)")
    p.add_argument("--pred-len", type=int, default=3)

    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.1)

    # preprocessing ablation
    p.add_argument(
        "--preproc-mode",
        type=str,
        default="log",
        choices=["nolog", "log", "log_clip", "log_clip_zscore"],
        help="preprocessing mode for ablation",
    )
    p.add_argument("--clip-q-lower", type=float, default=0.001)
    p.add_argument("--clip-q-upper", type=float, default=0.999)
    p.add_argument("--zscore-scope", type=str, default="global", choices=["global"])
    p.add_argument("--zscore-eps", type=float, default=1e-6)

    # graph
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--max-km", type=float, default=50.0, help="<=0 disables threshold")
    p.add_argument("--symmetric", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--self-loops", action=argparse.BooleanOptionalAction, default=True)

    # model
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--n-blocks", type=int, default=2)
    p.add_argument("--kt", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)

    # training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-train-batches", type=int, default=0, help="<=0 means all")
    p.add_argument("--max-eval-batches", type=int, default=0, help="<=0 means all")
    p.add_argument("--device", type=str, default="", help="e.g. cuda, cuda:0")

    # speed controls
    p.add_argument("--n-zip", type=int, default=0, help="<=0 means all ZIPs")

    # output
    p.add_argument("--save-json", type=str, default="", help="if set, write final JSON to this path")

    return p.parse_args()


def _sync(dev: Optional[torch.device]) -> None:
    if dev is not None and dev.type == "cuda":
        torch.cuda.synchronize()


def _log_epoch_enabled() -> bool:
    return os.getenv("HOUSETS_LOG_EPOCH", "0") == "1"


def _preproc_flags(mode: str) -> Tuple[bool, bool, bool]:
    """Return (log_enabled, clip_enabled, zscore_enabled)."""
    mode = mode.lower().strip()
    if mode == "nolog":
        return False, False, False
    if mode == "log":
        return True, False, False
    if mode == "log_clip":
        return True, True, False
    if mode == "log_clip_zscore":
        return True, True, True
    raise ValueError(f"Unknown preproc-mode: {mode}")


def _fit_clip_bounds(
    x_stage: np.ndarray,
    train_range: Tuple[int, int],
    clip_features: List[int],
    q_lower: float,
    q_upper: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit per-feature quantile clip bounds on train slice. x_stage: [N,T,F]."""
    N, T, F = x_stage.shape
    lo = np.full((F,), -np.inf, dtype=np.float32)
    hi = np.full((F,), np.inf, dtype=np.float32)
    t0, t1 = train_range
    x_tr = x_stage[:, t0:t1, :]  # [N,Tr,F]

    for j in clip_features:
        v = x_tr[:, :, j].reshape(-1)
        # handle NaN just in case
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        lo[j] = float(np.quantile(v, q_lower))
        hi[j] = float(np.quantile(v, q_upper))
    return lo, hi


def _apply_clip(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    x2 = x.copy()
    F = x2.shape[-1]
    for j in range(F):
        if np.isfinite(lo[j]) or np.isfinite(hi[j]):
            x2[:, :, j] = np.clip(x2[:, :, j], lo[j], hi[j])
    return x2


def _fit_zscore_stats(x_stage: np.ndarray, train_range: Tuple[int, int], eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Fit per-feature mean/std on train slice. x_stage: [N,T,F]."""
    t0, t1 = train_range
    x_tr = x_stage[:, t0:t1, :]  # [N,Tr,F]
    mu = np.mean(x_tr, axis=(0, 1)).astype(np.float32)  # [F]
    sd = np.std(x_tr, axis=(0, 1)).astype(np.float32)   # [F]
    sd = np.maximum(sd, float(eps)).astype(np.float32)
    return mu, sd


def _fit_zscore_target(y_stage: np.ndarray, train_range: Tuple[int, int], eps: float) -> Tuple[float, float]:
    """Fit mean/std for target only. y_stage: [N,T,1]."""
    t0, t1 = train_range
    y_tr = y_stage[:, t0:t1, 0].reshape(-1)
    y_tr = y_tr[np.isfinite(y_tr)]
    mu = float(np.mean(y_tr)) if y_tr.size else 0.0
    sd = float(np.std(y_tr)) if y_tr.size else 1.0
    sd = max(sd, float(eps))
    return mu, sd


def _build_preproc(
    *,
    x_raw: np.ndarray,              # [N,T,Fx]
    y_raw: np.ndarray,              # [N,T,1]
    train_range: Tuple[int, int],
    preproc_mode: str,
    price_feat_in_x: int,
    features_mode: str,
    clip_q_lower: float,
    clip_q_upper: float,
    zscore_eps: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object], str, Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:

    log_enabled, clip_enabled, zscore_enabled = _preproc_flags(preproc_mode)

    if log_enabled:
        x1 = np.log1p(np.clip(x_raw, 0.0, None))
        y1 = np.log1p(np.clip(y_raw, 0.0, None))
    else:
        x1 = x_raw.astype(np.float32, copy=True)
        y1 = y_raw.astype(np.float32, copy=True)

    clip_lo = None
    clip_hi = None
    if clip_enabled:
        if features_mode.upper() == "MS":
            clip_features = [j for j in range(x1.shape[-1]) if j != int(price_feat_in_x)]
        else:
            clip_features = []  # S => only target present, target not clipped
        lo, hi = _fit_clip_bounds(
            x_stage=x1,
            train_range=train_range,
            clip_features=clip_features,
            q_lower=float(clip_q_lower),
            q_upper=float(clip_q_upper),
        )
        x2 = _apply_clip(x1, lo, hi)
        clip_lo, clip_hi = lo, hi
    else:
        x2 = x1

    if zscore_enabled:
        mu_x, sd_x = _fit_zscore_stats(x2, train_range=train_range, eps=float(zscore_eps))
        x3 = (x2 - mu_x.reshape(1, 1, -1)) / sd_x.reshape(1, 1, -1)
        mu_y, sd_y = _fit_zscore_target(y1, train_range=train_range, eps=float(zscore_eps))
        y3 = (y1 - float(mu_y)) / float(sd_y)
    else:
        mu_x = np.zeros((x2.shape[-1],), dtype=np.float32)
        sd_x = np.ones((x2.shape[-1],), dtype=np.float32)
        mu_y, sd_y = 0.0, 1.0
        x3 = x2
        y3 = y1

    # pipeline desc
    if preproc_mode == "nolog":
        pipeline_desc = "nolog"
    elif preproc_mode == "log":
        pipeline_desc = "log1p"
    elif preproc_mode == "log_clip":
        pipeline_desc = f"log1p -> clip(q={clip_q_lower},{clip_q_upper})"
    else:
        pipeline_desc = f"log1p -> clip(q={clip_q_lower},{clip_q_upper}) -> zscore(scope=global)"

    meta: Dict[str, object] = {
        "mode": preproc_mode,
        "log_enabled": bool(log_enabled),
        "clip_enabled": bool(clip_enabled),
        "zscore_enabled": bool(zscore_enabled),
        "clip_q_lower": float(clip_q_lower),
        "clip_q_upper": float(clip_q_upper),
        "zscore_scope": "global",
        "zscore_eps": float(zscore_eps),
        "clip_applied_to": "x_non_target_features_only",
        "zscore_applied_to": "x_all_features_and_y_target",
    }
    return x3.astype(np.float32), y3.astype(np.float32), meta, pipeline_desc, (mu_y, sd_y), (mu_x, sd_x)


def _make_inverse_target(
    *,
    device: Optional[torch.device],
    log_enabled: bool,
    zscore_enabled: bool,
    mu_y: float,
    sd_y: float,
) :
    mu = torch.tensor(float(mu_y), dtype=torch.float32, device=device) if device is not None else float(mu_y)
    sd = torch.tensor(float(sd_y), dtype=torch.float32, device=device) if device is not None else float(sd_y)

    def inv(t: torch.Tensor) -> torch.Tensor:
        x = t
        if zscore_enabled:
            x = x * sd + mu
        if log_enabled:
            x = torch.expm1(x)
        return x

    return inv


@torch.no_grad()
def _evaluate_raw(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    A_norm,
    inv_y,
    device: Optional[torch.device],
    max_batches: Optional[int],
    eps: float = 1e-6,
) -> Metrics:
    se_sum = 0.0
    mape_sum = 0.0
    n = 0

    model.eval()
    for bidx, batch in enumerate(dataloader):
        if max_batches is not None and bidx >= int(max_batches):
            break

        x = batch["x"]
        y = batch["y"]
        if device is not None:
            x = x.to(device)
            y = y.to(device)

        y_hat = model(x, A_norm)

        # inverse to RAW
        y_raw = inv_y(y)
        yhat_raw = inv_y(y_hat)

        # logrmse on log1p
        log_true = torch.log1p(torch.clamp(y_raw, min=0.0))
        log_pred = torch.log1p(torch.clamp(yhat_raw, min=0.0))
        diff = (log_pred - log_true).float()
        se_sum += float(torch.sum(diff * diff).item())

        denom = torch.clamp(torch.abs(y_raw), min=eps)
        mape_sum += float(torch.sum(torch.abs(yhat_raw - y_raw) / denom).item())

        n += int(y_raw.numel())

    logrmse = float(np.sqrt(se_sum / max(n, 1)))
    mape = float(mape_sum / max(n, 1))
    return Metrics(logrmse=logrmse, mape=mape, n_points=int(n))


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    latlon_path = Path(args.zip_latlon)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if not latlon_path.exists():
        raise FileNotFoundError(f"zip_latlon not found: {latlon_path}")
    if args.label_len != 0:
        print("[warn] STGCN ignores --label-len; using 0.", flush=True)

    aligned = load_aligned(str(data_path), target_col=args.target_col, impute=True)

    # optional ZIP subsample
    if args.n_zip > 0 and aligned.n_zip > args.n_zip:
        zips = aligned.zipcodes[: args.n_zip]
        zip_mask = np.isin(np.array(aligned.zipcodes), np.array(zips))
        aligned.zipcodes = list(np.array(aligned.zipcodes)[zip_mask])
        aligned.values = aligned.values[zip_mask]

    cont_cols = list(aligned.schema.continuous_cols)
    if args.target_col not in cont_cols:
        raise ValueError("target_col not in continuous_cols")
    price_idx = cont_cols.index(args.target_col)

    if args.features_mode.upper() == "S":
        x_idx = [price_idx]
    else:
        x_idx = list(range(len(cont_cols)))

    # price position inside x feature dimension
    price_feat_in_x = x_idx.index(price_idx)

    x_raw = aligned.values[:, :, x_idx].astype(np.float32)          # [N,T,Fx]
    y_raw = aligned.values[:, :, [price_idx]].astype(np.float32)    # [N,T,1]

    split = make_ratio_split(aligned.n_time, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    spec = make_window_spec(seq_len=args.seq_len, pred_len=args.pred_len, label_len=0)
    train_range = split.train  # (t0,t1)

    # preprocessing
    x_proc, y_proc, preproc_meta, pipeline_desc, (mu_y, sd_y), _ = _build_preproc(
        x_raw=x_raw,
        y_raw=y_raw,
        train_range=train_range,
        preproc_mode=args.preproc_mode,
        price_feat_in_x=price_feat_in_x,
        features_mode=args.features_mode,
        clip_q_lower=args.clip_q_lower,
        clip_q_upper=args.clip_q_upper,
        zscore_eps=args.zscore_eps,
    )
    log_enabled, clip_enabled, zscore_enabled = _preproc_flags(args.preproc_mode)

    # build geo graph
    latlon_by_zip = read_zip_latlon_csv(latlon_path)
    graph = build_knn_geo_graph(
        aligned.zipcodes,
        latlon_by_zip,
        k=int(args.k),
        max_km=(float(args.max_km) if args.max_km > 0 else None),
        include_self_loops=bool(args.self_loops),
        symmetric=bool(args.symmetric),
    )
    src, dst, w = edges_from_graph(graph)

    dev = torch.device(args.device) if str(args.device).strip() else None

    # sparse adj normalized
    A = sparse_adj(src, dst, n_nodes=len(aligned.zipcodes), weight=w, device=dev)
    A_norm = normalize_adj_sym(A)

    # datasets
    ds_train = GraphWindowDataset(x_proc, y_proc, split=split.train, spec=spec, allow_history=False)
    ds_val = GraphWindowDataset(x_proc, y_proc, split=split.val, spec=spec, allow_history=True)
    ds_test = GraphWindowDataset(x_proc, y_proc, split=split.test, spec=spec, allow_history=True)

    collate = GraphWindowCollate()
    dl_train = DataLoader(ds_train, batch_size=int(args.batch_size), shuffle=True, num_workers=0, collate_fn=collate)
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)

    model = STGCN(
        input_dim=int(x_proc.shape[-1]),
        hidden_dim=int(args.hidden_dim),
        pred_len=int(args.pred_len),
        n_blocks=int(args.n_blocks),
        Kt=int(args.kt),
        dropout=float(args.dropout),
    )
    if dev is not None:
        model.to(dev)

    optim = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_fn = torch.nn.MSELoss()

    max_train_batches = None if args.max_train_batches <= 0 else int(args.max_train_batches)
    max_eval_batches = None if args.max_eval_batches <= 0 else int(args.max_eval_batches)

    log_epoch = _log_epoch_enabled()
    train_history = []
    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    t_fit0 = time.perf_counter()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        _sync(dev)
        t0 = time.perf_counter()

        train_losses = []
        for bidx, batch in enumerate(dl_train):
            if max_train_batches is not None and bidx >= max_train_batches:
                break
            x = batch["x"]
            y = batch["y"]
            if dev is not None:
                x = x.to(dev)
                y = y.to(dev)

            optim.zero_grad(set_to_none=True)
            y_hat = model(x, A_norm)
            loss = loss_fn(y_hat, y)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            optim.step()
            train_losses.append(float(loss.detach().cpu().item()))

        train_mse = float(np.mean(train_losses)) if train_losses else float("nan")

        # val mse (processed space)
        model.eval()
        with torch.no_grad():
            val_losses = []
            for bidx, batch in enumerate(dl_val):
                if max_eval_batches is not None and bidx >= max_eval_batches:
                    break
                x = batch["x"]
                y = batch["y"]
                if dev is not None:
                    x = x.to(dev)
                    y = y.to(dev)
                y_hat = model(x, A_norm)
                val_losses.append(float(loss_fn(y_hat, y).detach().cpu().item()))
            val_mse = float(np.mean(val_losses)) if val_losses else float("nan")

        _sync(dev)
        dt = time.perf_counter() - t0

        if log_epoch:
            print(
                f"[stgcn:{args.features_mode}|{args.preproc_mode}] epoch {epoch}/{args.epochs} "
                f"train_mse={train_mse:.6g} val_mse={val_mse:.6g} epoch_time={dt:.2f}s",
                flush=True,
            )

        train_history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse, "epoch_time_sec": float(dt)})

        if val_mse + 1e-12 < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(args.patience):
                break

    fit_sec = time.perf_counter() - t_fit0
    if best_state is not None:
        model.load_state_dict(best_state)
        if dev is not None:
            model.to(dev)

    inv_y = _make_inverse_target(
        device=dev,
        log_enabled=log_enabled,
        zscore_enabled=zscore_enabled,
        mu_y=mu_y,
        sd_y=sd_y,
    )

    # metrics 
    _sync(dev)
    t0 = time.perf_counter()
    val = _evaluate_raw(model=model, dataloader=dl_val, A_norm=A_norm, inv_y=inv_y, device=dev, max_batches=max_eval_batches)
    _sync(dev)
    val_sec = time.perf_counter() - t0

    _sync(dev)
    t0 = time.perf_counter()
    test = _evaluate_raw(model=model, dataloader=dl_test, A_norm=A_norm, inv_y=inv_y, device=dev, max_batches=max_eval_batches)
    _sync(dev)
    test_sec = time.perf_counter() - t0

    graph_info = {
        "type": "geo_knn",
        "k": int(args.k),
        "max_km": float(args.max_km),
        "symmetric": bool(args.symmetric),
        "self_loops": bool(args.self_loops),
        "n_nodes": int(len(aligned.zipcodes)),
        "n_edges": int(len(src)),
    }

    result: Dict[str, object] = {
        "model": "stgcn",
        "features_mode": args.features_mode,
        "preproc": preproc_meta,
        "graph": graph_info,
        "val": {"logrmse": val.logrmse, "mape": val.mape, "n_points": val.n_points},
        "test": {"logrmse": test.logrmse, "mape": test.mape, "n_points": test.n_points},
        "n_val": int(len(ds_val)),
        "n_test": int(len(ds_test)),
        "timing": {"fit_sec": float(fit_sec), "val_eval_sec": float(val_sec), "test_eval_sec": float(test_sec)},
        "hparams": {
            "seq_len": int(args.seq_len),
            "pred_len": int(args.pred_len),
            "hidden_dim": int(args.hidden_dim),
            "n_blocks": int(args.n_blocks),
            "kt": int(args.kt),
            "dropout": float(args.dropout),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "grad_clip": float(args.grad_clip),
            "patience": int(args.patience),
            "batch_size": int(args.batch_size),
        },
        "pipeline": pipeline_desc,
        "train_history": train_history,
    }

    # write json if requested
    if str(args.save_json).strip():
        outp = Path(args.save_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
