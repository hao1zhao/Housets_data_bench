from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from housets_bench.data.io import load_aligned
from housets_bench.data.split import make_ratio_split
from housets_bench.data.windowing import make_window_spec
from housets_bench.graph.geo_knn import build_knn_geo_graph
from housets_bench.data.graph_dataset import GraphWindowDataset, GraphWindowCollate
from housets_bench.models.gnn.gcn_tcn_geo import GeoGCN_TCN
from housets_bench.metrics.graph_evaluator import evaluate_graph_forecaster


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--data", type=str, default=str(REPO_ROOT / "data" / "raw" / "HouseTS.csv"))
    p.add_argument("--zip-latlon", type=str, default=str(REPO_ROOT / "data" / "processed" / "zip_latlon.csv"))
    p.add_argument("--target-col", type=str, default="price")
    p.add_argument("--features-mode", type=str, default="S", choices=["S", "MS"],
                  help="S: x=price only; MS: x=all features, y=price")

    p.add_argument("--seq-len", type=int, default=6)
    p.add_argument("--label-len", type=int, default=0, help="unused by this baseline (keep 0)")
    p.add_argument("--pred-len", type=int, default=3)

    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.1)

    # graph hyper-parameters
    p.add_argument("--k", type=int, default=10, help="kNN neighbors per node")
    p.add_argument("--max-km", type=float, default=50.0, help="distance threshold in km (<=0 to disable)")
    p.add_argument("--symmetric", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--self-loops", action=argparse.BooleanOptionalAction, default=True)

    # model hyper-parameters
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--tcn-kernel", type=int, default=3)

    # training hyper-parameters
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=5)

    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-train-batches", type=int, default=0, help="<=0 means all")
    p.add_argument("--max-eval-batches", type=int, default=0, help="<=0 means all")

    p.add_argument("--device", type=str, default="", help="e.g. 'cuda', 'cuda:0', or leave empty for CPU")

    # speed controls
    p.add_argument("--n-zip", type=int, default=0, help="<=0 means use all ZIPs")

    return p.parse_args()


def _sync(dev: Optional[torch.device]) -> None:
    if dev is not None and dev.type == "cuda":
        torch.cuda.synchronize()


def _log_epoch_enabled() -> bool:
    return os.getenv("HOUSETS_LOG_EPOCH", "0") == "1"


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    latlon_path = Path(args.zip_latlon)
    if not latlon_path.exists():
        raise FileNotFoundError(
            f"zip_latlon file not found: {latlon_path}. "
            f"Build it with: python scripts/make_zip_latlon_pgeocode.py --data {data_path} --out {latlon_path}"
        )

    if args.label_len != 0:
        # keep behaviour explicit: this baseline ignores decoder label context
        print(f"[warn] --label-len={args.label_len} is ignored by gcn_tcn_geo; using label_len=0.", flush=True)

    aligned = load_aligned(str(data_path), target_col=args.target_col, impute=True)

    # optional ZIP subsample (keeps node dimension smaller for debugging)
    if args.n_zip > 0 and aligned.n_zip > args.n_zip:
        zips = aligned.zipcodes[: args.n_zip]
        zip_mask = np.isin(np.array(aligned.zipcodes), np.array(zips))
        aligned.zipcodes = list(np.array(aligned.zipcodes)[zip_mask])
        aligned.values = aligned.values[zip_mask]

    # indices
    cont_cols = list(aligned.schema.continuous_cols)
    if args.target_col not in cont_cols:
        raise ValueError(f"target_col={args.target_col} not in continuous_cols: {cont_cols[:10]} ...")
    price_idx = cont_cols.index(args.target_col)

    if args.features_mode == "S":
        x_idx = [price_idx]
        pipeline_desc = "log1p(price)"
    else:
        x_idx = list(range(len(cont_cols)))
        pipeline_desc = "log(all) (x=all, y=price)"

    # processed values: log1p on inputs and on target
    x_raw = aligned.values[:, :, x_idx].astype(np.float32)  # [N,T,F]
    y_raw = aligned.values[:, :, [price_idx]].astype(np.float32)  # [N,T,1]
    x_proc = np.log1p(np.clip(x_raw, a_min=0.0, a_max=None))
    y_proc = np.log1p(np.clip(y_raw, a_min=0.0, a_max=None))

    split = make_ratio_split(aligned.n_time, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    spec = make_window_spec(seq_len=args.seq_len, pred_len=args.pred_len, label_len=0)

    # build geo kNN graph (node order matches aligned.zipcodes)
    zip_latlon_csv = Path(args.zip_latlon)
    if not zip_latlon_csv.exists():
        raise FileNotFoundError(f"zip_latlon_csv not found: {zip_latlon_csv}")

    df_ll = pd.read_csv(zip_latlon_csv)
    if "zipcode" not in df_ll.columns and "zip" in df_ll.columns:
        df_ll = df_ll.rename(columns={"zip": "zipcode"})
    need_cols = {"zipcode", "lat", "lon"}
    missing = need_cols - set(df_ll.columns)
    if missing:
        raise ValueError(f"zip_latlon_csv missing columns: {missing}. Need {need_cols}")

    df_ll["zipcode"] = df_ll["zipcode"].astype(str).str.zfill(5)
    df_ll = df_ll.dropna(subset=["lat", "lon"])

    latlon_by_zip = {
        z: (float(lat), float(lon))
        for z, lat, lon in zip(df_ll["zipcode"], df_ll["lat"], df_ll["lon"])
    }

    graph = build_knn_geo_graph(
        aligned.zipcodes,         
        latlon_by_zip,           
        k=int(args.k),
        max_km=float(args.max_km) if args.max_km > 0 else None,
        include_self_loops=bool(args.self_loops),
        symmetric=bool(args.symmetric),
)   

    dev = torch.device(args.device) if str(args.device).strip() else None

    # datasets: sample over time only
    ds_train = GraphWindowDataset(x_proc, y_proc, split=split.train, spec=spec, allow_history=False)
    ds_val = GraphWindowDataset(x_proc, y_proc, split=split.val, spec=spec, allow_history=True)
    ds_test = GraphWindowDataset(x_proc, y_proc, split=split.test, spec=spec, allow_history=True)

    collate = GraphWindowCollate()
    dl_train = DataLoader(ds_train, batch_size=int(args.batch_size), shuffle=True, num_workers=0, collate_fn=collate)
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)

    def _to_sparse_norm(graph, n_nodes: int, device):
        import numpy as np
        import torch

        # Prefer native method if available
        if hasattr(graph, "to_torch_sparse_norm"):
            return graph.to_torch_sparse_norm(device=device)

        # Extract edge list from common field names
        if hasattr(graph, "src") and hasattr(graph, "dst"):
            src = np.asarray(getattr(graph, "src"), dtype=np.int64)
            dst = np.asarray(getattr(graph, "dst"), dtype=np.int64)
            w = None
            for name in ("weight", "weights", "w"):
                if hasattr(graph, name):
                    w = np.asarray(getattr(graph, name), dtype=np.float32)
                    break
            if w is None:
                w = np.ones_like(src, dtype=np.float32)

        elif hasattr(graph, "edge_index"):
            ei = np.asarray(getattr(graph, "edge_index"))
            if ei.ndim == 2 and ei.shape[0] == 2:
                src, dst = ei[0], ei[1]
            else:
                src, dst = ei[:, 0], ei[:, 1]
            w = np.ones_like(src, dtype=np.float32)

        elif hasattr(graph, "edges"):
            edges = getattr(graph, "edges")
            src = np.fromiter((e[0] for e in edges), dtype=np.int64, count=len(edges))
            dst = np.fromiter((e[1] for e in edges), dtype=np.int64, count=len(edges))
            w = np.ones_like(src, dtype=np.float32)

        else:
            raise AttributeError(
                "GeoGraph has no to_torch_sparse_norm() and no edge list fields "
                "(src/dst/edge_index/edges)."
            )

        dev_ = device if device is not None else torch.device("cpu")
        idx = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long, device=dev_)
        val = torch.tensor(w, dtype=torch.float32, device=dev_)

        A = torch.sparse_coo_tensor(idx, val, size=(n_nodes, n_nodes)).coalesce()
        deg = torch.sparse.sum(A, dim=1).to_dense()
        deg_inv_sqrt = deg.clamp_min(1e-12).pow(-0.5)
        val_norm = val * deg_inv_sqrt[idx[0]] * deg_inv_sqrt[idx[1]]
        return torch.sparse_coo_tensor(idx, val_norm, size=(n_nodes, n_nodes)).coalesce()
    A_norm = _to_sparse_norm(graph, n_nodes=len(aligned.zipcodes), device=dev)

    model = GeoGCN_TCN(
    input_dim=int(x_proc.shape[-1]),
    hidden_dim=int(args.hidden_dim),
    pred_len=int(args.pred_len),
    dropout=float(args.dropout),
    tcn_kernel=int(args.tcn_kernel),
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

            x = batch["x"]  # [B,L,N,F]
            y = batch["y"]  # [B,H,N,1]
            if dev is not None:
                x = x.to(dev)
                y = y.to(dev)

            optim.zero_grad(set_to_none=True)
            y_hat = model(x, A_norm)  # [B,H,N,1]
            loss = loss_fn(y_hat, y)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            optim.step()
            train_losses.append(float(loss.detach().cpu().item()))

        train_mse = float(np.mean(train_losses)) if train_losses else float("nan")

        # val mse
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
                f"[gcn_tcn_geo:{args.features_mode}] epoch {epoch}/{args.epochs} "
                f"train_mse={train_mse:.6g} val_mse={val_mse:.6g} epoch_time={dt:.2f}s",
                flush=True,
            )

        train_history.append(
            {"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse, "epoch_time_sec": float(dt)}
        )

        # early stopping
        if val_mse + 1e-12 < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(args.patience):
                break

    fit_sec = time.perf_counter() - t_fit0

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # metrics
    _sync(dev)
    t0 = time.perf_counter()
    val = evaluate_graph_forecaster(model, dl_val, A_norm=A_norm, device=dev, max_batches=max_eval_batches)
    _sync(dev)
    val_sec = time.perf_counter() - t0

    _sync(dev)
    t0 = time.perf_counter()
    test = evaluate_graph_forecaster(model, dl_test, A_norm=A_norm, device=dev, max_batches=max_eval_batches)
    _sync(dev)
    test_sec = time.perf_counter() - t0

    result: Dict[str, object] = {
        "model": "gcn_tcn_geo",
        "features_mode": args.features_mode,
        "graph": {
          "type": "geo_knn",
          "k": int(args.k),
          "max_km": float(args.max_km),
          "symmetric": bool(args.symmetric),
          "self_loops": bool(args.self_loops),
          "n_nodes": int(len(aligned.zipcodes)),
          "n_edges": int(len(getattr(graph, "src"))) if hasattr(graph, "src") else None,
        },
        "val": val.__dict__,
        "test": test.__dict__,
        "n_val": len(ds_val),
        "n_test": len(ds_test),
        "timing": {
            "fit_sec": float(fit_sec),
            "val_eval_sec": float(val_sec),
            "test_eval_sec": float(test_sec),
        },
        "hparams": {
            "seq_len": int(args.seq_len),
            "label_len": 0,
            "pred_len": int(args.pred_len),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "tcn_kernel": int(args.tcn_kernel),
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

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
