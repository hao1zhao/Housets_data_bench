from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from housets_bench.data.io import load_aligned
from housets_bench.data.split import make_ratio_split
from housets_bench.data.windowing import make_window_spec
from housets_bench.transforms import LogTransform, ZScoreTransform, StageSpec, TransformPipeline
from housets_bench.bundles import build_proc_bundle
from housets_bench.metrics.evaluator import evaluate_forecaster
from housets_bench.metrics.loss import evaluate_mse_loss, extract_train_history, sync_device

# Ensure models are registered
import housets_bench.models  # noqa: F401
from housets_bench.models.registry import get as get_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--zscore",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply Z-score normalization after log1p (default: off for backwards compat)",
    )
    p.add_argument("--zscore-scope", type=str, default="global", choices=["global", "per_zip"])
    p.add_argument("--zscore-eps", type=float, default=1e-8)

    p.add_argument("--data", type=str, default=str(REPO_ROOT / "data" / "raw" / "HouseTS.csv"))
    p.add_argument("--target-col", type=str, default="price")
    p.add_argument("--features-mode", type=str, default="MS", choices=["S", "MS", "M"])

    p.add_argument("--seq-len", type=int, default=6)
    p.add_argument("--label-len", type=int, default=3)
    p.add_argument("--pred-len", type=int, default=3)

    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.1)

    p.add_argument(
        "--model",
        type=str,
        default="rnn",
        choices=["rnn", "lstm", "dlinear", "timemixer", "patchtst", "informer", "autoformer", "fedformer"],
    )

    # RNN/LSTM hyper-parameters (ignored by others)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)

    # DLinear / (Auto|FED)former decomposition hyper-parameters
    p.add_argument("--kernel-size", type=int, default=3, help="moving average kernel size (odd)")
    p.add_argument(
        "--individual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="per-channel linear layers (True) or shared (False)",
    )

    # TimeMixer hyper-parameters
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--n-blocks", type=int, default=2)
    p.add_argument("--scales", type=str, default="1,2,4", help="comma-separated list, e.g. '1,2,4'")

    # PatchTST hyper-parameters
    p.add_argument("--patch-len", type=int, default=3)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--ptst-d-model", type=int, default=128)
    p.add_argument("--ptst-n-heads", type=int, default=4)
    p.add_argument("--ptst-e-layers", type=int, default=2)
    p.add_argument("--ptst-d-ff", type=int, default=256)
    p.add_argument("--ptst-dropout", type=float, default=0.1)

    # Informer hyper-parameters (encoder-decoder transformer)
    p.add_argument("--inf-d-model", type=int, default=128)
    p.add_argument("--inf-n-heads", type=int, default=4)
    p.add_argument("--inf-e-layers", type=int, default=2)
    p.add_argument("--inf-d-layers", type=int, default=1)
    p.add_argument("--inf-d-ff", type=int, default=256)
    p.add_argument("--inf-dropout", type=float, default=0.1)

    # Autoformer hyper-parameters (encoder-decoder transformer + decomposition)
    p.add_argument("--af-d-model", type=int, default=128)
    p.add_argument("--af-n-heads", type=int, default=4)
    p.add_argument("--af-e-layers", type=int, default=2)
    p.add_argument("--af-d-layers", type=int, default=1)
    p.add_argument("--af-d-ff", type=int, default=256)
    p.add_argument("--af-dropout", type=float, default=0.1)

    # FEDformer hyper-parameters (FFT mixing + frequency cross-attention)
    p.add_argument("--fed-modes", type=int, default=16, help="number of Fourier modes to keep")
    p.add_argument("--fed-mode-select", type=str, default="low", choices=["low", "random"])
    p.add_argument("--fed-d-model", type=int, default=128)
    p.add_argument("--fed-n-heads", type=int, default=4)
    p.add_argument("--fed-e-layers", type=int, default=2)
    p.add_argument("--fed-d-layers", type=int, default=1)
    p.add_argument("--fed-d-ff", type=int, default=256)
    p.add_argument("--fed-dropout", type=float, default=0.1)

    # training hyper-parameters
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=5)

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-train-batches", type=int, default=0, help="<=0 means all")
    p.add_argument("--device", type=str, default="", help="e.g. 'cuda', 'cuda:0', or leave empty for CPU")

    # speed controls
    p.add_argument("--n-zip", type=int, default=50, help="<=0 means use all ZIPs")
    p.add_argument("--max-eval-batches", type=int, default=50, help="<=0 means evaluate all batches")

    return p.parse_args()


def _maybe_set(model: object, name: str, value: object) -> None:
    if hasattr(model, name):
        setattr(model, name, value)


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    aligned = load_aligned(str(data_path), target_col=args.target_col, impute=True)

    # optional ZIP subsample
    if args.n_zip > 0 and aligned.n_zip > args.n_zip:
        zips = aligned.zipcodes[: args.n_zip]
        zip_mask = np.isin(np.array(aligned.zipcodes), np.array(zips))
        aligned.zipcodes = list(np.array(aligned.zipcodes)[zip_mask])
        aligned.values = aligned.values[zip_mask]

    split = make_ratio_split(aligned.n_time, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    spec = make_window_spec(seq_len=args.seq_len, pred_len=args.pred_len, label_len=args.label_len)

    # -------------------------
    # transforms: log1p
    # -------------------------
    stages = [
        StageSpec(LogTransform(mode="log1p"), idx=None),
    ]
    if bool(args.zscore):
        stages.append(
            StageSpec(
                ZScoreTransform(scope=str(args.zscore_scope), eps=float(args.zscore_eps)),
                idx=None,
            )
        )
    pipeline = TransformPipeline(stages)

    bundle = build_proc_bundle(
        aligned,
        split=split,
        spec=spec,
        features_mode=args.features_mode,
        pipeline=pipeline,
        batch_size=int(args.batch_size),
        pad_to=None,
        shuffle_train=True,
    )

    model = get_model(args.model)

    # Apply hyper-parameter overrides (only if the model defines them)
    _maybe_set(model, "hidden_size", int(args.hidden_size))
    _maybe_set(model, "num_layers", int(args.num_layers))
    _maybe_set(model, "dropout", float(args.dropout))

    _maybe_set(model, "kernel_size", int(args.kernel_size))
    _maybe_set(model, "individual", bool(args.individual))

    _maybe_set(model, "d_model", int(args.d_model))
    _maybe_set(model, "d_ff", int(args.d_ff))
    _maybe_set(model, "n_blocks", int(args.n_blocks))
    _maybe_set(model, "scales", str(args.scales))

    # PatchTST overrides (avoid clobbering TimeMixer's d_model/d_ff when not used)
    if args.model == "patchtst":
        _maybe_set(model, "patch_len", int(args.patch_len))
        _maybe_set(model, "stride", int(args.stride))
        _maybe_set(model, "d_model", int(args.ptst_d_model))
        _maybe_set(model, "n_heads", int(args.ptst_n_heads))
        _maybe_set(model, "e_layers", int(args.ptst_e_layers))
        _maybe_set(model, "d_ff", int(args.ptst_d_ff))
        _maybe_set(model, "dropout", float(args.ptst_dropout))

    # Informer overrides
    if args.model == "informer":
        _maybe_set(model, "d_model", int(args.inf_d_model))
        _maybe_set(model, "n_heads", int(args.inf_n_heads))
        _maybe_set(model, "e_layers", int(args.inf_e_layers))
        _maybe_set(model, "d_layers", int(args.inf_d_layers))
        _maybe_set(model, "d_ff", int(args.inf_d_ff))
        _maybe_set(model, "dropout", float(args.inf_dropout))

    # Autoformer overrides
    if args.model == "autoformer":
        _maybe_set(model, "d_model", int(args.af_d_model))
        _maybe_set(model, "n_heads", int(args.af_n_heads))
        _maybe_set(model, "e_layers", int(args.af_e_layers))
        _maybe_set(model, "d_layers", int(args.af_d_layers))
        _maybe_set(model, "d_ff", int(args.af_d_ff))
        _maybe_set(model, "dropout", float(args.af_dropout))

    # FEDformer overrides
    if args.model == "fedformer":
        _maybe_set(model, "modes", int(args.fed_modes))
        _maybe_set(model, "mode_select_method", str(args.fed_mode_select))
        _maybe_set(model, "d_model", int(args.fed_d_model))
        _maybe_set(model, "n_heads", int(args.fed_n_heads))
        _maybe_set(model, "e_layers", int(args.fed_e_layers))
        _maybe_set(model, "d_layers", int(args.fed_d_layers))
        _maybe_set(model, "d_ff", int(args.fed_d_ff))
        _maybe_set(model, "dropout", float(args.fed_dropout))

    _maybe_set(model, "epochs", int(args.epochs))
    _maybe_set(model, "lr", float(args.lr))
    _maybe_set(model, "weight_decay", float(args.weight_decay))
    _maybe_set(model, "grad_clip", float(args.grad_clip))
    _maybe_set(model, "patience", int(args.patience))

    if args.max_train_batches > 0:
        _maybe_set(model, "max_train_batches", int(args.max_train_batches))
    else:
        _maybe_set(model, "max_train_batches", None)

    dev = torch.device(args.device) if str(args.device).strip() else None

    t0_total = time.perf_counter()

    # ---- fit timing ----
    sync_device(dev)
    t0_fit = time.perf_counter()
    model.fit(bundle, device=dev)
    sync_device(dev)
    fit_sec = time.perf_counter() - t0_fit

    max_eval = None if args.max_eval_batches <= 0 else int(args.max_eval_batches)

    # ---- eval timing ----
    sync_device(dev)
    t0_val = time.perf_counter()
    val = evaluate_forecaster(model, bundle, split="val", device=dev, max_batches=max_eval)
    sync_device(dev)
    val_eval_sec = time.perf_counter() - t0_val

    sync_device(dev)
    t0_test = time.perf_counter()
    test = evaluate_forecaster(model, bundle, split="test", device=dev, max_batches=max_eval)
    sync_device(dev)
    test_eval_sec = time.perf_counter() - t0_test

    # ---- processed-space horizon MSE loss (best-effort) ----
    def _safe_loss(split: str) -> dict:
        try:
            return evaluate_mse_loss(model, bundle, split=split, device=dev, max_batches=max_eval)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    loss_train = _safe_loss("train")
    loss_val = _safe_loss("val")
    loss_test = _safe_loss("test")

    total_sec = time.perf_counter() - t0_total

    result = {
        "model": args.model,
        "val": val.__dict__,
        "test": test.__dict__,
        "n_val": len(bundle.datasets["val"]),
        "n_test": len(bundle.datasets["test"]),
        "timing": {
            "fit_sec": float(fit_sec),
            "val_eval_sec": float(val_eval_sec),
            "test_eval_sec": float(test_eval_sec),
            "total_sec": float(total_sec),
        },
        "loss": {
            "space": "processed",
            "metric": "mse",
            "train": loss_train,
            "val": loss_val,
            "test": loss_test,
        },
        "pipeline": bundle.pipeline.summary(),
        "hparams": {
            # transforms
            "zscore": bool(args.zscore),
            "zscore_scope": str(args.zscore_scope),
            "zscore_eps": float(args.zscore_eps),
            # data/window
            "features_mode": str(args.features_mode),
            "seq_len": int(args.seq_len),
            "label_len": int(args.label_len),
            "pred_len": int(args.pred_len),
            # model knobs
            "hidden_size": int(args.hidden_size),
            "num_layers": int(args.num_layers),
            "dropout": float(args.dropout),
            "kernel_size": int(args.kernel_size),
            "individual": bool(args.individual),
            "d_model": int(args.d_model),
            "d_ff": int(args.d_ff),
            "n_blocks": int(args.n_blocks),
            "scales": str(args.scales),
            "patch_len": int(args.patch_len),
            "stride": int(args.stride),
            "ptst_d_model": int(args.ptst_d_model),
            "ptst_n_heads": int(args.ptst_n_heads),
            "ptst_e_layers": int(args.ptst_e_layers),
            "ptst_d_ff": int(args.ptst_d_ff),
            "ptst_dropout": float(args.ptst_dropout),
            "inf_d_model": int(args.inf_d_model),
            "inf_n_heads": int(args.inf_n_heads),
            "inf_e_layers": int(args.inf_e_layers),
            "inf_d_layers": int(args.inf_d_layers),
            "inf_d_ff": int(args.inf_d_ff),
            "inf_dropout": float(args.inf_dropout),
            "af_d_model": int(args.af_d_model),
            "af_n_heads": int(args.af_n_heads),
            "af_e_layers": int(args.af_e_layers),
            "af_d_layers": int(args.af_d_layers),
            "af_d_ff": int(args.af_d_ff),
            "af_dropout": float(args.af_dropout),
            "fed_modes": int(args.fed_modes),
            "fed_mode_select": str(args.fed_mode_select),
            "fed_d_model": int(args.fed_d_model),
            "fed_n_heads": int(args.fed_n_heads),
            "fed_e_layers": int(args.fed_e_layers),
            "fed_d_layers": int(args.fed_d_layers),
            "fed_d_ff": int(args.fed_d_ff),
            "fed_dropout": float(args.fed_dropout),
            # training
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "grad_clip": float(args.grad_clip),
            "patience": int(args.patience),
            "batch_size": int(args.batch_size),
            "max_train_batches": int(args.max_train_batches),
            "max_eval_batches": int(args.max_eval_batches),
            "n_zip": int(args.n_zip),
        },
    }

    hist = extract_train_history(model)
    if hist is not None:
        result["train_history"] = hist

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
