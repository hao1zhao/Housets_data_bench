from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from housets_bench.data.io import load_aligned
from housets_bench.data.split import make_ratio_split
from housets_bench.data.windowing import make_window_spec
from housets_bench.transforms import LogTransform, ZScoreTransform, PCATransform, StageSpec, TransformPipeline
from housets_bench.bundles import build_proc_bundle
from housets_bench.metrics.evaluator import evaluate_forecaster
from housets_bench.metrics.loss import evaluate_mse_loss, extract_train_history, sync_device

# Ensure models are registered
import housets_bench.models  # noqa: F401
from housets_bench.models.registry import get as get_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=str(REPO_ROOT / "data" / "raw" / "HouseTS.csv"))
    p.add_argument("--target-col", type=str, default="price")

    p.add_argument("--seq-len", type=int, default=6)
    p.add_argument("--label-len", type=int, default=3)
    p.add_argument("--pred-len", type=int, default=3)

    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.1)

    p.add_argument("--pca-components", type=int, default=16)
    p.add_argument("--zscore-scope", type=str, default="global", choices=["global", "per_zip"])

    p.add_argument("--model", type=str, default="arima", choices=["arima", "var", "rf", "xgb"])

    # model knobs (optional)
    p.add_argument("--arima-p", type=int, default=0, help="<=0 means use seq_len")
    p.add_argument("--var-p", type=int, default=1)
    p.add_argument("--max-train-samples", type=int, default=0, help="<=0 means use all train windows (RF/XGB only)")

    p.add_argument("--rf-n-estimators", type=int, default=200)
    p.add_argument("--xgb-n-estimators", type=int, default=300)

    # speed controls
    p.add_argument("--n-zip", type=int, default=50, help="<=0 means use all ZIPs")
    p.add_argument("--max-eval-batches", type=int, default=50, help="<=0 means evaluate all batches")

    return p.parse_args()


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

    pipeline = TransformPipeline([
        StageSpec(LogTransform(mode="log1p"), idx=None),
        StageSpec(ZScoreTransform(scope=args.zscore_scope), idx=None),
        StageSpec(PCATransform(n_components=args.pca_components, whiten=False, random_state=0), idx=None),
    ])

    bundle = build_proc_bundle(
        aligned,
        split=split,
        spec=spec,
        features_mode="MS",   
        pipeline=pipeline,
        batch_size=128,
        pad_to=None,        
        shuffle_train=False,
    )

    # instantiate + set optional knobs
    model = get_model(args.model)

    if args.model == "arima":
        if args.arima_p > 0:
            setattr(model, "p", int(args.arima_p))

    if args.model == "var":
        setattr(model, "p", int(args.var_p))

    if args.model == "rf":
        setattr(model, "n_estimators", int(args.rf_n_estimators))
        if args.max_train_samples > 0:
            setattr(model, "max_train_samples", int(args.max_train_samples))

    if args.model == "xgb":
        setattr(model, "n_estimators", int(args.xgb_n_estimators))
        if args.max_train_samples > 0:
            setattr(model, "max_train_samples", int(args.max_train_samples))

    dev = None
    t0_total = time.perf_counter()

    # ---- fit timing ----
    sync_device(dev)
    t0_fit = time.perf_counter()
    model.fit(bundle)
    sync_device(dev)
    fit_sec = time.perf_counter() - t0_fit

    max_eval = None if args.max_eval_batches <= 0 else int(args.max_eval_batches)
    # ---- eval timing ----
    sync_device(dev)
    t0_val = time.perf_counter()
    val = evaluate_forecaster(model, bundle, split="val", max_batches=max_eval)
    sync_device(dev)
    val_eval_sec = time.perf_counter() - t0_val

    sync_device(dev)
    t0_test = time.perf_counter()
    test = evaluate_forecaster(model, bundle, split="test", max_batches=max_eval)
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
        "pca_components": int(args.pca_components),
    }

    hist = extract_train_history(model)
    if hist is not None:
        result["train_history"] = hist

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
