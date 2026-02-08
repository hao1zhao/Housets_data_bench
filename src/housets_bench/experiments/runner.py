from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Optional

import time

import torch

from housets_bench.bundles.datatypes import ProcBundle
from housets_bench.metrics.evaluator import EvalResult, evaluate_forecaster
from housets_bench.metrics.loss import evaluate_mse_loss, extract_train_history, sync_device

import housets_bench.models  

from housets_bench.models.registry import get as get_model


def run_one(
    *,
    model_name: str,
    bundle: ProcBundle,
    device: Optional[str] = None,
    max_eval_batches: Optional[int] = None,
) -> Dict[str, object]:
    dev = torch.device(device) if device is not None else None

    t0_total = time.perf_counter()

    model = get_model(model_name)

    # ---- fit timing ----
    sync_device(dev)
    t0_fit = time.perf_counter()
    model.fit(bundle, device=dev)
    sync_device(dev)
    fit_sec = time.perf_counter() - t0_fit

    # ---- eval timing ----
    sync_device(dev)
    t0_val = time.perf_counter()
    val = evaluate_forecaster(model, bundle, split="val", device=dev, max_batches=max_eval_batches)
    sync_device(dev)
    val_eval_sec = time.perf_counter() - t0_val

    sync_device(dev)
    t0_test = time.perf_counter()
    test = evaluate_forecaster(model, bundle, split="test", device=dev, max_batches=max_eval_batches)
    sync_device(dev)
    test_eval_sec = time.perf_counter() - t0_test

    # ---- processed-space horizon MSE loss ----
    def _safe_loss(split: str) -> Dict[str, object]:
        try:
            return evaluate_mse_loss(model, bundle, split=split, device=dev, max_batches=max_eval_batches)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    loss_train = _safe_loss("train")
    loss_val = _safe_loss("val")
    loss_test = _safe_loss("test")

    total_sec = time.perf_counter() - t0_total

    out: Dict[str, object] = {
        "model": model_name,
        "val": asdict(val),
        "test": asdict(test),
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
    }

    hist = extract_train_history(model)
    if hist is not None:
        out["train_history"] = hist

    return out
