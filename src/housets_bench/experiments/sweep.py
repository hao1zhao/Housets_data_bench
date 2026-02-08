from __future__ import annotations
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple
import time
import copy
import json
import numpy as np
import torch
import housets_bench.models  

from housets_bench.bundles import build_proc_bundle
from housets_bench.bundles.datatypes import ProcBundle
from housets_bench.data.io import AlignedData, load_aligned
from housets_bench.data.split import make_ratio_split
from housets_bench.data.windowing import make_window_spec, WindowSpec
from housets_bench.metrics.evaluator import evaluate_forecaster
from housets_bench.metrics.loss import evaluate_mse_loss, extract_train_history, sync_device
from housets_bench.models.registry import get as get_model
from housets_bench.transforms import ClipTransform, LogTransform, PCATransform, StageSpec, TransformPipeline, ZScoreTransform


def _maybe_set(obj: object, name: str, value: Any) -> None:
    if hasattr(obj, name):
        try:
            setattr(obj, name, value)
        except Exception:
            pass


def apply_hparams(model: object, hparams: Dict[str, Any]) -> None:
    hparams = hparams or {}

    # Compatibility aliases
    if "mode_select" in hparams and "mode_select_method" not in hparams:
        hparams = dict(hparams)
        hparams["mode_select_method"] = hparams["mode_select"]

    for k, v in hparams.items():
        # Chronos stores point in a tiny config dataclass
        if k == "point" and hasattr(model, "point_cfg"):
            try:
                model.point_cfg.point = str(v)  
            except Exception:
                pass
            continue

        _maybe_set(model, k, v)

def build_pipeline_from_cfg(*, schema, cfg: Dict[str, Any]) -> TransformPipeline:
    """Build TransformPipeline from config dict following order [log, clip, zscore, pca]."""
    transforms_cfg = cfg.get("transforms", {}) or {}
    order = transforms_cfg.get("order", ["log", "clip", "zscore", "pca"])

    cont_names = list(schema.continuous_cols)
    target = schema.target_col
    if target not in cont_names:
        raise ValueError(f"target_col {target!r} not found in continuous_cols")
    target_idx = cont_names.index(target)
    all_idx = list(range(len(cont_names)))

    def _idx_from_apply_to(apply_to: Dict[str, Any]) -> Optional[List[int]]:
        tgt = bool((apply_to or {}).get("target", False))
        cont = bool((apply_to or {}).get("continuous_features", False))
        if cont and tgt:
            return None  # all
        if cont and not tgt:
            return [i for i in all_idx if i != target_idx]
        if (not cont) and tgt:
            return [target_idx]
        return []  

    stages: List[StageSpec] = []

    for name in order:
        if name == "log":
            c = transforms_cfg.get("log", {}) or {}
            if not c.get("enabled", True):
                continue
            mode = str(c.get("mode", "log1p"))
            idx = _idx_from_apply_to(c.get("apply_to", {"target": True, "continuous_features": True}))
            if idx == []:
                continue
            stages.append(StageSpec(LogTransform(mode=mode), idx=None if idx is None else idx))
            continue

        if name == "clip":
            c = transforms_cfg.get("clip", {}) or {}
            if not c.get("enabled", False):
                continue
            method = str(c.get("method", "quantile"))
            idx = _idx_from_apply_to(c.get("apply_to", {"target": False, "continuous_features": True}))
            if idx == []:
                continue

            if method == "quantile":
                qcfg = c.get("quantile", {}) or {}
                lower = float(qcfg.get("lower", 0.001))
                upper = float(qcfg.get("upper", 0.999))
                tr = ClipTransform(method="quantile", lower_q=lower, upper_q=upper)

            elif method == "sigma":
                scfg = c.get("sigma", {}) or {}
                k = float(scfg.get("k", 5.0))
                tr = ClipTransform(method="sigma", sigma_k=k)

            elif method == "absolute":
                acfg = c.get("absolute", {}) or {}
                lower = acfg.get("lower", None)
                upper = acfg.get("upper", None)
                tr = ClipTransform(method="absolute", abs_lower=lower, abs_upper=upper)

            else:
                raise ValueError(f"Unknown clip.method={method!r}")

            stages.append(StageSpec(tr, idx=None if idx is None else idx))
            continue

        if name == "zscore":
            c = transforms_cfg.get("zscore", {}) or {}
            if not c.get("enabled", False):
                continue
            scope = str(c.get("scope", "global"))
            eps = float(c.get("eps", 1e-6))
            idx = _idx_from_apply_to(c.get("apply_to", {"target": True, "continuous_features": True}))
            if idx == []:
                continue
            stages.append(StageSpec(ZScoreTransform(scope=scope, eps=eps), idx=None if idx is None else idx))
            continue

        if name == "pca":
            c = transforms_cfg.get("pca", {}) or {}
            if not c.get("enabled", False):
                continue
            n_components = int(c.get("n_components", 16))
            stages.append(StageSpec(PCATransform(n_components=n_components), idx=None))
            continue

        raise ValueError(f"Unknown transform stage {name!r}")

    return TransformPipeline(stages)


def build_bundle_from_cfg(
    *,
    aligned: AlignedData,
    cfg: Dict[str, Any],
) -> ProcBundle:
    split_cfg = cfg.get("split", {}) or {}
    split = make_ratio_split(
        aligned.n_time,
        train_ratio=float(split_cfg.get("train_ratio", 0.7)),
        val_ratio=float(split_cfg.get("val_ratio", 0.1)),
    )

    window_cfg = cfg.get("window", {}) or {}
    spec = make_window_spec(
        seq_len=int(window_cfg.get("seq_len", 6)),
        pred_len=int(window_cfg.get("pred_len", 3)),
        label_len=int(window_cfg.get("label_len", 3)),
    )

    # features mode
    features_mode = str(((cfg.get("task", {}) or {}).get("features_mode", cfg.get("features_mode", "MS")))).upper()

    # pipeline
    pipeline = build_pipeline_from_cfg(schema=aligned.schema, cfg=cfg)

    # dataloader
    dl_cfg = cfg.get("dataloader", {}) or {}
    batch_size = int(dl_cfg.get("batch_size", 64))
    num_workers = int(dl_cfg.get("num_workers", 0))
    pad_to = int(dl_cfg.get("pad_to", 0))
    pad_to_val: Optional[int] = None if pad_to <= 0 else pad_to

    bundle = build_proc_bundle(
        aligned,
        split=split,
        spec=spec,
        features_mode=features_mode,
        pipeline=pipeline,
        batch_size=batch_size,
        num_workers=num_workers,
        pad_to=pad_to_val,
    )
    return bundle


def run_one_cfg(
    *,
    cfg: Dict[str, Any],
    aligned: Optional[AlignedData] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    t0_total = time.perf_counter()

    data_cfg = cfg.get("data", {}) or {}
    if aligned is None:
        aligned = load_aligned(
            data_cfg.get("path"),
            target_col=str(data_cfg.get("target_col", "price")),
            impute=bool(data_cfg.get("impute", True)),
        )
        n_zip = int(data_cfg.get("n_zip", 0) or 0)
        if n_zip > 0 and aligned.n_zip > n_zip:
            zips = aligned.zipcodes[:n_zip]
            zip_mask = np.isin(np.array(aligned.zipcodes), np.array(zips))
            aligned = AlignedData(
                zipcodes=list(np.array(aligned.zipcodes)[zip_mask]),
                dates=aligned.dates,
                values=aligned.values[zip_mask],
                time_marks=aligned.time_marks,
                schema=aligned.schema,
            )
    bundle = build_bundle_from_cfg(aligned=aligned, cfg=cfg)

    model_cfg = cfg.get("model", {}) or {}
    model_name = str(model_cfg.get("name"))
    model = get_model(model_name)

    # apply hparams to model instance
    hparams = model_cfg.get("hparams", {}) or {}
    apply_hparams(model, hparams)

    run_cfg = cfg.get("run", {}) or {}
    dev = torch.device(device or run_cfg.get("device", "cpu"))

    max_eval = run_cfg.get("max_eval_batches", None)
    if max_eval is not None:
        max_eval = int(max_eval)
        if max_eval <= 0:
            max_eval = None

    # fit / eval
    sync_device(dev)
    t0_fit = time.perf_counter()
    model.fit(bundle, device=dev)
    sync_device(dev)
    fit_sec = time.perf_counter() - t0_fit

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

    def _safe_loss(split: str) -> Dict[str, object]:
        try:
            return evaluate_mse_loss(model, bundle, split=split, device=dev, max_batches=max_eval)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    loss_train = _safe_loss("train")
    loss_val = _safe_loss("val")
    loss_test = _safe_loss("test")

    total_sec = time.perf_counter() - t0_total

    out: Dict[str, Any] = {
        "model": model_name,
        "task": (cfg.get("task", {}) or {}).get("name"),
        "window": (cfg.get("window", {}) or {}).get("name"),
        "pipeline": bundle.pipeline.summary(),
        "val": asdict(val),
        "test": asdict(test),
        "n_train": len(bundle.datasets["train"]),
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
