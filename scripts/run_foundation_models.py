from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from housets_bench.bundles import build_proc_bundle
from housets_bench.data.io import load_aligned
from housets_bench.data.split import make_ratio_split
from housets_bench.data.windowing import make_window_spec
from housets_bench.metrics.evaluator import evaluate_forecaster
from housets_bench.transforms import LogTransform, StageSpec, TransformPipeline

# Ensure models are registered
import housets_bench.models  
from housets_bench.models.registry import get as get_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

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
        default="chronos2_zero",
        choices=[
            "timesfm_zero",
            "timesfm_ft",
            "timesfm_full_ft",
            "chronos_zero",
            "chronos_ft",
            "chronos_full_ft",
            "chronos2_zero",
            "chronos2_ft",
            "chronos2_full_ft",
        ],
    )

    # TimesFM options
    p.add_argument("--timesfm-repo-id", type=str, default="google/timesfm-1.0-200m-pytorch")
    p.add_argument("--timesfm-infer-bs", type=int, default=64)
    p.add_argument("--timesfm-max-calib-batches", type=int, default=200)

    # TimesFM full fine-tune options
    p.add_argument("--timesfm-ft-lr", type=float, default=1e-4)
    p.add_argument("--timesfm-ft-weight-decay", type=float, default=0.0)
    p.add_argument("--timesfm-ft-grad-clip", type=float, default=1.0)
    p.add_argument("--timesfm-ft-epochs", type=int, default=1)
    p.add_argument("--timesfm-ft-max-train-batches", type=int, default=200, help="<=0 means all")
    p.add_argument("--timesfm-ft-max-train-steps", type=int, default=0, help="<=0 means no explicit step cap")
    p.add_argument(
        "--timesfm-ft-freq-type",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="TimesFM frequency category: 0=daily-or-faster, 1=weekly/monthly, 2=quarterly/yearly",
    )
    p.add_argument("--timesfm-ft-load", type=str, default="", help="Optional path to load a fine-tuned state_dict")
    p.add_argument("--timesfm-ft-save", type=str, default="", help="Optional path to save fine-tuned state_dict")

    # optional stability knobs 
    p.add_argument(
        "--timesfm-ft-train-scope",
        type=str,
        default="all",
        choices=["all", "head", "last_block"],
        help="Which TimesFM parameters to fine-tune. 'head' is usually the most stable on small datasets.",
    )
    p.add_argument("--timesfm-ft-eval-every", type=int, default=1, help="Validate every N epochs (<=0 disables).")
    p.add_argument("--timesfm-ft-val-max-batches", type=int, default=50, help="Val batches per check (<=0 all).")
    p.add_argument("--timesfm-ft-patience", type=int, default=0, help="Early-stop patience (0 disables).")

    # Chronos options
    p.add_argument("--chronos-model-id", type=str, default="amazon/chronos-t5-small")
    p.add_argument("--chronos-infer-bs", type=int, default=128)
    p.add_argument("--chronos-point", type=str, default="median", choices=["median", "mean"])
    p.add_argument("--chronos-max-calib-batches", type=int, default=200)

    # Chronos full fine-tune options
    p.add_argument("--chronos-ft-lr", type=float, default=1e-5)
    p.add_argument("--chronos-ft-weight-decay", type=float, default=0.0)
    p.add_argument("--chronos-ft-grad-clip", type=float, default=1.0)
    p.add_argument("--chronos-ft-epochs", type=int, default=1)
    p.add_argument("--chronos-ft-max-train-batches", type=int, default=200, help="<=0 means all")
    p.add_argument("--chronos-ft-max-train-steps", type=int, default=0, help="<=0 means no explicit step cap")
    p.add_argument("--chronos-ft-load", type=str, default="", help="Optional path to load a fine-tuned state_dict")
    p.add_argument("--chronos-ft-save", type=str, default="", help="Optional path to save fine-tuned state_dict")

    # optional stability knobs 
    p.add_argument(
        "--chronos-ft-train-scope",
        type=str,
        default="all",
        choices=["all", "head", "last_block"],
        help="Which Chronos parameters to fine-tune.",
    )
    p.add_argument("--chronos-ft-eval-every", type=int, default=1, help="Validate every N epochs (<=0 disables).")
    p.add_argument("--chronos-ft-val-max-batches", type=int, default=50, help="Val batches per check (<=0 all).")
    p.add_argument("--chronos-ft-patience", type=int, default=0, help="Early-stop patience (0 disables).")

    # Chronos-2 inference option
    p.add_argument(
        "--chronos2-cross-learning",
        action="store_true",
        help="Enable Chronos-2 cross-learning at inference (shares information across the batch).",
    )

    # Chronos-2 fine-tune options
    p.add_argument("--chronos2-ft-mode", type=str, default="lora", choices=["full", "lora"])
    p.add_argument("--chronos2-ft-lr", type=float, default=1e-5)
    p.add_argument("--chronos2-ft-steps", type=int, default=500)
    p.add_argument("--chronos2-ft-batch-size", type=int, default=32)
    p.add_argument("--chronos2-ft-logging-steps", type=int, default=100)
    p.add_argument("--chronos2-ft-load", type=str, default="", help="Optional directory to load a fine-tuned pipeline")
    p.add_argument("--chronos2-ft-save", type=str, default="", help="Optional directory to save the fine-tuned pipeline")

    # speed controls
    p.add_argument("--n-zip", type=int, default=50, help="<=0 means use all ZIPs")
    p.add_argument("--max-eval-batches", type=int, default=50, help="<=0 means evaluate all batches")

    # misc
    p.add_argument(
        "--pad-to",
        type=int,
        default=32,
        help="Left-pad context to this length (0/negative disables). TimesFM legacy models often expect multiples of 32.",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", type=str, default="", help="e.g. 'cuda', 'cuda:0', or leave empty for CPU")

    return p.parse_args()


def _maybe_set(model: object, name: str, value: object) -> None:
    if hasattr(model, name):
        setattr(model, name, value)


def _as_none_if_empty(s: str) -> str | None:
    s = str(s).strip()
    return s if s else None


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    aligned = load_aligned(str(data_path), target_col=args.target_col, impute=True)

    if args.n_zip > 0 and aligned.n_zip > args.n_zip:
        zips = aligned.zipcodes[: args.n_zip]
        zip_mask = np.isin(np.array(aligned.zipcodes), np.array(zips))
        aligned.zipcodes = list(np.array(aligned.zipcodes)[zip_mask])
        aligned.values = aligned.values[zip_mask]

    split = make_ratio_split(aligned.n_time, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    spec = make_window_spec(seq_len=args.seq_len, pred_len=args.pred_len, label_len=args.label_len)

    pipeline = TransformPipeline([StageSpec(LogTransform(mode="log1p"), idx=None)])

    pad_to = int(args.pad_to)
    pad_to = None if pad_to <= 0 else pad_to

    bundle = build_proc_bundle(
        aligned,
        split=split,
        spec=spec,
        features_mode=args.features_mode,
        pipeline=pipeline,
        batch_size=int(args.batch_size),
        pad_to=pad_to,
        shuffle_train=True,
    )

    model = get_model(args.model)

    if args.model.startswith("timesfm"):
        _maybe_set(model, "repo_id", str(args.timesfm_repo_id))
        _maybe_set(model, "infer_batch_size", int(args.timesfm_infer_bs))
        _maybe_set(model, "max_calib_batches", int(args.timesfm_max_calib_batches))

        if args.model == "timesfm_full_ft" and hasattr(model, "ft_cfg"):
            ft = model.ft_cfg
            ft.lr = float(args.timesfm_ft_lr)
            ft.weight_decay = float(args.timesfm_ft_weight_decay)
            ft.grad_clip = float(args.timesfm_ft_grad_clip)
            ft.epochs = int(args.timesfm_ft_epochs)
            ft.max_train_batches = None if args.timesfm_ft_max_train_batches <= 0 else int(args.timesfm_ft_max_train_batches)
            ft.max_train_steps = None if args.timesfm_ft_max_train_steps <= 0 else int(args.timesfm_ft_max_train_steps)
            ft.freq_type = int(args.timesfm_ft_freq_type)
            ft.load_path = _as_none_if_empty(args.timesfm_ft_load)
            ft.save_path = _as_none_if_empty(args.timesfm_ft_save)

            # optional knobs
            ft.train_scope = str(args.timesfm_ft_train_scope)
            ft.eval_every = int(args.timesfm_ft_eval_every)
            ft.val_max_batches = None if args.timesfm_ft_val_max_batches <= 0 else int(args.timesfm_ft_val_max_batches)
            ft.patience = int(args.timesfm_ft_patience)

    if args.model.startswith("chronos"):
        _maybe_set(model, "model_id", str(args.chronos_model_id))
        _maybe_set(model, "infer_batch_size", int(args.chronos_infer_bs))
        _maybe_set(model, "max_calib_batches", int(args.chronos_max_calib_batches))

        # Point config can be stored in different places
        _maybe_set(model, "point", str(args.chronos_point))
        if hasattr(model, "point_cfg"):
            try:
                model.point_cfg.point = str(args.chronos_point)
            except Exception:
                pass

    # Chronos full fine-tune config
    if args.model == "chronos_full_ft" and hasattr(model, "ft_cfg"):
        ft = model.ft_cfg
        ft.lr = float(args.chronos_ft_lr)
        ft.weight_decay = float(args.chronos_ft_weight_decay)
        ft.grad_clip = float(args.chronos_ft_grad_clip)
        ft.epochs = int(args.chronos_ft_epochs)
        ft.max_train_batches = None if args.chronos_ft_max_train_batches <= 0 else int(args.chronos_ft_max_train_batches)
        ft.max_train_steps = None if args.chronos_ft_max_train_steps <= 0 else int(args.chronos_ft_max_train_steps)
        ft.load_path = _as_none_if_empty(args.chronos_ft_load)
        ft.save_path = _as_none_if_empty(args.chronos_ft_save)
        ft.train_scope = str(args.chronos_ft_train_scope)
        ft.eval_every = int(args.chronos_ft_eval_every)
        ft.val_max_batches = None if args.chronos_ft_val_max_batches <= 0 else int(args.chronos_ft_val_max_batches)
        ft.patience = int(args.chronos_ft_patience)

    # Chronos-2 inference option
    if args.model.startswith("chronos2"):
        _maybe_set(model, "cross_learning", bool(args.chronos2_cross_learning))

    # Chronos-2 full fine-tune config
    if args.model == "chronos2_full_ft" and hasattr(model, "ft_cfg"):
        ft = model.ft_cfg
        ft.finetune_mode = str(args.chronos2_ft_mode)
        ft.lr = float(args.chronos2_ft_lr)
        ft.num_steps = int(args.chronos2_ft_steps)
        ft.batch_size = int(args.chronos2_ft_batch_size)
        ft.logging_steps = int(args.chronos2_ft_logging_steps)
        ft.load_path = _as_none_if_empty(args.chronos2_ft_load)
        ft.save_path = _as_none_if_empty(args.chronos2_ft_save)

    dev = torch.device(args.device) if str(args.device).strip() else None

    model.fit(bundle, device=dev)

    max_eval = None if args.max_eval_batches <= 0 else int(args.max_eval_batches)
    val = evaluate_forecaster(model, bundle, split="val", device=dev, max_batches=max_eval)
    test = evaluate_forecaster(model, bundle, split="test", device=dev, max_batches=max_eval)

    finetune_cfg: Dict[str, Any] = {}
    if hasattr(model, "ft_cfg"):
        ft = getattr(model, "ft_cfg")
        try:
            finetune_cfg = {k: getattr(ft, k) for k in dir(ft) if not k.startswith("_")}
        except Exception:
            finetune_cfg = {}

    result = {
        "model": args.model,
        "val": val.__dict__,
        "test": test.__dict__,
        "n_val": len(bundle.datasets["val"]),
        "n_test": len(bundle.datasets["test"]),
        "pipeline": bundle.pipeline.summary(),
        "features_mode": str(args.features_mode),
        "pad_to": pad_to,
        "batch_size": int(args.batch_size),
        "device": str(dev) if dev is not None else "cpu",
        "foundation": {
            "timesfm_repo_id": str(args.timesfm_repo_id),
            "timesfm_infer_bs": int(args.timesfm_infer_bs),
            "timesfm_max_calib_batches": int(args.timesfm_max_calib_batches),
            "chronos_model_id": str(args.chronos_model_id),
            "chronos_infer_bs": int(args.chronos_infer_bs),
            "chronos_point": str(args.chronos_point),
            "chronos_max_calib_batches": int(args.chronos_max_calib_batches),
            "chronos2_cross_learning": bool(args.chronos2_cross_learning),
        },
    }
    if finetune_cfg:
        result["finetune"] = finetune_cfg

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

