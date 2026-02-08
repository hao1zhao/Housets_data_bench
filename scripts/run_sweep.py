from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd

from housets_bench.experiments.artifacts import collect_env, make_run_dir, save_json, save_yaml
from housets_bench.experiments.sweep import run_one_cfg
from housets_bench.metrics.reporting import collect_runs, pivot_metric
from housets_bench.utils.config import deep_update, load_yaml, pop_cli_overrides, resolve_relpaths


DEFAULT_WINDOWS = ["w6_h3", "w6_h6", "w6_h12", "w12_h3", "w12_h6", "w12_h12"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--config-dir", type=str, default=str(REPO_ROOT / "configs"))
    p.add_argument("--tasks", type=str, default="multivariate", help="comma list: univariate,multivariate")
    p.add_argument("--windows", type=str, default=",".join(DEFAULT_WINDOWS), help="comma list of window ids")
    p.add_argument("--models", type=str, default="ar_univariate,arima,var,rf,xgb,dlinear,timemixer,patchtst,timesfm_zero,chronos_zero", help="comma list of model config names")

    # global overrides
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--n-zip", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max-eval-batches", type=int, default=None)

    p.add_argument("--out-root", type=str, default=str(REPO_ROOT / "runs"))
    p.add_argument("--sweep-name", type=str, default="section4_sweep")

    p.add_argument("--resume", action="store_true", help="skip runs with existing metrics.json")

    p.add_argument("--set", dest="overrides", action="append", default=None)
    return p.parse_args()


def _split_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    cfg_dir = Path(args.config_dir)

    tasks = _split_csv(args.tasks)
    windows = _split_csv(args.windows)
    models = _split_csv(args.models)

    sweep_dir = Path(args.out_root) / args.sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    save_json(sweep_dir / "env.json", collect_env())

    results: List[Dict[str, Any]] = []

    for task in tasks:
        for window in windows:
            for model in models:
                cfg: Dict[str, Any] = {}
                deep_update(cfg, load_yaml(cfg_dir / "default.yaml"))
                deep_update(cfg, load_yaml(cfg_dir / "task" / f"{task}.yaml"))
                deep_update(cfg, load_yaml(cfg_dir / "windows" / f"{window}.yaml"))
                deep_update(cfg, load_yaml(cfg_dir / "models" / f"{model}.yaml"))

                # apply CLI scalar overrides
                if args.data is not None:
                    cfg.setdefault("data", {})["path"] = args.data
                if args.n_zip is not None:
                    cfg.setdefault("data", {})["n_zip"] = int(args.n_zip)
                if args.device is not None:
                    cfg.setdefault("run", {})["device"] = args.device
                if args.max_eval_batches is not None:
                    cfg.setdefault("run", {})["max_eval_batches"] = int(args.max_eval_batches)

                deep_update(cfg, pop_cli_overrides(args.overrides))
                resolve_relpaths(cfg, root=REPO_ROOT)

                run_name = f"{model}__{task}__{window}"
                run_dir = sweep_dir / run_name
                metrics_path = run_dir / "metrics.json"

                if args.resume and metrics_path.exists():
                    try:
                        with metrics_path.open("r", encoding="utf-8") as f:
                            results.append(json.load(f))
                        continue
                    except Exception:
                        pass

                paths = make_run_dir(root=sweep_dir, name=run_name, exist_ok=True)
                save_yaml(paths.config_path, cfg)

                res = run_one_cfg(cfg=cfg, device=cfg.get("run", {}).get("device", None))
                save_json(paths.metrics_path, res)
                results.append(res)

                print(json.dumps(res, ensure_ascii=False))

    # write a long-form summary csv
    df = collect_runs(sweep_dir)
    df.to_csv(sweep_dir / "summary_long.csv", index=False)

    # pivot tables for quick viewing
    for task in sorted(set(df["task"].dropna().unique().tolist())):
        for metric in ("logrmse", "mape"):
            piv = pivot_metric(df, task=task, split="test", metric=metric, window_order=DEFAULT_WINDOWS)
            if not piv.empty:
                piv.to_csv(sweep_dir / f"pivot_test_{metric}_{task}.csv")

    print(f"Wrote: {sweep_dir/'summary_long.csv'}")


if __name__ == "__main__":
    main()
