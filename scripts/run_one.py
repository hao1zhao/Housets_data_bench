from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from housets_bench.experiments.artifacts import collect_env, make_run_dir, save_json, save_yaml
from housets_bench.experiments.sweep import run_one_cfg
from housets_bench.utils.config import deep_update, load_yaml, pop_cli_overrides, resolve_relpaths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--config-dir", type=str, default=str(REPO_ROOT / "configs"))
    p.add_argument("--task", type=str, default="multivariate", choices=["univariate", "multivariate"])
    p.add_argument(
        "--window",
        type=str,
        default="w6_h3",
        choices=["w6_h3", "w6_h6", "w6_h12", "w12_h3", "w12_h6", "w12_h12"],
    )
    p.add_argument(
        "--model",
        type=str,
        default="xgb",
        help="model config name under configs/models",
    )

    # common overrides
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--n-zip", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max-eval-batches", type=int, default=None)

    # artifacts
    p.add_argument("--out-root", type=str, default=str(REPO_ROOT / "runs"))
    p.add_argument("--run-name", type=str, default=None, help="optional run dir name (otherwise auto)")

    # arbitrary overrides: --set a.b=1 --set x=true
    p.add_argument("--set", dest="overrides", action="append", default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_dir = Path(args.config_dir)

    cfg: Dict[str, Any] = {}
    deep_update(cfg, load_yaml(cfg_dir / "default.yaml"))
    deep_update(cfg, load_yaml(cfg_dir / "task" / f"{args.task}.yaml"))
    deep_update(cfg, load_yaml(cfg_dir / "windows" / f"{args.window}.yaml"))
    deep_update(cfg, load_yaml(cfg_dir / "models" / f"{args.model}.yaml"))

    # apply CLI scalar overrides
    if args.data is not None:
        cfg.setdefault("data", {})["path"] = args.data
    if args.n_zip is not None:
        cfg.setdefault("data", {})["n_zip"] = int(args.n_zip)
    if args.device is not None:
        cfg.setdefault("run", {})["device"] = args.device
    if args.max_eval_batches is not None:
        cfg.setdefault("run", {})["max_eval_batches"] = int(args.max_eval_batches)

    # apply --set overrides
    deep_update(cfg, pop_cli_overrides(args.overrides))

    # resolve relative paths
    resolve_relpaths(cfg, root=REPO_ROOT)

    # run
    result = run_one_cfg(cfg=cfg, device=cfg.get("run", {}).get("device", None))

    # name + save
    model = str(result.get("model"))
    task = str(result.get("task"))
    window = str(result.get("window"))

    run_name = args.run_name or f"{model}__{task}__{window}"
    paths = make_run_dir(root=args.out_root, name=run_name, exist_ok=True)

    save_yaml(paths.config_path, cfg)
    save_json(paths.metrics_path, result)
    save_json(paths.env_path, collect_env())

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
