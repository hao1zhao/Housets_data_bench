from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


def load_metrics_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_runs(run_root: str | Path) -> pd.DataFrame:
    root = Path(run_root)
    rows: List[Dict[str, Any]] = []
    for metrics_path in root.rglob("metrics.json"):
        try:
            m = load_metrics_json(metrics_path)
        except Exception:
            continue
        row_base: Dict[str, Any] = {
            "run_dir": str(metrics_path.parent),
            "model": m.get("model"),
            "task": m.get("task"),
            "window": m.get("window"),
            "split_train": m.get("n_train"),
            "split_val": m.get("n_val"),
            "split_test": m.get("n_test"),
            "pipeline": m.get("pipeline"),
        }
        for split in ("val", "test"):
            if split not in m:
                continue
            s = m[split] or {}
            row = dict(row_base)
            row["split"] = split
            row["logrmse"] = s.get("logrmse")
            row["mape"] = s.get("mape")
            row["n_points"] = s.get("n_points")
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def pivot_metric(
    df: pd.DataFrame,
    *,
    task: str,
    split: str,
    metric: str,
    window_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    sub = df[(df["task"] == task) & (df["split"] == split)].copy()
    if sub.empty:
        return pd.DataFrame()

    piv = sub.pivot_table(index="model", columns="window", values=metric, aggfunc="first")
    if window_order is not None:
        cols = [c for c in window_order if c in piv.columns]
        piv = piv.reindex(columns=cols)
    piv = piv.sort_index()
    return piv
