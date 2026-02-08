from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from torch.utils.data import DataLoader

from housets_bench.data.io import AlignedData
from housets_bench.data.split import TimeSplit
from housets_bench.data.windowing import WindowSpec
from housets_bench.data.dataset import WindowDataset
from housets_bench.transforms.pipeline import TransformPipeline


@dataclass(frozen=True)
class RawBundle:
    aligned: AlignedData
    split: TimeSplit
    spec: WindowSpec
    features_mode: str


@dataclass(frozen=True)
class ProcBundle:
    raw: RawBundle
    pipeline: TransformPipeline
    aligned_proc: AlignedData

    x_cols: Tuple[str, ...]
    y_cols: Tuple[str, ...]

    datasets: Dict[str, WindowDataset]
    dataloaders: Dict[str, DataLoader]
    raw_target_col: str
    raw_target_index: int
