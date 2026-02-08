
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train: Tuple[int, int]
    val: Tuple[int, int]
    test: Tuple[int, int]

    def range(self, name: str) -> Tuple[int, int]:
        name = name.lower()
        if name == "train":
            return self.train
        if name == "val":
            return self.val
        if name == "test":
            return self.test
        raise ValueError(f"Unknown split name: {name}")

    @property
    def n_time(self) -> int:
        return self.test[1]


def make_ratio_split(n_time: int, train_ratio: float = 0.7, val_ratio: float = 0.1) -> TimeSplit:
    if n_time <= 0:
        raise ValueError("n_time must be positive")
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be in (0,1)")
    if not (0 < val_ratio < 1):
        raise ValueError("val_ratio must be in (0,1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    train_end = int(round(n_time * train_ratio))
    val_end = int(round(n_time * (train_ratio + val_ratio)))

    # Ensure at least 1 time step in each split when possible
    train_end = max(1, min(train_end, n_time - 2))
    val_end = max(train_end + 1, min(val_end, n_time - 1))

    return TimeSplit(train=(0, train_end), val=(train_end, val_end), test=(val_end, n_time))


def make_date_split(
    dates: Sequence[pd.Timestamp],
    *,
    train_end: str,
    val_end: str,
) -> TimeSplit:
    if len(dates) == 0:
        raise ValueError("dates is empty")

    dts = [pd.Timestamp(d) for d in dates]
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    if train_end_ts >= val_end_ts:
        raise ValueError("train_end must be < val_end")

    # find first index > boundary
    def _end_exclusive(bound: pd.Timestamp) -> int:
        for i, d in enumerate(dts):
            if d > bound:
                return i
        return len(dts)

    train_end_ex = _end_exclusive(train_end_ts)
    val_end_ex = _end_exclusive(val_end_ts)

    if train_end_ex < 1:
        raise ValueError("train_end is before the first timestamp")
    if val_end_ex <= train_end_ex:
        raise ValueError("val_end is not after train_end in the provided date axis")
    if val_end_ex >= len(dts):
        # still allow, but then test would be empty
        pass

    return TimeSplit(train=(0, train_end_ex), val=(train_end_ex, val_end_ex), test=(val_end_ex, len(dts)))
