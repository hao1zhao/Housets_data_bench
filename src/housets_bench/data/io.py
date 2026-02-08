
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .schema import FeatureSchema, normalize_zipcode


@dataclass
class AlignedData:
    zipcodes: List[str]
    dates: List[pd.Timestamp]
    values: np.ndarray
    time_marks: np.ndarray
    schema: FeatureSchema

    @property
    def n_zip(self) -> int:
        return len(self.zipcodes)

    @property
    def n_time(self) -> int:
        return len(self.dates)

    @property
    def n_features(self) -> int:
        return int(self.values.shape[-1])


def read_table(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {suffix}. Use csv/parquet/xlsx.")


def clean_raw_table(df: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    df = df.copy()

    # Drop empty rows
    if schema.time_col not in df.columns or schema.id_col not in df.columns:
        raise ValueError(f"Input df must contain {schema.id_col!r} and {schema.time_col!r}")

    df = df.dropna(subset=[schema.id_col, schema.time_col], how="any")

    # Parse dates
    if not pd.api.types.is_datetime64_any_dtype(df[schema.time_col]):
        df[schema.time_col] = pd.to_datetime(df[schema.time_col], errors="coerce")
    df = df.dropna(subset=[schema.time_col], how="any")

    # Normalize zipcode
    df[schema.id_col] = df[schema.id_col].map(normalize_zipcode)
    df = df[df[schema.id_col] != ""]

    # Drop non-feature columns
    for c in schema.drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Add time markers
    df["year"] = df[schema.time_col].dt.year.astype(int)
    df["month"] = df[schema.time_col].dt.month.astype(int)

    # Sort
    df = df.sort_values([schema.id_col, schema.time_col]).reset_index(drop=True)

    return df


def three_stage_impute(
    values: np.ndarray,
    *,
    per_feature_global_median: Optional[np.ndarray] = None,
) -> np.ndarray:
    x = values.copy()

    Z, T, D = x.shape
    if per_feature_global_median is None:
        per_feature_global_median = np.nanmedian(x.reshape(-1, D), axis=0)
        per_feature_global_median = np.where(np.isfinite(per_feature_global_median), per_feature_global_median, 0.0)

    for z in range(Z):
        for d in range(D):
            s = x[z, :, d]
            if np.all(np.isnan(s)):
                x[z, :, d] = per_feature_global_median[d]
                continue
            ss = pd.Series(s)
            ss = ss.ffill()
            s2 = ss.to_numpy()
            if np.isnan(s2).any():
                med = np.nanmedian(s2)
                if np.isfinite(med):
                    s2 = np.where(np.isnan(s2), med, s2)
            if np.isnan(s2).any():
                s2 = np.where(np.isnan(s2), per_feature_global_median[d], s2)

            x[z, :, d] = s2

    return x


def align_to_tensor(
    df: pd.DataFrame,
    schema: FeatureSchema,
    *,
    impute: bool = True,
    coerce_negative_to_zero: bool = True,
) -> AlignedData:
    df = df.copy()

    # Ensure schema.continuous_cols exist
    missing_cols = [c for c in schema.continuous_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns required by schema.continuous_cols: {missing_cols}")

    dates = sorted(df[schema.time_col].unique())
    dates = [pd.Timestamp(d) for d in dates]
    date_to_t = {d: i for i, d in enumerate(dates)}

    zipcodes = sorted(df[schema.id_col].unique())
    zip_to_i = {z: i for i, z in enumerate(zipcodes)}

    Z = len(zipcodes)
    T = len(dates)
    D = len(schema.continuous_cols)

    values = np.full((Z, T, D), np.nan, dtype=np.float32)

    # time marks [year, month] from date axis
    tm = np.zeros((T, 2), dtype=np.int64)
    for t, d in enumerate(dates):
        tm[t, 0] = int(d.year)
        tm[t, 1] = int(d.month)

    cols = list(schema.continuous_cols)

    for _, row in df.iterrows():
        z = row[schema.id_col]
        d = row[schema.time_col]
        zi = zip_to_i[z]
        ti = date_to_t[pd.Timestamp(d)]
        vals = row[cols].to_numpy(dtype=np.float32, copy=False)
        values[zi, ti, :] = vals

    if coerce_negative_to_zero:
        values = np.where(values < 0, 0.0, values)

    if impute:
        values = three_stage_impute(values)

    return AlignedData(
        zipcodes=list(zipcodes),
        dates=dates,
        values=values.astype(np.float32, copy=False),
        time_marks=tm,
        schema=schema,
    )


def load_aligned(
    path: Union[str, Path],
    *,
    schema: Optional[FeatureSchema] = None,
    target_col: str = "price",
    impute: bool = True,
) -> AlignedData:
    df = read_table(path)

    if schema is None:
        schema = FeatureSchema.infer(df, target_col=target_col)

    df = clean_raw_table(df, schema)
    return align_to_tensor(df, schema, impute=impute)
