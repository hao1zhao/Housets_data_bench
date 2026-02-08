# HouseTS: A Large-Scale, Multimodal Spatiotemporal U.S. Housing Dataset + Benchmark

This repository contains the  **benchmark** for **HouseTS**, a large-scale multimodal spatiotemporal dataset for long-horizon housing-market forecasting at the U.S. ZIP-code level.

HouseTS aligns multiple modalities under a unified ZIP-month panel, including:
- **Monthly housing-market indicators**
- **Monthly POI counts**
- **Annual census / socioeconomic variables** aligned to the monthly timeline
- (Dataset also includes auxiliary modalities such as aerial imagery + derived annotations; see Kaggle for full contents.)

The benchmark supports **univariate** and **multivariate** forecasting with standardized train/val/test splitting, windowing, transforms, and evaluation.

---

## Dataset

HouseTS data (tabular signals) is available via Google Drive:

- Google Drive download: https://drive.google.com/file/d/1OC_PTXfaGuQ50-mu2LkfQRLdhjPUbyu7/view?usp=sharing

HouseTS aerial imagery data is hosted on Kaggle:

- Kaggle dataset page: https://www.kaggle.com/datasets/shengkunwang/housets-dataset

### Expected local path

By default, the benchmark expects:

- `data/raw/HouseTS.csv`

You can also point to `.csv`, `.parquet`, or `.xlsx` via config/CLI.

### Minimal schema

Your tabular file should include at least:
- `zipcode` (ZIP code; will be normalized to a 5-digit string)
- `date` (timestamp; will be parsed)
- `price` (forecast target; default `data.target_col`)

All other numeric columns are treated as continuous covariates for multivariate settings.

> Notes:
> - Non-feature columns like `city` / `city_full` (if present) are dropped by default.
> - The loader adds `year` and `month` time markers from `date`.
> - Missing values are handled with a benchmark imputation routine.


## Quick start

All examples below are run from the repository root.

### 1) Run a single experiment (config-driven)

The config runner merges:
- `configs/default.yaml`
- `configs/task/<task>.yaml`
- `configs/windows/<window>.yaml`
- `configs/models/<model>.yaml`

Example (multivariate, window `w6_h3`, model `dlinear`):

```bash
python scripts/run_one.py \
  --task multivariate \
  --window w6_h3 \
  --model dlinear \
  --data data/raw/HouseTS.csv \
  --device gpu
