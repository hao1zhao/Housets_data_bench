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
```
### 2) Run a univariate baseline

```bash
python scripts/run_one.py \
  --task univariate \
  --window w12_h6 \
  --model ar_univariate \
  --data data/raw/HouseTS.csv \
  --device cpu
```

---

## Window Presets

The repository currently provides the following window presets:

- `w6_h3`
- `w6_h6`
- `w6_h12`
- `w12_h3`
- `w12_h6`
- `w12_h12`

For example:

- `w6_h3`: `seq_len=6`, `label_len=3`, `pred_len=3`
- `w12_h6`: `seq_len=12`, `label_len=6`, `pred_len=6`

---

## Supported Model Configs

The current `configs/models/` directory includes the following model configs.

### Statistical baselines

- `ar_univariate`
- `ardl`
- `arima`
- `var`
- `var_ms`

### Classical machine learning

- `rf`
- `xgb`

### Deep learning

- `rnn`
- `lstm`
- `dlinear`
- `timemixer`
- `patchtst`
- `informer`
- `autoformer`
- `fedformer`

### Foundation-model variants

- `chronos2_zero`
- `chronos2_ft`
- `timesfm_xreg_zero`
- `timesfm_xreg_ft`

---

## Data Usage and Attribution

HouseTS integrates or aligns signals derived from several public data sources, including:

- housing-market time series
- OpenStreetMap-derived POI statistics
- U.S. Census / ACS socioeconomic variables
- USDA NAIP aerial imagery

Please review the paper and the upstream data-source licensing / attribution requirements before redistribution, publication of derivatives, or commercial use.

---

## Citation

If you use HouseTS or this benchmark code in your research, please cite:

```bibtex
@article{wang2025housets,
  title={HouseTS: A Large-Scale, Multimodal Spatiotemporal U.S. Housing Dataset and Benchmark},
  author={Wang, Shengkun and Sun, Yanshen and Chen, Fanglan and Wang, Linhan and Ramakrishnan, Naren and Lu, Chang-Tien and Chen, Yinlin},
  journal={arXiv preprint arXiv:2506.00765},
  year={2025}
}
```

---
