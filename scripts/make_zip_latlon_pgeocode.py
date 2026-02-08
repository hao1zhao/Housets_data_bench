from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="HouseTS.csv path")
    p.add_argument("--out", type=str, required=True, help="output csv: zipcode,lat,lon")
    p.add_argument("--country", type=str, default="us", help="pgeocode country code, e.g. 'us'")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path, usecols=["zipcode"])
    zips = sorted({str(z).zfill(5) for z in df["zipcode"].astype(str).tolist()})

    try:
        import pgeocode  
    except Exception as e:
        raise RuntimeError(
            "pgeocode is not installed. Install via: pip install pgeocode\n"
            f"Original error: {type(e).__name__}: {e}"
        )

    nomi = pgeocode.Nominatim(args.country)
    q = nomi.query_postal_code(zips)
    if not isinstance(q, pd.DataFrame):
        raise RuntimeError("Unexpected pgeocode output type")

    q = q.reset_index().rename(columns={"postal_code": "zipcode"})
    if "zipcode" not in q.columns:
        if "index" in q.columns:
            q = q.rename(columns={"index": "zipcode"})
        else:
            raise RuntimeError(f"Could not find zipcode column in pgeocode output. Columns={list(q.columns)}")

    q["zipcode"] = q["zipcode"].astype(str).str.zfill(5)
    out = q[["zipcode", "latitude", "longitude"]].rename(columns={"latitude": "lat", "longitude": "lon"})
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()
