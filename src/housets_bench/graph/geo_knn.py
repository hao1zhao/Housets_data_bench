from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

try:
    from sklearn.neighbors import NearestNeighbors  
except Exception:  # pragma: no cover
    NearestNeighbors = None 


EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class GeoGraph:
    edge_index: np.ndarray
    edge_weight: np.ndarray


def _normalize_zip(z: str) -> str:
    z = str(z).strip()
    if z.isdigit() and len(z) < 5:
        z = z.zfill(5)
    return z


def build_knn_geo_graph(
    zipcodes: Iterable[str],
    latlon_by_zip: dict[str, Tuple[float, float]],
    *,
    k: int = 10,
    max_km: Optional[float] = 50.0,
    include_self_loops: bool = True,
    symmetric: bool = True,
) -> GeoGraph:

    zips = [_normalize_zip(z) for z in zipcodes]
    n = len(zips)
    coords = np.full((n, 2), np.nan, dtype=np.float64)
    has = np.zeros((n,), dtype=bool)

    for i, z in enumerate(zips):
        if z in latlon_by_zip:
            lat, lon = latlon_by_zip[z]
            if np.isfinite(lat) and np.isfinite(lon):
                coords[i, 0] = float(lat)
                coords[i, 1] = float(lon)
                has[i] = True

    # Build edges among nodes that have coords.
    idx_has = np.where(has)[0]
    coords_has = coords[idx_has]

    edges_src: list[int] = []
    edges_dst: list[int] = []

    if include_self_loops:
        for i in range(n):
            edges_src.append(i)
            edges_dst.append(i)

    if len(idx_has) > 0 and k > 0:
        if NearestNeighbors is None:
            rad = np.deg2rad(coords_has)  # [M,2]
            M = rad.shape[0]
            for ii in range(M):
                lat1, lon1 = rad[ii]
                dlat = rad[:, 0] - lat1
                dlon = rad[:, 1] - lon1
                a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(rad[:, 0]) * np.sin(dlon / 2.0) ** 2
                c = 2 * np.arcsin(np.sqrt(a))
                km = EARTH_RADIUS_KM * c
                # Exclude self by setting inf
                km[ii] = np.inf
                nn = np.argpartition(km, kth=min(k, M - 1))[: min(k, M - 1)]
                for jj in nn:
                    if (max_km is None) or (km[jj] <= max_km):
                        src = int(idx_has[ii])
                        dst = int(idx_has[jj])
                        edges_src.append(src)
                        edges_dst.append(dst)
        else:
            # sklearn uses radians for haversine; returns radians distance
            rad = np.deg2rad(coords_has)  # [M,2]
            nn = NearestNeighbors(
                n_neighbors=min(k + 1, len(idx_has)),
                algorithm="ball_tree",
                metric="haversine",
            )
            nn.fit(rad)
            dist_rad, nbrs = nn.kneighbors(rad, return_distance=True)
            dist_km = dist_rad * EARTH_RADIUS_KM

            for row in range(nbrs.shape[0]):
                src = int(idx_has[row])
                for col in range(1, nbrs.shape[1]): 
                    j = int(nbrs[row, col])
                    km = float(dist_km[row, col])
                    if (max_km is not None) and (km > max_km):
                        continue
                    dst = int(idx_has[j])
                    edges_src.append(src)
                    edges_dst.append(dst)

    if symmetric:
        # Add reverse edges
        rev_src = list(edges_dst)
        rev_dst = list(edges_src)
        edges_src.extend(rev_src)
        edges_dst.extend(rev_dst)

    edge_index = np.stack([np.array(edges_src, dtype=np.int64), np.array(edges_dst, dtype=np.int64)], axis=0)

    deg = np.zeros((n,), dtype=np.float64)
    for dst in edge_index[1]:
        deg[int(dst)] += 1.0
    deg = np.maximum(deg, 1.0) 

    src = edge_index[0].astype(np.int64)
    dst = edge_index[1].astype(np.int64)
    w = 1.0 / np.sqrt(deg[src] * deg[dst])
    edge_weight = w.astype(np.float32)

    return GeoGraph(edge_index=edge_index, edge_weight=edge_weight)
