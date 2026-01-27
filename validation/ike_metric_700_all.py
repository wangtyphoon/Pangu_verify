from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from subset import extract_subset
from tc_tracker_bst_msl import align_lon_system, surface_to_dict, to_1d_lat_lon


PANGU_CHANNEL_LEVEL_MAP = {
    "u700": (3, 3),
    "v700": (4, 3),
    "t700": (2, 3),
}
SURFACE_T_FIELDS = ("msl", "t2m")
GAS_CONSTANT_DRY_AIR = 287.05  # J/(kg*K)
P700_PA = 70000.0


@dataclass(frozen=True)
class Config:
    csv_path: Path = Path("tc_aurora_metrics35_step12.csv")
    results_dir: Path = Path("../results")
    era5_dir: Path = Path("../forecasts")
    lat_path: Path = Path("lat.npy")
    lon_path: Path = Path("lon.npy")
    upper_nc_path: Path = Path("../upper.nc")
    output_path: Path = Path("tc_aurora_metrics35_step12.csv")
    radius_km: float = 750.0
    thickness_m: float = 1000.0


def parse_ymdh(value: float | int | str) -> datetime:
    text = str(int(float(value)))
    if len(text) != 10:
        raise ValueError(f"Expected YYYYMMDDHH format, got {text!r}")
    return datetime.strptime(text, "%Y%m%d%H")


def dt_to_run_dir(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d-%H-00")


def dt_to_upper_name(dt: datetime) -> str:
    return dt.strftime("output_upper_%Y-%m-%d-%H-%M.npy")


def dt_to_surface_name(dt: datetime) -> str:
    return dt.strftime("output_surface_%Y-%m-%d-%H-%M.npy")


def dt_to_era5_dir(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d-%H-00")


def build_global_lat_lon(H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    lat = np.linspace(90.0, -90.0, H)
    lon = np.linspace(0.0, 360.0 - 360.0 / W, W)
    return lat, lon


def select_bbox_slices(lat1d: np.ndarray, lon1d: np.ndarray, lat_range, lon_range) -> tuple[slice, slice]:
    lat_min, lat_max = min(lat_range), max(lat_range)
    lon_min, lon_max = min(lon_range), max(lon_range)
    lat_mask = (lat1d >= lat_min) & (lat1d <= lat_max)
    lon_mask = (lon1d >= lon_min) & (lon1d <= lon_max)
    if not np.any(lat_mask) or not np.any(lon_mask):
        raise ValueError("Subset range does not overlap grid.")
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    return slice(lat_idx.min(), lat_idx.max() + 1), slice(lon_idx.min(), lon_idx.max() + 1)


def subset_global_upper(
    data: np.ndarray,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
) -> dict[str, np.ndarray]:
    if data.ndim != 4:
        raise ValueError(f"Unsupported upper shape: {data.shape}")
    max_ch = max(ch for ch, _ in PANGU_CHANNEL_LEVEL_MAP.values()) + 1
    max_lv = max(lv for _, lv in PANGU_CHANNEL_LEVEL_MAP.values()) + 1
    if data.shape[0] >= max_ch and data.shape[1] >= max_lv:
        arr = data
    elif data.shape[1] >= max_ch and data.shape[0] >= max_lv:
        arr = np.transpose(data, (1, 0, 2, 3))
    else:
        raise ValueError(f"Unsupported upper shape: {data.shape}")
    H, W = arr.shape[-2:]
    lat1d, lon1d = build_global_lat_lon(H, W)
    lat_sl, lon_sl = select_bbox_slices(lat1d, lon1d, lat_range, lon_range)
    fields: dict[str, np.ndarray] = {}
    for name, (ch, lv) in PANGU_CHANNEL_LEVEL_MAP.items():
        fields[name] = arr[ch, lv, lat_sl, lon_sl]
    return fields


def upper_to_fields(
    data: np.ndarray,
    H: int,
    W: int,
    lat1d: np.ndarray,
    lon1d: np.ndarray,
    cfg: Config,
) -> dict[str, np.ndarray]:
    try:
        arr = np.asarray(data)
        if arr.ndim == 4 and arr.shape[-2:] == (H, W):
            if arr.shape[0] >= max(ch for ch, _ in PANGU_CHANNEL_LEVEL_MAP.values()) + 1 and arr.shape[1] >= max(
                lv for _, lv in PANGU_CHANNEL_LEVEL_MAP.values()
            ) + 1:
                clhw = arr
            else:
                clhw = np.transpose(arr, (1, 0, 2, 3))
            return {name: clhw[ch, lv, :, :] for name, (ch, lv) in PANGU_CHANNEL_LEVEL_MAP.items()}
        raise ValueError("Grid mismatch")
    except ValueError:
        arr = np.asarray(data)
        if arr.ndim == 4:
            lat_range = (float(np.nanmin(lat1d)), float(np.nanmax(lat1d)))
            lon_mod = np.mod(lon1d, 360.0)
            lon_range = (float(np.nanmin(lon_mod)), float(np.nanmax(lon_mod)))
            try:
                return subset_global_upper(arr, lat_range, lon_range)
            except ValueError:
                pass

        if not cfg.upper_nc_path.exists():
            raise FileNotFoundError(f"Missing upper.nc for target subset: {cfg.upper_nc_path}")
        return extract_subset(data, str(cfg.upper_nc_path), PANGU_CHANNEL_LEVEL_MAP)


def adjust_lon_grid(lon1d: np.ndarray, center_lon: float) -> tuple[np.ndarray, float]:
    lon_min, lon_max = float(np.nanmin(lon1d)), float(np.nanmax(lon1d))
    lon_center = align_lon_system(lon1d, center_lon)
    lon_grid = lon1d.copy()
    if lon_min >= 0 and lon_center < 0:
        lon_grid = lon_grid + 360.0
        lon_center += 360.0
    if lon_max <= 180 and lon_center > 180:
        lon_grid = lon_grid - 360.0
        lon_center -= 360.0
    return lon_grid, lon_center


def haversine_km(lat2d: np.ndarray, lon2d: np.ndarray, lat0: float, lon0: float) -> np.ndarray:
    radius_km = 6371.0
    lat0_rad = np.deg2rad(lat0)
    lon0_rad = np.deg2rad(lon0)
    lat2_rad = np.deg2rad(lat2d)
    lon2_rad = np.deg2rad(lon2d)
    dlat = lat2_rad - lat0_rad
    dlon = lon2_rad - lon0_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat0_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return radius_km * c


def grid_cell_area(lat1d: np.ndarray, lon1d: np.ndarray) -> np.ndarray:
    lat_rad = np.deg2rad(lat1d)
    lon_rad = np.deg2rad(lon1d)
    lat_edges = np.empty(lat_rad.size + 1)
    lon_edges = np.empty(lon_rad.size + 1)
    lat_edges[1:-1] = 0.5 * (lat_rad[:-1] + lat_rad[1:])
    lon_edges[1:-1] = 0.5 * (lon_rad[:-1] + lon_rad[1:])
    lat_edges[0] = lat_rad[0] - 0.5 * (lat_rad[1] - lat_rad[0])
    lat_edges[-1] = lat_rad[-1] + 0.5 * (lat_rad[-1] - lat_rad[-2])
    lon_edges[0] = lon_rad[0] - 0.5 * (lon_rad[1] - lon_rad[0])
    lon_edges[-1] = lon_rad[-1] + 0.5 * (lon_rad[-1] - lon_rad[-2])

    dlon = np.diff(lon_edges)
    lat_band = np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1])
    area = (6371000.0**2) * lat_band[:, None] * dlon[None, :]
    return np.abs(area)


def compute_ike(
    u: np.ndarray,
    v: np.ndarray,
    area_m2: np.ndarray,
    mask: np.ndarray,
    rho: np.ndarray,
    thickness_m: float,
) -> float:
    speed2 = np.square(u) + np.square(v)
    rho_arr = np.asarray(rho, dtype=float)
    valid = mask & np.isfinite(speed2) & np.isfinite(area_m2) & np.isfinite(rho_arr)
    ike = 0.5 * float(np.sum(rho_arr[valid] * speed2[valid] * area_m2[valid]))
    if thickness_m > 0.0:
        ike *= thickness_m
    return ike


def compute_ike_for_center(
    u: np.ndarray,
    v: np.ndarray,
    lat1d: np.ndarray,
    lon1d: np.ndarray,
    center_lat: float,
    center_lon: float,
    rho: np.ndarray,
    cfg: Config,
) -> float:
    lon1d_adj, center_lon_adj = adjust_lon_grid(lon1d, center_lon)
    lon2d = np.broadcast_to(lon1d_adj[None, :], (lat1d.size, lon1d.size))
    lat2d = np.broadcast_to(lat1d[:, None], (lat1d.size, lon1d.size))
    dist_km = haversine_km(lat2d, lon2d, center_lat, center_lon_adj)
    mask = dist_km <= cfg.radius_km
    area_m2 = grid_cell_area(lat1d, lon1d)
    return compute_ike(u, v, area_m2, mask, rho, cfg.thickness_m)


def main() -> None:
    cfg = Config()

    df = pd.read_csv(cfg.csv_path)
    df = df.copy()

    base_lat1d, base_lon1d = to_1d_lat_lon(np.load(cfg.lat_path), np.load(cfg.lon_path))

    cache_latlon: dict[Path, tuple[np.ndarray, np.ndarray]] = {}

    forecast_ike700 = []
    forecast_ike_surface = []
    target_ike700 = []
    target_ike_surface = []

    for _, row in df.iterrows():
        f_ike700 = float("nan")
        f_ikesfc = float("nan")
        t_ike700 = float("nan")
        t_ikesfc = float("nan")

        init_dt = parse_ymdh(row["init_time"])
        valid_dt = parse_ymdh(row["time"])

        fc_lat = row.get("forecast_lat", np.nan)
        fc_lon = row.get("forecast_lon", np.nan)
        if np.isfinite(fc_lat) and np.isfinite(fc_lon):
            run_dir = cfg.results_dir / dt_to_run_dir(init_dt)
            upper_path = run_dir / dt_to_upper_name(valid_dt)
            surface_path = run_dir / dt_to_surface_name(valid_dt)
            if upper_path.exists() and surface_path.exists():
                if run_dir not in cache_latlon:
                    lat_path = run_dir / "latitudes_subset.npy"
                    lon_path = run_dir / "longitudes_subset.npy"
                    if lat_path.exists() and lon_path.exists():
                        lat1d, lon1d = to_1d_lat_lon(np.load(lat_path), np.load(lon_path))
                        cache_latlon[run_dir] = (lat1d, lon1d)
                    else:
                        cache_latlon[run_dir] = (base_lat1d, base_lon1d)
                lat1d, lon1d = cache_latlon[run_dir]

                fields = upper_to_fields(np.load(upper_path), lat1d.size, lon1d.size, lat1d, lon1d, cfg)
                sfc = surface_to_dict(np.load(surface_path), lat1d.size, lon1d.size)
                rho700 = P700_PA / (GAS_CONSTANT_DRY_AIR * fields["t700"])
                rho10 = sfc[SURFACE_T_FIELDS[0]] / (GAS_CONSTANT_DRY_AIR * sfc[SURFACE_T_FIELDS[1]])

                f_ike700 = compute_ike_for_center(
                    fields["u700"],
                    fields["v700"],
                    lat1d,
                    lon1d,
                    float(fc_lat),
                    float(fc_lon),
                    rho700,
                    cfg,
                )
                f_ikesfc = compute_ike_for_center(
                    sfc["u10"],
                    sfc["v10"],
                    lat1d,
                    lon1d,
                    float(fc_lat),
                    float(fc_lon),
                    rho10,
                    cfg,
                )

        bst_lat = row.get("bst_lat", np.nan)
        bst_lon = row.get("bst_lon", np.nan)
        if np.isfinite(bst_lat) and np.isfinite(bst_lon):
            era5_dir = cfg.era5_dir / dt_to_era5_dir(valid_dt)
            target_upper = era5_dir / "input_upper.npy"
            target_surface = era5_dir / "input_surface.npy"
            if target_upper.exists() and target_surface.exists():
                fields = upper_to_fields(np.load(target_upper), base_lat1d.size, base_lon1d.size, base_lat1d, base_lon1d, cfg)
                sfc = surface_to_dict(np.load(target_surface), base_lat1d.size, base_lon1d.size)
                rho700 = P700_PA / (GAS_CONSTANT_DRY_AIR * fields["t700"])
                rho10 = sfc[SURFACE_T_FIELDS[0]] / (GAS_CONSTANT_DRY_AIR * sfc[SURFACE_T_FIELDS[1]])

                t_ike700 = compute_ike_for_center(
                    fields["u700"],
                    fields["v700"],
                    base_lat1d,
                    base_lon1d,
                    float(bst_lat),
                    float(bst_lon),
                    rho700,
                    cfg,
                )
                t_ikesfc = compute_ike_for_center(
                    sfc["u10"],
                    sfc["v10"],
                    base_lat1d,
                    base_lon1d,
                    float(bst_lat),
                    float(bst_lon),
                    rho10,
                    cfg,
                )

        forecast_ike700.append(f_ike700)
        forecast_ike_surface.append(f_ikesfc)
        target_ike700.append(t_ike700)
        target_ike_surface.append(t_ikesfc)

    df["forecast_ike700"] = forecast_ike700
    df["forecast_ike_surface"] = forecast_ike_surface
    df["target_ike700"] = target_ike700
    df["target_ike_surface"] = target_ike_surface

    df.to_csv(cfg.output_path, index=False)
    print(f"Saved: {cfg.output_path}")


if __name__ == "__main__":
    main()
