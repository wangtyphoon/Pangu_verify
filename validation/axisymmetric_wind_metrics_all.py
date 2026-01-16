from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tc_tracker_bst_msl import (
    align_lon_system,
    compute_wind_speed,
    surface_to_dict,
    to_1d_lat_lon,
)


@dataclass(frozen=True)
class Config:
    csv_path: Path = Path("tc_bst_msl_metrics35_step12.csv")
    results_dir: Path = Path("../results")
    era5_dir: Path = Path("../forecasts")
    lat_path: Path = Path("lat.npy")
    lon_path: Path = Path("lon.npy")
    output_path: Path = Path("tc_bst_msl_metrics35_step12.csv")
    r8_threshold: float = 8.0
    max_radius_km: float = 2000.0
    radial_bin_km: float = 25.0
    smooth_window: int = 5


def parse_ymdh(value: float | int | str) -> datetime:
    text = str(int(float(value)))
    if len(text) != 10:
        raise ValueError(f"Expected YYYYMMDDHH format, got {text!r}")
    return datetime.strptime(text, "%Y%m%d%H")


def dt_to_run_dir(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d-%H-00")


def dt_to_surface_name(dt: datetime) -> str:
    return dt.strftime("output_surface_%Y-%m-%d-%H-%M.npy")


def dt_to_era5_dir(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d-%H-00")


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


def radial_profile(
    distance_km: np.ndarray,
    values: np.ndarray,
    max_radius_km: float,
    bin_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    bins = np.arange(0.0, max_radius_km + bin_km, bin_km)
    centers = 0.5 * (bins[:-1] + bins[1:])
    profile = np.full(centers.shape, np.nan)

    dist_flat = distance_km.ravel()
    val_flat = values.ravel()
    valid = np.isfinite(dist_flat) & np.isfinite(val_flat)
    if not np.any(valid):
        return centers, profile

    dist_flat = dist_flat[valid]
    val_flat = val_flat[valid]
    bin_index = np.digitize(dist_flat, bins) - 1

    for i in range(centers.size):
        mask = bin_index == i
        if np.any(mask):
            profile[i] = float(np.mean(val_flat[mask]))
    return centers, profile


def smooth_profile(profile: np.ndarray, window: int) -> np.ndarray:
    if window < 3 or window % 2 == 0:
        return profile.copy()
    kernel = np.ones(window, dtype=float)
    valid = np.isfinite(profile).astype(float)
    num = np.convolve(np.nan_to_num(profile) * valid, kernel, mode="same")
    den = np.convolve(valid, kernel, mode="same")
    smooth = np.full(profile.shape, np.nan)
    mask = den > 0
    smooth[mask] = num[mask] / den[mask]
    return smooth


def compute_metrics(
    s10: np.ndarray,
    lat1d: np.ndarray,
    lon1d: np.ndarray,
    center_lat: float,
    center_lon: float,
    cfg: Config,
) -> tuple[float, float, float]:
    lon1d_adj, center_lon_adj = adjust_lon_grid(lon1d, center_lon)
    H, W = lat1d.size, lon1d.size
    lon2d = np.broadcast_to(lon1d_adj[None, :], (H, W))
    lat2d = np.broadcast_to(lat1d[:, None], (H, W))

    dist_km = haversine_km(lat2d, lon2d, center_lat, center_lon_adj)
    within = dist_km <= cfg.max_radius_km
    if not np.any(within):
        return float("nan"), float("nan"), float("nan")

    r_km, profile = radial_profile(dist_km, s10, cfg.max_radius_km, cfg.radial_bin_km)
    profile_smooth = smooth_profile(profile, cfg.smooth_window)

    if np.any(np.isfinite(profile_smooth)):
        vmax = float(np.nanmax(profile_smooth))
    else:
        vmax = float("nan")

    if np.any(np.isfinite(profile_smooth)):
        rmax_km = float(r_km[int(np.nanargmax(profile_smooth))])
    else:
        rmax_km = float("nan")

    r8_mask = np.isfinite(profile_smooth) & (profile_smooth >= cfg.r8_threshold)
    r8_km = float(np.nanmax(r_km[r8_mask])) if np.any(r8_mask) else float("nan")
    return vmax, rmax_km, r8_km


def main() -> None:
    cfg = Config()

    df = pd.read_csv(cfg.csv_path)
    df = df.copy()

    base_lat = np.load(cfg.lat_path)
    base_lon = np.load(cfg.lon_path)
    base_lat1d, base_lon1d = to_1d_lat_lon(base_lat, base_lon)

    cache_latlon: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    cache_s10: dict[Path, np.ndarray] = {}

    forecast_vmax = []
    forecast_rmax = []
    forecast_r8 = []
    target_vmax = []
    target_rmax = []
    target_r8 = []

    for _, row in df.iterrows():
        f_vmax = float("nan")
        f_rmax = float("nan")
        f_r8 = float("nan")
        t_vmax = float("nan")
        t_rmax = float("nan")
        t_r8 = float("nan")

        init_dt = parse_ymdh(row["init_time"])
        valid_dt = parse_ymdh(row["time"])

        fc_lat = row.get("forecast_lat", np.nan)
        fc_lon = row.get("forecast_lon", np.nan)
        if np.isfinite(fc_lat) and np.isfinite(fc_lon):
            run_dir = cfg.results_dir / dt_to_run_dir(init_dt)
            surface_path = run_dir / dt_to_surface_name(valid_dt)
            if surface_path.exists():
                if run_dir not in cache_latlon:
                    lat_path = run_dir / "latitudes_subset.npy"
                    lon_path = run_dir / "longitudes_subset.npy"
                    if lat_path.exists() and lon_path.exists():
                        lat1d, lon1d = to_1d_lat_lon(np.load(lat_path), np.load(lon_path))
                        cache_latlon[run_dir] = (lat1d, lon1d)
                    else:
                        cache_latlon[run_dir] = (base_lat1d, base_lon1d)
                lat1d, lon1d = cache_latlon[run_dir]

                if surface_path not in cache_s10:
                    surface = np.load(surface_path)
                    H, W = lat1d.size, lon1d.size
                    sfc = surface_to_dict(surface, H, W)
                    cache_s10[surface_path] = compute_wind_speed(sfc["u10"], sfc["v10"])
                s10 = cache_s10[surface_path]

                f_vmax, f_rmax, f_r8 = compute_metrics(
                    s10,
                    lat1d,
                    lon1d,
                    float(fc_lat),
                    float(fc_lon),
                    cfg,
                )

        bst_lat = row.get("bst_lat", np.nan)
        bst_lon = row.get("bst_lon", np.nan)
        if np.isfinite(bst_lat) and np.isfinite(bst_lon):
            era5_dir = cfg.era5_dir / dt_to_era5_dir(valid_dt)
            target_surface = era5_dir / "input_surface.npy"
            if target_surface.exists():
                if target_surface not in cache_s10:
                    surface = np.load(target_surface)
                    H, W = base_lat1d.size, base_lon1d.size
                    sfc = surface_to_dict(surface, H, W)
                    cache_s10[target_surface] = compute_wind_speed(sfc["u10"], sfc["v10"])
                s10 = cache_s10[target_surface]

                t_vmax, t_rmax, t_r8 = compute_metrics(
                    s10,
                    base_lat1d,
                    base_lon1d,
                    float(bst_lat),
                    float(bst_lon),
                    cfg,
                )

        forecast_vmax.append(f_vmax)
        forecast_rmax.append(f_rmax)
        forecast_r8.append(f_r8)
        target_vmax.append(t_vmax)
        target_rmax.append(t_rmax)
        target_r8.append(t_r8)

    df["forecast_vmax"] = forecast_vmax
    df["forecast_rmax"] = forecast_rmax
    df["forecast_r8"] = forecast_r8
    df["target_vmax"] = target_vmax
    df["target_rmax"] = target_rmax
    df["target_r8"] = target_r8

    df.to_csv(cfg.output_path, index=False)
    print(f"Saved: {cfg.output_path}")


if __name__ == "__main__":
    main()
