from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
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
    step_target: int = 1
    row_index: int = 0
    center_source: str = "forecast"  # "forecast" or "bst"
    r8_threshold: float = 8.0
    max_radius_km: float = 2000.0
    radial_bin_km: float = 25.0
    smooth_window: int = 5
    output_png: Path = Path("axisymmetric_s10_step12.png")


def parse_ymdh(value: float | int | str) -> datetime:
    text = str(int(float(value)))
    if len(text) != 10:
        raise ValueError(f"Expected YYYYMMDDHH format, got {text!r}")
    return datetime.strptime(text, "%Y%m%d%H")


def dt_to_run_dir(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d-%H-00")


def dt_to_surface_name(dt: datetime) -> str:
    return dt.strftime("output_surface_%Y-%m-%d-%H-%M.npy")


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


def choose_center(row: pd.Series, source: str) -> tuple[float, float]:
    if source == "bst":
        lat, lon = row["bst_lat"], row["bst_lon"]
    else:
        lat, lon = row["forecast_lat"], row["forecast_lon"]
    if not np.isfinite(lat) or not np.isfinite(lon):
        lat, lon = row["bst_lat"], row["bst_lon"]
    if not np.isfinite(lat) or not np.isfinite(lon):
        raise ValueError("Center location is NaN for both forecast and bst.")
    return float(lat), float(lon)


def circle_latlon(
    center_lat: float,
    center_lon: float,
    radius_km: float,
    lon_axis_1d: np.ndarray,
    points: int = 180,
) -> tuple[np.ndarray, np.ndarray]:
    radius_earth_km = 6371.0
    lat0 = np.deg2rad(center_lat)
    lon0 = np.deg2rad(center_lon)
    bearings = np.deg2rad(np.linspace(0.0, 360.0, points))
    ang = radius_km / radius_earth_km

    sin_lat0 = np.sin(lat0)
    cos_lat0 = np.cos(lat0)
    sin_ang = np.sin(ang)
    cos_ang = np.cos(ang)

    lat = np.arcsin(sin_lat0 * cos_ang + cos_lat0 * sin_ang * np.cos(bearings))
    lon = lon0 + np.arctan2(
        np.sin(bearings) * sin_ang * cos_lat0,
        cos_ang - sin_lat0 * np.sin(lat),
    )

    lat_deg = np.rad2deg(lat)
    lon_deg = np.rad2deg(lon)
    lon_min, lon_max = float(np.nanmin(lon_axis_1d)), float(np.nanmax(lon_axis_1d))
    if lon_min >= 0:
        lon_deg = np.mod(lon_deg, 360.0)
    elif lon_max <= 180:
        lon_deg = (lon_deg + 180.0) % 360.0 - 180.0
    return lat_deg, lon_deg


def main() -> None:
    cfg = Config()

    df = pd.read_csv(cfg.csv_path)
    df_step = df[df["step"] == cfg.step_target].reset_index(drop=True)
    if df_step.empty:
        raise ValueError(f"No rows found for step={cfg.step_target}.")
    if cfg.row_index >= len(df_step):
        raise IndexError(f"row_index={cfg.row_index} out of range for step={cfg.step_target}.")

    row = df_step.loc[cfg.row_index]
    init_dt = parse_ymdh(row["init_time"])
    valid_dt = parse_ymdh(row["time"])
    run_dir = cfg.results_dir / dt_to_run_dir(init_dt)
    surface_path = run_dir / dt_to_surface_name(valid_dt)

    if not surface_path.exists():
        raise FileNotFoundError(f"Surface file not found: {surface_path}")

    lat_path = run_dir / "latitudes_subset.npy"
    lon_path = run_dir / "longitudes_subset.npy"
    if not lat_path.exists() or not lon_path.exists():
        raise FileNotFoundError(f"Missing lat/lon subset files under {run_dir}")

    lat_raw = np.load(lat_path)
    lon_raw = np.load(lon_path)
    lat1d, lon1d = to_1d_lat_lon(lat_raw, lon_raw)
    H, W = lat1d.size, lon1d.size

    surface = np.load(surface_path)
    sfc = surface_to_dict(surface, H, W)
    u10 = sfc["u10"]
    v10 = sfc["v10"]
    s10 = compute_wind_speed(u10, v10)

    center_lat, center_lon = choose_center(row, cfg.center_source)
    lon1d_adj, center_lon_adj = adjust_lon_grid(lon1d, center_lon)
    lon2d = np.broadcast_to(lon1d_adj[None, :], (H, W))
    lat2d = np.broadcast_to(lat1d[:, None], (H, W))

    dist_km = haversine_km(lat2d, lon2d, center_lat, center_lon_adj)
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    pcm = axes[0].pcolormesh(lon2d, lat2d, s10, shading="auto", cmap="turbo")
    axes[0].plot(center_lon_adj, center_lat, "ko", markersize=4)
    axes[0].set_title("S10 (m/s) with Rmax / R8")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    fig.colorbar(pcm, ax=axes[0], label="m/s")

    if np.isfinite(rmax_km):
        c_lat, c_lon = circle_latlon(center_lat, center_lon_adj, rmax_km, lon1d_adj)
        axes[0].plot(c_lon, c_lat, "k--", linewidth=1.2, label=f"Rmax {rmax_km:.0f} km")
    if np.isfinite(r8_km):
        c_lat, c_lon = circle_latlon(center_lat, center_lon_adj, r8_km, lon1d_adj)
        axes[0].plot(c_lon, c_lat, "k-", linewidth=1.2, label=f"R8 {r8_km:.0f} km")
    axes[0].legend(loc="upper right")

    axes[1].plot(r_km, profile, color="0.5", label="axisymmetric")
    axes[1].plot(r_km, profile_smooth, color="k", label="smoothed")
    if np.isfinite(rmax_km):
        axes[1].axvline(rmax_km, color="k", linestyle="--", linewidth=1.0)
    if np.isfinite(r8_km):
        axes[1].axvline(r8_km, color="k", linestyle="-", linewidth=1.0)
    axes[1].set_xlim(0.0, cfg.max_radius_km)
    axes[1].set_xlabel("Radius (km)")
    axes[1].set_ylabel("Wind speed (m/s)")
    axes[1].set_title("Axisymmetric S10 profile")
    axes[1].legend(loc="upper right")

    fig.suptitle(
        f"Init {init_dt:%Y-%m-%d %H} | Valid {valid_dt:%Y-%m-%d %H} | "
        f"step={cfg.step_target} row={cfg.row_index}"
    )
    #fig.savefig(cfg.output_png, dpi=150)
    #plt.close(fig)
    plt.show()

    print(f"Saved: {cfg.output_png}")
    print(f"Center: lat={center_lat:.3f}, lon={center_lon_adj:.3f}")
    print(f"Vmax_axisymmetric={vmax:.2f} m/s")
    print(f"Rmax_km={rmax_km:.1f}, R8_km={r8_km:.1f}")


if __name__ == "__main__":
    main()
