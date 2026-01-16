from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from subset import extract_subset
from tc_tracker_bst_msl import align_lon_system, to_1d_lat_lon


PANGU_CHANNEL_LEVEL_MAP = {
    "z700": (0, 3),
    "u700": (3, 3),
    "v700": (4, 3),
}
PANGU_CHANNELS = 5
PANGU_LEVELS = 13


@dataclass(frozen=True)
class Config:
    csv_path: Path = Path("tc_bst_msl_metrics35_step12.csv")
    results_dir: Path = Path("../results")
    era5_dir: Path = Path("../forecasts")
    lat_path: Path = Path("lat.npy")
    lon_path: Path = Path("lon.npy")
    upper_nc_path: Path = Path("../upper.nc")
    step_target: int = 1
    row_index: int = 0
    max_radius_km: float = 2000.0
    radial_bin_km: float = 25.0
    target_radius_km: float = 100.0
    z_is_geopotential: bool = True


def parse_ymdh(value: float | int | str) -> datetime:
    text = str(int(float(value)))
    if len(text) != 10:
        raise ValueError(f"Expected YYYYMMDDHH format, got {text!r}")
    return datetime.strptime(text, "%Y%m%d%H")


def dt_to_run_dir(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d-%H-00")


def dt_to_upper_name(dt: datetime) -> str:
    return dt.strftime("output_upper_%Y-%m-%d-%H-%M.npy")


def dt_to_era5_dir(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d-%H-00")


def normalize_lon(lon_axis_1d: np.ndarray, lon_value: float) -> float:
    v = align_lon_system(lon_axis_1d, lon_value)
    lo_min, lo_max = np.nanmin(lon_axis_1d), np.nanmax(lon_axis_1d)
    if lo_min >= 0 and lo_max > 180:
        v = v % 360.0
    elif lo_max <= 180:
        v = ((v + 180.0) % 360.0) - 180.0
    return float(v)


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


def axisymmetric_mean(
    field2d: np.ndarray,
    lat1d: np.ndarray,
    lon1d: np.ndarray,
    center_lat: float,
    center_lon: float,
    bin_km: float,
    max_radius_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    bins = np.arange(0.0, max_radius_km + bin_km, bin_km)
    centers = 0.5 * (bins[:-1] + bins[1:])
    profile = np.full(centers.shape, np.nan)

    lat2d = lat1d[:, None]
    lon2d = lon1d[None, :]
    dist_km = haversine_km(lat2d, lon2d, center_lat, center_lon)

    for i in range(centers.size):
        mask = (dist_km >= bins[i]) & (dist_km < bins[i + 1])
        if np.any(mask):
            profile[i] = float(np.nanmean(field2d[mask]))
    return centers, profile


def interp_profile(r_km: np.ndarray, profile: np.ndarray, target_km: float) -> float:
    mask = np.isfinite(profile) & np.isfinite(r_km)
    if not np.any(mask):
        return float("nan")
    r_valid = r_km[mask]
    p_valid = profile[mask]
    return float(np.interp(target_km, r_valid, p_valid))


def nearest_grid_value(
    field2d: np.ndarray,
    lat1d: np.ndarray,
    lon1d: np.ndarray,
    center_lat: float,
    center_lon: float,
) -> float:
    i = int(np.nanargmin(np.abs(lat1d - center_lat)))
    j = int(np.nanargmin(np.abs(lon1d - center_lon)))
    return float(field2d[i, j])


def normalize_to_clhw(arr: np.ndarray, H: int, W: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[-2:] == (H, W):
        if arr.shape[0] == PANGU_CHANNELS:
            return arr
        if arr.shape[1] == PANGU_CHANNELS:
            return np.transpose(arr, (1, 0, 2, 3))
    if arr.ndim == 3 and arr.shape[-1] == H * W:
        return arr.reshape(arr.shape[0], arr.shape[1], H, W)
    raise ValueError(f"Unsupported upper shape: {arr.shape}")


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
    if data.shape[0] == PANGU_CHANNELS and data.shape[1] >= PANGU_LEVELS:
        arr = data
    elif data.shape[1] == PANGU_CHANNELS and data.shape[0] >= PANGU_LEVELS:
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


def upper_to_fields(data: np.ndarray, H: int, W: int, lat1d: np.ndarray, lon1d: np.ndarray, cfg: Config) -> dict[str, np.ndarray]:
    try:
        clhw = normalize_to_clhw(data, H, W)
        return {name: clhw[ch, lv, :, :] for name, (ch, lv) in PANGU_CHANNEL_LEVEL_MAP.items()}
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


def tangential_wind(
    u: np.ndarray,
    v: np.ndarray,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    center_lat: float,
    center_lon: float,
) -> np.ndarray:
    km_per_deg = 111.32
    lat0_rad = np.deg2rad(center_lat)
    dy = (lat2d - center_lat) * km_per_deg
    dx = (lon2d - center_lon) * km_per_deg * np.cos(lat0_rad)
    theta = np.arctan2(dy, dx)
    return -u * np.sin(theta) + v * np.cos(theta)


def compute_gwb(
    z700: np.ndarray,
    u700: np.ndarray,
    v700: np.ndarray,
    lat1d: np.ndarray,
    lon1d: np.ndarray,
    center_lat: float,
    center_lon: float,
    cfg: Config,
) -> dict[str, float]:
    center_lon_norm = normalize_lon(lon1d, center_lon)
    H, W = lat1d.size, lon1d.size
    lon2d = np.broadcast_to(lon1d[None, :], (H, W))
    lat2d = np.broadcast_to(lat1d[:, None], (H, W))

    vtheta = tangential_wind(u700, v700, lat2d, lon2d, center_lat, center_lon_norm)

    max_bin_radius_km = cfg.target_radius_km + cfg.radial_bin_km
    r_km, vtheta_profile = axisymmetric_mean(
        vtheta,
        lat1d,
        lon1d,
        center_lat,
        center_lon_norm,
        cfg.radial_bin_km,
        max_bin_radius_km,
    )
    _, z_profile = axisymmetric_mean(
        z700,
        lat1d,
        lon1d,
        center_lat,
        center_lon_norm,
        cfg.radial_bin_km,
        max_bin_radius_km,
    )

    z0 = nearest_grid_value(z700, lat1d, lon1d, center_lat, center_lon_norm)
    zR = interp_profile(r_km, z_profile, cfg.target_radius_km)
    delta_z_sim = zR - z0 if np.isfinite(zR) and np.isfinite(z0) else float("nan")

    omega = 7.2921159e-5
    f = 2.0 * omega * np.sin(np.deg2rad(center_lat))
    r_m = r_km * 1000.0
    r_safe = np.where(r_m > 0.0, r_m, np.nan)
    integrand = vtheta_profile * vtheta_profile / r_safe + f * vtheta_profile
    max_integral_km = cfg.target_radius_km + cfg.radial_bin_km
    mask = np.isfinite(integrand) & np.isfinite(r_m) & (r_km <= max_integral_km)
    if np.sum(mask) >= 2:
        delta_z_gwb = float(np.trapz(integrand[mask], r_m[mask]) / 9.80665)
    else:
        delta_z_gwb = float("nan")

    return {
        "delta_z_sim": float(delta_z_sim),
        "delta_z_gwb": float(delta_z_gwb),
        "z0": float(z0),
        "zR": float(zR),
        "center_lat": float(center_lat),
        "center_lon": float(center_lon_norm),
    }


def load_upper_fields(
    upper_path: Path,
    lat1d: np.ndarray,
    lon1d: np.ndarray,
    cfg: Config,
) -> dict[str, np.ndarray]:
    if not upper_path.exists():
        raise FileNotFoundError(f"Upper file not found: {upper_path}")
    upper = np.load(upper_path)
    fields = upper_to_fields(upper, lat1d.size, lon1d.size, lat1d, lon1d, cfg)
    if cfg.z_is_geopotential:
        fields["z700"] = fields["z700"] / 9.80665
    return fields


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


def print_result(label: str, result: dict[str, float], radius_km: float) -> None:
    print(f"[{label}] center lat={result['center_lat']:.3f}, lon={result['center_lon']:.3f}")
    print(f"[{label}] Z0={result['z0']:.3f}, ZR({radius_km:.0f}km)={result['zR']:.3f}")
    print(f"[{label}] delta_z_sim={result['delta_z_sim']:.3f}")
    print(f"[{label}] delta_z_gwb={result['delta_z_gwb']:.3f}")


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
    upper_path = run_dir / dt_to_upper_name(valid_dt)

    lat_path = run_dir / "latitudes_subset.npy"
    lon_path = run_dir / "longitudes_subset.npy"
    if lat_path.exists() and lon_path.exists():
        lat1d, lon1d = to_1d_lat_lon(np.load(lat_path), np.load(lon_path))
    else:
        lat1d, lon1d = to_1d_lat_lon(np.load(cfg.lat_path), np.load(cfg.lon_path))
    print(upper_path)
    forecast_fields = load_upper_fields(upper_path, lat1d, lon1d, cfg)
    forecast_center = choose_center(row, "forecast")
    forecast_result = compute_gwb(
        forecast_fields["z700"],
        forecast_fields["u700"],
        forecast_fields["v700"],
        lat1d,
        lon1d,
        forecast_center[0],
        forecast_center[1],
        cfg,
    )

    target_result = None
    era5_dir = cfg.era5_dir / dt_to_era5_dir(valid_dt)
    era5_upper = era5_dir / "input_upper.npy"
    if era5_upper.exists():
        base_lat1d, base_lon1d = to_1d_lat_lon(np.load(cfg.lat_path), np.load(cfg.lon_path))
        target_fields = load_upper_fields(era5_upper, base_lat1d, base_lon1d, cfg)
        target_center = choose_center(row, "bst")
        target_result = compute_gwb(
            target_fields["z700"],
            target_fields["u700"],
            target_fields["v700"],
            base_lat1d,
            base_lon1d,
            target_center[0],
            target_center[1],
            cfg,
        )

    fig, axes = plt.subplots(1, 2 if target_result is not None else 1, figsize=(12, 5), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    forecast_center_lon_norm = normalize_lon(lon1d, forecast_center[1])
    lon2d = np.broadcast_to(lon1d[None, :], (lat1d.size, lon1d.size))
    lat2d = np.broadcast_to(lat1d[:, None], (lat1d.size, lon1d.size))
    pcm = axes[0].pcolormesh(lon2d, lat2d, forecast_fields["z700"], shading="auto", cmap="viridis")
    axes[0].plot(forecast_center_lon_norm, forecast_center[0], "ko", markersize=4)
    axes[0].set_title("Forecast z700")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    fig.colorbar(pcm, ax=axes[0], label="m")

    if target_result is not None:
        base_lat1d, base_lon1d = to_1d_lat_lon(np.load(cfg.lat_path), np.load(cfg.lon_path))
        target_center_lon_norm = normalize_lon(base_lon1d, choose_center(row, "bst")[1])
        lon2d = np.broadcast_to(base_lon1d[None, :], (base_lat1d.size, base_lon1d.size))
        lat2d = np.broadcast_to(base_lat1d[:, None], (base_lat1d.size, base_lon1d.size))
        pcm2 = axes[1].pcolormesh(lon2d, lat2d, target_fields["z700"], shading="auto", cmap="viridis")
        axes[1].plot(target_center_lon_norm, choose_center(row, "bst")[0], "ko", markersize=4)
        axes[1].set_title("Target z700")
        axes[1].set_xlabel("Longitude")
        axes[1].set_ylabel("Latitude")
        fig.colorbar(pcm2, ax=axes[1], label="m")

    plt.show()

    print_result("forecast", forecast_result, cfg.target_radius_km)
    if target_result is not None:
        print_result("target", target_result, cfg.target_radius_km)
    else:
        print("[target] input_upper.npy not found; target GWB not computed.")


if __name__ == "__main__":
    main()
