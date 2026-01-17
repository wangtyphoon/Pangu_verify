from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from subset import extract_subset
from tc_tracker_bst_msl import align_lon_system, to_1d_lat_lon


PANGU_PRESSURE_LEVELS = {
    1000: 0,
    925: 1,
    850: 2,
    700: 3,
    600: 4,
    500: 5,
    400: 6,
    300: 7,
    250: 8,
    200: 9,
    150: 10,
    100: 11,
    50: 12,
}
PANGU_CHANNELS = 5
PANGU_LEVELS = 13

LOWER_LAYER = (925.0, 600.0)
UPPER_LAYER = (600.0, 300.0)


@dataclass(frozen=True)
class Config:
    csv_path: Path = Path("tc_bst_msl_metrics35_step12.csv")
    results_dir: Path = Path("../results")
    era5_dir: Path = Path("../forecasts")
    lat_path: Path = Path("lat.npy")
    lon_path: Path = Path("lon.npy")
    upper_nc_path: Path = Path("../upper.nc")
    step_target: int = 8
    row_index: int = 0
    max_radius_km: float = 500.0
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
    channel_level_map: dict[str, tuple[int, int]],
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
    for name, (ch, lv) in channel_level_map.items():
        fields[name] = arr[ch, lv, lat_sl, lon_sl]
    return fields


def upper_to_fields(
    data: np.ndarray,
    H: int,
    W: int,
    lat1d: np.ndarray,
    lon1d: np.ndarray,
    cfg: Config,
    channel_level_map: dict[str, tuple[int, int]],
) -> dict[str, np.ndarray]:
    try:
        clhw = normalize_to_clhw(data, H, W)
        return {name: clhw[ch, lv, :, :] for name, (ch, lv) in channel_level_map.items()}
    except ValueError:
        arr = np.asarray(data)
        if arr.ndim == 4:
            lat_range = (float(np.nanmin(lat1d)), float(np.nanmax(lat1d)))
            lon_mod = np.mod(lon1d, 360.0)
            lon_range = (float(np.nanmin(lon_mod)), float(np.nanmax(lon_mod)))
            try:
                return subset_global_upper(arr, lat_range, lon_range, channel_level_map)
            except ValueError:
                pass

        if not cfg.upper_nc_path.exists():
            raise FileNotFoundError(f"Missing upper.nc for target subset: {cfg.upper_nc_path}")
        return extract_subset(data, str(cfg.upper_nc_path), channel_level_map)


def load_upper_fields(
    upper_path: Path,
    lat1d: np.ndarray,
    lon1d: np.ndarray,
    cfg: Config,
    channel_level_map: dict[str, tuple[int, int]],
) -> dict[str, np.ndarray]:
    if not upper_path.exists():
        raise FileNotFoundError(f"Upper file not found: {upper_path}")
    upper = np.load(upper_path)
    fields = upper_to_fields(upper, lat1d.size, lon1d.size, lat1d, lon1d, cfg, channel_level_map)
    if cfg.z_is_geopotential:
        for name in fields:
            fields[name] = fields[name] / 9.80665
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


def build_channel_map(pressures: list[int]) -> dict[str, tuple[int, int]]:
    channel_map: dict[str, tuple[int, int]] = {}
    for p in pressures:
        if p not in PANGU_PRESSURE_LEVELS:
            raise KeyError(f"Pressure level {p} hPa not in PANGU_PRESSURE_LEVELS.")
        channel_map[f"z{p}"] = (0, PANGU_PRESSURE_LEVELS[p])
    return channel_map


def compute_delta_z_profile(
    fields: dict[str, np.ndarray],
    lat1d: np.ndarray,
    lon1d: np.ndarray,
    center_lat: float,
    center_lon: float,
    max_radius_km: float,
) -> dict[int, float]:
    lon1d_adj, center_lon_adj = adjust_lon_grid(lon1d, center_lon)
    H, W = lat1d.size, lon1d.size
    lon2d = np.broadcast_to(lon1d_adj[None, :], (H, W))
    lat2d = np.broadcast_to(lat1d[:, None], (H, W))

    dist_km = haversine_km(lat2d, lon2d, center_lat, center_lon_adj)
    within = dist_km <= max_radius_km
    if not np.any(within):
        return {}

    profile: dict[int, float] = {}
    for key, field in fields.items():
        valid = within & np.isfinite(field)
        if not np.any(valid):
            profile[int(key[1:])] = float("nan")
            continue
        z_max = float(np.nanmax(field[valid]))
        z_min = float(np.nanmin(field[valid]))
        profile[int(key[1:])] = z_max - z_min
    return profile


def build_pressure_grid(p_min: float, p_max: float, step: float = 50.0) -> np.ndarray:
    start = np.ceil(p_min / step) * step
    end = np.floor(p_max / step) * step
    if end < start:
        return np.array([])
    return np.arange(start, end + step * 0.5, step)


def interpolate_profile(
    pressures: np.ndarray,
    delta_z: np.ndarray,
    p_grid: np.ndarray,
) -> np.ndarray:
    if p_grid.size == 0:
        return np.array([])
    valid = np.isfinite(pressures) & np.isfinite(delta_z)
    if np.count_nonzero(valid) < 2:
        return np.full(p_grid.shape, np.nan)
    p_vals = pressures[valid]
    dz_vals = delta_z[valid]
    order = np.argsort(p_vals)
    p_vals = p_vals[order]
    dz_vals = dz_vals[order]
    return np.interp(p_grid, p_vals, dz_vals, left=np.nan, right=np.nan)


def regression_slope_ln_p(p_grid: np.ndarray, delta_grid: np.ndarray) -> float:
    valid = np.isfinite(p_grid) & np.isfinite(delta_grid)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    lnp = np.log(p_grid[valid])
    dz = delta_grid[valid]
    slope, _ = np.polyfit(lnp, dz, 1)
    return float(slope)


def compute_vt_indicators(delta_z_profile: dict[int, float]) -> dict[str, float]:
    pressures = np.array(sorted(delta_z_profile.keys()), dtype=float)
    delta_vals = np.array([delta_z_profile[int(p)] for p in pressures], dtype=float)

    lower_grid = build_pressure_grid(LOWER_LAYER[1], LOWER_LAYER[0], step=50.0)
    upper_grid = build_pressure_grid(UPPER_LAYER[1], UPPER_LAYER[0], step=50.0)

    lower_interp = interpolate_profile(pressures, delta_vals, lower_grid)
    upper_interp = interpolate_profile(pressures, delta_vals, upper_grid)

    lower_slope = regression_slope_ln_p(lower_grid, lower_interp)
    upper_slope = regression_slope_ln_p(upper_grid, upper_interp)

    return {
        "lower_slope": lower_slope,
        "upper_slope": upper_slope,
        "neg_vt_lower": -lower_slope if np.isfinite(lower_slope) else float("nan"),
        "neg_vt_upper": -upper_slope if np.isfinite(upper_slope) else float("nan"),
    }


def print_profile(label: str, delta_z_profile: dict[int, float]) -> None:
    for p in sorted(delta_z_profile.keys(), reverse=True):
        dz = delta_z_profile[p]
        if np.isfinite(dz):
            print(f"[{label}] ΔZ({p}) = {dz:.2f} m")
        else:
            print(f"[{label}] ΔZ({p}) = nan")


def print_result(label: str, center_lat: float, center_lon: float, vt: dict[str, float]) -> None:
    print(f"[{label}] center lat={center_lat:.3f}, lon={center_lon:.3f}")
    print(
        f"[{label}] slope d(ΔZ)/dlnp lower(925-600) = {vt['lower_slope']:.4f}, -VT^L = {vt['neg_vt_lower']:.4f}"
    )
    print(
        f"[{label}] slope d(ΔZ)/dlnp upper(600-300) = {vt['upper_slope']:.4f}, -VT^U = {vt['neg_vt_upper']:.4f}"
    )


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

    level_pressures = [925, 850, 700, 600, 500, 400, 300]
    channel_map = build_channel_map(level_pressures)

    run_dir = cfg.results_dir / dt_to_run_dir(init_dt)
    upper_path = run_dir / dt_to_upper_name(valid_dt)
    print(f"Loading forecast upper from: {upper_path}")
    lat_path = run_dir / "latitudes_subset.npy"
    lon_path = run_dir / "longitudes_subset.npy"
    if lat_path.exists() and lon_path.exists():
        lat1d, lon1d = to_1d_lat_lon(np.load(lat_path), np.load(lon_path))
    else:
        lat1d, lon1d = to_1d_lat_lon(np.load(cfg.lat_path), np.load(cfg.lon_path))

    forecast_fields = load_upper_fields(upper_path, lat1d, lon1d, cfg, channel_map)
    forecast_center = choose_center(row, "forecast")
    forecast_profile = compute_delta_z_profile(
        forecast_fields,
        lat1d,
        lon1d,
        forecast_center[0],
        forecast_center[1],
        cfg.max_radius_km,
    )
    forecast_vt = compute_vt_indicators(forecast_profile)
    print_profile("forecast", forecast_profile)
    print_result(
        "forecast",
        forecast_center[0],
        normalize_lon(lon1d, forecast_center[1]),
        forecast_vt,
    )

    era5_dir = cfg.era5_dir / dt_to_era5_dir(valid_dt)
    era5_upper = era5_dir / "input_upper.npy"
    if era5_upper.exists():
        base_lat1d, base_lon1d = to_1d_lat_lon(np.load(cfg.lat_path), np.load(cfg.lon_path))
        target_fields = load_upper_fields(era5_upper, base_lat1d, base_lon1d, cfg, channel_map)
        target_center = choose_center(row, "bst")
        target_profile = compute_delta_z_profile(
            target_fields,
            base_lat1d,
            base_lon1d,
            target_center[0],
            target_center[1],
            cfg.max_radius_km,
        )
        target_vt = compute_vt_indicators(target_profile)
        print_profile("target", target_profile)
        print_result(
            "target",
            target_center[0],
            normalize_lon(base_lon1d, target_center[1]),
            target_vt,
        )
    else:
        print("[target] input_upper.npy not found; target ΔZ/VT not computed.")


if __name__ == "__main__":
    main()
