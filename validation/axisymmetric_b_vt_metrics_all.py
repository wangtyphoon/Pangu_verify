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
    csv_path: Path = Path("tc_aurora_metrics35_step12.csv")
    results_dir: Path = Path("../results")
    era5_dir: Path = Path("../forecasts")
    lat_path: Path = Path("lat.npy")
    lon_path: Path = Path("lon.npy")
    upper_nc_path: Path = Path("../upper.nc")
    output_path: Path = Path("tc_aurora_metrics35_step12.csv")
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


def fields_from_upper(
    upper: np.ndarray,
    lat1d: np.ndarray,
    lon1d: np.ndarray,
    cfg: Config,
    channel_level_map: dict[str, tuple[int, int]],
) -> dict[str, np.ndarray]:
    fields = upper_to_fields(upper, lat1d.size, lon1d.size, lat1d, lon1d, cfg, channel_level_map)
    if cfg.z_is_geopotential:
        for name in fields:
            fields[name] = fields[name] / 9.80665
    return fields


def motion_vector_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    ref_lat: float,
) -> tuple[float, float]:
    if not (
        np.isfinite(lat1)
        and np.isfinite(lon1)
        and np.isfinite(lat2)
        and np.isfinite(lon2)
        and np.isfinite(ref_lat)
    ):
        return float("nan"), float("nan")
    dlon = ((lon2 - lon1 + 180.0) % 360.0) - 180.0
    lat0 = np.deg2rad(ref_lat)
    km_per_deg = 111.32
    dx = dlon * km_per_deg * np.cos(lat0)
    dy = (lat2 - lat1) * km_per_deg
    return float(dx), float(dy)


def compute_motion_for_row(
    df_group: pd.DataFrame,
    step_value: int,
    lat_col: str,
    lon_col: str,
) -> dict[str, float | str | np.ndarray]:
    group_sorted = df_group.sort_values("step").reset_index(drop=True)
    idx_list = group_sorted.index[group_sorted["step"] == step_value].tolist()
    if not idx_list:
        return {
            "dx_km": float("nan"),
            "dy_km": float("nan"),
            "unit": np.array([np.nan, np.nan]),
            "method": "na",
            "heading_deg": float("nan"),
        }
    idx = idx_list[0]
    ref_lat = float(group_sorted.loc[idx, lat_col])
    if len(group_sorted) < 2:
        return {
            "dx_km": float("nan"),
            "dy_km": float("nan"),
            "unit": np.array([np.nan, np.nan]),
            "method": "na",
            "heading_deg": float("nan"),
        }
    if idx == 0:
        lat1, lon1 = group_sorted.loc[idx, [lat_col, lon_col]]
        lat2, lon2 = group_sorted.loc[idx + 1, [lat_col, lon_col]]
        method = "forward"
    elif idx == len(group_sorted) - 1:
        lat1, lon1 = group_sorted.loc[idx - 1, [lat_col, lon_col]]
        lat2, lon2 = group_sorted.loc[idx, [lat_col, lon_col]]
        method = "backward"
    else:
        lat1, lon1 = group_sorted.loc[idx - 1, [lat_col, lon_col]]
        lat2, lon2 = group_sorted.loc[idx + 1, [lat_col, lon_col]]
        method = "central"
    dx, dy = motion_vector_km(float(lat1), float(lon1), float(lat2), float(lon2), ref_lat)
    mag = np.hypot(dx, dy)
    if mag == 0 or not np.isfinite(mag):
        unit = np.array([np.nan, np.nan])
    else:
        unit = np.array([dx / mag, dy / mag])
    if np.isfinite(dx) and np.isfinite(dy):
        heading_deg = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
    else:
        heading_deg = float("nan")
    return {"dx_km": dx, "dy_km": dy, "unit": unit, "method": method, "heading_deg": heading_deg}


def compute_b_parameter(
    z925: np.ndarray,
    z600: np.ndarray,
    lat1d: np.ndarray,
    lon1d: np.ndarray,
    center_lat: float,
    center_lon: float,
    motion_unit: np.ndarray,
    max_radius_km: float,
) -> dict[str, float]:
    if not np.all(np.isfinite(motion_unit)):
        return {"left_mean": float("nan"), "right_mean": float("nan"), "b_param": float("nan")}

    lon1d_adj, center_lon_adj = adjust_lon_grid(lon1d, center_lon)
    H, W = lat1d.size, lon1d.size
    lon2d = np.broadcast_to(lon1d_adj[None, :], (H, W))
    lat2d = np.broadcast_to(lat1d[:, None], (H, W))

    dist_km = haversine_km(lat2d, lon2d, center_lat, center_lon_adj)
    within = dist_km <= max_radius_km

    km_per_deg = 111.32
    lat0_rad = np.deg2rad(center_lat)
    dx = (lon2d - center_lon_adj) * km_per_deg * np.cos(lat0_rad)
    dy = (lat2d - center_lat) * km_per_deg
    cross = motion_unit[0] * dy - motion_unit[1] * dx

    thickness = z600 - z925
    valid = np.isfinite(thickness) & within

    left_mask = valid & (cross > 0)
    right_mask = valid & (cross < 0)
    left_mean = float(np.nanmean(thickness[left_mask])) if np.any(left_mask) else float("nan")
    right_mean = float(np.nanmean(thickness[right_mask])) if np.any(right_mask) else float("nan")
    b_param = right_mean - left_mean if np.isfinite(left_mean) and np.isfinite(right_mean) else float("nan")
    return {"left_mean": left_mean, "right_mean": right_mean, "b_param": b_param}


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
    result = compute_native_vt(list(delta_z_profile), list(delta_z_profile.values()))
    return {**result, 'lower_slope': result['vt_lower'], 'upper_slope': result['vt_upper']}


LOWER_PRESSURES = (925, 850, 700, 600)
UPPER_PRESSURES = (600, 500, 400, 300)

def native_delta_z(levels, deltas, requested):
    """Select measured height ranges only: never interpolate missing layers."""
    lookup = dict(zip(levels, deltas))
    return np.asarray([lookup.get(p, np.nan) for p in requested], dtype=float)

def compute_native_vt(levels, deltas):
    """Signed -V_T: equal-weight OLS of 500-km height range against ln(p/1 hPa)."""
    lower = native_delta_z(levels, deltas, LOWER_PRESSURES)
    upper = native_delta_z(levels, deltas, UPPER_PRESSURES)
    return {
        'vt_lower': regression_slope_ln_p(np.asarray(LOWER_PRESSURES), lower),
        'vt_upper': regression_slope_ln_p(np.asarray(UPPER_PRESSURES), upper),
        'lower_valid_levels': int(np.isfinite(lower).sum()),
        'upper_valid_levels': int(np.isfinite(upper).sum()),
    }

def write_thermal_metadata(output_path, counts):
    """Keep legacy CSV columns intact; save valid-layer counts and method beside it."""
    import json
    path = Path(output_path)
    pd.DataFrame(counts, columns=[
        'forecast_lower_valid_levels', 'forecast_upper_valid_levels',
        'target_lower_valid_levels', 'target_upper_valid_levels',
    ]).to_csv(path.with_suffix('.thermal_wind_counts.csv'), index_label='row')
    path.with_suffix('.thermal_wind_method.json').write_text(json.dumps({
        'method': 'signed-native-pressure-levels-v1', 'radius_km': 500,
        'lower_pressures_hpa': LOWER_PRESSURES, 'upper_pressures_hpa': UPPER_PRESSURES,
        'regression': 'equal-weight OLS of Delta Z_core against ln(p/1 hPa)',
        'interpolation': False, 'minimum_finite_layers': 2, 'units': 'm',
        'positive': 'warm core', 'negative': 'cold core',
    }, indent=2), encoding='utf-8')

def main() -> None:
    cfg = Config()

    df = pd.read_csv(cfg.csv_path)
    df = df.copy()

    base_lat1d, base_lon1d = to_1d_lat_lon(np.load(cfg.lat_path), np.load(cfg.lon_path))

    cache_latlon: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    group_cache: dict[tuple[float | int, float | int], pd.DataFrame] = {}

    b_channel_map = build_channel_map([925, 600])
    vt_channel_map = build_channel_map([925, 850, 700, 600, 500, 400, 300])

    forecast_b_param = []
    forecast_vtu = []
    forecast_vtl = []
    target_b_param = []
    target_vtu = []
    target_vtl = []

    thermal_counts = []
    for _, row in df.iterrows():
        f_counts = t_counts = (0, 0)
        f_b = float("nan")
        f_vtu = float("nan")
        f_vtl = float("nan")
        t_b = float("nan")
        t_vtu = float("nan")
        t_vtl = float("nan")

        init_dt = parse_ymdh(row["init_time"])
        valid_dt = parse_ymdh(row["time"])
        step_value = int(row["step"])

        group_key = (row["init_time"], row["cyclone_num"])
        if group_key in group_cache:
            df_group = group_cache[group_key]
        else:
            df_group = df[(df["init_time"] == group_key[0]) & (df["cyclone_num"] == group_key[1])]
            group_cache[group_key] = df_group

        fc_lat = row.get("forecast_lat", np.nan)
        fc_lon = row.get("forecast_lon", np.nan)
        if np.isfinite(fc_lat) and np.isfinite(fc_lon):
            run_dir = cfg.results_dir / dt_to_run_dir(init_dt)
            upper_path = run_dir / dt_to_upper_name(valid_dt)
            if upper_path.exists():
                if run_dir not in cache_latlon:
                    lat_path = run_dir / "latitudes_subset.npy"
                    lon_path = run_dir / "longitudes_subset.npy"
                    if lat_path.exists() and lon_path.exists():
                        lat1d, lon1d = to_1d_lat_lon(np.load(lat_path), np.load(lon_path))
                        cache_latlon[run_dir] = (lat1d, lon1d)
                    else:
                        cache_latlon[run_dir] = (base_lat1d, base_lon1d)
                lat1d, lon1d = cache_latlon[run_dir]

                upper = np.load(upper_path)
                fields_b = fields_from_upper(upper, lat1d, lon1d, cfg, b_channel_map)
                motion = compute_motion_for_row(df_group, step_value, "forecast_lat", "forecast_lon")
                b_result = compute_b_parameter(
                    fields_b["z925"],
                    fields_b["z600"],
                    lat1d,
                    lon1d,
                    float(fc_lat),
                    float(fc_lon),
                    np.asarray(motion["unit"]),
                    cfg.max_radius_km,
                )
                f_b = b_result["b_param"]

                fields_vt = fields_from_upper(upper, lat1d, lon1d, cfg, vt_channel_map)
                vt_profile = compute_delta_z_profile(
                    fields_vt,
                    lat1d,
                    lon1d,
                    float(fc_lat),
                    float(fc_lon),
                    cfg.max_radius_km,
                )
                vt_result = compute_vt_indicators(vt_profile)
                f_vtu = vt_result["vt_upper"]
                f_vtl = vt_result['vt_lower']
                f_counts = (vt_result['lower_valid_levels'], vt_result['upper_valid_levels'])

        bst_lat = row.get("bst_lat", np.nan)
        bst_lon = row.get("bst_lon", np.nan)
        if np.isfinite(bst_lat) and np.isfinite(bst_lon):
            era5_dir = cfg.era5_dir / dt_to_era5_dir(valid_dt)
            upper_path = era5_dir / "input_upper.npy"
            if upper_path.exists():
                upper = np.load(upper_path)
                fields_b = fields_from_upper(upper, base_lat1d, base_lon1d, cfg, b_channel_map)
                motion = compute_motion_for_row(df_group, step_value, "bst_lat", "bst_lon")
                b_result = compute_b_parameter(
                    fields_b["z925"],
                    fields_b["z600"],
                    base_lat1d,
                    base_lon1d,
                    float(bst_lat),
                    float(bst_lon),
                    np.asarray(motion["unit"]),
                    cfg.max_radius_km,
                )
                t_b = b_result["b_param"]

                fields_vt = fields_from_upper(upper, base_lat1d, base_lon1d, cfg, vt_channel_map)
                vt_profile = compute_delta_z_profile(
                    fields_vt,
                    base_lat1d,
                    base_lon1d,
                    float(bst_lat),
                    float(bst_lon),
                    cfg.max_radius_km,
                )
                vt_result = compute_vt_indicators(vt_profile)
                t_vtu = vt_result["vt_upper"]
                t_vtl = vt_result['vt_lower']
                t_counts = (vt_result['lower_valid_levels'], vt_result['upper_valid_levels'])

        thermal_counts.append((*f_counts, *t_counts))
        forecast_b_param.append(f_b)
        forecast_vtu.append(f_vtu)
        forecast_vtl.append(f_vtl)
        target_b_param.append(t_b)
        target_vtu.append(t_vtu)
        target_vtl.append(t_vtl)

    df["forecast_b_param"] = forecast_b_param
    df["forecast_vtu"] = forecast_vtu
    df["forecast_vtl"] = forecast_vtl
    df["target_b_param"] = target_b_param
    df["target_vtu"] = target_vtu
    df["target_vtl"] = target_vtl

    df.to_csv(cfg.output_path, index=False)
    write_thermal_metadata(cfg.output_path, thermal_counts)
    print(f"Saved: {cfg.output_path}")


if __name__ == "__main__":
    main()
