from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from subset import extract_subset
from tc_tracker_bst_msl import surface_to_dict


@dataclass(frozen=True)
class Config:
    csv_path: Path = Path("tc_aurora_metrics35_step12.csv")
    results_dir: Path = Path("../results")
    era5_dir: Path = Path("../forecasts")
    lat_path: Path = Path("lat.npy")
    lon_path: Path = Path("lon.npy")
    upper_nc_path: Path = Path("../upper.nc")
    step_target: int = 1
    row_index: int = 0
    center_source: str = "forecast"  # "forecast" or "bst"
    radius_km: float = 750.0
    thickness_m: float = 1000.0

PANGU_CHANNEL_LEVEL_MAP = {
    "u700": (3, 3),
    "v700": (4, 3),
    "t700": (2, 3),
}
SURFACE_T_FIELDS = ("msl", "t2m")
GAS_CONSTANT_DRY_AIR = 287.05  # J/(kg*K)
P700_PA = 70000.0

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


def to_1d_lat_lon(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if lat.ndim == 2:
        lat1d = lat[:, 0]
    elif lat.ndim == 1:
        lat1d = lat
    else:
        raise ValueError(f"Unsupported lat shape: {lat.shape}")

    if lon.ndim == 2:
        lon1d = lon[0, :]
    elif lon.ndim == 1:
        lon1d = lon
    else:
        raise ValueError(f"Unsupported lon shape: {lon.shape}")

    return lat1d.astype(float), lon1d.astype(float)


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


def align_lon_system(lon_axis_1d: np.ndarray, lon_value: float) -> float:
    lo_min, lo_max = float(np.nanmin(lon_axis_1d)), float(np.nanmax(lon_axis_1d))
    v = float(lon_value)
    if lo_min >= 0 and v < 0:
        v = v + 360.0
    if lo_max <= 180 and v > 180:
        v = v - 360.0
    return v


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


def extract_upper_fields(data: np.ndarray, H: int, W: int) -> dict[str, np.ndarray]:
    arr = np.asarray(data)
    if arr.ndim != 4:
        raise ValueError(f"Unsupported upper shape: {arr.shape}")

    max_ch = max(ch for ch, _ in PANGU_CHANNEL_LEVEL_MAP.values()) + 1
    max_lv = max(lv for _, lv in PANGU_CHANNEL_LEVEL_MAP.values()) + 1

    if arr.shape[-2:] == (H, W):
        if arr.shape[0] >= max_ch and arr.shape[1] >= max_lv:
            arr_clhw = arr
        elif arr.shape[1] >= max_ch and arr.shape[0] >= max_lv:
            arr_clhw = np.transpose(arr, (1, 0, 2, 3))
        else:
            raise ValueError(f"Unsupported upper shape: {arr.shape}")
    else:
        raise ValueError(f"Upper grid shape {arr.shape[-2:]} != {(H, W)}")

    fields: dict[str, np.ndarray] = {}
    for name, (ch, lv) in PANGU_CHANNEL_LEVEL_MAP.items():
        fields[name] = arr_clhw[ch, lv, :, :]
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
        return extract_upper_fields(data, H, W)
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


def compute_ike(
    u: np.ndarray,
    v: np.ndarray,
    area_m2: np.ndarray,
    mask: np.ndarray,
    rho: float | np.ndarray,
    thickness_m: float,
) -> tuple[float, int]:
    speed2 = np.square(u) + np.square(v)
    rho_arr = np.asarray(rho, dtype=float)
    valid = mask & np.isfinite(speed2) & np.isfinite(area_m2) & np.isfinite(rho_arr)
    ike = 0.5 * float(np.sum(rho_arr[valid] * speed2[valid] * area_m2[valid]))
    if thickness_m > 0.0:
        ike *= thickness_m
    return ike, int(np.count_nonzero(valid))


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
    ike, _ = compute_ike(u, v, area_m2, mask, rho, cfg.thickness_m)
    return ike


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
    surface_path = run_dir / dt_to_surface_name(valid_dt)

    if not upper_path.exists():
        raise FileNotFoundError(f"Upper file not found: {upper_path}")
    if not surface_path.exists():
        raise FileNotFoundError(f"Surface file not found: {surface_path}")

    lat_path = run_dir / "latitudes_subset.npy"
    lon_path = run_dir / "longitudes_subset.npy"
    if lat_path.exists() and lon_path.exists():
        lat1d, lon1d = to_1d_lat_lon(np.load(lat_path), np.load(lon_path))
    else:
        lat1d, lon1d = to_1d_lat_lon(np.load(cfg.lat_path), np.load(cfg.lon_path))
    H, W = lat1d.size, lon1d.size

    upper = np.load(upper_path)
    fields = upper_to_fields(upper, H, W, lat1d, lon1d, cfg)
    u700 = fields["u700"]
    v700 = fields["v700"]
    t700 = fields["t700"]

    surface = np.load(surface_path)
    sfc = surface_to_dict(surface, H, W)
    u10 = sfc["u10"]
    v10 = sfc["v10"]
    msl = sfc[SURFACE_T_FIELDS[0]]
    t2m = sfc[SURFACE_T_FIELDS[1]]

    forecast_center = choose_center(row, cfg.center_source)
    rho700_fc = P700_PA / (GAS_CONSTANT_DRY_AIR * t700)
    rho10_fc = msl / (GAS_CONSTANT_DRY_AIR * t2m)
    ike700_fc = compute_ike_for_center(
        u700,
        v700,
        lat1d,
        lon1d,
        forecast_center[0],
        forecast_center[1],
        rho700_fc,
        cfg,
    )
    ike10_fc = compute_ike_for_center(
        u10,
        v10,
        lat1d,
        lon1d,
        forecast_center[0],
        forecast_center[1],
        rho10_fc,
        cfg,
    )

    target_result = None
    era5_dir = cfg.era5_dir / dt_to_era5_dir(valid_dt)
    era5_upper = era5_dir / "input_upper.npy"
    era5_surface = era5_dir / "input_surface.npy"
    if era5_upper.exists() and era5_surface.exists():
        base_lat1d, base_lon1d = to_1d_lat_lon(np.load(cfg.lat_path), np.load(cfg.lon_path))
        base_H, base_W = base_lat1d.size, base_lon1d.size
        target_upper = np.load(era5_upper)
        target_surface = np.load(era5_surface)
        target_fields = upper_to_fields(target_upper, base_H, base_W, base_lat1d, base_lon1d, cfg)
        target_sfc = surface_to_dict(target_surface, base_H, base_W)
        target_center = choose_center(row, "bst")
        t700_tg = target_fields["t700"]
        msl_tg = target_sfc[SURFACE_T_FIELDS[0]]
        t2m_tg = target_sfc[SURFACE_T_FIELDS[1]]
        rho700_tg = P700_PA / (GAS_CONSTANT_DRY_AIR * t700_tg)
        rho10_tg = msl_tg / (GAS_CONSTANT_DRY_AIR * t2m_tg)
        ike700_tg = compute_ike_for_center(
            target_fields["u700"],
            target_fields["v700"],
            base_lat1d,
            base_lon1d,
            target_center[0],
            target_center[1],
            rho700_tg,
            cfg,
        )
        ike10_tg = compute_ike_for_center(
            target_sfc["u10"],
            target_sfc["v10"],
            base_lat1d,
            base_lon1d,
            target_center[0],
            target_center[1],
            rho10_tg,
            cfg,
        )
        target_result = (ike700_tg, ike10_tg)

    unit = "J" if cfg.thickness_m > 0.0 else "J/m"
    print(f"{'level_hpa':>8}  {'ike_forecast':>12}  {'ike_target':>12}")
    ike700_text = f"{ike700_fc:.6e}"
    ike10_text = f"{ike10_fc:.6e}"
    if target_result is not None:
        ike700_tg, ike10_tg = target_result
        ike700_tg_text = f"{ike700_tg:.6e}"
        ike10_tg_text = f"{ike10_tg:.6e}"
    else:
        ike700_tg_text = "nan"
        ike10_tg_text = "nan"
    print(f"{'700':>8}  {ike700_text:>12}  {ike700_tg_text:>12}")
    print(f"{'u10v10':>8}  {ike10_text:>12}  {ike10_tg_text:>12}")
    print(f"IKE unit: {unit}")
    print(f"Total IKE forecast ({unit}): {(ike700_fc + ike10_fc):.6e}")
    if target_result is not None:
        print(f"Total IKE target ({unit}): {(ike700_tg + ike10_tg):.6e}")
    else:
        print("[target] input_upper.npy or input_surface.npy not found; target IKE not computed.")


if __name__ == "__main__":
    main()
