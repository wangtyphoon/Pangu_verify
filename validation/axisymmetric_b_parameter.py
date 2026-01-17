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
    "z925": (0, 1),
    "z600": (0, 4),
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


def upper_to_fields(
    data: np.ndarray,
    H: int,
    W: int,
    lat1d: np.ndarray,
    lon1d: np.ndarray,
    cfg: Config,
) -> dict[str, np.ndarray]:
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
        for name in ("z925", "z600"):
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
        return {"dx_km": float("nan"), "dy_km": float("nan"), "unit": np.array([np.nan, np.nan]), "method": "na", "heading_deg": float("nan")}
    idx = idx_list[0]
    ref_lat = float(group_sorted.loc[idx, lat_col])
    if len(group_sorted) < 2:
        return {"dx_km": float("nan"), "dy_km": float("nan"), "unit": np.array([np.nan, np.nan]), "method": "na", "heading_deg": float("nan")}
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


def normalize_track_lons(lon1d: np.ndarray, lons: np.ndarray) -> np.ndarray:
    return np.array([normalize_lon(lon1d, float(lon)) if np.isfinite(lon) else np.nan for lon in lons])


def build_circle_lat_lon(center_lat: float, center_lon: float, radius_km: float, lon1d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, 181)
    km_per_deg = 111.32
    lat0_rad = np.deg2rad(center_lat)
    dx = radius_km * np.cos(angles)
    dy = radius_km * np.sin(angles)
    dlat = dy / km_per_deg
    dlon = dx / (km_per_deg * np.cos(lat0_rad))
    lats = center_lat + dlat
    lons = center_lon + dlon
    lons = normalize_track_lons(lon1d, lons)
    return lats, lons


def add_motion_divider(
    ax: plt.Axes,
    center_lat: float,
    center_lon: float,
    motion_unit: np.ndarray,
    lon1d: np.ndarray,
    radius_km: float,
) -> None:
    if not np.all(np.isfinite(motion_unit)):
        return
    km_per_deg = 111.32
    lat0_rad = np.deg2rad(center_lat)
    dx = motion_unit[0] * radius_km
    dy = motion_unit[1] * radius_km
    dlat = dy / km_per_deg
    dlon = dx / (km_per_deg * np.cos(lat0_rad))
    x0 = normalize_lon(lon1d, center_lon - dlon)
    x1 = normalize_lon(lon1d, center_lon + dlon)
    y0 = center_lat - dlat
    y1 = center_lat + dlat
    ax.plot([x0, x1], [y0, y1], "k--", linewidth=1.2)
    ax.text(x1, y1, "R", fontsize=9, fontweight="bold", ha="left", va="bottom")
    ax.text(x0, y0, "L", fontsize=9, fontweight="bold", ha="right", va="top")


def plot_geopotential_fields(
    axes: tuple[plt.Axes, plt.Axes],
    label: str,
    fields: dict[str, np.ndarray],
    lat1d: np.ndarray,
    lon1d: np.ndarray,
    center_lat: float,
    center_lon: float,
    motion_unit: np.ndarray,
    track_lats: np.ndarray,
    track_lons: np.ndarray,
    cfg: Config,
) -> None:
    lon2d = np.broadcast_to(lon1d[None, :], (lat1d.size, lon1d.size))
    lat2d = np.broadcast_to(lat1d[:, None], (lat1d.size, lon1d.size))
    center_lon_norm = normalize_lon(lon1d, center_lon)

    for ax, key, title in zip(axes, ("z925", "z600"), ("Z925", "Z600")):
        pcm = ax.pcolormesh(lon2d, lat2d, fields[key], shading="auto", cmap="viridis")
        ax.plot(center_lon_norm, center_lat, "ko", markersize=4)
        if track_lats.size:
            ax.plot(track_lons, track_lats, "w-", linewidth=1.5, alpha=0.8)
            ax.plot(track_lons, track_lats, "k--", linewidth=0.8, alpha=0.6)
        circle_lats, circle_lons = build_circle_lat_lon(center_lat, center_lon_norm, cfg.max_radius_km, lon1d)
        ax.plot(circle_lons, circle_lats, "k-", linewidth=1.0, alpha=0.8)
        add_motion_divider(ax, center_lat, center_lon_norm, motion_unit, lon1d, cfg.max_radius_km)
        ax.set_title(f"{label} {title}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.figure.colorbar(pcm, ax=ax, label="m")


def print_result(
    label: str,
    center_lat: float,
    center_lon: float,
    motion: dict[str, float | str | np.ndarray],
    result: dict[str, float],
) -> None:
    print(f"[{label}] center lat={center_lat:.3f}, lon={center_lon:.3f}")
    print(
        f"[{label}] motion dx_km={motion['dx_km']:.2f}, dy_km={motion['dy_km']:.2f}, method={motion['method']}"
    )
    print(f"[{label}] heading_deg={motion['heading_deg']:.1f}")
    print(
        f"[{label}] thickness L/R (m) = {result['left_mean']:.2f} / {result['right_mean']:.2f},"
        f" B={result['b_param']:.2f}"
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

    run_dir = cfg.results_dir / dt_to_run_dir(init_dt)
    upper_path = run_dir / dt_to_upper_name(valid_dt)
    print(f"Loading forecast upper from: {upper_path}")
    lat_path = run_dir / "latitudes_subset.npy"
    lon_path = run_dir / "longitudes_subset.npy"
    if lat_path.exists() and lon_path.exists():
        lat1d, lon1d = to_1d_lat_lon(np.load(lat_path), np.load(lon_path))
    else:
        lat1d, lon1d = to_1d_lat_lon(np.load(cfg.lat_path), np.load(cfg.lon_path))

    forecast_fields = load_upper_fields(upper_path, lat1d, lon1d, cfg)
    forecast_center = choose_center(row, "forecast")

    group_key = (row["init_time"], row["cyclone_num"])
    df_group = df[(df["init_time"] == group_key[0]) & (df["cyclone_num"] == group_key[1])]
    forecast_motion = compute_motion_for_row(df_group, int(row["step"]), "forecast_lat", "forecast_lon")
    forecast_result = compute_b_parameter(
        forecast_fields["z925"],
        forecast_fields["z600"],
        lat1d,
        lon1d,
        forecast_center[0],
        forecast_center[1],
        np.asarray(forecast_motion["unit"]),
        cfg.max_radius_km,
    )

    print_result(
        "forecast",
        forecast_center[0],
        normalize_lon(lon1d, forecast_center[1]),
        forecast_motion,
        forecast_result,
    )
    forecast_track_lat = df_group["forecast_lat"].to_numpy()
    forecast_track_lon = normalize_track_lons(lon1d, df_group["forecast_lon"].to_numpy())
    forecast_track_mask = np.isfinite(forecast_track_lat) & np.isfinite(forecast_track_lon)

    era5_dir = cfg.era5_dir / dt_to_era5_dir(valid_dt)
    era5_upper = era5_dir / "input_upper.npy"
    if era5_upper.exists():
        base_lat1d, base_lon1d = to_1d_lat_lon(np.load(cfg.lat_path), np.load(cfg.lon_path))
        target_fields = load_upper_fields(era5_upper, base_lat1d, base_lon1d, cfg)
        target_center = choose_center(row, "bst")
        target_motion = compute_motion_for_row(df_group, int(row["step"]), "bst_lat", "bst_lon")
        target_result = compute_b_parameter(
            target_fields["z925"],
            target_fields["z600"],
            base_lat1d,
            base_lon1d,
            target_center[0],
            target_center[1],
            np.asarray(target_motion["unit"]),
            cfg.max_radius_km,
        )
        print_result(
            "target",
            target_center[0],
            normalize_lon(base_lon1d, target_center[1]),
            target_motion,
            target_result,
        )
        target_track_lat = df_group["bst_lat"].to_numpy()
        target_track_lon = normalize_track_lons(base_lon1d, df_group["bst_lon"].to_numpy())
        target_track_mask = np.isfinite(target_track_lat) & np.isfinite(target_track_lon)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
        plot_geopotential_fields(
            (axes[0, 0], axes[0, 1]),
            "Forecast",
            forecast_fields,
            lat1d,
            lon1d,
            forecast_center[0],
            forecast_center[1],
            np.asarray(forecast_motion["unit"]),
            track_lats=forecast_track_lat[forecast_track_mask],
            track_lons=forecast_track_lon[forecast_track_mask],
            cfg=cfg,
        )
        plot_geopotential_fields(
            (axes[1, 0], axes[1, 1]),
            "Target",
            target_fields,
            base_lat1d,
            base_lon1d,
            target_center[0],
            target_center[1],
            np.asarray(target_motion["unit"]),
            track_lats=target_track_lat[target_track_mask],
            track_lons=target_track_lon[target_track_mask],
            cfg=cfg,
        )
        plt.show()
    else:
        print("[target] input_upper.npy not found; target B parameter not computed.")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        plot_geopotential_fields(
            (axes[0], axes[1]),
            "Forecast",
            forecast_fields,
            lat1d,
            lon1d,
            forecast_center[0],
            forecast_center[1],
            np.asarray(forecast_motion["unit"]),
            track_lats=forecast_track_lat[forecast_track_mask],
            track_lons=forecast_track_lon[forecast_track_mask],
            cfg=cfg,
        )
        plt.show()


if __name__ == "__main__":
    main()
