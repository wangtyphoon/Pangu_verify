from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from subset import extract_subset, extract_subset_surface

# ------------------ Config ------------------
START_TIME_TAG = "2024111200"
# RUN_DIR derived from START_TIME_TAG to keep time consistent.
RUN_DIR = Path("../results") / datetime.strptime(START_TIME_TAG, "%Y%m%d%H").strftime("%Y-%m-%d-%H-%M")
# If multiple cyclones share the same start time, set CYCLONE_NUM to disambiguate.
CYCLONE_NUM = 25  # e.g., 1, 2, ...
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "cyclone_data_2024.csv"
LAT_NPY = BASE_DIR / "lat.npy"
LON_NPY = BASE_DIR / "lon.npy"

STEP_HOURS = 6
WINDOW_RADIUS = 25
CENTROID_RADIUS = 6
OUTPUT_DIR = RUN_DIR / "figures"
FORECAST_PREFIX = "output_upper_"
SURFACE_FORECAST_PREFIX = "output_surface_"
FORECAST_DT_FORMAT = "%Y-%m-%d-%H-%M"
PANGU_CHANNEL_LEVEL_MAP = {
    "u850": (3, 2),
    "v850": (4, 2),
    "z700": (0, 3),
    "z850": (0, 2),
}
SURFACE_CHANNEL_MAP = {
    "msl": 0,
    "u10": 1,
    "v10": 2,
    "t2m": 3,
}
ERA5_ROOT = BASE_DIR.parent / "forecasts"
UPPER_NC_PATH = BASE_DIR.parent / "upper.nc"
SURFACE_NC_PATH = BASE_DIR.parent / "surface.nc"
INTENSITY_RADIUS = 8
SUBSET_LAT_RANGE = (0.0, 47.75)
SUBSET_LON_RANGE = (102.25, 150.0)

# ------------------ Utilities ------------------


def parse_forecast_datetime(path: Path) -> datetime:
    dt_str = path.stem[len(FORECAST_PREFIX) :]
    return datetime.strptime(dt_str, FORECAST_DT_FORMAT)


def normalize_to_CLHW(arr: np.ndarray, H: int, W: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 4:
        if arr.shape[-2:] != (H, W):
            raise ValueError(f"Grid mismatch: {arr.shape[-2:]} vs {(H, W)}")
        return arr
    if arr.ndim == 3 and arr.shape[-1] == H * W:
        return arr.reshape(arr.shape[0], arr.shape[1], H, W)
    raise ValueError(f"Unsupported forecast shape: {arr.shape}")


def normalize_to_CHW(arr: np.ndarray, H: int, W: int, channels: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        if arr.shape[0] == channels and arr.shape[1:] == (H, W):
            return arr
        if arr.shape[-1] == channels and arr.shape[:2] == (H, W):
            return np.transpose(arr, (2, 0, 1))
    if arr.ndim == 2:
        if arr.shape == (H * W, channels):
            return arr.reshape(H, W, channels).transpose(2, 0, 1)
        if arr.shape == (channels, H * W):
            return arr.reshape(channels, H, W)
    raise ValueError(f"Unsupported surface shape: {arr.shape}")


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


def align_lon_system(lon_axis_1d: np.ndarray, lon_value: float) -> float:
    lo_min, lo_max = np.nanmin(lon_axis_1d), np.nanmax(lon_axis_1d)
    v = float(lon_value)
    if lo_min >= 0 and v < 0:
        v = v + 360.0
    if lo_max <= 180 and v > 180:
        v = v - 360.0
    return v


def safe_window_center(lat1d: np.ndarray, lon1d: np.ndarray, lat0: float, lon0: float) -> tuple[int, int]:
    i = int(np.argmin(np.abs(lat1d - lat0)))
    lon0_adj = align_lon_system(lon1d, lon0)
    j = int(np.argmin(np.abs(lon1d - lon0_adj)))
    i = max(0, min(lat1d.size - 1, i))
    j = max(0, min(lon1d.size - 1, j))
    return i, j


def make_window(i: int, j: int, radius: int, H: int, W: int) -> tuple[int, int, int, int]:
    i0, i1 = max(0, i - radius), min(H, i + radius + 1)
    j0, j1 = max(0, j - radius), min(W, j + radius + 1)
    if i0 >= i1:
        i0, i1 = i, i + 1
    if j0 >= j1:
        j0, j1 = j, j + 1
    return i0, i1, j0, j1


def weighted_centroid_latlon(
    field: np.ndarray,
    lat1d: np.ndarray,
    lon1d: np.ndarray,
    center_i: int,
    center_j: int,
    radius: int,
    mode: str,
) -> tuple[float, float]:
    H, W = field.shape
    i0, i1, j0, j1 = make_window(center_i, center_j, radius, H, W)
    sub = field[i0:i1, j0:j1]
    if np.all(np.isnan(sub)):
        return float(lat1d[center_i]), float(lon1d[center_j])

    if mode == "min":
        weights = np.nanmax(sub) - sub
    else:
        weights = sub - np.nanmin(sub)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    if not np.any(weights > 0):
        return float(lat1d[center_i]), float(lon1d[center_j])

    lat_grid = lat1d[i0:i1][:, None]
    lon_grid = lon1d[j0:j1][None, :]
    wsum = float(np.sum(weights))
    lat_c = float(np.sum(weights * lat_grid) / wsum)
    lon_c = float(np.sum(weights * lon_grid) / wsum)
    return lat_c, lon_c


def extract_pangu_fields(forecast: np.ndarray, H: int, W: int) -> dict[str, np.ndarray]:
    data = normalize_to_CLHW(forecast, H, W)
    fields: dict[str, np.ndarray] = {}
    for name, (ch, lv) in PANGU_CHANNEL_LEVEL_MAP.items():
        fields[name] = data[ch, lv, :, :]
    return fields


def surface_to_dict(data: np.ndarray, H: int, W: int) -> dict[str, np.ndarray]:
    channels = len(SURFACE_CHANNEL_MAP)
    try:
        chw = normalize_to_CHW(data, H, W, channels)
        return {name: chw[ch, :, :] for name, ch in SURFACE_CHANNEL_MAP.items()}
    except ValueError:
        arr = np.asarray(data)
        if arr.ndim == 3:
            if arr.shape[0] == channels:
                chw = arr
            elif arr.shape[-1] == channels:
                chw = np.transpose(arr, (2, 0, 1))
            else:
                chw = None
            if chw is not None:
                sub = subset_global_chw(chw, SUBSET_LAT_RANGE, SUBSET_LON_RANGE)
                return {name: sub[ch, :, :] for name, ch in SURFACE_CHANNEL_MAP.items()}

        if not SURFACE_NC_PATH.exists():
            raise FileNotFoundError(f"Missing surface.nc for target subset: {SURFACE_NC_PATH}")
        return extract_subset_surface(data, str(SURFACE_NC_PATH), SURFACE_CHANNEL_MAP)


def upper_to_dict(data: np.ndarray, H: int, W: int) -> dict[str, np.ndarray]:
    try:
        return extract_pangu_fields(data, H, W)
    except ValueError:
        arr = np.asarray(data)
        if arr.ndim == 4 and (arr.shape[-2], arr.shape[-1]) != (H, W):
            try:
                return subset_global_upper(arr, SUBSET_LAT_RANGE, SUBSET_LON_RANGE)
            except ValueError:
                pass

        if not UPPER_NC_PATH.exists():
            raise FileNotFoundError(f"Missing upper.nc for target subset: {UPPER_NC_PATH}")
        return extract_subset(data, str(UPPER_NC_PATH), PANGU_CHANNEL_LEVEL_MAP)


def compute_wind_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.sqrt(np.square(u) + np.square(v))


def safe_nanmin(arr: np.ndarray) -> float:
    return float(np.nanmin(arr)) if np.any(np.isfinite(arr)) else float("nan")


def safe_nanmax(arr: np.ndarray) -> float:
    return float(np.nanmax(arr)) if np.any(np.isfinite(arr)) else float("nan")


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


def subset_global_chw(
    data_chw: np.ndarray,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
) -> np.ndarray:
    _, H, W = data_chw.shape
    lat1d, lon1d = build_global_lat_lon(H, W)
    lat_sl, lon_sl = select_bbox_slices(lat1d, lon1d, lat_range, lon_range)
    return data_chw[:, lat_sl, lon_sl]


def subset_global_upper(
    data: np.ndarray,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
) -> dict[str, np.ndarray]:
    if data.ndim != 4:
        raise ValueError(f"Unsupported upper shape: {data.shape}")
    max_ch = max(ch for ch, _ in PANGU_CHANNEL_LEVEL_MAP.values()) + 1
    max_lv = max(lv for _, lv in PANGU_CHANNEL_LEVEL_MAP.values()) + 1
    if data.shape[0] == max_ch and data.shape[1] >= max_lv:
        arr = data
    elif data.shape[1] == max_ch and data.shape[0] >= max_lv:
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


def collect_forecast_steps(run_dir: Path, start_time: datetime) -> list[dict]:
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    files = sorted(run_dir.glob(f"{FORECAST_PREFIX}*.npy"), key=parse_forecast_datetime)
    if not files:
        raise FileNotFoundError(f"No {FORECAST_PREFIX}*.npy found in {run_dir}")

    steps: list[dict] = []
    for f in files:
        valid_time = parse_forecast_datetime(f)
        if valid_time < start_time:
            continue
        delta_hours = (valid_time - start_time).total_seconds() / 3600.0
        step_float = delta_hours / STEP_HOURS
        step_idx = int(round(step_float))
        if not math.isclose(step_float, step_idx, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"Time step mismatch for {f.name}: {delta_hours} hours from start.")
        steps.append({"path": f, "step": step_idx, "valid_time": valid_time})

    if not steps:
        raise FileNotFoundError(f"No forecast steps on/after {start_time:%Y-%m-%d %H:%M}")
    return steps


def compute_relative_vorticity(u: np.ndarray, v: np.ndarray, lat1d: np.ndarray, lon1d: np.ndarray) -> np.ndarray:
    dlat = np.deg2rad(np.mean(np.diff(lat1d)))
    dlon = np.deg2rad(np.mean(np.diff(lon1d)))
    if dlat == 0 or dlon == 0:
        raise ValueError("Latitude/longitude spacing is zero; cannot compute vorticity.")

    earth_radius = 6371000.0
    dy = earth_radius * dlat
    dx_row = earth_radius * np.cos(np.deg2rad(lat1d)) * dlon

    dv_dx = np.gradient(v, axis=1) / dx_row[:, None]
    du_dy = np.gradient(u, axis=0) / dy
    return dv_dx - du_dy


def select_cyclone_track(df: pd.DataFrame, start_time: datetime, cyclone_num: int | None) -> pd.DataFrame:
    if cyclone_num is not None:
        df = df[df["CYCLONE_NUM"] == cyclone_num]
    df = df.sort_values("DATE_TIME")
    df_after = df[df["DATE_TIME"] >= start_time].copy()
    if df_after.empty:
        raise ValueError("No track points found at/after the start time.")
    return df_after


def track_with_vorticity(
    step_files: list[dict],
    lat1d: np.ndarray,
    lon1d: np.ndarray,
    start_lat: float,
    start_lon: float,
) -> list[dict]:
    H, W = lat1d.size, lon1d.size
    prev_i, prev_j = safe_window_center(lat1d, lon1d, start_lat, start_lon)
    print(f"Starting tracking at lat={start_lat}, lon={start_lon} -> grid i={prev_i}, j={prev_j}")

    records: list[dict] = []

    for step in step_files:
        step_idx = int(step["step"])
        valid_time = step["valid_time"]
        forecast_raw = np.load(step["path"])
        fields = extract_pangu_fields(forecast_raw, H, W)

        u850 = fields["u850"]
        v850 = fields["v850"]
        vort = compute_relative_vorticity(u850, v850, lat1d, lon1d)
        z700 = fields["z700"]
        z850 = fields["z850"]

        i0, i1, j0, j1 = make_window(prev_i, prev_j, WINDOW_RADIUS, H, W)
        window = vort[i0:i1, j0:j1]
        if np.all(np.isnan(window)):
            i, j = np.unravel_index(np.nanargmax(vort), vort.shape)
        else:
            local_idx = np.nanargmax(window)
            di, dj = np.unravel_index(local_idx, window.shape)
            i, j = i0 + di, j0 + dj

        centroid_lat, centroid_lon = weighted_centroid_latlon(
            vort,
            lat1d,
            lon1d,
            i,
            j,
            CENTROID_RADIUS,
            mode="max",
        )
        print(centroid_lat, centroid_lon)

        z700_window = z700[i0:i1, j0:j1]
        if np.all(np.isnan(z700_window)):
            z700_i, z700_j = np.unravel_index(np.nanargmin(z700), z700.shape)
        else:
            z700_local = np.nanargmin(z700_window)
            z700_di, z700_dj = np.unravel_index(z700_local, z700_window.shape)
            z700_i, z700_j = i0 + z700_di, j0 + z700_dj

        z850_window = z850[i0:i1, j0:j1]
        if np.all(np.isnan(z850_window)):
            z850_i, z850_j = np.unravel_index(np.nanargmin(z850), z850.shape)
        else:
            z850_local = np.nanargmin(z850_window)
            z850_di, z850_dj = np.unravel_index(z850_local, z850_window.shape)
            z850_i, z850_j = i0 + z850_di, j0 + z850_dj

        prev_i, prev_j = safe_window_center(lat1d, lon1d, centroid_lat, centroid_lon)
        records.append(
            {
                "time": valid_time,
                "step": step_idx,
                "lat": float(centroid_lat),
                "lon": float(centroid_lon),
                "z700_min_lat": float(lat1d[z700_i]),
                "z700_min_lon": float(lon1d[z700_j]),
                "z850_min_lat": float(lat1d[z850_i]),
                "z850_min_lon": float(lon1d[z850_j]),
                "source_file": step["path"].name,
            }
        )

    return records


def compute_intensity_metrics(
    msl: np.ndarray,
    u10: np.ndarray,
    v10: np.ndarray,
    u850: np.ndarray,
    v850: np.ndarray,
    center_i: int,
    center_j: int,
    radius: int,
) -> tuple[float, float, float]:
    H, W = msl.shape
    i0, i1, j0, j1 = make_window(center_i, center_j, radius, H, W)
    s10 = compute_wind_speed(u10, v10)
    s850 = compute_wind_speed(u850, v850)

    msl_min_hpa = safe_nanmin(msl[i0:i1, j0:j1]) / 100.0
    s10_max = safe_nanmax(s10[i0:i1, j0:j1])
    s850_max = safe_nanmax(s850[i0:i1, j0:j1])
    return msl_min_hpa, s10_max, s850_max


def collect_intensity_timeseries(
    step_files: list[dict],
    pred_track: pd.DataFrame,
    truth_track: pd.DataFrame,
    lat1d: np.ndarray,
    lon1d: np.ndarray,
) -> pd.DataFrame:
    pred_idx = pred_track.set_index("time")[["lat", "lon"]]
    truth_idx = truth_track.set_index("DATE_TIME")[["LAT", "LON"]]
    H, W = lat1d.size, lon1d.size
    records: list[dict] = []

    for step in step_files:
        valid_time = step["valid_time"]
        lead_hours = int(step["step"]) * STEP_HOURS
        dt_str = step["path"].stem[len(FORECAST_PREFIX) :]

        rec = {
            "time": valid_time,
            "lead_hours": lead_hours,
            "forecast_msl_min_hPa": float("nan"),
            "forecast_s10_max": float("nan"),
            "forecast_s850_max": float("nan"),
            "target_msl_min_hPa": float("nan"),
            "target_s10_max": float("nan"),
            "target_s850_max": float("nan"),
        }

        if valid_time in pred_idx.index:
            fc_surface_path = step["path"].with_name(f"{SURFACE_FORECAST_PREFIX}{dt_str}.npy")
            if fc_surface_path.exists():
                fc_surface = np.load(fc_surface_path)
                fc_upper = np.load(step["path"])
                fc_sfc = surface_to_dict(fc_surface, H, W)
                fc_prs = upper_to_dict(fc_upper, H, W)

                f_lat = float(pred_idx.loc[valid_time, "lat"])
                f_lon = float(pred_idx.loc[valid_time, "lon"])
                f_i, f_j = safe_window_center(lat1d, lon1d, f_lat, f_lon)
                rec["forecast_msl_min_hPa"], rec["forecast_s10_max"], rec["forecast_s850_max"] = (
                    compute_intensity_metrics(
                        fc_sfc["msl"],
                        fc_sfc["u10"],
                        fc_sfc["v10"],
                        fc_prs["u850"],
                        fc_prs["v850"],
                        f_i,
                        f_j,
                        INTENSITY_RADIUS,
                    )
                )
            else:
                print(f"[WARN] Missing forecast surface: {fc_surface_path}")

        if valid_time in truth_idx.index:
            er_dir = ERA5_ROOT / dt_str
            er_surface_path = er_dir / "input_surface.npy"
            er_upper_path = er_dir / "input_upper.npy"
            if er_surface_path.exists() and er_upper_path.exists():
                er_surface = np.load(er_surface_path)
                er_upper = np.load(er_upper_path)
                er_sfc = surface_to_dict(er_surface, H, W)
                er_prs = upper_to_dict(er_upper, H, W)

                t_lat = float(truth_idx.loc[valid_time, "LAT"])
                t_lon = float(truth_idx.loc[valid_time, "LON"])
                t_i, t_j = safe_window_center(lat1d, lon1d, t_lat, t_lon)
                rec["target_msl_min_hPa"], rec["target_s10_max"], rec["target_s850_max"] = (
                    compute_intensity_metrics(
                        er_sfc["msl"],
                        er_sfc["u10"],
                        er_sfc["v10"],
                        er_prs["u850"],
                        er_prs["v850"],
                        t_i,
                        t_j,
                        INTENSITY_RADIUS,
                    )
                )
            else:
                if not er_surface_path.exists():
                    print(f"[WARN] Missing target surface: {er_surface_path}")
                if not er_upper_path.exists():
                    print(f"[WARN] Missing target upper: {er_upper_path}")

        records.append(rec)

    df = pd.DataFrame.from_records(records).sort_values("lead_hours")
    return df


def plot_intensity_timeseries(
    intensity_df: pd.DataFrame,
    output_path: Path,
    start_time_tag: str,
    cyclone_num: int,
) -> None:
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("matplotlib is required. Install with `pip install matplotlib`.") from exc

    #matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 8.5), sharex=True)
    x = intensity_df["lead_hours"].to_numpy()

    axes[0].plot(x, intensity_df["forecast_s10_max"], "-o", label="Forecast")
    axes[0].plot(x, intensity_df["target_s10_max"], "-o", label="Target (BST)")
    axes[0].set_ylabel("10m wind max (m/s)")
    axes[0].grid(alpha=0.4)
    axes[0].legend(loc="best")

    axes[1].plot(x, intensity_df["forecast_s850_max"], "-o", label="Forecast")
    axes[1].plot(x, intensity_df["target_s850_max"], "-o", label="Target (BST)")
    axes[1].set_ylabel("850 hPa wind max (m/s)")
    axes[1].grid(alpha=0.4)

    axes[2].plot(x, intensity_df["forecast_msl_min_hPa"], "-o", label="Forecast")
    axes[2].plot(x, intensity_df["target_msl_min_hPa"], "-o", label="Target (BST)")
    axes[2].set_ylabel("MSL min (hPa)")
    axes[2].set_xlabel("Lead hours")
    axes[2].grid(alpha=0.4)

    fig.suptitle(f"TC Intensity (Init: {start_time_tag}, Cyclone {cyclone_num})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_tracks(truth: pd.DataFrame, pred: pd.DataFrame, output_path: Path, start_time_tag: str) -> None:
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("matplotlib is required. Install with `pip install matplotlib`.") from exc
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("cartopy is required. Install with `pip install cartopy`.") from exc

    #matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7.5, 6.5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, lw=0.8)
    ax.add_feature(cfeature.BORDERS, lw=0.4)
    ax.add_feature(cfeature.LAND, facecolor="0.95")
    ax.add_feature(cfeature.OCEAN, facecolor="white")

    ax.plot(
        truth["LON"].to_numpy(),
        truth["LAT"].to_numpy(),
        "-o",
        label="Best Track",
        lw=2,
        transform=ccrs.PlateCarree(),
    )
    ax.plot(
        pred["lon"].to_numpy(),
        pred["lat"].to_numpy(),
        "-o",
        label="Forecast (850 hPa vorticity)",
        lw=2,
        transform=ccrs.PlateCarree(),
    )
    ax.plot(
        pred["z700_min_lon"].to_numpy(),
        pred["z700_min_lat"].to_numpy(),
        "-^",
        label="Forecast (700 hPa Z min)",
        lw=1.5,
        transform=ccrs.PlateCarree(),
    )
    ax.plot(
        pred["z850_min_lon"].to_numpy(),
        pred["z850_min_lat"].to_numpy(),
        "-s",
        label="Forecast (850 hPa Z min)",
        lw=1.5,
        transform=ccrs.PlateCarree(),
    )

    ax.scatter(
        truth["LON"].iloc[0],
        truth["LAT"].iloc[0],
        c="k",
        s=50,
        label="Start",
        transform=ccrs.PlateCarree(),
    )
    all_lats = np.concatenate([truth["LAT"].to_numpy(), pred["lat"].to_numpy()])
    all_lons = np.concatenate([truth["LON"].to_numpy(), pred["lon"].to_numpy()])
    if all_lats.size > 0 and all_lons.size > 0:
        pad = 3.0
        ax.set_extent(
            [
                float(np.nanmin(all_lons) - pad),
                float(np.nanmax(all_lons) + pad),
                float(np.nanmin(all_lats) - pad),
                float(np.nanmax(all_lats) + pad),
            ],
            crs=ccrs.PlateCarree(),
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Tropical Cyclone Track (Init: {start_time_tag})")
    ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.4)
    ax.legend(loc="upper right", fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.show()
    #fig.savefig(output_path, dpi=150, bbox_inches="tight")
    #plt.close(fig)


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], format="%Y%m%d%H")

    start_time = datetime.strptime(START_TIME_TAG, "%Y%m%d%H")
    df_time = df[df["DATE_TIME"] == start_time]
    if df_time.empty:
        raise ValueError(f"No cyclone record found at start time {START_TIME_TAG}")

    if CYCLONE_NUM is None:
        unique_nums = df_time["CYCLONE_NUM"].dropna().unique()
        if unique_nums.size != 1:
            raise ValueError(
                f"Multiple cyclones found at {START_TIME_TAG}; set CYCLONE_NUM explicitly."
            )
        cyclone_num = int(unique_nums[0])
    else:
        cyclone_num = int(CYCLONE_NUM)

    df_start = df_time[df_time["CYCLONE_NUM"] == cyclone_num]
    if df_start.empty:
        raise ValueError(
            f"No cyclone record found at start time {START_TIME_TAG} for CYCLONE_NUM={cyclone_num}"
        )

    df_cyc = df[df["CYCLONE_NUM"] == cyclone_num]
    truth_track = select_cyclone_track(df_cyc, start_time, cyclone_num)

    lat_raw = np.load(LAT_NPY)
    lon_raw = np.load(LON_NPY)
    lat1d, lon1d = to_1d_lat_lon(lat_raw, lon_raw)

    step_files = collect_forecast_steps(RUN_DIR, start_time)

    start_lat = float(df_start["LAT"].iloc[0])
    start_lon = float(df_start["LON"].iloc[0])

    pred_records = track_with_vorticity(
        step_files,
        lat1d,
        lon1d,
        start_lat,
        start_lon,
    )
    pred_track = pd.DataFrame(pred_records)

    output_path = OUTPUT_DIR / f"tc_track_{START_TIME_TAG}_cyclone{cyclone_num}.png"
    plot_tracks(truth_track, pred_track, output_path, START_TIME_TAG)
    pred_csv = OUTPUT_DIR / f"tc_track_{START_TIME_TAG}_cyclone{cyclone_num}_pred.csv"
    # pred_track.to_csv(pred_csv, index=False, date_format="%Y%m%d%H")

    print(f"Saved track figure: {output_path}")
    print(f"Saved predicted track CSV: {pred_csv}")

    intensity_df = collect_intensity_timeseries(
        step_files,
        pred_track,
        truth_track,
        lat1d,
        lon1d,
    )
    if intensity_df.empty:
        print("[WARN] No intensity records to plot.")
    else:
        intensity_path = OUTPUT_DIR / f"tc_intensity_{START_TIME_TAG}_cyclone{cyclone_num}.png"
        plot_intensity_timeseries(intensity_df, intensity_path, START_TIME_TAG, cyclone_num)
        print(f"Saved intensity figure: {intensity_path}")


if __name__ == "__main__":
    main()
