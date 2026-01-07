from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from subset import extract_subset, extract_subset_surface

# ------------------ Config ------------------
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "cyclone_data_2024.csv"
RESULTS_DIR = BASE_DIR.parent / "results"
ERA5_ROOT = BASE_DIR.parent / "forecasts"
LAT_NPY = BASE_DIR / "lat.npy"
LON_NPY = BASE_DIR / "lon.npy"
UPPER_NC_PATH = BASE_DIR.parent / "upper.nc"
SURFACE_NC_PATH = BASE_DIR.parent / "surface.nc"

FORECAST_PREFIX = "output_upper_"
SURFACE_FORECAST_PREFIX = "output_surface_"
FORECAST_DT_FORMAT = "%Y-%m-%d-%H-%M"
STEP_HOURS = 6
REQUIRED_FORECAST_STEPS = 28

WINDOW_RADIUS_LAT = 12
WINDOW_RADIUS_LON = 16
CENTROID_RADIUS = 6
INTENSITY_RADIUS = 6

SUBSET_LAT_RANGE = (0.0, 47.75)
SUBSET_LON_RANGE = (102.25, 150.0)

OUTPUT_CSV = BASE_DIR / "tc_bst_z700_metrics.csv"

BOUNDARY_MARGIN_DEG = 5.0
VMAX_MIN = 25.0
INERTIA_WEIGHT = 1
MAX_SEED_DISPLACEMENT_DEG = 5.5

PANGU_CHANNEL_LEVEL_MAP = {
    "u850": (3, 2),
    "v850": (4, 2),
    "z700": (0, 3),
}
SURFACE_CHANNEL_MAP = {
    "msl": 0,
    "u10": 1,
    "v10": 2,
    "t2m": 3,
}


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


def make_window_rect(
    i: int,
    j: int,
    radius_i: int,
    radius_j: int,
    H: int,
    W: int,
) -> tuple[int, int, int, int]:
    i0, i1 = max(0, i - radius_i), min(H, i + radius_i + 1)
    j0, j1 = max(0, j - radius_j), min(W, j + radius_j + 1)
    if i0 >= i1:
        i0, i1 = i, i + 1
    if j0 >= j1:
        j0, j1 = j, j + 1
    return i0, i1, j0, j1


def deg_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    lon_axis_1d: np.ndarray,
) -> float:
    lon1_adj = align_lon_system(lon_axis_1d, lon1)
    lon2_adj = align_lon_system(lon_axis_1d, lon2)
    return float(math.hypot(lat2 - lat1, lon2_adj - lon1_adj))


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


def find_z700_weighted_center(
    z700: np.ndarray,
    lat1d: np.ndarray,
    lon1d: np.ndarray,
    seed_i: int,
    seed_j: int,
    search_radius: int,
    centroid_radius: int,
) -> tuple[float, float]:
    H, W = z700.shape
    i0, i1, j0, j1 = make_window(seed_i, seed_j, search_radius, H, W)
    window = z700[i0:i1, j0:j1]

    if np.all(np.isnan(window)):
        i, j = np.unravel_index(np.nanargmin(z700), z700.shape)
    else:
        local_idx = np.nanargmin(window)
        di, dj = np.unravel_index(local_idx, window.shape)
        i, j = i0 + di, j0 + dj

    return weighted_centroid_latlon(
        z700,
        lat1d,
        lon1d,
        i,
        j,
        centroid_radius,
        mode="min",
    )


def extract_pangu_fields(forecast: np.ndarray, H: int, W: int) -> dict[str, np.ndarray]:
    data = normalize_to_CLHW(forecast, H, W)
    fields: dict[str, np.ndarray] = {}
    for name, (ch, lv) in PANGU_CHANNEL_LEVEL_MAP.items():
        fields[name] = data[ch, lv, :, :]
    return fields


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
        steps.append(
            {
                "path": f,
                "step": step_idx,
                "valid_time": valid_time,
                "lead_hours": int(round(delta_hours)),
            }
        )

    if not steps:
        raise FileNotFoundError(f"No forecast steps on/after {start_time:%Y-%m-%d %H:%M}")
    return steps


def has_required_steps(step_files: list[dict], required_steps: int | None) -> bool:
    if required_steps is None:
        return True
    if required_steps <= 0 or not step_files:
        return False
    max_step = max(int(step["step"]) for step in step_files)
    return max_step == required_steps


def collect_all_forecast_steps(results_dir: Path, required_steps: int | None) -> list[dict]:
    if not results_dir.exists():
        raise FileNotFoundError(f"Results dir not found: {results_dir}")

    steps: list[dict] = []
    for run_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        try:
            init_time = datetime.strptime(run_dir.name, FORECAST_DT_FORMAT)
        except ValueError:
            continue

        try:
            run_steps = collect_forecast_steps(run_dir, init_time)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[WARN] {exc}")
            continue

        if not has_required_steps(run_steps, required_steps):
            max_step = max(int(step["step"]) for step in run_steps) - 1
            print(
                f"[INFO] Skip {run_dir} (max_step={max_step}, "
                f"required_steps={required_steps})"
            )
            continue

        for step in run_steps:
            step["init_time"] = init_time
            steps.append(step)

    return steps


def build_valid_time_index(step_files: list[dict]) -> dict[datetime, list[dict]]:
    index: dict[datetime, list[dict]] = {}
    for step in step_files:
        index.setdefault(step["valid_time"], []).append(step)
    for steps in index.values():
        steps.sort(key=lambda item: (item.get("step", 0), item["path"].name))
    return index


def track_with_z700(
    step_files: list[dict],
    lat1d: np.ndarray,
    lon1d: np.ndarray,
    start_lat: float,
    start_lon: float,
    bst_track: pd.DataFrame,
) -> list[dict]:
    H, W = lat1d.size, lon1d.size
    prev_i, prev_j = safe_window_center(lat1d, lon1d, start_lat, start_lon)
    prev_lat, prev_lon = float(start_lat), float(start_lon)
    bst_idx = bst_track.set_index("DATE_TIME")[["LAT", "LON"]]
    pred_history: list[tuple[float, float]] = []
    bst_history: list[tuple[float, float]] = []

    records: list[dict] = []
    for step in step_files:
        step_idx = int(step["step"])
        valid_time = step["valid_time"]
        bst_lat = float("nan")
        bst_lon = float("nan")
        if valid_time in bst_idx.index:
            bst_lat = float(bst_idx.loc[valid_time, "LAT"])
            bst_lon = float(bst_idx.loc[valid_time, "LON"])

        if len(pred_history) >= 2:
            lat_last, lon_last = pred_history[-1]
            lat_prev, lon_prev = pred_history[-2]
            seed_lat = lat_last + INERTIA_WEIGHT * (lat_last - lat_prev)
            seed_lon = lon_last + INERTIA_WEIGHT * (lon_last - lon_prev)
        elif len(bst_history) >= 2:
            lat_last, lon_last = bst_history[-1]
            lat_prev, lon_prev = bst_history[-2]
            seed_lat = lat_last + INERTIA_WEIGHT * (lat_last - lat_prev)
            seed_lon = lon_last + INERTIA_WEIGHT * (lon_last - lon_prev)
        elif np.isfinite(bst_lat) and np.isfinite(bst_lon):
            seed_lat, seed_lon = bst_lat, bst_lon
        else:
            seed_lat, seed_lon = prev_lat, prev_lon

        if (
            deg_distance(prev_lat, prev_lon, seed_lat, seed_lon, lon1d)
            > MAX_SEED_DISPLACEMENT_DEG
        ):
            seed_lat, seed_lon = prev_lat, prev_lon

        forecast_raw = np.load(step["path"])
        fields = upper_to_dict(forecast_raw, H, W)
        z700 = fields["z700"]

        seed_i, seed_j = safe_window_center(lat1d, lon1d, seed_lat, seed_lon)
        i0, i1, j0, j1 = make_window_rect(
            seed_i,
            seed_j,
            WINDOW_RADIUS_LAT,
            WINDOW_RADIUS_LON,
            H,
            W,
        )
        window = z700[i0:i1, j0:j1]
        if np.all(np.isnan(window)):
            i, j = np.unravel_index(np.nanargmin(z700), z700.shape)
        else:
            local_idx = np.nanargmin(window)
            di, dj = np.unravel_index(local_idx, window.shape)
            i, j = i0 + di, j0 + dj

        centroid_lat, centroid_lon = weighted_centroid_latlon(
            z700,
            lat1d,
            lon1d,
            i,
            j,
            CENTROID_RADIUS,
            mode="min",
        )
        prev_i, prev_j = safe_window_center(lat1d, lon1d, centroid_lat, centroid_lon)
        prev_lat, prev_lon = float(centroid_lat), float(centroid_lon)
        pred_history.append((prev_lat, prev_lon))
        if np.isfinite(bst_lat) and np.isfinite(bst_lon):
            bst_history.append((bst_lat, bst_lon))

        dist_idx = float(math.hypot(i - seed_i, j - seed_j))
        seed_far_flag = int(dist_idx > 0.8 * max(WINDOW_RADIUS_LAT, WINDOW_RADIUS_LON))
        records.append(
            {
                "time": valid_time,
                "step": step_idx,
                "lat": float(centroid_lat),
                "lon": float(centroid_lon),
                "seed_far_flag": seed_far_flag,
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


def parse_best_track_datetime(df: pd.DataFrame) -> pd.DataFrame:
    dt1 = pd.to_datetime(df["DATE_TIME"].astype(str), format="%Y%m%d%H", errors="coerce")
    if dt1.isna().any():
        dt2 = pd.to_datetime(df["DATE_TIME"].astype(str), format="%Y%m%d%H%M", errors="coerce")
        dt1 = dt1.fillna(dt2)
    if dt1.isna().any():
        dt3 = pd.to_datetime(df["DATE_TIME"].astype(str), format="%Y%m%d%H%M%S", errors="coerce")
        dt1 = dt1.fillna(dt3)
    if dt1.isna().any():
        bad = df.loc[dt1.isna(), "DATE_TIME"].head(5).tolist()
        raise ValueError(f"Unparsed DATE_TIME values (first 5): {bad}")
    df = df.copy()
    df["DATE_TIME"] = dt1
    return df


def filter_bst_records(
    df: pd.DataFrame,
    lat1d: np.ndarray,
    lon1d: np.ndarray,
    margin_deg: float,
    vmax_min: float,
) -> pd.DataFrame:
    lat_min, lat_max = float(np.nanmin(lat1d)), float(np.nanmax(lat1d))
    lon_min, lon_max = float(np.nanmin(lon1d)), float(np.nanmax(lon1d))
    lat_lo, lat_hi = min(lat_min, lat_max) + margin_deg, max(lat_min, lat_max) - margin_deg
    lon_lo, lon_hi = min(lon_min, lon_max) + margin_deg, max(lon_min, lon_max) - margin_deg

    lat_vals = pd.to_numeric(df["LAT"], errors="coerce")
    lon_vals = pd.to_numeric(df["LON"], errors="coerce")
    lon_adj = lon_vals.apply(lambda v: align_lon_system(lon1d, v) if np.isfinite(v) else v)
    if "VMAX" not in df.columns:
        raise KeyError("Missing VMAX column in cyclone data.")
    vmax_vals = pd.to_numeric(df["VMAX"], errors="coerce")

    mask = (
        lat_vals.between(lat_lo, lat_hi)
        & lon_adj.between(lon_lo, lon_hi)
        & (vmax_vals >= vmax_min)
    )
    return df.loc[mask].copy()


def collect_intensity_records(
    step_files: list[dict],
    pred_track: pd.DataFrame,
    truth_track: pd.DataFrame,
    lat1d: np.ndarray,
    lon1d: np.ndarray,
    init_time: datetime,
    cyclone_num: int,
) -> list[dict]:
    pred_idx = pred_track.set_index("time")[["lat", "lon", "seed_far_flag"]]
    truth_idx = truth_track.set_index("DATE_TIME")[["LAT", "LON"]]
    H, W = lat1d.size, lon1d.size
    records: list[dict] = []

    for step in step_files:
        valid_time = step["valid_time"]
        lead_hours = int(step["step"]) * STEP_HOURS
        dt_str = step["path"].stem[len(FORECAST_PREFIX) :]

        rec = {
            "init_time": init_time,
            "cyclone_num": cyclone_num,
            "time": valid_time,
            "lead_hours": lead_hours,
            "step": int(step["step"]),
            "forecast_lat": float("nan"),
            "forecast_lon": float("nan"),
            "seed_far_flag": 0,
            "bst_lat": float("nan"),
            "bst_lon": float("nan"),
            "forecast_msl_min_hPa": float("nan"),
            "forecast_s10_max": float("nan"),
            "forecast_s850_max": float("nan"),
            "target_msl_min_hPa": float("nan"),
            "target_s10_max": float("nan"),
            "target_s850_max": float("nan"),
        }

        if valid_time in pred_idx.index:
            rec["forecast_lat"] = float(pred_idx.loc[valid_time, "lat"])
            rec["forecast_lon"] = float(pred_idx.loc[valid_time, "lon"])
            rec["seed_far_flag"] = int(pred_idx.loc[valid_time, "seed_far_flag"])

        if valid_time in truth_idx.index:
            rec["bst_lat"] = float(truth_idx.loc[valid_time, "LAT"])
            rec["bst_lon"] = float(truth_idx.loc[valid_time, "LON"])

        if np.isfinite(rec["forecast_lat"]) and np.isfinite(rec["forecast_lon"]):
            fc_surface_path = step["path"].with_name(f"{SURFACE_FORECAST_PREFIX}{dt_str}.npy")
            if fc_surface_path.exists():
                fc_surface = np.load(fc_surface_path)
                fc_upper = np.load(step["path"])
                fc_sfc = surface_to_dict(fc_surface, H, W)
                fc_prs = upper_to_dict(fc_upper, H, W)

                f_i, f_j = safe_window_center(lat1d, lon1d, rec["forecast_lat"], rec["forecast_lon"])
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

        if np.isfinite(rec["bst_lat"]) and np.isfinite(rec["bst_lon"]):
            er_dir = ERA5_ROOT / dt_str
            er_surface_path = er_dir / "input_surface.npy"
            er_upper_path = er_dir / "input_upper.npy"
            if er_surface_path.exists() and er_upper_path.exists():
                er_surface = np.load(er_surface_path)
                er_upper = np.load(er_upper_path)
                er_sfc = surface_to_dict(er_surface, H, W)
                er_prs = upper_to_dict(er_upper, H, W)

                t_i, t_j = safe_window_center(lat1d, lon1d, rec["bst_lat"], rec["bst_lon"])
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

    return records


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df = parse_best_track_datetime(df)
    if "TECH" in df.columns:
        df = df[df["TECH"].astype(str).str.upper() == "BEST"].copy()

    lat_raw = np.load(LAT_NPY)
    lon_raw = np.load(LON_NPY)
    lat1d, lon1d = to_1d_lat_lon(lat_raw, lon_raw)
    df = filter_bst_records(df, lat1d, lon1d, BOUNDARY_MARGIN_DEG, VMAX_MIN)

    step_files = collect_all_forecast_steps(RESULTS_DIR, REQUIRED_FORECAST_STEPS)
    if not step_files:
        raise FileNotFoundError(f"No forecast files found under {RESULTS_DIR}")
    steps_by_init: dict[datetime, list[dict]] = {}
    for step in step_files:
        steps_by_init.setdefault(step["init_time"], []).append(step)
    for steps in steps_by_init.values():
        steps.sort(key=lambda item: int(item.get("step", 0)))

    records: list[dict] = []
    cyclone_groups: dict[int, pd.DataFrame] = {}
    for cyclone_num, group in df.groupby("CYCLONE_NUM"):
        cyclone_groups[int(cyclone_num)] = group.sort_values("DATE_TIME")

    for init_time, run_steps in sorted(steps_by_init.items(), key=lambda item: item[0]):
        for cyclone_num, track_df in cyclone_groups.items():
            start_match = track_df[track_df["DATE_TIME"] == init_time]
            if start_match.empty:
                continue
            start_lat = float(start_match.iloc[0]["LAT"])
            start_lon = float(start_match.iloc[0]["LON"])
            print(f"Processing init {init_time:%Y-%m-%d %H:%M}, cyclone #{cyclone_num}")

            pred_records = track_with_z700(
                run_steps,
                lat1d,
                lon1d,
                start_lat,
                start_lon,
                track_df,
            )
            pred_track = pd.DataFrame.from_records(pred_records)
            records.extend(
                collect_intensity_records(
                    run_steps,
                    pred_track,
                    track_df,
                    lat1d,
                    lon1d,
                    init_time,
                    cyclone_num,
                )
            )
            
    if not records:
        print("[WARN] No records collected; check BST times and results directories.")
        return

    out_df = pd.DataFrame.from_records(records)
    out_df = out_df.sort_values(["init_time", "cyclone_num", "lead_hours"])
    out_df.to_csv(OUTPUT_CSV, index=False, date_format="%Y%m%d%H")
    print(f"Saved: {OUTPUT_CSV} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
