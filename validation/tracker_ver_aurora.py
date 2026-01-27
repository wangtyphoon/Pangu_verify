"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import logging
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, minimum_filter

from subset import extract_subset, extract_subset_surface

__all__ = ["Tracker"]

logger = logging.getLogger(__file__)


class NoEyeException(Exception):
    """Raised when no eye can be found."""


def get_box(
    variable: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
):
    """Get a square box for a variable."""
    # Make latitude selection.
    lat_mask = (lat_min <= lats) & (lats <= lat_max)
    box = variable[..., lat_mask, :]
    lats = lats[lat_mask]

    # Make longitude selection. Be careful when wrapping around.
    lon_min = lon_min % 360
    lon_max = lon_max % 360
    if lon_min <= lon_max:
        lon_mask = (lon_min <= lons) & (lons <= lon_max)
        box = box[..., lon_mask]
        lons = lons[lon_mask]
    else:
        lon_mask1 = lon_min <= lons
        lon_mask2 = lons <= lon_max
        box = np.concatenate((box[..., lon_mask1], box[..., lon_mask2]), axis=-1)
        lons = np.concatenate((lons[lon_mask1], lons[lon_mask2]))

    return lats, lons, box


def havdist(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance between two latitude-longitude coordinates."""
    lat1, lat2 = np.deg2rad(lat1), np.deg2rad(lat2)
    lon1, lon2 = np.deg2rad(lon1), np.deg2rad(lon2)
    rad_earth_km = 6371
    inner = 1 - np.cos(lat2 - lat1) + np.cos(lat1) * np.cos(lat2) * (1 - np.cos(lon2 - lon1))
    return 2 * rad_earth_km * np.arcsin(np.sqrt(0.5 * inner))


def get_closest_min(
    variable: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    lat: float,
    lon: float,
    delta_lat: float = 5,
    delta_lon: float = 5,
    minimum_cap_size: int = 8,
) -> tuple[float, float]:
    """Get the minimum in `variable` that is closest to `lat` and `lon`."""
    # Create a box centred around the current latitude and longitude.
    lats, lons, box = get_box(
        variable,
        lats,
        lons,
        lat - delta_lat,
        lat + delta_lat,
        lon - delta_lon,
        lon + delta_lon,
    )

    # Smooth to avoid local minima due to noise.
    box = gaussian_filter(box, sigma=1)

    # Find local minima.
    if box.size == 0 or box.shape[-2] == 0 or box.shape[-1] == 0:
        raise NoEyeException()
    local_minima = minimum_filter(box, size=(minimum_cap_size, minimum_cap_size)) == box

    # Remove minima at the edges: these occur when the tracker fails.
    local_minima[0, :] = 0
    local_minima[-1, :] = 0
    local_minima[:, 0] = 0
    local_minima[:, -1] = 0

    # If no local minima are left, no eye can be found. Try the next one.
    if local_minima.sum() == 0:
        raise NoEyeException()

    # Return the latitude and longitude of the closest local minimum.
    lat_inds, lon_inds = zip(*np.argwhere(local_minima))
    dists = havdist(lats[list(lat_inds)], lons[list(lon_inds)], lat, lon)
    i = np.argmin(dists)

    return lats[lat_inds[i]], lons[lon_inds[i]]


def extrapolate(lats: list[float], lons: list[float]) -> tuple[float, float]:
    """Guess an initial latitude and longitude by extrapolating `lats` and `lons`."""
    assert len(lats) == len(lons)
    if len(lats) == 0:
        raise ValueError("Cannot extrapolate from empty lists.")
    elif len(lats) == 1:
        return lats[0], lons[0]
    else:
        # Linearly extrapolate using the last eight points.
        lats = lats[-8:]
        lons = lons[-8:]
        n = len(lats)
        fit = np.polyfit(np.arange(n), np.stack((lats, lons), axis=-1), 1)
        return np.polyval(fit, n)


class Tracker:
    """Simple tropical cyclone tracker.

    This algorithm was originally designed and implemented by Anna Allen. This particular
    implementation is by Wessel Bruinsma and features various improvements over the original design.
    """

    def __init__(
        self,
        init_lat: float | None = None,
        init_lon: float | None = None,
        init_time: datetime | None = None,
    ) -> None:
        self.tracked_times: list[datetime] = []
        self.tracked_lats: list[float] = []
        self.tracked_lons: list[float] = []
        self.tracked_msls: list[float] = []
        self.tracked_winds: list[float] = []
        if init_lat is not None or init_lon is not None or init_time is not None:
            if init_lat is None or init_lon is None or init_time is None:
                raise ValueError("init_lat/init_lon/init_time must all be set or all be None.")
            self.tracked_times.append(init_time)
            self.tracked_lats.append(init_lat)
            self.tracked_lons.append(init_lon)
            self.tracked_msls.append(np.nan)
            self.tracked_winds.append(np.nan)
        self.fails: int = 0

    def results(self) -> pd.DataFrame:
        """Assemble the track into a convenient DataFrame."""
        return pd.DataFrame(
            {
                "time": self.tracked_times,
                "lat": self.tracked_lats,
                "lon": self.tracked_lons,
                "msl": self.tracked_msls,
                "wind": self.tracked_winds,
            }
        )

    def _track_step(
        self,
        z700: np.ndarray,
        msl: np.ndarray,
        u10: np.ndarray,
        v10: np.ndarray,
        lsm: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        time: datetime,
        guess: tuple[float, float] | None = None,
    ) -> None:
        wind = np.sqrt(u10 * u10 + v10 * v10)
        if guess is None:
            if not self.tracked_lats:
                raise ValueError("No previous track available; provide guess for the first step.")
            lat, lon = extrapolate(self.tracked_lats, self.tracked_lons)
        else:
            lat, lon = guess
        lat = max(min(lat, 90), -90)
        lon = lon % 360

        def is_clear(lat: float, lon: float, delta: float) -> bool:
            """Is a box centred at `lat` and `lon` with "radius" `delta` clear of land?"""
            _, _, lsm_box = get_box(
                lsm,
                lats,
                lons,
                lat - delta,
                lat + delta,
                lon - delta,
                lon + delta,
            )
            if lsm_box.size == 0:
                return False
            return lsm_box.max() < 0.5

        # Did we "snap" from the guess to a real nearby minimum?
        snap = False

        # Try MSL with increasingly small boxes.
        for delta in [5, 4, 3, 2, 1.5]:
            try:
                if is_clear(lat, lon, delta):
                    lat, lon = get_closest_min(
                        msl,
                        lats,
                        lons,
                        lat,
                        lon,
                        delta_lat=delta,
                        delta_lon=delta,
                    )
                    snap = True
                    break
            except NoEyeException:
                pass

        if not snap:
            # MSL didn't work. Try Z700. If it works, try to refine with MSL.
            try:
                lat, lon = get_closest_min(
                    z700,
                    lats,
                    lons,
                    lat,
                    lon,
                    delta_lat=5,
                    delta_lon=5,
                )
                snap = True

                for delta in [5, 4, 3, 2, 1.5]:
                    try:
                        if is_clear(lat, lon, delta):
                            lat, lon = get_closest_min(
                                msl,
                                lats,
                                lons,
                                lat,
                                lon,
                                delta_lat=delta,
                                delta_lon=delta,
                            )
                            break
                    except NoEyeException:
                        pass
            except NoEyeException:
                pass

        if not snap:
            self.fails += 1
            if len(self.tracked_lats) > 1:
                logger.info(f"Failed at time {time}. Extrapolating in a silly way.")
            else:
                raise NoEyeException("Completely failed at the first step.")

        lat_axis = lats[:, 0] if lats.ndim == 2 else lats
        lon_axis = lons[0, :] if lons.ndim == 2 else lons
        center_i, center_j = safe_window_center(lat_axis, lon_axis, lat, lon)
        lat, lon = weighted_centroid_latlon(
            msl,
            lat_axis,
            lon_axis,
            center_i,
            center_j,
            CENTROID_RADIUS,
            mode="min",
        )

        self.tracked_times.append(time)
        self.tracked_lats.append(lat)
        self.tracked_lons.append(lon)

        # Extract minimum MSL and maximum wind speed from a crop around the TC.
        _, _, msl_crop = get_box(
            msl,
            lats,
            lons,
            lat - 1.5,
            lat + 1.5,
            lon - 1.5,
            lon + 1.5,
        )
        _, _, wind_crop = get_box(
            wind,
            lats,
            lons,
            lat - 1.5,
            lat + 1.5,
            lon - 1.5,
            lon + 1.5,
        )
        self.tracked_msls.append(msl_crop.min())
        self.tracked_winds.append(wind_crop.max())

    def step(self, batch) -> None:
        """Track the next step.

        Args:
            batch (:class:`aurora.Batch`): Prediction.
        """
        # Check that there is only one prediction. We don't support batched tracking.
        if len(batch.metadata.time) != 1:
            raise RuntimeError("Predictions don't have batch size one.")

        # No need to do tracking on the GPU. It's cheap.
        batch = batch.to("cpu")

        # Extract the relevant variables from the prediction.
        z700_index = list(batch.metadata.atmos_levels).index(700)
        z700 = batch.atmos_vars["z"][0, 0, z700_index].numpy()
        msl = batch.surf_vars["msl"][0, 0].numpy()
        u10 = batch.surf_vars["10u"][0, 0].numpy()
        v10 = batch.surf_vars["10v"][0, 0].numpy()
        lsm = batch.static_vars["lsm"].numpy()
        lats = batch.metadata.lat.numpy()
        lons = batch.metadata.lon.numpy()
        time = batch.metadata.time[0]
        self._track_step(z700, msl, u10, v10, lsm, lats, lons, time)

    def step_fields(
        self,
        z700: np.ndarray,
        msl: np.ndarray,
        u10: np.ndarray,
        v10: np.ndarray,
        lsm: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        time: datetime,
        guess: tuple[float, float] | None = None,
    ) -> None:
        """Track the next step from numpy arrays."""
        self._track_step(z700, msl, u10, v10, lsm, lats, lons, time, guess=guess)


# ------------------ Config ------------------
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "cyclone_data_2024.csv"
RESULTS_DIR = BASE_DIR.parent / "results"
ERA5_ROOT = BASE_DIR.parent / "forecasts"
LAT_NPY = BASE_DIR / "lat.npy"
LON_NPY = BASE_DIR / "lon.npy"
LSM_NPY = BASE_DIR / "lsm.npy"
UPPER_NC_PATH = BASE_DIR.parent / "upper.nc"
SURFACE_NC_PATH = BASE_DIR.parent / "surface.nc"

FORECAST_PREFIX = "output_upper_"
SURFACE_FORECAST_PREFIX = "output_surface_"
FORECAST_DT_FORMAT = "%Y-%m-%d-%H-%M"
STEP_HOURS = 6
REQUIRED_FORECAST_STEPS = 28
CENTROID_RADIUS = 6
INTENSITY_RADIUS = 6
BOUNDARY_MARGIN_DEG = 5.0
VMAX_MIN = 35.0

SUBSET_LAT_RANGE = (0.0, 47.75)
SUBSET_LON_RANGE = (102.25, 150.0)
OUTPUT_CSV = BASE_DIR / "tc_aurora_metrics35_step28.csv"

PANGU_CHANNEL_LEVEL_MAP = {
    "z700": (0, 3),
    "u850": (3, 2),
    "v850": (4, 2),
}
SURFACE_CHANNEL_MAP = {
    "msl": 0,
    "u10": 1,
    "v10": 2,
    "t2m": 3,
}


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


def build_global_lat_lon(H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    lat = np.linspace(90.0, -90.0, H)
    lon = np.linspace(0.0, 360.0 - 360.0 / W, W)
    return lat, lon


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


def subset_global_field(
    field: np.ndarray,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
) -> np.ndarray:
    H, W = field.shape
    lat1d, lon1d = build_global_lat_lon(H, W)
    lat_sl, lon_sl = select_bbox_slices(lat1d, lon1d, lat_range, lon_range)
    return field[lat_sl, lon_sl]


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
    step_indices = [int(step["step"]) for step in step_files]
    max_step = max(step_indices)
    if max_step < required_steps:
        return False
    if step_indices.count(required_steps) != 1:
        return False
    return True


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

        if required_steps is not None:
            run_steps = [step for step in run_steps if int(step["step"]) <= required_steps]

        for step in run_steps:
            step["init_time"] = init_time
            steps.append(step)

    return steps


def load_lsm(lat1d: np.ndarray, lon1d: np.ndarray) -> np.ndarray:
    lsm = np.load(LSM_NPY)
    target_shape = (lat1d.size, lon1d.size)
    if lsm.shape == target_shape:
        return lsm
    lat_range = (float(np.nanmin(lat1d)), float(np.nanmax(lat1d)))
    lon_range = (float(np.nanmin(lon1d)), float(np.nanmax(lon1d)))
    if lsm.ndim != 2:
        raise ValueError(f"Unexpected LSM shape: {lsm.shape}")
    lsm_sub = subset_global_field(lsm, lat_range, lon_range)
    if lsm_sub.shape != target_shape:
        raise ValueError(f"LSM shape mismatch: {lsm.shape} -> {lsm_sub.shape}, expected {target_shape}")
    return lsm_sub


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df = parse_best_track_datetime(df)
    if "TECH" in df.columns:
        df = df[df["TECH"].astype(str).str.upper() == "BEST"].copy()

    lat_raw = np.load(LAT_NPY)
    lon_raw = np.load(LON_NPY)
    lat1d, lon1d = to_1d_lat_lon(lat_raw, lon_raw)
    if np.nanmin(lon1d) < 0 and np.nanmax(lon1d) <= 180:
        lon1d = np.mod(lon1d, 360.0)

    df = filter_bst_records(df, lat1d, lon1d, BOUNDARY_MARGIN_DEG, VMAX_MIN)

    lsm = load_lsm(lat1d, lon1d)
    H, W = lat1d.size, lon1d.size

    step_files = collect_all_forecast_steps(RESULTS_DIR, REQUIRED_FORECAST_STEPS)
    if not step_files:
        raise FileNotFoundError(f"No forecast files found under {RESULTS_DIR}")
    steps_by_init: dict[datetime, list[dict]] = {}
    for step in step_files:
        steps_by_init.setdefault(step["init_time"], []).append(step)
    for steps in steps_by_init.values():
        steps.sort(key=lambda item: int(item.get("step", 0)))

    cyclone_groups: dict[int, pd.DataFrame] = {}
    for cyclone_num, group in df.groupby("CYCLONE_NUM"):
        cyclone_groups[int(cyclone_num)] = group.sort_values("DATE_TIME")

    records: list[dict] = []
    for init_time, run_steps in sorted(steps_by_init.items(), key=lambda item: item[0]):
        if not run_steps:
            continue
        for cyclone_num, track_df in cyclone_groups.items():
            start_match = track_df[track_df["DATE_TIME"] == init_time]
            if start_match.empty:
                continue
            bst_idx = track_df.set_index("DATE_TIME")[["LAT", "LON"]]
            seed_lat = float(start_match.iloc[0]["LAT"])
            seed_lon = float(start_match.iloc[0]["LON"])
            seed_lon = align_lon_system(lon1d, seed_lon)

            print(
                f"Processing init {init_time:%Y-%m-%d %H:%M}, cyclone #{cyclone_num}"
            )
            tracker = Tracker()

            for idx, step in enumerate(run_steps):
                valid_time = step["valid_time"]
                dt_str = step["path"].stem[len(FORECAST_PREFIX) :]
                surface_path = step["path"].with_name(f"{SURFACE_FORECAST_PREFIX}{dt_str}.npy")
                if not surface_path.exists():
                    print(f"[WARN] Missing forecast surface: {surface_path}")
                    break

                upper_raw = np.load(step["path"])
                surface_raw = np.load(surface_path)
                upper_fields = upper_to_dict(upper_raw, H, W)
                sfc_fields = surface_to_dict(surface_raw, H, W)

                guess = (seed_lat, seed_lon) if idx == 0 else None
                try:
                    tracker.step_fields(
                        upper_fields["z700"],
                        sfc_fields["msl"],
                        sfc_fields["u10"],
                        sfc_fields["v10"],
                        lsm,
                        lat1d,
                        lon1d,
                        valid_time,
                        guess=guess,
                    )
                except NoEyeException as exc:
                    print(f"[WARN] Tracking failed at {valid_time:%Y-%m-%d %H:%M}: {exc}")
                    break

                bst_lat = float("nan")
                bst_lon = float("nan")
                if valid_time in bst_idx.index:
                    bst_lat = float(bst_idx.loc[valid_time, "LAT"])
                    bst_lon = float(bst_idx.loc[valid_time, "LON"])
                    bst_lon = align_lon_system(lon1d, bst_lon)

                rec = {
                    "init_time": init_time,
                    "cyclone_num": cyclone_num,
                    "time": valid_time,
                    "lead_hours": int(step["step"]) * STEP_HOURS,
                    "step": int(step["step"]),
                    "forecast_lat": float(tracker.tracked_lats[-1]),
                    "forecast_lon": float(tracker.tracked_lons[-1]),
                    "bst_lat": bst_lat,
                    "bst_lon": bst_lon,
                    "forecast_msl_min_hPa": float("nan"),
                    "forecast_s10_max": float("nan"),
                    "forecast_s850_max": float("nan"),
                    "target_msl_min_hPa": float("nan"),
                    "target_s10_max": float("nan"),
                    "target_s850_max": float("nan"),
                }

                f_i, f_j = safe_window_center(lat1d, lon1d, rec["forecast_lat"], rec["forecast_lon"])
                rec["forecast_msl_min_hPa"], rec["forecast_s10_max"], rec["forecast_s850_max"] = (
                    compute_intensity_metrics(
                        sfc_fields["msl"],
                        sfc_fields["u10"],
                        sfc_fields["v10"],
                        upper_fields["u850"],
                        upper_fields["v850"],
                        f_i,
                        f_j,
                        INTENSITY_RADIUS,
                    )
                )

                if np.isfinite(bst_lat) and np.isfinite(bst_lon):
                    er_dir = ERA5_ROOT / dt_str
                    er_surface_path = er_dir / "input_surface.npy"
                    er_upper_path = er_dir / "input_upper.npy"
                    if er_surface_path.exists() and er_upper_path.exists():
                        er_surface = np.load(er_surface_path)
                        er_upper = np.load(er_upper_path)
                        er_sfc = surface_to_dict(er_surface, H, W)
                        er_prs = upper_to_dict(er_upper, H, W)

                        t_i, t_j = safe_window_center(lat1d, lon1d, bst_lat, bst_lon)
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

    if not records:
        print("[WARN] No records collected; check BST times and results directories.")
        return

    out_df = pd.DataFrame.from_records(records)
    out_df = out_df.sort_values(["init_time", "cyclone_num", "lead_hours"])
    out_df.to_csv(OUTPUT_CSV, index=False, date_format="%Y%m%d%H")
    print(f"Saved: {OUTPUT_CSV} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
