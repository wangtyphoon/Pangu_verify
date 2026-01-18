import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Allow running from repo root
SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

try:
    from loss import create_spatial_weights  # type: ignore
except Exception:
    def create_spatial_weights(H=192, W=192, boundary=8, device=None):
        lats_deg = np.linspace(0, 47.75, H)
        lat_w = np.cos(np.deg2rad(lats_deg))
        w2d = np.tile(lat_w[:, None], (1, W))
        w2d = np.flipud(w2d).copy()

        y_idx, x_idx = np.indices((H, W))
        dist_top = y_idx
        dist_bottom = H - 1 - y_idx
        dist_left = x_idx
        dist_right = W - 1 - x_idx
        min_dist = np.minimum.reduce([dist_top, dist_bottom, dist_left, dist_right])
        w2d[min_dist <= boundary] = 0.0
        w2d = w2d / np.mean(w2d)
        return w2d

from subset import extract_subset_surface

# Surface channel mapping
channel_map_surface = {
    "msl": 0,
    "u10": 1,
    "v10": 2,
    "t2m": 3,
}

climate_var_map = {
    "msl": "msl",
    "u10": "u10",
    "v10": "v10",
    "t2m": "t2m",
}

# Paths (relative to repo root)
results_root = os.path.join(BASE_DIR, "results")
era5_root = os.path.join(BASE_DIR, "forecasts")
climate_dir = os.path.join(SCRIPT_DIR, "climate_meteology")
reference_nc = "../surface.nc"
reference_path = r"D:\RDL_TC_MODEL\npz\test\prs"

# Forecast step settings
FORECAST_STEP_HOURS = 6
FORECAST_STEPS = 28
FORECAST_TOTAL_HOURS = FORECAST_STEPS * FORECAST_STEP_HOURS

# Spatial weights
weight_2d = create_spatial_weights(H=192, W=192, boundary=8, device="cpu")
sum_w = float(np.sum(weight_2d))
WORKERS = 6


def parse_stamp(stamp: str) -> datetime:
    return datetime.strptime(stamp, "%Y-%m-%d-%H-%M")


def consecutive_6h_steps(files, ascending=True):
    files_sorted = sorted(files, reverse=not ascending)
    times = [datetime.strptime(f[:10], "%Y%m%d%H") for f in files_sorted]
    step_list = [0]
    for i in range(1, len(times)):
        delta = times[i] - times[i - 1]
        target = timedelta(hours=6) if ascending else timedelta(hours=-6)
        if delta == target:
            step_list.append(step_list[-1] + 1)
        else:
            step_list.append(0)
    return step_list if ascending else step_list[::-1]


def filename_to_datetime(fn: str) -> datetime:
    return datetime.strptime(fn[:10], "%Y%m%d%H")


def build_valid_start_set() -> set[str]:
    all_samples = sorted([f for f in os.listdir(reference_path) if f.endswith(".npz")])
    asc_steps = consecutive_6h_steps(all_samples, ascending=True)
    desc_steps = consecutive_6h_steps(all_samples, ascending=False)

    df = pd.DataFrame(
        {
            "filename": all_samples,
            "asc_steps": asc_steps,
            "desc_steps": desc_steps,
        }
    )
    df["date_time"] = df["filename"].apply(filename_to_datetime)
    mask2024 = (
        (df["date_time"].dt.year == 2024)
        & (df["asc_steps"] >= 1)
        & (df["desc_steps"] >= FORECAST_STEPS)
    )
    df2024 = df.loc[mask2024].copy()
    return set(dt.strftime("%Y-%m-%d-%H-%M") for dt in df2024["date_time"])


def weighted_mean(field: np.ndarray) -> float:
    return float(np.sum(field * weight_2d) / sum_w)


def weighted_rms(field: np.ndarray) -> float:
    return float(np.sqrt(np.sum(np.square(field) * weight_2d) / sum_w))


def compute_acc(f_anom: np.ndarray, o_anom: np.ndarray) -> float:
    f_demean = f_anom - weighted_mean(f_anom)
    o_demean = o_anom - weighted_mean(o_anom)
    denom = np.sqrt(
        np.sum(weight_2d * np.square(f_demean))
        * np.sum(weight_2d * np.square(o_demean))
    )
    if denom == 0:
        return float("nan")
    return float(np.sum(weight_2d * f_demean * o_demean) / denom)


def climate_path_from_datetime(dt: datetime) -> str:
    day = dt.timetuple().tm_yday
    hour = dt.hour
    fname = f"day{day:03d}_hour{hour:02d}.npz"
    return os.path.join(climate_dir, fname)


def surface_to_dict(data, surface_nc_path):
    H, W = weight_2d.shape
    if data.ndim == 3:
        if data.shape[0] == len(channel_map_surface) and data.shape[1:] == (H, W):
            return {
                name: np.asarray(data[ch], dtype=np.float32)
                for name, ch in channel_map_surface.items()
            }
        if data.shape[-1] == len(channel_map_surface) and data.shape[:2] == (H, W):
            return {
                name: np.asarray(data[:, :, ch], dtype=np.float32)
                for name, ch in channel_map_surface.items()
            }
    return extract_subset_surface(data, surface_nc_path, channel_map_surface)


def evaluate_run(run_dir):
    run_name = os.path.basename(run_dir)
    start_str = run_name.split("to", 1)[0] if "to" in run_name else run_name
    try:
        run_start = parse_stamp(start_str)
    except ValueError:
        run_start = None

    records = []
    for fn in sorted(os.listdir(run_dir)):
        if not fn.startswith("output_surface_") or not fn.endswith(".npy"):
            continue
        dt_str = fn[len("output_surface_") : -4]

        fc_path = os.path.join(run_dir, fn)
        er_path = os.path.join(era5_root, dt_str, "input_surface.npy")
        if not os.path.exists(er_path):
            print(f"[WARN] Skip {dt_str}: missing ERA5 {er_path}")
            continue
        if not os.path.exists(reference_nc):
            print(f"[WARN] Skip {dt_str}: missing ERA5 NC {reference_nc}")
            continue

        try:
            dt_obj = parse_stamp(dt_str)
        except ValueError:
            print(f"[WARN] Skip {dt_str}: unsupported datetime format")
            continue

        clim_path = climate_path_from_datetime(dt_obj)
        if not os.path.exists(clim_path):
            print(f"[WARN] Skip {dt_str}: missing climate file {clim_path}")
            continue

        fc_data = np.load(fc_path)
        er_data = np.load(er_path)
        fc_dict = surface_to_dict(fc_data, reference_nc)
        er_dict = surface_to_dict(er_data, reference_nc)

        with np.load(clim_path) as clim:
            acc_dict = {}
            activity_dict = {}
            for var, clim_key in climate_var_map.items():
                if clim_key not in clim:
                    print(f"[WARN] Skip {dt_str} {var}: missing climate key {clim_key}")
                    continue
                c = np.asarray(clim[clim_key])
                f = np.asarray(fc_dict[var])
                o = np.asarray(er_dict[var])

                f_anom = f - c
                o_anom = o - c

                acc_dict[f"{var}_acc"] = compute_acc(f_anom, o_anom)
                activity_dict[f"{var}_activity"] = weighted_rms(
                    f_anom - weighted_mean(f_anom)
                )

        if not acc_dict and not activity_dict:
            continue

        record = {}
        record.update(acc_dict)
        record.update(activity_dict)
        if acc_dict:
            record["overall_acc"] = float(np.nanmean(list(acc_dict.values())))
        if activity_dict:
            record["overall_activity"] = float(
                np.nanmean(list(activity_dict.values()))
            )
        record["datetime"] = dt_str
        record["run"] = run_name

        if run_start is not None:
            lead_h = int((parse_stamp(dt_str) - run_start).total_seconds() // 3600)
            record["lead_hours"] = lead_h
        records.append(record)

    return records


def main(max_workers: int | None = None):
    valid_starts = build_valid_start_set()
    if not valid_starts:
        print("No valid start times found.")
        return

    all_records = []
    run_dirs = []
    for folder in sorted(os.listdir(results_root)):
        if not folder.startswith("2024"):
            continue
        run_dir = os.path.join(results_root, folder)
        if not os.path.isdir(run_dir):
            continue

        start_str = folder.split("to", 1)[0] if "to" in folder else folder
        if start_str not in valid_starts:
            continue
        run_dirs.append(run_dir)

    if max_workers == 1:
        for rd in run_dirs:
            print(f"[DEBUG] Running evaluate_run({rd}) in main process")
            recs = evaluate_run(rd)
            if recs:
                all_records.extend(recs)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(evaluate_run, rd): rd for rd in run_dirs}
            for future in as_completed(future_map):
                recs = future.result()
                if recs:
                    run_name = os.path.basename(future_map[future])
                    print(f"Evaluating {run_name} ...")
                    all_records.extend(recs)

    if not all_records:
        print("No valid records found.")
        return

    df = pd.DataFrame(all_records).set_index("datetime").sort_index()

    out_full = f"acc_activity_sfc{FORECAST_TOTAL_HOURS}.csv"
    df.to_csv(out_full, float_format="%.8f", encoding="utf-8-sig")
    print(f"Saved per-sample results: {out_full}")

    if "lead_hours" in df.columns:
        acc_cols = [c for c in df.columns if c.endswith("_acc")]
        activity_cols = [c for c in df.columns if c.endswith("_activity")]
        lead_df = None
        if acc_cols:
            acc_df = df.groupby("lead_hours")[acc_cols].mean()
            lead_df = acc_df
        if activity_cols:
            activity_df = df.groupby("lead_hours")[activity_cols].mean()
            lead_df = activity_df if lead_df is None else pd.concat(
                [lead_df, activity_df], axis=1
            )
        if lead_df is not None:
            lead_df = lead_df.sort_index()
            out_lead = f"acc_activity_sfc{FORECAST_TOTAL_HOURS}_by_lead.csv"
            lead_df.to_csv(out_lead, float_format="%.8f", encoding="utf-8-sig")
            print(f"Saved lead-mean results: {out_lead}")


if __name__ == "__main__":
    main(max_workers=WORKERS)
