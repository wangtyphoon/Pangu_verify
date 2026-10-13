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

from subset import extract_subset

# Channel mapping
z500_channel = 0
q700_channel = 1
t850_channel = 2
u850_channel = 3
v850_channel = 4
channel_level_map = {
    "z500": (z500_channel, 5),
    "q700": (q700_channel, 3),
    "t850": (t850_channel, 2),
    "u850": (u850_channel, 2),
    "v850": (v850_channel, 2),
}

climate_var_map = {
    "z500": "500z",
    "q700": "700q",
    "t850": "850t",
    "u850": "850u",
    "v850": "850v",
}

# Paths (relative to repo root)
results_root = os.path.join(BASE_DIR, "results")
era5_root = os.path.join(BASE_DIR, "forecasts")
climate_dir = os.path.join(SCRIPT_DIR, "climate_meteology")
reference_nc = "../upper.nc"
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


def weighted_mean(field):
    field = np.asarray(field,dtype=np.float64)
    weights = np.asarray(weight_2d,dtype=np.float64)
    keep = weights>0
    if weights[keep].sum()<=1e-12 or not np.isfinite(field[keep]).all():
        return float('nan')
    return float(np.sum(weights[keep]*field[keep])/weights[keep].sum())


def weighted_rms(field):
    field = np.asarray(field,dtype=np.float64)
    weights = np.asarray(weight_2d,dtype=np.float64)
    keep = weights>0
    if weights[keep].sum()<=1e-12 or not np.isfinite(field[keep]).all():
        return float('nan')
    return float(np.sqrt(np.sum(weights[keep]*field[keep]**2)/weights[keep].sum()))


def compute_acc(f_anom, o_anom):
    f = np.asarray(f_anom,dtype=np.float64)
    o = np.asarray(o_anom,dtype=np.float64)
    weights = np.asarray(weight_2d,dtype=np.float64)
    if not np.isfinite(weights).all() or (weights<0).any():
        raise ValueError('Weights must be finite and nonnegative')
    keep = weights>0
    f,o,w = f[keep],o[keep],weights[keep]
    if not np.isfinite(f).all() or not np.isfinite(o).all() or w.sum()<=1e-12:
        return float('nan')
    den = np.sqrt(np.sum(w*f*f)*np.sum(w*o*o))
    return float(np.sum(w*f*o)/den) if den>1e-12 else float('nan')


def climate_path_from_datetime(dt: datetime) -> str:
    day = dt.timetuple().tm_yday
    hour = dt.hour
    fname = f"day{day:03d}_hour{hour:02d}.npz"
    return os.path.join(climate_dir, fname)


def forecast_to_dict(forecast_data):
    fc_dict = {}
    for name, (ch, lv) in channel_level_map.items():
        fc_dict[name] = forecast_data[ch, lv]
    return fc_dict


def evaluate_run(run_dir):
    run_name = os.path.basename(run_dir)
    start_str = run_name.split("to", 1)[0] if "to" in run_name else run_name
    try:
        run_start = parse_stamp(start_str)
    except ValueError:
        run_start = None

    records = []
    for fn in sorted(os.listdir(run_dir)):
        if not fn.startswith("output_upper_") or not fn.endswith(".npy"):
            continue
        dt_str = fn[len("output_upper_") : -4]

        fc_path = os.path.join(run_dir, fn)
        er_path = os.path.join(era5_root, dt_str, "input_upper.npy")
        if not os.path.exists(er_path):
            print(f"[WARN] Skip {dt_str}: missing ERA5 {er_path}")
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

        fc_data = np.load(fc_path, mmap_mode='r')
        er_data = np.load(er_path, mmap_mode='r')
        fc_dict = forecast_to_dict(fc_data)
        er_dict = extract_subset(er_data, reference_nc, channel_level_map)

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

    # ===== Debug 模式：max_workers == 1 時改成同步跑 =====
    if max_workers == 1:
        for rd in run_dirs:
            print(f"[DEBUG] Running evaluate_run({rd}) in main process")
            recs = evaluate_run(rd)  # <-- 這裡的 raise 會直接把整個程式中斷
            if recs:
                all_records.extend(recs)
    else:
        # ===== 平常生產模式：用 ProcessPoolExecutor =====
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(evaluate_run, rd): rd for rd in run_dirs}
            for future in as_completed(future_map):
                recs = future.result()  # 若 evaluate_run 裡 raise，這裡才會噴錯
                if recs:
                    run_name = os.path.basename(future_map[future])
                    print(f"Evaluating {run_name} ...")
                    all_records.extend(recs)


    if not all_records:
        print("No valid records found.")
        return

    df = pd.DataFrame(all_records).set_index("datetime").sort_index()

    out_full = f"acc_activity_prs{FORECAST_TOTAL_HOURS}.csv"
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
            activity_df = df.groupby('lead_hours')[activity_cols].agg(lambda values: np.sqrt(np.nanmean(np.square(values))))
            lead_df = activity_df if lead_df is None else pd.concat(
                [lead_df, activity_df], axis=1
            )
        if lead_df is not None:
            lead_df = lead_df.sort_index()
            out_lead = f"acc_activity_prs{FORECAST_TOTAL_HOURS}_by_lead.csv"
            lead_df.to_csv(out_lead, float_format="%.8f", encoding="utf-8-sig")
            print(f"Saved lead-mean results: {out_lead}")


if __name__ == "__main__":
    main(max_workers=WORKERS)
