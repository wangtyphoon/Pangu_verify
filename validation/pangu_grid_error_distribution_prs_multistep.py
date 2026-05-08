import os
import re
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# 讓腳本可直接從 validation 或專案根目錄執行
SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

try:
    from loss import create_spatial_weights  # type: ignore
except Exception:
    # 簡化版，不依賴 torch，與 validation/loss.py 的實作等價
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

try:
    from subset import extract_subset as external_extract_subset  # type: ignore
except Exception:
    external_extract_subset = None


# 通道與層次對應（與 weight_rmse_prs_multistep.py 一致）
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

PRESSURE_VARIABLES = tuple(channel_level_map.keys()) + ("wind850",)

# 預設路徑與多步設定
DEFAULT_RESULTS_ROOT = os.path.join(BASE_DIR, "results")
DEFAULT_ERA5_ROOT = os.path.join(BASE_DIR, "forecasts")
DEFAULT_REFERENCE_NC = os.path.join(BASE_DIR, "upper.nc")
DEFAULT_REFERENCE_PATH = r"D:\RDL_TC_MODEL\npz\test\prs"
DEFAULT_OUTPUT_PREFIX = "pangu_grid_error_distribution_prs168"
YEAR = 2024
USE_VALID_START_FILTER = True
MAX_RUNS = None  # 設成整數可只處理前 N 個 run，方便快速測試

FORECAST_STEP_HOURS = 6
FORECAST_STEPS = 1
FORECAST_TOTAL_HOURS = FORECAST_STEP_HOURS * FORECAST_STEPS
GRID_H = 192
GRID_W = 192
BOUNDARY = 8

QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
_SUBSET_SLICE_CACHE = {}


def resolve_path(path: str) -> str:
    """支援 Windows 路徑在 WSL 下自動轉成 /mnt/<drive>/...。"""
    if os.path.exists(path):
        return path

    m = re.match(r"^([A-Za-z]):[\\/](.*)$", path)
    if not m:
        return path

    drive = m.group(1).lower()
    rest = m.group(2).replace("\\", "/")
    wsl_path = f"/mnt/{drive}/{rest}"
    return wsl_path if os.path.exists(wsl_path) else path


def parse_stamp(stamp: str) -> datetime:
    return datetime.strptime(stamp, "%Y-%m-%d-%H-%M")


def filename_to_datetime(fn: str) -> datetime:
    return datetime.strptime(fn[:10], "%Y%m%d%H")


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


def build_valid_start_set(reference_path: str, year: int) -> set[str]:
    reference_path = resolve_path(reference_path)
    all_samples = sorted(
        [f for f in os.listdir(reference_path) if f.endswith(".npz")]
    )
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
    mask_year = (
        (df["date_time"].dt.year == year)
        & (df["asc_steps"] >= 1)
        & (df["desc_steps"] >= FORECAST_STEPS)
    )
    df_year = df.loc[mask_year].copy()
    return set(dt.strftime("%Y-%m-%d-%H-%M") for dt in df_year["date_time"])


def compute_wind_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.sqrt(np.square(u) + np.square(v))


def read_lat_lon_from_nc(nc_path: str):
    try:
        from netCDF4 import Dataset  # type: ignore

        with Dataset(nc_path) as ds:
            return np.asarray(ds.variables["latitude"][:]), np.asarray(ds.variables["longitude"][:])
    except Exception:
        pass

    try:
        import xarray as xr  # type: ignore

        with xr.open_dataset(nc_path) as ds:
            return np.asarray(ds["latitude"].values), np.asarray(ds["longitude"].values)
    except Exception:
        return None, None


def extract_subset_local(
    data: np.ndarray,
    nc_path: str,
    mapping: dict[str, tuple[int, int]],
    lat_range=(0.0, 47.75),
    lon_range=(102.25, 150.0),
):
    global external_extract_subset
    if external_extract_subset is not None:
        try:
            return external_extract_subset(data, nc_path, mapping, lat_range=lat_range, lon_range=lon_range)
        except OSError as exc:
            print(f"[warn] subset.py 讀取 reference_nc 失敗，改用本地裁切: {exc}")
            external_extract_subset = None

    cache_key = (nc_path, tuple(lat_range), tuple(lon_range), data.shape[-2:])
    if cache_key not in _SUBSET_SLICE_CACHE:
        lats, lons = read_lat_lon_from_nc(nc_path)
        if lats is not None and lons is not None:
            lons = np.mod(lons, 360.0)
            lat_idx = np.where((lats >= lat_range[0]) & (lats <= lat_range[1]))[0]
            lon_idx = np.where((lons >= lon_range[0]) & (lons <= lon_range[1]))[0]
            if len(lat_idx) == 0 or len(lon_idx) == 0:
                raise ValueError("reference_nc 選不到指定經緯度範圍")
            lat_sl = slice(lat_idx.min(), lat_idx.max() + 1)
            lon_sl = slice(lon_idx.min(), lon_idx.max() + 1)
        elif data.shape[-2:] == (721, 1440):
            # Pangu 0.25 度全球網格: latitude 90 -> -90, longitude 0 -> 359.75。
            lat_sl = slice(169, 361)  # 47.75N -> 0N, 共 192 格
            lon_sl = slice(409, 601)  # 102.25E -> 150E, 共 192 格
        else:
            raise RuntimeError(
                "無法讀取 reference_nc，且資料不是已知的 Pangu 全球網格 721x1440；"
                "請安裝 netCDF4/xarray 可用 backend，或提供可讀的 --reference-nc。"
            )
        _SUBSET_SLICE_CACHE[cache_key] = (lat_sl, lon_sl)

    lat_sl, lon_sl = _SUBSET_SLICE_CACHE[cache_key]

    return {name: data[ch, lv, lat_sl, lon_sl] for name, (ch, lv) in mapping.items()}


def data_to_pressure_dict(data: np.ndarray, reference_nc: str) -> dict[str, np.ndarray]:
    """
    將 Pangu upper data 轉成變數 dict。
    若輸入已是裁切後 192x192，直接取 channel/level；否則依 upper.nc 裁切。
    """
    if data.ndim != 4:
        raise ValueError(f"upper data 應為 4 維 (C,L,H,W)，目前 shape={data.shape}")

    if data.shape[-2:] == (GRID_H, GRID_W):
        out = {
            name: np.asarray(data[ch, lv], dtype=np.float32)
            for name, (ch, lv) in channel_level_map.items()
        }
    else:
        out = {
            name: np.asarray(arr, dtype=np.float32)
            for name, arr in extract_subset_local(data, reference_nc, channel_level_map).items()
        }

    out["wind850"] = compute_wind_speed(out["u850"], out["v850"]).astype(np.float32)
    return out


def boundary_distance_map(h: int, w: int) -> np.ndarray:
    y_idx, x_idx = np.indices((h, w))
    dist_top = y_idx
    dist_bottom = h - 1 - y_idx
    dist_left = x_idx
    dist_right = w - 1 - x_idx
    return np.minimum.reduce([dist_top, dist_bottom, dist_left, dist_right]).astype(np.int16)


def empty_step_stats() -> dict[str, np.ndarray | int]:
    shape = (GRID_H, GRID_W, len(PRESSURE_VARIABLES))
    return {
        "sample_count": 0,
        "squared_error_sum": np.zeros(shape, dtype=np.float64),
        "absolute_error_sum": np.zeros(shape, dtype=np.float64),
        "bias_sum": np.zeros(shape, dtype=np.float64),
    }


def get_run_start(run_name: str) -> datetime | None:
    start_str = run_name.split("to", 1)[0] if "to" in run_name else run_name
    try:
        return parse_stamp(start_str)
    except ValueError:
        return None


def iter_run_dirs(results_root: str, year: int, valid_starts: set[str] | None):
    for folder in sorted(os.listdir(results_root)):
        if not folder.startswith(str(year)):
            continue
        run_dir = os.path.join(results_root, folder)
        if not os.path.isdir(run_dir):
            continue

        start_str = folder.split("to", 1)[0] if "to" in folder else folder
        if valid_starts is not None and start_str not in valid_starts:
            continue
        yield run_dir


def accumulate_run(run_dir: str, era5_root: str, reference_nc: str, step_stats):
    run_name = os.path.basename(run_dir)
    run_start = get_run_start(run_name)
    if run_start is None:
        print(f"[skip] 無法解析 run start: {run_name}")
        return

    used = 0
    for fn in sorted(os.listdir(run_dir)):
        if not fn.startswith("output_upper_") or not fn.endswith(".npy"):
            continue

        dt_str = fn[len("output_upper_"):-4]
        try:
            valid_time = parse_stamp(dt_str)
        except ValueError:
            print(f"[skip] 無法解析預報時間: {fn}")
            continue

        lead_hours = int((valid_time - run_start).total_seconds() // 3600)
        if lead_hours <= 0 or lead_hours > FORECAST_TOTAL_HOURS:
            continue
        if lead_hours % FORECAST_STEP_HOURS != 0:
            print(f"[skip] 非 {FORECAST_STEP_HOURS} 小時間隔 lead: {run_name}/{fn}")
            continue

        fc_path = os.path.join(run_dir, fn)
        er_path = os.path.join(era5_root, dt_str, "input_upper.npy")
        if not os.path.exists(er_path):
            print(f"[warn] 缺少 ERA5，略過: {er_path}")
            continue

        fc_dict = data_to_pressure_dict(np.load(fc_path), reference_nc)
        er_dict = data_to_pressure_dict(np.load(er_path), reference_nc)

        if lead_hours not in step_stats:
            step_stats[lead_hours] = empty_step_stats()

        for var_idx, var_name in enumerate(PRESSURE_VARIABLES):
            diff = fc_dict[var_name].astype(np.float32) - er_dict[var_name].astype(np.float32)
            step_stats[lead_hours]["squared_error_sum"][..., var_idx] += np.square(
                diff, dtype=np.float32
            )
            step_stats[lead_hours]["absolute_error_sum"][..., var_idx] += np.abs(diff)
            step_stats[lead_hours]["bias_sum"][..., var_idx] += diff

        step_stats[lead_hours]["sample_count"] += 1
        used += 1

    print(f"[run] {run_name}: used={used}")


def safe_weighted_mean(grid: np.ndarray, weight_2d: np.ndarray) -> float:
    return float(np.sum(grid * weight_2d) / np.sum(weight_2d))


def build_distribution_records(
    lead_hours: np.ndarray,
    variable_names: np.ndarray,
    sample_count: np.ndarray,
    mse_grid: np.ndarray,
    rmse_grid: np.ndarray,
    mae_grid: np.ndarray,
    bias_grid: np.ndarray,
    weight_2d: np.ndarray,
) -> list[dict[str, float | int | str]]:
    records = []
    for step_idx, lead in enumerate(lead_hours):
        for var_idx, var_name in enumerate(variable_names):
            mse_2d = mse_grid[step_idx, ..., var_idx]
            rmse_2d = rmse_grid[step_idx, ..., var_idx]
            mae_2d = mae_grid[step_idx, ..., var_idx]
            bias_2d = bias_grid[step_idx, ..., var_idx]

            rec: dict[str, float | int | str] = {
                "lead_hours": int(lead),
                "step": int(lead // FORECAST_STEP_HOURS),
                "variable": str(var_name),
                "sample_count": int(sample_count[step_idx]),
                "rmse_grid_mean": float(np.mean(rmse_2d)),
                "rmse_grid_std": float(np.std(rmse_2d)),
                "mae_grid_mean": float(np.mean(mae_2d)),
                "bias_grid_mean": float(np.mean(bias_2d)),
                "domain_rmse_weighted": float(np.sqrt(safe_weighted_mean(mse_2d, weight_2d))),
                "domain_mae_weighted": safe_weighted_mean(mae_2d, weight_2d),
                "domain_bias_weighted": safe_weighted_mean(bias_2d, weight_2d),
            }
            for q, value in zip(QUANTILES, np.quantile(rmse_2d, QUANTILES)):
                rec[f"rmse_q{int(q * 100):02d}"] = float(value)
            records.append(rec)
    return records


def save_outputs(step_stats, output_prefix: str):
    if not step_stats:
        raise RuntimeError("沒有任何可用資料可輸出")

    lead_hours = np.array(sorted(step_stats.keys()), dtype=np.int32)
    sample_count = np.array(
        [step_stats[int(lead)]["sample_count"] for lead in lead_hours], dtype=np.int64
    )
    squared_error_sum = np.stack(
        [step_stats[int(lead)]["squared_error_sum"] for lead in lead_hours], axis=0
    )
    absolute_error_sum = np.stack(
        [step_stats[int(lead)]["absolute_error_sum"] for lead in lead_hours], axis=0
    )
    bias_sum = np.stack(
        [step_stats[int(lead)]["bias_sum"] for lead in lead_hours], axis=0
    )

    denom = np.maximum(sample_count[:, None, None, None].astype(np.float64), 1.0)
    mse_grid = squared_error_sum / denom
    mae_grid = absolute_error_sum / denom
    bias_grid = bias_sum / denom
    rmse_grid = np.sqrt(mse_grid)

    rmse_grid_variable_mean = np.sqrt(mse_grid.mean(axis=-1)).astype(np.float32)
    mae_grid_variable_mean = mae_grid.mean(axis=-1).astype(np.float32)
    bias_grid_variable_mean = bias_grid.mean(axis=-1).astype(np.float32)

    variable_names = np.array(PRESSURE_VARIABLES, dtype=object)
    weight_2d = np.asarray(
        create_spatial_weights(H=GRID_H, W=GRID_W, boundary=BOUNDARY, device="cpu"),
        dtype=np.float32,
    )
    boundary_dist = boundary_distance_map(GRID_H, GRID_W)

    out_npz = f"{output_prefix}.npz"
    np.savez_compressed(
        out_npz,
        lead_hours=lead_hours,
        steps=(lead_hours // FORECAST_STEP_HOURS).astype(np.int32),
        sample_count=sample_count,
        variable_names=variable_names,
        rmse_grid=rmse_grid.astype(np.float32),
        mae_grid=mae_grid.astype(np.float32),
        bias_grid=bias_grid.astype(np.float32),
        rmse_grid_variable_mean=rmse_grid_variable_mean,
        mae_grid_variable_mean=mae_grid_variable_mean,
        bias_grid_variable_mean=bias_grid_variable_mean,
        boundary_distance=boundary_dist,
        spatial_weight=weight_2d,
        h=np.int32(GRID_H),
        w=np.int32(GRID_W),
    )

    records = build_distribution_records(
        lead_hours=lead_hours,
        variable_names=variable_names,
        sample_count=sample_count,
        mse_grid=mse_grid,
        rmse_grid=rmse_grid,
        mae_grid=mae_grid,
        bias_grid=bias_grid,
        weight_2d=weight_2d,
    )
    out_csv = f"{output_prefix}_by_lead_variable.csv"
    pd.DataFrame(records).to_csv(out_csv, index=False, float_format="%.8f", encoding="utf-8-sig")

    print(f"已輸出格點統計: {out_npz}")
    print(f"已輸出分布摘要: {out_csv}")
    print(f"lead_hours={lead_hours.tolist()}, shape(rmse_grid)={rmse_grid.shape}")


def main():
    results_root = resolve_path(DEFAULT_RESULTS_ROOT)
    era5_root = resolve_path(DEFAULT_ERA5_ROOT)
    reference_nc = resolve_path(DEFAULT_REFERENCE_NC)

    valid_starts = None
    if USE_VALID_START_FILTER:
        reference_path = resolve_path(DEFAULT_REFERENCE_PATH)
        if os.path.exists(reference_path):
            valid_starts = build_valid_start_set(reference_path, YEAR)
            print(f"valid_starts={len(valid_starts)}")
        else:
            print(f"[warn] reference_path 不存在，改為不過濾 start: {reference_path}")

    run_dirs = list(iter_run_dirs(results_root, YEAR, valid_starts))
    if MAX_RUNS is not None:
        run_dirs = run_dirs[:MAX_RUNS]
    if not run_dirs:
        print("No valid run directories found.")
        return

    print(f"runs={len(run_dirs)}")
    step_stats = {}
    for idx, run_dir in enumerate(run_dirs, 1):
        print(f"[{idx}/{len(run_dirs)}] {os.path.basename(run_dir)}")
        accumulate_run(run_dir, era5_root, reference_nc, step_stats)

    save_outputs(step_stats, DEFAULT_OUTPUT_PREFIX)


if __name__ == "__main__":
    main()
