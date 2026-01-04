import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# 讓腳本可直接從專案根目錄執行
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

from subset import extract_subset

# 通道與層次對應（與推理/原始評估一致）
z500_channel = 0
q700_channel = 1
t850_channel = 2
u850_channel = 3
v850_channel = 4
channel_level_map = {
    'z500': (z500_channel, 5),
    'q700': (q700_channel, 3),
    't850': (t850_channel, 2),
    'u850': (u850_channel, 2),
    'v850': (v850_channel, 2),
}

# 路徑設定（以專案根目錄為基準）
results_root = os.path.join(BASE_DIR, 'results')
era5_root = os.path.join(BASE_DIR, 'forecasts')
reference_nc = '../upper.nc'

# 預報步長設定
FORECAST_STEP_HOURS = 6
FORECAST_STEPS = 28
FORECAST_TOTAL_HOURS = FORECAST_STEPS * FORECAST_STEP_HOURS

# 空間權重（輸出檔案已經裁切過，因此這裡只用來給邊界與緯度加權）
weight_2d = create_spatial_weights(H=192, W=192, boundary=8, device='cpu')
reference_path = r"D:\RDL_TC_MODEL\npz\test\prs"
WORKERS = 6  # 可改成整數，指定平行 worker 數


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
    mask2024 = (
        (df["date_time"].dt.year == 2024)
        & (df["asc_steps"] >= 1)
        & (df["desc_steps"] >= FORECAST_STEPS)
    )
    df2024 = df.loc[mask2024].copy()
    return set(dt.strftime("%Y-%m-%d-%H-%M") for dt in df2024["date_time"])


def compute_wind_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """由 u、v 分量計算風速。"""
    return np.sqrt(np.square(u) + np.square(v))


def weighted_mse_dict(forecast_dict, era5_dict, weight):
    sum_w = np.sum(weight)
    mse_per_var = {}
    for var in forecast_dict:
        err2 = (forecast_dict[var] - era5_dict[var]) ** 2
        mse_per_var[var] = float(np.sum(err2 * weight) / sum_w)
    mse_overall = float(np.mean(list(mse_per_var.values())))
    return mse_per_var, mse_overall


def weighted_bias_dict(forecast_dict, era5_dict, weight):
    sum_w = np.sum(weight)
    bias_per_var = {}
    for var in forecast_dict:
        err = forecast_dict[var] - era5_dict[var]
        bias_per_var[var] = float(np.sum(err * weight) / sum_w)
    bias_overall = float(np.mean(list(bias_per_var.values())))
    return bias_per_var, bias_overall


def forecast_to_dict(forecast_data):
    """
    直接從已裁切的推理輸出取得各變數 2D 權重網格。
    forecast_data 形狀預期為 (5, 13, 192, 192)
    """
    fc_dict = {}
    for name, (ch, lv) in channel_level_map.items():
        fc_dict[name] = forecast_data[ch, lv]
    return fc_dict


def add_wind_speed(var_dict):
    """在 dict 上新增 wind850（需已含 u850/v850）。"""
    u = np.asarray(var_dict["u850"])
    v = np.asarray(var_dict["v850"])
    var_dict["wind850"] = compute_wind_speed(u, v)


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
        dt_str = fn[len("output_upper_"):-4]

        fc_path = os.path.join(run_dir, fn)
        er_path = os.path.join(era5_root, dt_str, 'input_upper.npy')
        if not os.path.exists(er_path):
            print(f"[WARN] Skip {dt_str}: missing ERA5 {er_path}")
            continue

        fc_data = np.load(fc_path)
        er_data = np.load(er_path)

        fc_dict = forecast_to_dict(fc_data)
        er_dict = extract_subset(er_data, reference_nc, channel_level_map)
        add_wind_speed(fc_dict)
        add_wind_speed(er_dict)

        mse_dict, mse_all = weighted_mse_dict(fc_dict, er_dict, weight_2d)
        bias_dict, bias_all = weighted_bias_dict(fc_dict, er_dict, weight_2d)
        rmse_dict = {f"{var}_rmse": np.sqrt(mse) for var, mse in mse_dict.items()}
        rmse_dict.update({f"{var}_bias": bias for var, bias in bias_dict.items()})
        rmse_dict["overall_rmse"] = np.sqrt(mse_all)
        rmse_dict["overall_bias"] = bias_all
        rmse_dict["datetime"] = dt_str
        rmse_dict["run"] = run_name

        if run_start is not None:
            lead_h = int((parse_stamp(dt_str) - run_start).total_seconds() // 3600)
            rmse_dict["lead_hours"] = lead_h
        records.append(rmse_dict)

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

    # 逐個時間點/樣本的 RMSE
    out_full = f"rmse_multistep_prs{FORECAST_TOTAL_HOURS}.csv"
    df.to_csv(out_full, float_format="%.8f", encoding="utf-8-sig")
    print(f"已輸出逐樣本結果：{out_full}")
    # 依 lead_hours 分組平均（缺少 lead 時略過）
    if "lead_hours" in df.columns:
        # 先將 RMSE 還原為 MSE 再平均，最後開根號回到 RMSE
        rmse_cols = [c for c in df.columns if c.endswith("_rmse")]
        bias_cols = [c for c in df.columns if c.endswith("_bias")]
        lead_df = None
        if rmse_cols:
            rmse_df = df.groupby("lead_hours")[rmse_cols].apply(
                lambda g: np.sqrt((g ** 2).mean())
            )
            lead_df = rmse_df
        if bias_cols:
            bias_df = df.groupby("lead_hours")[bias_cols].mean()
            lead_df = bias_df if lead_df is None else pd.concat([lead_df, bias_df], axis=1)
        if lead_df is not None:
            lead_df = lead_df.sort_index()
            out_lead = f"rmse_multistep_prs{FORECAST_TOTAL_HOURS}_by_lead.csv"
            lead_df.to_csv(out_lead, float_format="%.8f", encoding="utf-8-sig")
            print(f"已輸出依 lead 平均結果：{out_lead}")


if __name__ == "__main__":
    main(max_workers=WORKERS)
