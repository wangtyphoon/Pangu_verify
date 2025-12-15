import os
import sys
from datetime import datetime
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

from subset import extract_subset_surface

# 地表層通道映射
channel_map_surface = {
    'msl': 0,
    'u10': 1,
    'v10': 2,
    't2m': 3,
}

# 路徑設定（以專案根目錄為基準）
results_root = os.path.join(BASE_DIR, 'results')
era5_root = os.path.join(BASE_DIR, 'forecasts')

# 空間權重（輸出檔案已經裁切過，因此這裡只用來給邊界與緯度加權）
weight_2d = create_spatial_weights(H=192, W=192, boundary=8, device='cpu')


def parse_stamp(stamp: str) -> datetime:
    return datetime.strptime(stamp, "%Y-%m-%d-%H-%M")


def weighted_mse_dict(forecast_dict, era5_dict, weight):
    sum_w = np.sum(weight)
    mse_per_var = {}
    for var in forecast_dict:
        err2 = (forecast_dict[var] - era5_dict[var]) ** 2
        mse_per_var[var] = float(np.sum(err2 * weight) / sum_w)
    mse_overall = float(np.mean(list(mse_per_var.values())))
    return mse_per_var, mse_overall


def surface_to_dict(data, surface_nc_path):
    # 若資料已裁切到目標區域，直接依通道映射回傳
    H, W = weight_2d.shape
    if data.ndim == 3:
        if data.shape[0] == len(channel_map_surface) and data.shape[1:] == (H, W):
            return {name: np.asarray(data[ch], dtype=np.float32) for name, ch in channel_map_surface.items()}
        if data.shape[-1] == len(channel_map_surface) and data.shape[:2] == (H, W):
            return {name: np.asarray(data[:, :, ch], dtype=np.float32) for name, ch in channel_map_surface.items()}

    # 否則假設是全球 ERA5，需依 NC 切子區域
    return extract_subset_surface(data, surface_nc_path, channel_map_surface)

def evaluate_run(run_dir):
    run_name = os.path.basename(run_dir)
    if "to" in run_name:
        start_str, _ = run_name.split("to", 1)
        try:
            run_start = parse_stamp(start_str)
        except ValueError:
            run_start = None
            
    else:
        run_start = None

    records = []
    nc_path = '../surface.nc'
    for fn in sorted(os.listdir(run_dir)):
        if not fn.startswith("output_surface_") or not fn.endswith(".npy"):
            continue
        dt_str = fn[len("output_surface_"):-4]

        fc_path = os.path.join(run_dir, fn)
        er_dir = os.path.join(era5_root, dt_str)
        er_path = os.path.join(er_dir, 'input_surface.npy')
        if not os.path.exists(er_path):
            print(f"[WARN] Skip {dt_str}: missing ERA5 {er_path}")
            continue
        if not os.path.exists(nc_path):
            print(f"[WARN] Skip {dt_str}: missing ERA5 NC {nc_path}")
            continue

        fc_data = np.load(fc_path)
        er_data = np.load(er_path)

        fc_dict = surface_to_dict(fc_data, nc_path)
        er_dict = surface_to_dict(er_data, nc_path)

        mse_dict, mse_all = weighted_mse_dict(fc_dict, er_dict, weight_2d)
        rmse_dict = {f"{var}_rmse": np.sqrt(mse) for var, mse in mse_dict.items()}
        rmse_dict["overall_rmse"] = np.sqrt(mse_all)
        rmse_dict["datetime"] = dt_str
        rmse_dict["run"] = run_name

        if run_start is not None:
            lead_h = int((parse_stamp(dt_str) - run_start).total_seconds() // 3600)
            rmse_dict["lead_hours"] = lead_h
        records.append(rmse_dict)

    return records


def main():
    all_records = []
    for folder in sorted(os.listdir(results_root)):
        if not folder.startswith("2024"):
            continue
        run_dir = os.path.join(results_root, folder)
        if not os.path.isdir(run_dir):
            continue

        print(f"Evaluating {folder} ...")
        recs = evaluate_run(run_dir)
        all_records.extend(recs)

    if not all_records:
        print("No valid records found.")
        return

    df = pd.DataFrame(all_records).set_index("datetime").sort_index()

    # 逐個時間點/樣本的 RMSE
    out_full = "rmse_multistep_sfc72.csv"
    df.to_csv(out_full, float_format="%.8f", encoding="utf-8-sig")
    print(f"已輸出逐樣本結果：{out_full}")
    # 依 lead_hours 分組平均（缺少 lead 時略過）
    if "lead_hours" in df.columns:
        # 先將 RMSE 還原為 MSE 再平均，最後開根號回到 RMSE
        rmse_cols = [c for c in df.columns if c.endswith("_rmse")]
        if rmse_cols:
            lead_df = (
                df.groupby("lead_hours")[rmse_cols]
                .apply(lambda g: np.sqrt((g ** 2).mean()))
                .sort_index()
            )
            out_lead = "rmse_multistep_sfc72_by_lead.csv"
            lead_df.to_csv(out_lead, float_format="%.8f", encoding="utf-8-sig")
            print(f"已輸出依 lead 平均結果：{out_lead}")


if __name__ == "__main__":
    main()
