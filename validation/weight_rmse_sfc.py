import numpy as np
import pandas as pd
import os
from loss import create_spatial_weights
from subset import extract_subset_surface  # ← 新增：改用 surface 版
# -----------------------------------------------

# 地表層通道映射
channel_map_surface = {
    'msl': 0,
    'u10': 1,
    'v10': 2,
    't2m': 3,
}

forecast_dir = '../results'
era5_dir     = '../forecasts'
nc_path      = r'../forecasts\2020-05-05-00-00\surface.nc'  # ← 換成 surface

weight = create_spatial_weights(H=192, W=192, boundary=8, device='cpu')  # (H,W)

records = []

def weighted_mse_dict(forecast_dict, era5_dict, weight):
    sum_w = np.sum(weight)
    mse_per_var = {}
    for var in forecast_dict:
        err2 = (forecast_dict[var] - era5_dict[var]) ** 2
        mse_per_var[var] = np.sum(err2 * weight) / sum_w
    mse_overall = np.mean(list(mse_per_var.values()))
    return mse_per_var, mse_overall

for folder in os.listdir(forecast_dir):
    # 假設資料夾命名仍是 ...output_surface_<dt>.npy 對應 ../forecasts/<dt>/input_surface.npy
    if not folder.startswith("2024"):  # 仅处理 2024 年的数据
        continue
    dt = folder.split('o')[1]
    print(f"Processing {dt}...")

    fc_path = os.path.join(forecast_dir, folder, f'output_surface_{dt}.npy')
    er_path = os.path.join(era5_dir, dt, 'input_surface.npy')

    fc_data = np.load(fc_path)   # shape: (4,H,W) 或 (H,W,4)
    er_data = np.load(er_path)   # shape: (4,H,W) 或 (H,W,4)

    fc_sub  = extract_subset_surface(fc_data, nc_path, channel_map_surface)
    er_sub  = extract_subset_surface(er_data, nc_path, channel_map_surface)

    mse_dict, mse_all = weighted_mse_dict(fc_sub, er_sub, weight)
    # rmse_dict = {f"{var}_rmse": np.sqrt(mse) for var, mse in mse_dict.items()}
    # rmse_dict["overall_rmse"] = np.sqrt(mse_all)
    rmse_dict = {f"{var}_mse": mse for var, mse in mse_dict.items()}
    rmse_dict["overall_mse"] = mse_all
    rmse_dict["datetime"] = dt
    records.append(rmse_dict)

df = pd.DataFrame(records).set_index("datetime").sort_index()
# mean_series = df.mean(axis=0).rename("mean_rmse")
mean_series = df.mean(axis=0).rename("mean_rmse")
mean_series = np.sqrt(mean_series).rename("mean_rmse")  # 開根號 → RMSE
df.loc["平均"] = mean_series

out_csv = "rmse_summary_surface8_macro2024.csv"
df.to_csv(out_csv, float_format="%.8f", encoding="utf-8-sig")
print(f"結果已保存到 {out_csv}")