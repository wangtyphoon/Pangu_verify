import numpy as np
import pandas as pd
import os
from loss import create_spatial_weights
from subset import extract_subset

# 通道映射与路径配置
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

forecast_dir = '../results'
era5_dir     = '../forecasts'
nc_path      = r'../forecasts\2020-05-05-00-00\upper.nc'

# 计算权重
weight = create_spatial_weights(H=192, W=192, boundary=8, device='cpu')  # shape (H, W)

# 用于存储每个时刻的 RMSE
records = []

def weighted_mse_dict(forecast_dict, era5_dict, weight):
    sum_w = np.sum(weight)
    mse_per_var = {}
    for var in forecast_dict:
        err2 = (forecast_dict[var] - era5_dict[var]) ** 2
        mse_per_var[var] = np.sum(err2 * weight) / sum_w
    mse_overall = np.mean(list(mse_per_var.values()))
    return mse_per_var, mse_overall

# 遍历所有时刻文件
for folder in os.listdir(forecast_dir):
    if not folder.startswith("2024"):  # 仅处理 2024 年的数据
        continue
    dt = folder.split('o')[1]
    print(f"Processing {dt}...")
    fc_path  = os.path.join(forecast_dir, folder,   f'output_upper_{dt}.npy')
    er_path  = os.path.join(era5_dir,     dt,        'input_upper.npy')
    
    # 载入并提取子集
    fc_data  = np.load(fc_path)
    er_data  = np.load(er_path)
    fc_sub   = extract_subset(fc_data, nc_path, channel_level_map)
    er_sub   = extract_subset(er_data, nc_path, channel_level_map)
    
    # 计算加权 MSE/ RMSE
    mse_dict, mse_all = weighted_mse_dict(fc_sub, er_sub, weight)
    # rmse_dict = {f"{var}_rmse": np.sqrt(mse) for var, mse in mse_dict.items()}
    # rmse_dict["overall_rmse"] = np.sqrt(mse_all)
    rmse_dict = {f"{var}_mse": mse for var, mse in mse_dict.items()}
    rmse_dict["overall_mse"] = mse_all
    rmse_dict["datetime"] = dt
    
    records.append(rmse_dict)

# 构建 DataFrame
df = pd.DataFrame(records)
df = df.set_index("datetime").sort_index()

# 计算平均 RMSE（按列求均值）
mean_series = df.mean(axis=0).rename("mean_rmse")
mean_series = np.sqrt(mean_series).rename("mean_rmse")  # 開根號 → RMSE
df.loc["平均"] = mean_series

# 保存到 CSV
out_csv = "rmse_summary_prs8_macro2024.csv"
df.to_csv(out_csv, float_format="%.8f", encoding="utf-8-sig")

print(f"结果已保存到 {out_csv}")
