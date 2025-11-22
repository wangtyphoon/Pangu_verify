import pandas as pd
ds = pd.read_csv("rmse_summary_prs8_macro2024.csv")
# 移除 datetime 欄位不是日期格式的列（例如「平均」）
ds = ds[pd.to_datetime(ds['datetime'], format='%Y-%m-%d-%H-%M', errors='coerce').notnull()]

ds['datetime'] = pd.to_datetime(ds['datetime'], format='%Y-%m-%d-%H-%M')
ds['month'] = ds['datetime'].dt.month

# 計算各月份平均
monthly_mean = ds.groupby('month').mean(numeric_only=True)
#計算標準差
monthly_std = ds.groupby('month').std(numeric_only=True)
monthly_mean.to_csv("monthly_mean_rmse_sfc.csv", float_format="%.8f", encoding="utf-8-sig")
monthly_std.to_csv("monthly_std_rmse_sfc.csv", float_format="%.8f", encoding="utf-8-sig")
ds['hour'] = ds['datetime'].dt.hour
#回退6小時
ds['hour'] = (ds['hour'] - 6) % 24  # 確保小時在 0-23 範圍內
# 計算各小時平均
hourly_mean = ds.groupby('hour').mean(numeric_only=True)
# 計算標準差
hourly_std = ds.groupby('hour').std(numeric_only=True)
hourly_mean.to_csv("hourly_mean_rmse_prs.csv", float_format="%.8f", encoding="utf-8-sig")
hourly_std.to_csv("hourly_std_rmse_prs.csv", float_format="%.8f", encoding="utf-8-sig")

