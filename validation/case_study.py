import pandas as pd

# 讀取 CSV
df = pd.read_csv("tc_intensity_extracted.csv")

# 計算差距欄位
df["diff_msl"]  = (df["forecast_msl_min_hPa"] - df["target_msl_min_hPa"]).abs()
df["diff_s10"]  = (df["forecast_s10_max"]     - df["target_s10_max"]).abs()
df["diff_s850"] = (df["forecast_s850_max"]    - df["target_s850_max"]).abs()

# ========== 分別找 Top10 ==========
top10_msl  = df.sort_values("diff_msl", ascending=False).head(10)
top10_s10  = df.sort_values("diff_s10", ascending=False).head(10)
top10_s850 = df.sort_values("diff_s850", ascending=False).head(10)

# 輸出結果
print("==== MSL Top10 ====")
print(top10_msl[["time", "lat", "lon", 
                 "forecast_msl_min_hPa", "target_msl_min_hPa", "diff_msl"]])

print("\n==== S10 Top10 ====")
print(top10_s10[["time", "lat", "lon", 
                 "forecast_s10_max", "target_s10_max", "diff_s10"]])

print("\n==== S850 Top10 ====")
print(top10_s850[["time", "lat", "lon", 
                  "forecast_s850_max", "target_s850_max", "diff_s850"]])

import pandas as pd
import numpy as np

# 讀取 CSV
df = pd.read_csv("tc_intensity_extracted.csv")

# 計算差距
df["diff_msl"]  = df["forecast_msl_min_hPa"] - df["target_msl_min_hPa"]
df["diff_s10"]  = df["forecast_s10_max"]     - df["target_s10_max"]
df["diff_s850"] = df["forecast_s850_max"]    - df["target_s850_max"]

# ===== MSE =====
mse_msl  = (df["diff_msl"]**2).mean()
mse_s10  = (df["diff_s10"]**2).mean()
mse_s850 = (df["diff_s850"]**2).mean()

# ===== RMSE =====
rmse_msl  = np.sqrt(mse_msl)
rmse_s10  = np.sqrt(mse_s10)
rmse_s850 = np.sqrt(mse_s850)

print("====== RMSE 結果 ======")
print(f"MSL  RMSE = {rmse_msl:.4f}")
print(f"S10  RMSE = {rmse_s10:.4f}")
print(f"S850 RMSE = {rmse_s850:.4f}")

# 計算差距（不取絕對值，保留正負號）
df["diff_msl"]  = df["forecast_msl_min_hPa"] - df["target_msl_min_hPa"]
df["diff_s10"]  = df["forecast_s10_max"]     - df["target_s10_max"]
df["diff_s850"] = df["forecast_s850_max"]    - df["target_s850_max"]

# ===== Bias =====
bias_msl  = df["diff_msl"].mean()
bias_s10  = df["diff_s10"].mean()
bias_s850 = df["diff_s850"].mean()

print("====== Bias 結果 ======")
print(f"MSL  Bias = {bias_msl:.4f}")
print(f"S10  Bias = {bias_s10:.4f}")
print(f"S850 Bias = {bias_s850:.4f}")

import pandas as pd
import matplotlib.pyplot as plt

# 讀取 CSV
df = pd.read_csv("tc_intensity_extracted.csv")

# 定義要畫的三組變數
pairs = [
    ("forecast_msl_min_hPa", "target_msl_min_hPa", "MSL min (hPa)"),
    ("forecast_s10_max", "target_s10_max", "S10 max wind"),
    ("forecast_s850_max", "target_s850_max", "S850 max wind"),
]

plt.figure(figsize=(15, 5))

for i, (f_col, t_col, title) in enumerate(pairs, 1):
    plt.subplot(1, 3, i)
    plt.scatter(df[t_col], df[f_col], alpha=0.5, s=15)
    plt.plot([df[t_col].min(), df[t_col].max()],
             [df[t_col].min(), df[t_col].max()],
             'r--', label="y = x (perfect)")
    plt.xlabel("Target")
    plt.ylabel("Forecast")
    plt.title(f"{title}: Forecast vs Target")
    plt.legend()

plt.tight_layout()
plt.show()

# ===== 整理成 DataFrame =====
metrics = pd.DataFrame({
    "Variable": ["MSL", "S10", "S850"],
    "RMSE": [rmse_msl, rmse_s10, rmse_s850],
    "Bias": [bias_msl, bias_s10, bias_s850]
})

# 輸出 CSV
metrics.to_csv("tc_rmse_bias_results.csv", index=False, float_format="%.4f")

print("已輸出 tc_rmse_bias_results.csv")
print(metrics)