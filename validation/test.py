import numpy as np  
import matplotlib.pyplot as plt
from subset import extract_subset
import os
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

target_dt = '2024-10-28-06-00'
current_dt = '2024-10-28-00-00'
folder = "2024-10-28-00-00to2024-10-28-06-00"
fc_path  = os.path.join(forecast_dir, folder,   f'output_upper_{target_dt}.npy')
current_path = os.path.join(era5_dir,     current_dt,        'input_upper.npy') 
target_path  = os.path.join(era5_dir,     target_dt,        'input_upper.npy')

 # 载入并提取子集
fc_data  = np.load(fc_path)
current_data  = np.load(current_path)
target_data  = np.load(target_path)
fc_sub   = extract_subset(fc_data, nc_path, channel_level_map)
current_sub   = extract_subset(current_data, nc_path, channel_level_map)
target_sub   = extract_subset(target_data, nc_path, channel_level_map)

import numpy as np
import matplotlib.pyplot as plt

# 保證資料為 (192, 192)
def to_hw(arr):
    if arr.ndim == 2 and arr.shape == (192, 192):
        return arr
    if arr.ndim == 1 and arr.size == 192*192:
        return arr.reshape(192, 192)
    if arr.ndim == 2 and (arr.shape[0] == 192*192 or arr.shape[1] == 192*192):
        if arr.shape[0] == 192*192:
            return arr.reshape(192, 192)
        else:
            return arr.T.reshape(192, 192)
    return arr  # 若已正確，原樣返回

vars_order = ['z500', 'q700', 't850', 'u850', 'v850']

fig, axes = plt.subplots(len(vars_order), 4, figsize=(20, 22))
fig.suptitle(
    f"{current_dt} (Current) → {target_dt} (Truth) — 5 Vars × (Current / Forecast / Truth / Error)",
    y=0.995, fontsize=16
)

for i, var in enumerate(vars_order):
    C = to_hw(current_sub[var])   # Current
    F = to_hw(fc_sub[var])        # Forecast (從 current_dt 起報到 target_dt 的預測)
    T = to_hw(target_sub[var])    # Truth（target_dt 的 ERA5）

    # 三張圖統一色階
    vmin = float(np.nanmin([C.min(), F.min(), T.min()]))
    vmax = float(np.nanmax([C.max(), F.max(), T.max()]))

    im0 = axes[i, 0].imshow(C, cmap='jet', vmin=vmin, vmax=vmax)
    axes[i, 0].set_title(f"{var} - Current")
    plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.02)

    im1 = axes[i, 1].imshow(F, cmap='jet', vmin=vmin, vmax=vmax)
    axes[i, 1].set_title(f"{var} - Forecast")
    plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.02)

    im2 = axes[i, 2].imshow(T, cmap='jet', vmin=vmin, vmax=vmax)
    axes[i, 2].set_title(f"{var} - Truth")
    plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.02)

    # 誤差（F - T）用對稱色階
    E = F - T
    eabs = float(np.nanmax(np.abs(E)))
    if eabs == 0:
        eabs = 1e-9  # 避免 vmin=vmax
    im3 = axes[i, 3].imshow(E, cmap='bwr', vmin=-eabs, vmax=eabs)
    rmse = float(np.sqrt(np.nanmean((F - T) ** 2)))
    axes[i, 3].set_title(f"{var} - Error (F - T)")
    # 在右上角標 RMSE
    axes[i, 3].text(0.98, 0.02, f"RMSE={rmse:.4g}", ha='right', va='bottom',
                    transform=axes[i, 3].transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
    plt.colorbar(im3, ax=axes[i, 3], fraction=0.046, pad=0.02)

    # 美化：移除刻度
    for j in range(4):
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])

plt.tight_layout(rect=[0, 0, 1, 0.98])

out_png = f"{current_dt}_to_{target_dt}_5x4.png"
plt.savefig(out_png, dpi=150)
plt.show()
print(f"Saved figure to {out_png}")

# === 存檔 (npz) ===
# out_npz = f"{current_dt}_to_{target_dt}_data.npz"
# np.savez(
#     out_npz,
#     current_sub=current_sub,
#     forecast_sub=fc_sub,
#     truth_sub=target_sub,
#     vars_order=vars_order
# )
# print(f"Saved data to {out_npz}")