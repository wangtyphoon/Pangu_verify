import numpy as np
import netCDF4 as nc
data_up = np.load(r'results\2020-05-05-06-00to2020-05-05-12-00\output_upper_2020-05-05-12-00.npy')
data_surf = np.load(r'results\2020-05-05-06-00to2020-05-05-12-00\output_surface_2020-05-05-12-00.npy')  

origin_nc = nc.Dataset(r'forecasts\2020-05-05-00-00\upper.nc')

# ------------ 1) 讀取經緯度座標 ------------
nc_path = r'forecasts\2020-05-05-00-00\upper.nc'
with nc.Dataset(nc_path) as ds:
    # 根據檔案實際的變數名稱替換 'latitude'、'longitude'
    lats = ds.variables['latitude'][:]          # 1-D (n_lat,)
    lons = ds.variables['longitude'][:]         # 1-D (n_lon,)

# 若經度範圍是 0–360°，保持不變；若是 −180–180°，轉成 0–360° 方便比較
lons = np.mod(lons, 360)

# ------------ 2) 找出目標區域的索引 ------------
lat_min, lat_max = 0.0, 47.75
lon_min, lon_max = 102.25, 150.0

lat_idx = np.where((lats >= lat_min) & (lats <= lat_max))[0]
lon_idx = np.where((lons >= lon_min) & (lons <= lon_max))[0]

lat_sl = slice(lat_idx.min(), lat_idx.max() + 1)
lon_sl = slice(lon_idx.min(), lon_idx.max() + 1)

import numpy as np
import torch
import pandas as pd

# ---------- 1) 通道權重 ----------
def create_channel_weights(device):
    """
    讀取 next_feature_table_weight.csv，產生長度 116 的通道權重向量
    （其餘未在 CSV 出現的通道預設為 1.0）。
    """
    base = torch.ones(116, device=device)
    df   = pd.read_csv("next_feature_table_weight.csv")
    base[:len(df)] = torch.tensor(df['weight'].values, device=device)
    return base         # shape: (C,)

# ---------- 2) 空間權重 ----------
def create_spatial_weights(H=192, W=192, boundary=8, device=torch.device('cpu')):
    """
    緯度餘弦加權 + 邊界距離≤boundary 的格點置 0。
    回傳 shape = (H, W) 的 torch.Tensor
    """
    # cos(φ) 緯度加權（φ=90°–緯度；若資料自北到南，需 flip）
    lats_deg = np.linspace(0, 47.75, H)          # 依實際格網修改
    lat_w    = np.cos(np.deg2rad(lats_deg))      # shape (H,)
    w2d      = np.tile(lat_w[:, None], (1, W))   # (H, W)
    w2d = np.flipud(w2d).copy()             # ERA-I/ERA5 分段常見 N→S 需反轉

    # 距離四邊最近的格點距離
    y_idx, x_idx   = np.indices((H, W))
    dist_top       = y_idx
    dist_bottom    = H - 1 - y_idx
    dist_left      = x_idx
    dist_right     = W - 1 - x_idx
    min_dist       = np.minimum.reduce([dist_top, dist_bottom, dist_left, dist_right])

    # 將距離 ≤ boundary 的格點權重設定為 0
    w2d[min_dist <= boundary] = 0.0

    return torch.tensor(w2d, dtype=torch.float32, device=device)  # (H, W)

# ---------- 3) 加權 RMSE ----------
def weighted_rmse_loss(pred, label, channel_weights, spatial_weights, epsilon=1e-8):
    """
    pred, label:    (batch*node, C)
    channel_weights: (C,)
    spatial_weights_2d: (H, W)
    """

    Bn, C = pred.shape

    # 1) 展平空间权重
    node = spatial_weights.numel()           # H*W
    assert Bn % node == 0, "batch*node 必须能被 node 整除"
    batch = Bn // node

    # flatten to (node,)
    spatial_flat = spatial_weights.view(-1)  # (node,)

    # repeat across batch -> (batch*node,)
    spatial_repeat = spatial_flat.repeat(batch) # (batch*node,)

    # 2) 构造广播权重矩阵 (batch*node, C)
    # channel_weights: (C,) -> (1, C)
    w_c = channel_weights.unsqueeze(0)         # (1, C)
    # spatial_repeat: (batch*node,) -> (batch*node, 1)
    w_s = spatial_repeat.unsqueeze(1)          # (batch*node, 1)

    # outer product 广播得到 (batch*node, C)
    total_w = w_s * w_c                        # (batch*node, C)

    # 3) 加权 RMSE
    diff2 = (pred - label)**2                  # (batch*node, C)
    weighted = diff2 * total_w                 # (batch*node, C)
    mse = weighted.mean()
    rmse = torch.sqrt(mse + epsilon)
    return rmse
import matplotlib.pyplot as plt
# lat = np.load("../../../era5/lat.npy")  # 假設緯度數據存儲在此
w2d = create_spatial_weights(H=192, W=192, boundary=8, device=torch.device('cpu'))  # (H, W)

plt.figure(figsize=(6,5))
plt.imshow(w2d.cpu().numpy(), 
           cmap='viridis')    # 或你喜歡的其他 colormap
plt.colorbar(label='Spatial weight')
plt.title('2D Spatial Weight Map')
plt.xlabel('X index')
plt.ylabel('Y index')
plt.tight_layout()
plt.show()