import numpy as np
from netCDF4 import Dataset

def extract_subset(data,
                   nc_path,
                   channel_level_map,
                   lat_range=(0.0, 47.75),
                   lon_range=(102.25, 150.0)):
    """
    從多通道、多層次的 data 陣列中，提取指定經緯度區域和指定通道/層次的子集。

    Parameters
    ----------
    data : np.ndarray
        形狀 (n_channels, n_levels, n_lat, n_lon) 的資料陣列。
    nc_path : str
        包含 latitude, longitude 變數的 NetCDF 檔案路徑，用來抓取座標。
    channel_level_map : dict
        欲提取變數到 (channel_idx, level_idx) 的對應，例如：
        {
            'z500': (z500_channel, 5),
            'q700': (q700_channel, 3),
            't850': (t850_channel, 2),
            'u850': (u850_channel, 2),
            'v850': (v850_channel, 2),
        }
    lat_range : tuple(float, float)
        提取前的原始緯度範圍 (min_lat, max_lat)。
    lon_range : tuple(float, float)
        提取前的原始經度範圍 (min_lon, max_lon)。
    edge_interval : float
        邊界內縮量 (預設 0.25*9)。

    Returns
    -------
    subset : dict
        包含各變數子陣列及對應經緯度：
        {
            'z500': …,
            'q700': …,
            …,
            'latitudes': array([...]),
            'longitudes': array([...])
        }
    """
    # 1) 讀取經緯度
    with Dataset(nc_path) as ds:
        lats = ds.variables['latitude'][:]      # (n_lat,)
        lons = ds.variables['longitude'][:]     # (n_lon,)
    # 統一轉成 0–360° 範圍
    lons = np.mod(lons, 360)

    # 2) 計算實際邊界
    lat_min = lat_range[0] 
    lat_max = lat_range[1] 
    lon_min = lon_range[0]
    lon_max = lon_range[1] 

    # 3) 找出索引
    lat_idx = np.where((lats >= lat_min) & (lats <= lat_max))[0]
    lon_idx = np.where((lons >= lon_min) & (lons <= lon_max))[0]
    lat_sl = slice(lat_idx.min(), lat_idx.max() + 1)
    lon_sl = slice(lon_idx.min(), lon_idx.max() + 1)

    # 4) 依照對應表抽子集
    subset = {}
    for name, (ch, lv) in channel_level_map.items():
        subset[name] = data[ch, lv, lat_sl, lon_sl]

    return subset

import numpy as np
from netCDF4 import Dataset

def extract_subset_surface(data,
                           nc_path,
                           channel_map_surface,
                           lat_range=(0.0, 47.75),
                           lon_range=(102.25, 150.0)):
    """
    從地表層多通道資料中，提取指定經緯度區域與指定通道的子集。
    
    Parameters
    ----------
    data : np.ndarray
        形狀 (n_channels, n_lat, n_lon) 或 (n_lat, n_lon, n_channels) 的資料陣列。
    nc_path : str
        包含 latitude, longitude 變數的 NetCDF 路徑（用來抓座標切片）。
    channel_map_surface : dict
        變數名到通道索引的對應，例如：
            {'u10': 0, 'v10': 1, 'msl': 2, 't2m': 3}
    lat_range : (float, float)
        緯度範圍 (min_lat, max_lat)。
    lon_range : (float, float)
        經度範圍 (min_lon, max_lon)，假設在 0–360 座標系內。

    Returns
    -------
    subset : dict
        {
            'u10': (H, W), 'v10': (H, W), 'msl': (H, W), 't2m': (H, W),
            'latitudes': array([...]),
            'longitudes': array([...])
        }
    """

    # -------- 1) 讀取經緯度座標（假設為 1D） --------
    with Dataset(nc_path) as ds:
        lats = ds.variables['latitude'][:]    # (n_lat,)
        lons = ds.variables['longitude'][:]   # (n_lon,)

    # 經度統一到 0–360（使 lon_range=102.25~150.0 可直接使用）
    lons = np.mod(lons, 360.0)

    # -------- 2) 計算切片範圍 --------
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range

    lat_mask = (lats >= lat_min) & (lats <= lat_max)
    if not np.any(lat_mask):
        raise ValueError("lat_range 選不到任何格點，請檢查 LAT 範圍或 nc 檔經緯度")

    lon_mask = (lons >= lon_min) & (lons <= lon_max)
    if not np.any(lon_mask):
        raise ValueError("lon_range 選不到任何格點，請檢查 LON 範圍或 nc 檔經緯度")

    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    lat_sl = slice(lat_idx.min(), lat_idx.max() + 1)
    lon_sl = slice(lon_idx.min(), lon_idx.max() + 1)

    # -------- 3) 判斷資料通道位置 (C,H,W) 或 (H,W,C) --------
    if data.ndim != 3:
        raise ValueError(f"surface 資料應為 3 維 (C,H,W) 或 (H,W,C)，目前是 {data.shape}")

    nC_first = data.shape[0]
    nC_last  = data.shape[-1]
    if nC_first == len(channel_map_surface):
        layout = "CHW"
    elif nC_last == len(channel_map_surface):
        layout = "HWC"
    else:
        raise ValueError(
            f"無法判定通道維度；資料 shape={data.shape}，"
            f"但 channel_map_surface 長度={len(channel_map_surface)}"
        )

    # -------- 4) 依照對應表抽子集 --------
    subset = {}
    for name, ch in channel_map_surface.items():
        if layout == "CHW":
            sub2d = data[ch, lat_sl, lon_sl]
        else:  # "HWC"
            sub2d = data[lat_sl, lon_sl, ch]

        if sub2d.ndim != 2:
            raise ValueError(f"{name} 子陣列不是 2 維，得到 shape={sub2d.shape}")

        subset[name] = np.asarray(sub2d, dtype=np.float32)

    return subset
