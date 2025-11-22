import numpy as np
import pandas as pd
import os
from subset import extract_subset
import matplotlib.pyplot as plt

# ---------------- 基本設定（保持你的設定不變） ----------------
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

LAT_RANGE = (2.25, 45.50)
LON_RANGE = (104.5, 147.75)

# ---------------- 面積權重工具函式（等經緯度格網） ----------------
R_EARTH = 6371.0  # km

def select_bbox_indices(lat2d, lon2d, lat_range, lon_range):
    lat_min, lat_max = min(lat_range), max(lat_range)
    lon_min, lon_max = min(lon_range), max(lon_range)
    mask = (lat2d >= lat_min) & (lat2d <= lat_max) & (lon2d >= lon_min) & (lon2d <= lon_max)
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return slice(y0, y1), slice(x0, x1)

def cell_area_weights(lat2d, lon2d):
    """
    在規則經緯度網格上，用 sin(phi) 帶狀積分近似每格面積：
    A_i,j ≈ R^2 * (sin(phi_n) - sin(phi_s)) * dlon
    備註：只需相對權重，因此無需乘上常數比例因子。
    """
    if lat2d.ndim == 1 and lon2d.ndim == 1:
        lat2d, lon2d = np.meshgrid(lat2d, lon2d, indexing='ij')

    lat1d = lat2d[:, 0]
    lon1d = lon2d[0, :]

    # 度→弧度的步長（假設規則格網）
    dlat = np.deg2rad(np.abs(np.diff(lat1d).mean()))
    dlon = np.deg2rad(np.abs(np.diff(lon1d).mean()))

    phi_c = np.deg2rad(lat1d)
    phi_n = phi_c + 0.5 * dlat
    phi_s = phi_c - 0.5 * dlat

    strip = (np.sin(phi_n) - np.sin(phi_s)) * dlon      # 每條緯帶的面積比例
    A = np.repeat(strip[:, None], lon1d.size, axis=1)   # 擴展到 2D
    return A

# ---------------- 只加入「面積加權」的徑向譜 ----------------
def radial_spectrum_2d_areaweighted(field2d, area2d):
    """
    與你原本的 radial_spectrum_2d 幾乎相同，
    差別：FFT 之前做「面積加權去均值」並乘上 sqrt(area)。
    其餘（徑向分箱、k 定義）完全不變。
    """
    H, W = field2d.shape
    assert area2d.shape == (H, W), "area2d 形狀需與 field2d 相同"

    # 規模化面積以避免數值過大/過小，但保留相對權重
    A = area2d / np.mean(area2d)

    # 面積加權去均值
    mean_w = np.sum(field2d * A) / np.sum(A)
    f = (field2d - mean_w) * np.sqrt(A)

    # 與原本相同的 FFT → 幅度 → 徑向分箱
    fft2 = np.fft.fftshift(np.fft.fft2(f))
    amp = np.abs(fft2)

    cy, cx = H // 2, W // 2
    y, x = np.indices((H, W))
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    max_r = min(cx, cy)

    bin_sum   = np.bincount(r.ravel(), amp.ravel(), minlength=max_r+1)
    bin_count = np.bincount(r.ravel(),               minlength=max_r+1)
    radial_mean = bin_sum[:max_r+1] / np.maximum(bin_count[:max_r+1], 1)
    k = np.arange(max_r+1)  # 與你原本一致（像素半徑）
    return k, radial_mean

# --------- 載入經緯度，先把子區的面積權重算好（一次即可） ---------
LAT_ALL = np.load("../lat.npy")   # 可為 1D 或 2D
LON_ALL = np.load("../lon.npy")   # 可為 1D 或 2D
if LAT_ALL.ndim == 1 and LON_ALL.ndim == 1:
    LAT_ALL, LON_ALL = np.meshgrid(LAT_ALL, LON_ALL, indexing='ij')

ys, xs = select_bbox_indices(LAT_ALL, LON_ALL, LAT_RANGE, LON_RANGE)
LAT_SUB = LAT_ALL[ys, xs].copy()
LON_SUB = LON_ALL[ys, xs].copy()
AREA_SUB = cell_area_weights(LAT_SUB, LON_SUB)  # 與 extract_subset 同一子區

# ---------------- 你的原流程，僅替換呼叫的頻譜函式 ----------------
variables = list(channel_level_map.keys())
spectra_list = {
    'forecast': {v: [] for v in variables},
    'era5':     {v: [] for v in variables},
}

for folder in os.listdir(forecast_dir):
    dt = folder.split('o')[1]
    if not folder.startswith("2024"):  # 仅处理 2024 年的数据
        continue
    print(f"Processing {dt}...")
    fc_path  = os.path.join(forecast_dir, folder,   f'output_upper_{dt}.npy')
    er_path  = os.path.join(era5_dir,     dt,        'input_upper.npy')

    fc_data  = np.load(fc_path)
    er_data  = np.load(er_path)

    fc_sub   = extract_subset(fc_data, nc_path, channel_level_map,
                              lat_range=LAT_RANGE, lon_range=LON_RANGE)
    er_sub   = extract_subset(er_data, nc_path, channel_level_map,
                              lat_range=LAT_RANGE, lon_range=LON_RANGE)

    for var_name in variables:
        fc_field = fc_sub[var_name]  # 2D
        er_field = er_sub[var_name]  # 2D

        # ★ 只把 radial_spectrum_2d 換成 areaweighted 版本
        k_fc, spec_fc = radial_spectrum_2d_areaweighted(fc_field, AREA_SUB)
        k_er, spec_er = radial_spectrum_2d_areaweighted(er_field, AREA_SUB)

        spectra_list['forecast'][var_name].append((k_fc, spec_fc))
        spectra_list['era5'][var_name].append((k_er, spec_er))

# 3) 計算平均譜與標準差（保持不變）
mean_spectra = {'forecast': {}, 'era5': {}}
std_spectra  = {'forecast': {}, 'era5': {}}

for var in variables:
    k = spectra_list['forecast'][var][0][0]
    all_fc = np.stack([sp[1] for sp in spectra_list['forecast'][var]], axis=0)
    all_er = np.stack([sp[1] for sp in spectra_list['era5'][var]],     axis=0)

    mean_fc = np.mean(all_fc, axis=0)
    mean_er = np.mean(all_er, axis=0)
    std_fc  = np.std(all_fc,  axis=0)
    std_er  = np.std(all_er,  axis=0)

    mean_spectra['forecast'][var] = (k, mean_fc)
    mean_spectra['era5'][var]     = (k, mean_er)
    std_spectra['forecast'][var]  = (k, std_fc)
    std_spectra['era5'][var]      = (k, std_er)

# 4) 輸出 CSV（保持欄位與你原本一致）
for var in variables:
    k, mean_er = mean_spectra['era5'][var]
    _, std_er  = std_spectra ['era5'][var]
    _, mean_fc = mean_spectra['forecast'][var]
    _, std_fc  = std_spectra ['forecast'][var]

    df = pd.DataFrame({
        'k':          k,
        'era5_mean':  mean_er,
        'era5_std':   std_er,
        'pangu_mean': mean_fc,
        'pangu_std':  std_fc,
    })
    df.to_csv(f'{var}_spectral_stats.csv', index=False)
    print(f'Saved {var}_spectral_stats.csv')

# 5) 輸出 NPZ（保持鍵名與你原本一致）
npz_dict = {}
for var in variables:
    k, mean_er = mean_spectra['era5'][var]
    _, std_er  = std_spectra ['era5'][var]
    _, mean_fc = mean_spectra['forecast'][var]
    _, std_fc  = std_spectra ['forecast'][var]
    npz_dict[f'k_{var}']          = k
    npz_dict[f'era5_mean_{var}']  = mean_er
    npz_dict[f'era5_std_{var}']   = std_er
    npz_dict[f'pangu_mean_{var}'] = mean_fc
    npz_dict[f'pangu_std_{var}']  = std_fc

np.savez('all_spectral_data.npz', **npz_dict)
print('Saved all_spectral_data.npz')
