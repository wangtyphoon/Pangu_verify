import numpy as np
import pandas as pd
import os
from subset import extract_subset

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
APPLY_LAT_WEIGHT = False      # 是否套用緯度面積權重（cos(lat)）
AMPLITUDE_SPECTRUM = False    # True: 幅度譜；False: 功率譜
DETREND_MEAN = False          # 是否去除（加權）平均
DENSITY_NORM = False         # 是否做頻譜密度正規化

# ---------------- 面積權重工具函式（等經緯度格網） ----------------
def select_bbox_indices(lat2d, lon2d, lat_range, lon_range):
    lat_min, lat_max = min(lat_range), max(lat_range)
    lon_min, lon_max = min(lon_range), max(lon_range)
    mask = (lat2d >= lat_min) & (lat2d <= lat_max) & (lon2d >= lon_min) & (lon2d <= lon_max)
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return slice(y0, y1), slice(x0, x1)

# ---------------- 徑向譜計算（參考 spectrum_same） ----------------
def radial_power_spectrum(field2d,
                          area2d=None,
                          detrend_mean=True,
                          amplitude=True,
                          density_norm=False):
    """
    面積加權幅度/功率徑向譜：
      1) (可選) 面積加權去均值，乘上 sqrt(A_norm)
      2) 2D FFT → |F| 或 |F|^2
      3) 以像素半徑做徑向平均
    """
    H, W = field2d.shape

    if area2d is not None:
        if area2d.shape != (H, W):
            raise ValueError(f"area2d 形狀需為 {(H, W)}，目前是 {area2d.shape}")
        A = area2d.astype(np.float64, copy=False)
        A = A / np.nanmean(A)
        if detrend_mean:
            mean_w = np.nansum(field2d * A) / np.nansum(A)
        else:
            mean_w = 0.0
        x = (field2d.astype(np.float64) - mean_w) * np.sqrt(A)
    else:
        x = field2d.astype(np.float64, copy=False)
        if detrend_mean:
            x = x - np.nanmean(x)

    F = np.fft.fft2(x)
    F = np.fft.fftshift(F)

    if amplitude:
        P = np.abs(F)
        if density_norm:
            P = P / (H * W)
    else:
        P = np.abs(F) ** 2
        if density_norm:
            P = P / (H * W) ** 2

    cy, cx = H // 2, W // 2
    y, xg = np.indices((H, W))
    r = np.sqrt((xg - cx)**2 + (y - cy)**2).astype(int)
    kmax = min(cx, cy) + 1

    spec_sum = np.bincount(r.ravel(), weights=P.ravel(), minlength=kmax)
    spec_cnt = np.bincount(r.ravel(), minlength=kmax)
    spec = spec_sum[:kmax] / np.maximum(spec_cnt[:kmax], 1)

    k = np.arange(kmax)
    return k, spec

# --------- 載入經緯度，先把子區的面積權重算好（一次即可） ---------
LAT_ALL = np.load("../lat.npy")   # 可為 1D 或 2D
LON_ALL = np.load("../lon.npy")   # 可為 1D 或 2D
if LAT_ALL.ndim == 1 and LON_ALL.ndim == 1:
    LAT_ALL, LON_ALL = np.meshgrid(LAT_ALL, LON_ALL, indexing='ij')

ys, xs = select_bbox_indices(LAT_ALL, LON_ALL, LAT_RANGE, LON_RANGE)
LAT_SUB = LAT_ALL[ys, xs].copy()
AREA_SUB = np.cos(np.deg2rad(LAT_SUB)) if APPLY_LAT_WEIGHT else None

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

        k_fc, spec_fc = radial_power_spectrum(
            fc_field,
            area2d=AREA_SUB,
            detrend_mean=DETREND_MEAN,
            amplitude=AMPLITUDE_SPECTRUM,
            density_norm=DENSITY_NORM,
        )
        k_er, spec_er = radial_power_spectrum(
            er_field,
            area2d=AREA_SUB,
            detrend_mean=DETREND_MEAN,
            amplitude=AMPLITUDE_SPECTRUM,
            density_norm=DENSITY_NORM,
        )

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

np.savez('all_energy_data.npz', **npz_dict)
print('Saved all_energy_data.npz')