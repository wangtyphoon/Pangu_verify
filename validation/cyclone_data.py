import os
import numpy as np
import pandas as pd
from subset import extract_subset,extract_subset_surface
# ------------------ 設定 ------------------
CSV_PATH = "cyclone_data_2024.csv"        # 颱風路徑資料
RESULTS_DIR = "../results"                 # 預報輸出根目錄（注意：是 results 不是 result）
ERA5_DIR   = "../forecasts"               # ERA5 輸入根目錄
FEATURE_TABLE = "next_feature_table_weight.csv"  # 若 upper 是 CHW，從這裡找 u850/v850 的索引
LAT_NPY = "lat.npy"
LON_NPY = "lon.npy"
WINDOW_RADIUS = 8                          # 搜尋範圍 ±格點數
H, W = 192, 192                            # 網格大小

# ------------------ 工具函式 ------------------
def get_channel_map(feature_csv, varnames):
    """
    從 feature 表中找到指定變數所在的通道索引
    varnames: dict[name] = [keyword1, keyword2, ...]
    回傳 dict {name: idx}
    """
    df = pd.read_csv(feature_csv)
    features = df['feature'].astype(str).tolist()

    result = {}
    for name, keys in varnames.items():
        found = None
        for i, f in enumerate(features):
            if any(k.lower() in f.lower() for k in keys):
                found = i
                break
        if found is None:
            raise ValueError(f"{name} 找不到，請檢查 feature 表（keywords={keys}）")
        result[name] = found
    return result

def normalize_to_CHW(arr, H, W):
    """
    接受以下任一形狀並轉成 (C, H, W)：
    - (C, H, W)
    - (H, W, C)
    - (H*W, C)
    - (C, H*W)
    """
    arr = np.asarray(arr)
    if arr.ndim == 3:
        if arr.shape[1] == H and arr.shape[2] == W:  # (C,H,W)
            return arr
        if arr.shape[0] == H and arr.shape[1] == W:  # (H,W,C)
            return np.transpose(arr, (2, 0, 1))
        raise ValueError(f"無法辨識的3D形狀: {arr.shape}，期望 H={H}, W={W}")

    if arr.ndim == 2:
        n0, n1 = arr.shape
        if n0 == H*W:  # (H*W, C)
            C = n1
            return np.transpose(arr.reshape(H, W, C), (2, 0, 1))
        if n1 == H*W:  # (C, H*W)
            C = n0
            return arr.reshape(C, H, W)
        raise ValueError(f"2D形狀不匹配: {arr.shape}，無法還原到 H*W={H*W}")

    raise ValueError(f"僅支援 2D/3D 陣列，收到 {arr.ndim}D，shape={arr.shape}")

def wind_speed(u, v):
    return np.sqrt(u*u + v*v)

def local_min(val, i0, i1, j0, j1):
    sub = val[i0:i1, j0:j1]
    if sub.size == 0:
        return val[i0, j0]
    k = np.argmin(sub)
    di, dj = np.unravel_index(k, sub.shape)
    return sub[di, dj]

def local_max(val, i0, i1, j0, j1):
    sub = val[i0:i1, j0:j1]
    if sub.size == 0:
        return val[i0, j0]
    k = np.argmax(sub)
    di, dj = np.unravel_index(k, sub.shape)
    return sub[di, dj]

def to_1d_lat_lon(lat, lon):
    """
    將 lat, lon 轉為 1D 軸：lat1d.shape=(H,), lon1d.shape=(W,)
    - 若 2D（meshgrid），取 lat[:,0] 與 lon[0,:]
    - 若 1D，原樣返回
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    if lat.ndim == 2:
        lat1d = lat[:, 0]
    elif lat.ndim == 1:
        lat1d = lat
    else:
        raise ValueError(f"lat 維度不支援: {lat.shape}")

    if lon.ndim == 2:
        lon1d = lon[0, :]
    elif lon.ndim == 1:
        lon1d = lon
    else:
        raise ValueError(f"lon 維度不支援: {lon.shape}")

    return lat1d.astype(float), lon1d.astype(float)

def align_lon_system(lon_axis_1d, lon_value):
    """
    對齊經度系統：
    - 若 lon_axis 在 [0,360) 而 lon_value 在 [-180,180]，則 lon_value+=360（若為負）
    - 若 lon_axis 在 [-180,180] 而 lon_value 在 [0,360)，則 lon_value-=360（若 >180）
    """
    lo_min, lo_max = np.nanmin(lon_axis_1d), np.nanmax(lon_axis_1d)
    v = float(lon_value)
    if lo_min >= 0 and v < 0:
        v = v + 360.0
    if lo_max <= 180 and v > 180:
        v = v - 360.0
    return v

def safe_window_center(lat1d, lon1d, lat0, lon0):
    """回傳格點中心索引 (i, j)，保證落在 [0..H-1], [0..W-1]"""
    i = int(np.argmin(np.abs(lat1d - lat0)))
    lon0_adj = align_lon_system(lon1d, lon0)
    j = int(np.argmin(np.abs(lon1d - lon0_adj)))
    i = max(0, min(H - 1, i))
    j = max(0, min(W - 1, j))
    return i, j

def make_window(i, j, radius, H, W):
    i0, i1 = max(0, i - radius), min(H, i + radius + 1)
    j0, j1 = max(0, j - radius), min(W, j + radius + 1)
    if i0 >= i1:
        i0, i1 = i, i + 1
    if j0 >= j1:
        j0, j1 = j, j + 1
    return i0, i1, j0, j1

# ---- 針對不同 table 的抽取工具 ----
# surface 的通道固定順序（你的設定）：msl=0, u10=1, v10=2, t2m=3
SURFACE_MAP = {'msl': 0, 'u10': 1, 'v10': 2, 't2m': 3}

# upper 若是 (C,L,H,W) 結構，從這裡抓 level
# 通道順序（你的設定）：z=0, q=1, t=2, u=3, v=4
# level 索引（你的設定）：例如 850 對應到 2
UPPER_CL_MAP = {
    'u850': (3, 2),  # (u channel, 850 level)
    'v850': (4, 2),
}

def extract_surface(arr_surface):
    """回傳 (msl, u10, v10) -> (H,W)"""
    chw = normalize_to_CHW(arr_surface, H, W)
    msl = chw[SURFACE_MAP['msl'], :, :]
    u10 = chw[SURFACE_MAP['u10'], :, :]
    v10 = chw[SURFACE_MAP['v10'], :, :]
    return msl, u10, v10

# ------------------ 主程式 ------------------
def main():
    nc_path      = r'../forecasts\2020-05-05-00-00\surface.nc'  # ← 換成 surface
    # 讀取颱風資料，保留 Datetime+方便格式字串
    df = pd.read_csv(CSV_PATH)
    # 解析成 datetime（優先 YYYYMMDDHH；失敗再試 YYYYMMDDHHMM / YYYYMMDDHHMMSS）
    dt1 = pd.to_datetime(df["DATE_TIME"].astype(str), format="%Y%m%d%H", errors="coerce")
    if dt1.isna().any():
        dt2 = pd.to_datetime(df["DATE_TIME"].astype(str), format="%Y%m%d%H%M", errors="coerce")
        dt1 = dt1.fillna(dt2)
    if dt1.isna().any():
        dt3 = pd.to_datetime(df["DATE_TIME"].astype(str), format="%Y%m%d%H%M%S", errors="coerce")
        dt1 = dt1.fillna(dt3)
    if dt1.isna().any():
        bad = df.loc[dt1.isna(), "DATE_TIME"].head(5).tolist()
        raise ValueError(f"仍有無法解析的 DATE_TIME（前5筆）：{bad}")

    df["DT_END"] = dt1
    df["DT_START"] = df["DT_END"] - pd.Timedelta(hours=6)
    df["DATE_TIME_STR"] = df["DT_END"].dt.strftime("%Y%m%d%H%M%S")  # 保留一份含秒字串

    # 讀取經緯度
    lat_raw = np.load(LAT_NPY)
    lon_raw = np.load(LON_NPY)
    lat1d, lon1d = to_1d_lat_lon(lat_raw, lon_raw)

    records = []

    for row in df.itertuples(index=False):
        lat0 = float(row.LAT)
        lon0 = float(row.LON)
        dt_end = row.DT_END
        dt_start = row.DT_START
        print(dt_end)

        # 視窗索引
        i, j = safe_window_center(lat1d, lon1d, lat0, lon0)
        i0, i1, j0, j1 = make_window(i, j, WINDOW_RADIUS, H, W)

        # ===== 路徑（forecast / results）=====
        fc_folder_name = f"{dt_start:%Y-%m-%d-%H-%M}to{dt_end:%Y-%m-%d-%H-%M}"
        fc_folder_path = os.path.join(RESULTS_DIR, fc_folder_name)
        fc_upper_path   = os.path.join(fc_folder_path, f"output_upper_{dt_end:%Y-%m-%d-%H-%M}.npy")
        fc_surface_path = os.path.join(fc_folder_path, f"output_surface_{dt_end:%Y-%m-%d-%H-%M}.npy")

        # ===== 路徑（ERA5 / forecasts）=====
        er_folder_path = os.path.join(ERA5_DIR, f"{dt_end:%Y-%m-%d-%H-%M}")
        er_upper_path   = os.path.join(er_folder_path, "input_upper.npy")
        er_surface_path = os.path.join(er_folder_path, "input_surface.npy")

        # 檢查檔案
        missing = [p for p in [fc_upper_path, fc_surface_path, er_upper_path, er_surface_path] if not os.path.exists(p)]
        if missing:
            print("[WARN] 缺檔，略過：")
            for p in missing:
                print("   -", p)
            continue

        # 讀檔
        fc_upper   = np.load(fc_upper_path, allow_pickle=False)
        fc_surface = np.load(fc_surface_path, allow_pickle=False)
        er_upper   = np.load(er_upper_path, allow_pickle=False)
        er_surface = np.load(er_surface_path, allow_pickle=False)



        # surface: MSL / U10 / V10
        fc_sfc_sub = extract_subset_surface(fc_surface,nc_path,channel_map_surface=SURFACE_MAP)
        er_sfc_sub = extract_subset_surface(er_surface,nc_path,channel_map_surface=SURFACE_MAP)
        f_msl, f_u10, f_v10 = fc_sfc_sub['msl'], fc_sfc_sub['u10'], fc_sfc_sub['v10']
        t_msl, t_u10, t_v10 = er_sfc_sub['msl'], er_sfc_sub['u10'], er_sfc_sub['v10']

        # upper: U850 / V850
        fc_prs_sub = extract_subset(fc_upper, nc_path, channel_level_map=UPPER_CL_MAP)
        er_prs_sub = extract_subset(er_upper, nc_path, channel_level_map=UPPER_CL_MAP)
        f_u850, f_v850 = fc_prs_sub['u850'], fc_prs_sub['v850']
        t_u850, t_v850 = er_prs_sub['u850'], er_prs_sub['v850']

        # 指標（視窗內）
        f_msl_min  = local_min(f_msl, i0, i1, j0, j1) / 100.0  # Pa→hPa
        f_s10_max  = local_max(wind_speed(f_u10,  f_v10 ), i0, i1, j0, j1)
        f_s850_max = local_max(wind_speed(f_u850, f_v850), i0, i1, j0, j1)

        t_msl_min  = local_min(t_msl, i0, i1, j0, j1) / 100.0  # Pa→hPa
        t_s10_max  = local_max(wind_speed(t_u10,  t_v10 ), i0, i1, j0, j1)
        t_s850_max = local_max(wind_speed(t_u850, t_v850), i0, i1, j0, j1)

        records.append(dict(
            time=row.DATE_TIME_STR, lat=lat0, lon=lon0,
            forecast_msl_min_hPa=f_msl_min,
            forecast_s10_max=f_s10_max,
            forecast_s850_max=f_s850_max,
            target_msl_min_hPa=t_msl_min,
            target_s10_max=t_s10_max,
            target_s850_max=t_s850_max
        ))
        # break

    out_df = pd.DataFrame(records)
    out_df['CYCLONE_NUM'] = df['CYCLONE_NUM']
    out_df.to_csv("tc_intensity_extracted.csv", index=False, float_format="%.3f")
    print("已輸出 tc_intensity_extracted.csv，筆數:", len(out_df))

if __name__ == "__main__":
    main()
