"""
多 step 預報的徑向頻譜分析。
流程參考目前的 spectrum.py，並加入依 lead_hours 分組的統計，
實作與 spectrum_all_stat 的分步統計概念相似。

輸入：
    - ../results/<run>/output_upper_YYYY-MM-DD-HH-MM.npy
    - ../forecasts/YYY.../input_upper.npy
    - 需要 ../upper.nc、../lat.npy、../lon.npy

輸出（皆在 validation/ 下）：
    - overall: {var}_spectral_stats_overall.csv
    - 每個 lead: {var}_spectral_stats_leadXX.csv（XX=小時）
    - all_energy_multistep.npz：包含 k 及每個 var 的 mean/std（overall 與各 lead）
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from subset import extract_subset

# ---------------- 基本設定（與現有 spectrum.py 保持一致，可依需求調整） ----------------
z500_channel = 0
q700_channel = 1
t850_channel = 2
u850_channel = 3
v850_channel = 4
channel_level_map = {
    "z500": (z500_channel, 5),
    "q700": (q700_channel, 3),
    "t850": (t850_channel, 2),
    "u850": (u850_channel, 2),
    "v850": (v850_channel, 2),
}

LAT_RANGE = (2.25, 45.50)
LON_RANGE = (104.5, 147.75)
APPLY_LAT_WEIGHT = False  # 是否套用 cos(lat) 面積權重
AMPLITUDE_SPECTRUM = False  # True: 幅度譜；False: 功率譜
DETREND_MEAN = False  # 是否去除（加權）平均
DENSITY_NORM = False  # 是否做頻譜密度正規化

# ---------------- 路徑 ----------------
ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
RESULTS_ROOT = BASE / "results"
ERA5_ROOT = BASE / "forecasts"
NC_PATH = "..//upper.nc"

# ---------------- 面積權重工具 ----------------
def select_bbox_indices(lat2d: np.ndarray, lon2d: np.ndarray, lat_range, lon_range):
    lat_min, lat_max = min(lat_range), max(lat_range)
    lon_min, lon_max = min(lon_range), max(lon_range)
    mask = (lat2d >= lat_min) & (lat2d <= lat_max) & (lon2d >= lon_min) & (lon2d <= lon_max)
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return slice(y0, y1), slice(x0, x1)


def radial_power_spectrum(
    field2d: np.ndarray,
    area2d: np.ndarray | None = None,
    detrend_mean: bool = True,
    amplitude: bool = True,
    density_norm: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    面積加權幅度/功率徑向譜。
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
        mean_w = np.nansum(field2d * A) / np.nansum(A) if detrend_mean else 0.0
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
    r = np.sqrt((xg - cx) ** 2 + (y - cy) ** 2).astype(int)
    kmax = min(cx, cy) + 1

    spec_sum = np.bincount(r.ravel(), weights=P.ravel(), minlength=kmax)
    spec_cnt = np.bincount(r.ravel(), minlength=kmax)
    spec = spec_sum[:kmax] / np.maximum(spec_cnt[:kmax], 1)

    k = np.arange(kmax)
    return k, spec

def forecast_to_dict(forecast_data):
    """
    直接從已裁切的推理輸出取得各變數 2D 權重網格。
    forecast_data 形狀預期為 (5, 13, 192, 192)
    """

    #從192*192裁減到 174*174
    fc_dict = {}
    for name, (ch, lv) in channel_level_map.items():
        fc_dict[name] = forecast_data[ch, lv, 9:183, 9:183]
    return fc_dict


# --------- 一次性載入經緯度，對應子區面積權重 ---------
LAT_ALL = np.load(BASE / "lat.npy")
LON_ALL = np.load(BASE / "lon.npy")
if LAT_ALL.ndim == 1 and LON_ALL.ndim == 1:
    LAT_ALL, LON_ALL = np.meshgrid(LAT_ALL, LON_ALL, indexing="ij")
ys, xs = select_bbox_indices(LAT_ALL, LON_ALL, LAT_RANGE, LON_RANGE)
LAT_SUB = LAT_ALL[ys, xs].copy()
AREA_SUB = np.cos(np.deg2rad(LAT_SUB)) if APPLY_LAT_WEIGHT else None

# ---------------- helper ----------------
def parse_stamp(stamp: str) -> datetime:
    return datetime.strptime(stamp, "%Y-%m-%d-%H-%M")


def compute_stats(spec_list: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """將 [(k, spec), ...] 匯總成 (k, mean, std)。"""
    if not spec_list:
        raise ValueError("spec_list 為空，無法計算統計量")
    k = spec_list[0][0]
    arr = np.stack([sp[1] for sp in spec_list], axis=0)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    return k, mean, std


def add_record(store: Dict, lead: int | None, source: str, var: str, k: np.ndarray, spec: np.ndarray) -> None:
    """
    將頻譜結果存入指定分組。
    store 的層級：lead -> source -> var -> list[(k, spec)]
    """
    grp = store.setdefault(lead, {"forecast": {v: [] for v in channel_level_map}, "era5": {v: [] for v in channel_level_map}})
    grp[source][var].append((k, spec))


def main() -> None:
    all_specs: Dict[int | None, Dict] = {}  # None 表示 overall（所有 lead 一起）

    # 逐 run 掃描
    for run_folder in sorted(RESULTS_ROOT.iterdir()):
        if not run_folder.is_dir():
            continue
        run_name = run_folder.name
        if "to" not in run_name:
            continue
        start_str, _ = run_name.split("to", 1)
        try:
            run_start = parse_stamp(start_str)
        except ValueError:
            print(f"[WARN] 無法解析起報時間，跳過 {run_name}")
            continue

        print(f"處理 {run_name} ...")
        for fn in sorted(run_folder.iterdir()):
            name = fn.name
            if not (name.startswith("output_upper_") and name.endswith(".npy")):
                continue
            dt_str = name[len("output_upper_") : -4]
            era_path = ERA5_ROOT / dt_str / "input_upper.npy"
            if not era_path.exists():
                print(f"[WARN] 缺少 ERA5 {era_path}, 跳過 {dt_str}")
                continue

            fc_data = np.load(fn)
            er_data = np.load(era_path)

            fc_sub = forecast_to_dict(fc_data)
            er_sub = extract_subset(
                er_data,
                str(NC_PATH),
                channel_level_map,
                lat_range=LAT_RANGE,
                lon_range=LON_RANGE,
            )

            lead_hours = int((parse_stamp(dt_str) - run_start).total_seconds() // 3600)

            for var in channel_level_map:
                fc_field = fc_sub[var]
                er_field = er_sub[var]

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

                # overall（None）：所有 lead 一起
                add_record(all_specs, None, "forecast", var, k_fc, spec_fc)
                add_record(all_specs, None, "era5", var, k_er, spec_er)
                # 個別 lead
                add_record(all_specs, lead_hours, "forecast", var, k_fc, spec_fc)
                add_record(all_specs, lead_hours, "era5", var, k_er, spec_er)

    if not all_specs:
        print("未找到任何頻譜資料。")
        return

    npz_payload: Dict[str, np.ndarray] = {}
    k_ref: np.ndarray | None = None

    def dump_group(lead_key: int | None, specs: Dict) -> None:
        nonlocal k_ref
        suffix = "overall" if lead_key is None else f"lead{int(lead_key):02d}"
        for var in channel_level_map:
            fc_list = specs["forecast"][var]
            er_list = specs["era5"][var]
            if not fc_list or not er_list:
                continue

            k, mean_fc, std_fc = compute_stats(fc_list)
            _, mean_er, std_er = compute_stats(er_list)

            # 以各向量的最小長度對齊，避免 ERA5 與預報的 k 長度不一致
            target_len = min(len(k), len(mean_fc), len(std_fc), len(mean_er), len(std_er))
            if k_ref is None:
                k_ref = k[:target_len]
                npz_payload["k"] = k_ref
            else:
                target_len = min(target_len, len(k_ref))

            k = k_ref[:target_len]
            mean_fc = mean_fc[:target_len]
            std_fc = std_fc[:target_len]
            mean_er = mean_er[:target_len]
            std_er = std_er[:target_len]

            df = pd.DataFrame(
                {
                    "k": k,
                    "era5_mean": mean_er,
                    "era5_std": std_er,
                    "pangu_mean": mean_fc,
                    "pangu_std": std_fc,
                }
            )
            out_csv = ROOT / f"{var}_spectral_stats_{suffix}.csv"
            # df.to_csv(out_csv, index=False)
            print(f"已輸出 {out_csv.name}")

            npz_payload[f"era5_mean_{var}_{suffix}"] = mean_er
            npz_payload[f"era5_std_{var}_{suffix}"] = std_er
            npz_payload[f"pangu_mean_{var}_{suffix}"] = mean_fc
            npz_payload[f"pangu_std_{var}_{suffix}"] = std_fc

    # 先輸出 overall（所有 lead 混在一起）
    overall_specs = all_specs.get(None)
    if overall_specs is not None:
        dump_group(None, overall_specs)
    else:
        print("警告：找不到 overall (None) 分組，略過 overall。") 

    # 再輸出每一個實際出現的 lead
    lead_keys = sorted(k for k in all_specs.keys() if k is not None)
    for lead in lead_keys:
        dump_group(lead, all_specs[lead])

    npz_path = ROOT / "all_energy_multistep72.npz"
    np.savez(npz_path, **npz_payload)
    print(f"已輸出 {npz_path.name}（包含 overall 及所有 lead 的 mean/std）")


if __name__ == "__main__":
    main()
