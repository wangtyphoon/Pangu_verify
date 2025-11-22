#!/usr/bin/env python3
"""
Boundary RMSE vs distance analysis for Pangu data (five channels only).

- Keeps the distance-binning and side/region logic from boundary_loss_analysis.py
- Reads *.npy forecast/era files like weight_rmse_prs.py and extracts only:
  z500, q700, t850, u850, v850 (order is fixed)
- Produces a CSV similar to the original boundary script, but for 5 channels.

References:
- boundary_distance binning & aggregation pattern: boundary_loss_analysis.py  :contentReference[oaicite:2]{index=2}
- Pangu subset extraction & channel mapping idea: weight_rmse_prs.py          :contentReference[oaicite:3]{index=3}
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import os
import numpy as np
import pandas as pd

# === 設定區（請依實際路徑調整） ===
FORECAST_DIR = Path("../results")        # e.g. ../results/2024-05-05-00-00/output_upper_2024-05-05-00-00.npy
ERA5_DIR     = Path("../forecasts")      # e.g. ../forecasts/2024-05-05-00-00/input_upper.npy
NC_PATH      = Path("../forecasts/2020-05-05-00-00/upper.nc")  # for extract_subset alignment
DATETIME_PREFIX = "2024"                 # 只處理此年前綴的資料夾

GRID_HEIGHT = 192
GRID_WIDTH  = 192
BOUNDARY_WIDTH = 8

# 輸出檔名
SUMMARY_CSV = Path("pangu_boundary_loss_summary.csv")

# === 五個通道的固定順序與對應（沿用 weight_rmse_prs 的設計） ===
# channel_level_map 中的 key 與輸出欄位名稱
ORDERED_VARS = ["z500", "q700", "t850", "u850", "v850"]

# 直接沿用你現有專案中的工具（與 weight_rmse_prs.py 相同的導入）
# create_spatial_weights 並不直接用在這支分析（因為我們做每個距離 bin 的 RMSE），
# 但 extract_subset 是必要的。
from loss import create_spatial_weights          # noqa: F401  (kept available if needed)
from subset import extract_subset                # noqa: F401


# ====== 取自原 boundary_loss_analysis.py 的核心輔助函數（小幅整理） ======
def compute_distance_maps(height: int, width: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    y_idx, x_idx = np.indices((height, width))
    dist_top = y_idx
    dist_bottom = height - 1 - y_idx
    dist_left = x_idx
    dist_right = width - 1 - x_idx
    dist_maps: Dict[str, np.ndarray] = {
        "top": dist_top,
        "bottom": dist_bottom,
        "left": dist_left,
        "right": dist_right,
    }
    min_dist = np.minimum.reduce([dist_top, dist_bottom, dist_left, dist_right])
    return dist_maps, min_dist


def accumulate(summary: Dict[Tuple[str, int, str, int, str], Dict[str, float]],
               key: Tuple[str, int, str, int, str],
               pixel_count: int,
               sq_sum: float) -> None:
    rec = summary.get(key)
    if rec is None:
        summary[key] = {"pixel_count": float(pixel_count), "sum_sq_error": float(sq_sum)}
    else:
        rec["pixel_count"] += float(pixel_count)
        rec["sum_sq_error"] += float(sq_sum)


def build_summary(summary: Dict[Tuple[str, int, str, int, str], Dict[str, float]]) -> pd.DataFrame:
    if not summary:
        return pd.DataFrame(columns=[
            "side", "distance", "region",
            "channel_index", "channel_name",
            "pixel_count", "sum_sq_error", "mean_mse", "rmse",
        ])
    rows = []
    for (side, distance, region, ch_idx, ch_name), rec in summary.items():
        pix = rec["pixel_count"]
        sse = rec["sum_sq_error"]
        mean_mse = sse / pix if pix > 0 else np.nan
        rmse = np.sqrt(mean_mse) if mean_mse >= 0 else np.nan
        rows.append({
            "side": side,
            "distance": distance,
            "region": region,
            "channel_index": ch_idx,
            "channel_name": ch_name,
            "pixel_count": float(pix),
            "sum_sq_error": float(sse),
            "mean_mse": mean_mse,
            "rmse": rmse,
        })
    df = pd.DataFrame(rows)
    df.sort_values(["side", "region", "distance", "channel_index"], inplace=True)
    return df
# ====== /取自原 boundary_loss_analysis.py ======


def list_datetime_folders(root: Path, prefix: str) -> List[Path]:
    """列出形如 root/<datetime> 的資料夾；可用 prefix 篩選（例如 '2024'）。"""
    if not root.exists():
        return []
    items = []
    for name in os.listdir(root):
        full = root / name
        if not full.is_dir():
            continue
        if prefix and not name.startswith(prefix):
            continue
        items.append(full)
    items.sort()
    return items


def load_pangu_pair(dt_folder: Path, era_root: Path) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    讀取一個時刻資料夾的 (forecast, era) 並抽取五通道子集，回傳
      pred_5: (H, W, 5)
      era_5 : (H, W, 5)
      dt_str: 對齊的時間字串（從檔名推得）
    這裡沿用 weight_rmse_prs.py 的命名慣例：
      - forecast: <dt_folder>/output_upper_<datetime>.npy
      - target  : <era_root>/<datetime>/input_upper.npy
    """
    dt = dt_folder.name  # e.g., "2024-05-05-00-00"
    dt = dt.split('o')[1]
    print(f"Loading {dt_folder.name}...")
    fc_path = dt_folder / f"output_upper_{dt}.npy"
    er_path = era_root / dt / "input_upper.npy"

    if not fc_path.exists():
        raise FileNotFoundError(f"Missing forecast file: {fc_path}")
    if not er_path.exists():
        raise FileNotFoundError(f"Missing target file:   {er_path}")

    fc_data = np.load(fc_path)
    er_data = np.load(er_path)

    # 透過 extract_subset + upper.nc，抽出五個指定變數，得到 dict[var] -> (H,W)
    fc_sub = extract_subset(fc_data, str(NC_PATH), {
        'z500': (0, 5),
        'q700': (1, 3),
        't850': (2, 2),
        'u850': (3, 2),
        'v850': (4, 2),
    })
    er_sub = extract_subset(er_data, str(NC_PATH), {
        'z500': (0, 5),
        'q700': (1, 3),
        't850': (2, 2),
        'u850': (3, 2),
        'v850': (4, 2),
    })

    # 疊成 (H,W,5)；順序固定為 ORDERED_VARS
    pred_5 = np.stack([fc_sub[k] for k in ORDERED_VARS], axis=-1)
    era_5  = np.stack([er_sub[k] for k in ORDERED_VARS], axis=-1)

    return pred_5, era_5, dt


def analyze_pangu(pred_hw5: np.ndarray,
                  era_hw5: np.ndarray,
                  height: int,
                  width: int,
                  boundary: int,
                  dist_maps: Dict[str, np.ndarray],
                  min_dist: np.ndarray,
                  channel_names: List[str],
                  summary: Dict[Tuple[str, int, str, int, str], Dict[str, float]]) -> None:
    """
    套用原 boundary 分析流程，但輸入固定為單一時刻 (H,W,5)。
    """
    # 檢查網格
    h, w, c = pred_hw5.shape
    if h != height or w != width:
        raise ValueError(f"Grid mismatch: got ({h},{w}) vs ({height},{width})")
    if c != 5:
        raise ValueError(f"Expected 5 channels, got {c}")

    # 轉 (B,H,W,C) 以服用原本計算邏輯（B=1）
    pred = pred_hw5[None, ...].astype(np.float64)
    era  = era_hw5[None, ...].astype(np.float64)

    # err^2 展平
    err = (pred - era) ** 2                            # (1,H,W,5)
    err_flat = err.reshape(-1, c)                      # (H*W,5)

    node = height * width
    repeat = err_flat.shape[0] // node                 # here = 1
    dist_flat_maps = {side: np.tile(d.reshape(-1), repeat)
                      for side, d in dist_maps.items()}
    min_dist_flat = np.tile(min_dist.reshape(-1), repeat)

    region_masks = {
        "outer": (min_dist_flat <= boundary),
        "inner": (min_dist_flat > boundary),
    }

    for region_name, region_mask_flat in region_masks.items():
        if not np.any(region_mask_flat):
            continue

        err_region = err_flat[region_mask_flat]

        for side, dist_full in dist_flat_maps.items():
            dist_region = dist_full[region_mask_flat]
            if dist_region.size == 0:
                continue

            order = np.argsort(dist_region, kind="mergesort")
            dist_sorted = dist_region[order]
            err_sorted = err_region[order]

            unique_dists, first_indices = np.unique(dist_sorted, return_index=True)
            split_indices = np.append(first_indices, dist_sorted.size)

            for idx in range(len(unique_dists)):
                start = split_indices[idx]
                end = split_indices[idx + 1]
                dist_val = int(unique_dists[idx])

                err_block = err_sorted[start:end]
                pixel_count = err_block.shape[0]
                if pixel_count == 0:
                    continue

                # 逐通道累積 SSE
                for ch_idx in range(c):
                    sq_sum = float(err_block[:, ch_idx].sum())
                    ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"ch{ch_idx}"
                    key = (side, dist_val, region_name, ch_idx, ch_name)
                    accumulate(summary, key, pixel_count, sq_sum)


def main() -> None:
    # 1) 路徑檢查
    if not FORECAST_DIR.is_dir():
        raise FileNotFoundError(f"FORECAST_DIR not found: {FORECAST_DIR.resolve()}")
    if not ERA5_DIR.is_dir():
        raise FileNotFoundError(f"ERA5_DIR not found: {ERA5_DIR.resolve()}")
    if not NC_PATH.exists():
        raise FileNotFoundError(f"NC_PATH not found: {NC_PATH.resolve()}")

    # 2) 距離圖準備
    height, width = GRID_HEIGHT, GRID_WIDTH
    if height <= 0 or width <= 0:
        raise ValueError("Height and width must be positive.")
    dist_maps, min_dist = compute_distance_maps(height, width)

    # 3) 取待處理的時刻資料夾
    dt_folders = list_datetime_folders(FORECAST_DIR, DATETIME_PREFIX)
    if not dt_folders:
        print("No datetime folders found under FORECAST_DIR with given prefix.")
        return

    summary: Dict[Tuple[str, int, str, int, str], Dict[str, float]] = {}
    channel_names = ORDERED_VARS[:]  # ["z500","q700","t850","u850","v850"]

    # 4) 逐時刻累積
    for dt_folder in dt_folders:
        try:
            pred_hw5, era_hw5, dt_str = load_pangu_pair(dt_folder, ERA5_DIR)
        except Exception as e:
            print(f"[Skip] {dt_folder.name}: {e}")
            continue

        # 安全檢查大小
        if pred_hw5.shape[0] != height or pred_hw5.shape[1] != width:
            print(f"[Skip] {dt_folder.name}: grid mismatch {pred_hw5.shape[:2]} vs {(height, width)}")
            continue

        analyze_pangu(pred_hw5, era_hw5,
                      height, width, BOUNDARY_WIDTH,
                      dist_maps, min_dist,
                      channel_names, summary)

    # 5) 輸出彙總
    df = build_summary(summary)
    df.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved Pangu boundary analysis to {SUMMARY_CSV.resolve()}")


if __name__ == "__main__":
    main()
