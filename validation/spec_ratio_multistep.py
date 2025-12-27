"""
多 lead 預報的譜能量比值 (Pangu / ERA5)。
參考 validation/spec_ratio.py（單一 overall）與 plot_spectrum_multistep.py（多 lead）。

輸入：validation/all_energy_multistep.npz（由 spectrum_multistep.py 產生）
輸出：顯示兩張圖
  1) Ratio vs wavenumber k（每個 subplot 對應一個變數，線條為不同 lead/overall）
  2) Ratio vs wavelength λ（同上）
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


VARIABLES = ["z500", "q700", "t850", "u850", "v850"]
#VARIABLES = ["msl", "u10", "v10", "t2m"]

DATA_FILE = Path(__file__).with_name("all_energy_multistep72.npz")
#SFC_DATA_FILE = Path(__file__).with_name("all_energy_multistep_sfc72.npz")
LAT_PATH = Path(__file__).resolve().parent.parent / "lat.npy"
LON_PATH = Path(__file__).resolve().parent.parent / "lon.npy"
R = 6371.0  # km
EPS = 1e-12
# 若要裁掉高波數噪聲，設定波數上限；None 表示不裁切
K_CUTOFF = 56
LAT_RANGE = (2.25, 45.50)
LON_RANGE = (104.5, 147.75)


def select_bbox_indices(lat2d: np.ndarray, lon2d: np.ndarray, lat_range, lon_range):
    lat_min, lat_max = min(lat_range), max(lat_range)
    lon_min, lon_max = min(lon_range), max(lon_range)
    mask = (lat2d >= lat_min) & (lat2d <= lat_max) & (lon2d >= lon_min) & (lon2d <= lon_max)
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("LAT/LON range selection is empty.")
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return slice(y0, y1), slice(x0, x1)


def cell_area_equal_area(lat2d: np.ndarray, lon2d: np.ndarray) -> np.ndarray:
    """Compute equal-area cell size (km^2) on a lat/lon grid."""
    if lat2d.ndim == 1 and lon2d.ndim == 1:
        lat2d, lon2d = np.meshgrid(lat2d, lon2d, indexing="ij")
    lat1d = lat2d[:, 0]
    lon1d = lon2d[0, :]
    dlat = np.deg2rad(np.abs(np.diff(lat1d).mean()))
    dlon = np.deg2rad(np.abs(np.diff(lon1d).mean()))
    phi_c = np.deg2rad(lat1d)
    phi_n = phi_c + 0.5 * dlat
    phi_s = phi_c - 0.5 * dlat
    strip = (R**2) * (np.sin(phi_n) - np.sin(phi_s)) * dlon
    return np.repeat(strip[:, None], lon1d.size, axis=1)


def effective_grid_km_equal_area(lat2d: np.ndarray, lon2d: np.ndarray) -> float:
    area = cell_area_equal_area(lat2d, lon2d)
    return float(np.sqrt(area.mean()))


def sort_suffix_key(suffix: str) -> tuple[int, str]:
    if suffix == "overall":
        return (-1, suffix)
    match = re.search(r"\d+", suffix)
    if match:
        return (int(match.group()), suffix)
    return (10_000, suffix)


def discover_suffixes(npz_obj: np.lib.npyio.NpzFile) -> list[str]:
    """Find common suffixes (overall/leadXX) across all variables."""
    per_var = {var: set() for var in VARIABLES}
    pattern = re.compile(r"pangu_mean_(?P<var>[a-z0-9]+)_(?P<sfx>.+)")
    for key in npz_obj.files:
        m = pattern.match(key)
        if m and m.group("var") in per_var:
            per_var[m.group("var")].add(m.group("sfx"))

    common = set.intersection(*per_var.values()) if all(per_var.values()) else set()
    if not common:
        raise RuntimeError("No common suffix (overall/leadXX) found across variables.")

    ordered = ["overall"] if "overall" in common else []
    ordered += sorted((s for s in common if s != "overall"), key=sort_suffix_key)
    return ordered


def compute_ratio(mean_er: np.ndarray, std_er: np.ndarray, mean_fc: np.ndarray, std_fc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pangu/ERA5 power ratio and propagated std (energy input)."""
    P_e = np.clip(mean_er, EPS, None)
    P_f = np.clip(mean_fc, EPS, None)
    sP_e = np.clip(std_er, 0, None)
    sP_f = np.clip(std_fc, 0, None)

    ratio = np.clip(P_f / (P_e + EPS), EPS, None)
    rel_e = np.clip(sP_e / (P_e + EPS), 0, None)
    rel_f = np.clip(sP_f / (P_f + EPS), 0, None)
    s_ratio = ratio * np.sqrt(rel_e**2 + rel_f**2)
    return ratio, s_ratio


def get_era5_stats(npz_obj, var: str, suffix: str, mask) -> tuple[np.ndarray, np.ndarray]:
    era_suffix = suffix
    if suffix != "overall":
        mean_key = f"era5_mean_{var}_{suffix}"
        std_key = f"era5_std_{var}_{suffix}"
        if mean_key not in npz_obj.files or std_key not in npz_obj.files:
            era_suffix = "overall"
    mean_er = npz_obj[f"era5_mean_{var}_{era_suffix}"][mask]
    std_er = npz_obj[f"era5_std_{var}_{era_suffix}"][mask]
    return mean_er, std_er


def format_suffix(sfx: str) -> str:
    if sfx == "overall":
        return "Overall"
    m = re.search(r"\d+", sfx)
    return f"Lead +{int(m.group()):02d}h" if m else sfx


def plot_ratio_wavenumber(npz_obj, suffixes: list[str], k: np.ndarray) -> None:
    plt.figure(figsize=(20, 4))
    lead_suffixes = suffixes  # include overall first (already ordered)
    mask_k = (k <= K_CUTOFF) if K_CUTOFF is not None else slice(None)
    for idx, var in enumerate(VARIABLES, 1):
        ax = plt.subplot(1, len(VARIABLES), idx)
        for sfx in lead_suffixes:
            mean_er, std_er = get_era5_stats(npz_obj, var, sfx, mask_k)
            mean_fc = npz_obj[f"pangu_mean_{var}_{sfx}"][mask_k]
            std_fc = npz_obj[f"pangu_std_{var}_{sfx}"][mask_k]
            ratio, s_ratio = compute_ratio(mean_er, std_er, mean_fc, std_fc)

            label = format_suffix(sfx)
            ax.plot(k[mask_k], ratio, label=label)
            # ax.fill_between(k, ratio - s_ratio, ratio + s_ratio, alpha=0.15)

        ax.axhline(1.0, linestyle=":", linewidth=1, color="gray")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-1, 1e1)
        ax.set_xlabel("Wavenumber k")
        ax.set_ylabel("Power Ratio (Pangu/ERA5)")
        ax.set_title(var)
        ax.grid(True, which="both", linestyle=":")
        ax.legend(fontsize="x-small", loc="best")

    plt.tight_layout()


def plot_ratio_wavelength(npz_obj, suffixes: list[str], k: np.ndarray, L_eff: float) -> None:
    plt.figure(figsize=(20, 4))
    lead_suffixes = suffixes

    mask_k = (k > 0)
    if K_CUTOFF is not None:
        mask_k &= (k <= K_CUTOFF)
    lam_base = L_eff / k[mask_k]

    for idx, var in enumerate(VARIABLES, 1):
        ax = plt.subplot(1, len(VARIABLES), idx)
        for sfx in lead_suffixes:
            mean_er, std_er = get_era5_stats(npz_obj, var, sfx, mask_k)
            mean_fc = npz_obj[f"pangu_mean_{var}_{sfx}"][mask_k]
            std_fc = npz_obj[f"pangu_std_{var}_{sfx}"][mask_k]
            ratio, s_ratio = compute_ratio(mean_er, std_er, mean_fc, std_fc)

            label = format_suffix(sfx)
            ax.plot(lam_base, ratio, label=label)
            # ax.fill_between(lam_base, ratio - s_ratio, ratio + s_ratio, alpha=0.15)

        ax.axhline(1.0, linestyle=":", linewidth=1, color="gray")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-1, 1e1)
        ax.set_xlim(lam_base.min(), lam_base.max())
        ax.invert_xaxis()
        ax.set_xlabel("Wavelength λ (km)")
        ax.set_ylabel("Power Ratio (Pangu/ERA5)")
        ax.set_title(var)
        ax.grid(True, which="both", linestyle=":")
        ax.legend(fontsize="x-small", loc="best")

    plt.tight_layout()


def main() -> None:
    npz_obj = np.load(DATA_FILE)
    k = npz_obj["k"]

    # ---- L_eff 計算邏輯對齊 spec_ratio.py ----
    try:
        lat = np.load(LAT_PATH)
        lon = np.load(LON_PATH)

        # 和 spec_ratio.py 一樣：若是 1D lat/lon 則轉成 2D 網格
        if lat.ndim == 1 and lon.ndim == 1:
            lat, lon = np.meshgrid(lat, lon, indexing="ij")

        ys, xs = select_bbox_indices(lat, lon, LAT_RANGE, LON_RANGE)
        lat_sub = lat[ys, xs]
        lon_sub = lon[ys, xs]

        s_eff = effective_grid_km_equal_area(lat_sub, lon_sub)
        H_sub, W_sub = lat_sub.shape
        L_eff = s_eff * min(H_sub, W_sub)
    except (FileNotFoundError, ValueError):
        # fallback 也對齊 spec_ratio.py
        L_eff = 192 * 28.0
        print(f"[Info] LAT/LON not found or invalid, using fallback L_eff={L_eff:.1f} km")

    # ---- lead suffix 的選擇 ----
    # 原本有 discover_suffixes(npz_obj)，但你現在手動鎖成 lead24
    # 如果想要自動偵測，再加回 discover_suffixes(npz_obj) 即可
    # suffixes = discover_suffixes(npz_obj)[
    #lead 6-72
    suffixes = ['lead06','lead12','lead18','lead24','lead30','lead36','lead42','lead48','lead54','lead60','lead66','lead72']
    suffixes = ['lead24','lead48','lead72']  # 保留你目前的設定

    # ---- 繪圖：k-domain & λ-domain ----
    plot_ratio_wavenumber(npz_obj, suffixes, k)
    plot_ratio_wavelength(npz_obj, suffixes, k, L_eff)

    plt.show()


if __name__ == "__main__":
    main()
