import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

VARIABLES = ["z500", "q700", "t850", "u850", "v850"]
DATA_FILE = Path(__file__).with_name("all_energy_multistep.npz")
LAT_PATH = Path(__file__).resolve().parent.parent / "lat.npy"
LON_PATH = Path(__file__).resolve().parent.parent / "lon.npy"
R = 6371.0  # km


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
    """Find common suffixes (overall/lead) across all variables."""
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


def format_suffix(suffix: str) -> str:
    if suffix == "overall":
        return "Overall"
    m = re.search(r"\d+", suffix)
    if m:
        return f"Lead +{int(m.group()):02d}h"
    return suffix


def plot_wavenumber(npz_obj, suffixes: list[str], k: np.ndarray) -> None:
    """
    單一張圖：ERA5 + 多個 lead 的 Pangu 預報（wavenumber）。
    不畫 std 陰影，只用 std 當作閾值畫散點。
    """
    lead_suffixes = [s for s in suffixes if s != "overall"]

    plt.figure(figsize=(20, 4))
    for idx, var in enumerate(VARIABLES, 1):
        mean_er = npz_obj[f"era5_mean_{var}_overall"]
        std_er = npz_obj[f"era5_std_{var}_overall"]

        ax = plt.subplot(1, len(VARIABLES), idx)
        ax.plot(k, mean_er, linestyle="-", label="ERA5 mean")

        # 用來彙總所有 lead 的 k1/k2/k3
        k1 = k2 = k3 = None

        # 只讓散點圖的 legend 各出現一次
        added_1 = added_2 = added_3 = False

        for sfx in lead_suffixes:
            mean_fc = npz_obj[f"pangu_mean_{var}_{sfx}"]
            std_fc = npz_obj[f"pangu_std_{var}_{sfx}"]

            diff = mean_fc - mean_er
            mask1 = np.abs(diff) > 1 * std_er
            mask2 = np.abs(diff) > 2 * std_er
            mask3 = np.abs(diff) > 3 * std_er

            ax.plot(k, mean_fc, linestyle="--", label=f"Pangu mean ({format_suffix(sfx)})")

            # 標記超過 std 的位置（不畫帶狀，只畫點）
            if mask1.any():
                if k1 is None:
                    k1 = k[mask1].min()
                ax.scatter(
                    k[mask1],
                    mean_fc[mask1],
                    s=10,
                    c="orange",
                    label=">1 std" if not added_1 else None,
                )
                added_1 = True

            if mask2.any():
                if k2 is None:
                    k2 = k[mask2].min()
                ax.scatter(
                    k[mask2],
                    mean_fc[mask2],
                    s=10,
                    c="green",
                    label=">2 std" if not added_2 else None,
                )
                added_2 = True

            if mask3.any():
                if k3 is None:
                    k3 = k[mask3].min()
                ax.scatter(
                    k[mask3],
                    mean_fc[mask3],
                    s=20,
                    c="red",
                    marker="*",
                    label=">3 std" if not added_3 else None,
                )
                added_3 = True

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("k")
        ax.set_ylabel("Amplitude")
        ax.grid(True, which="both", linestyle=":")

        k_texts = []
        if k1 is not None:
            k_texts.append(f"k1={k1:.1f}")
        if k2 is not None:
            k_texts.append(f"k2={k2:.1f}")
        if k3 is not None:
            k_texts.append(f"k3={k3:.1f}")
        k_text = " | ".join(k_texts) if k_texts else "no exceed"

        ax.set_title(f"Wavenumber: {var}\n({k_text})")
        ax.legend(fontsize="x-small", loc="best")

    plt.tight_layout()


def plot_wavelength(npz_obj, suffixes: list[str], k: np.ndarray, L_eff: float) -> None:
    """
    單一張圖：ERA5 + 多個 lead 的 Pangu 預報（wavelength）。
    不畫 std 陰影，只用 std 當作閾值畫散點。
    """
    lead_suffixes = [s for s in suffixes if s != "overall"]

    plt.figure(figsize=(20, 4))
    for idx, var in enumerate(VARIABLES, 1):
        mean_er = npz_obj[f"era5_mean_{var}_overall"]
        std_er = npz_obj[f"era5_std_{var}_overall"]

        ax = plt.subplot(1, len(VARIABLES), idx)

        mask_k = k > 0
        lam_base = L_eff / k[mask_k]

        ax.plot(lam_base, mean_er[mask_k], linestyle="-", label="ERA5 mean")

        # 彙總所有 lead 的 lam1/lam2/lam3（取對應閾值下最長波長）
        lam1 = lam2 = lam3 = None

        added_1 = added_2 = added_3 = False

        for sfx in lead_suffixes:
            mean_fc = npz_obj[f"pangu_mean_{var}_{sfx}"]
            std_fc = npz_obj[f"pangu_std_{var}_{sfx}"]

            diff = mean_fc[mask_k] - mean_er[mask_k]
            mask1 = np.abs(diff) > 1 * std_er[mask_k]
            mask2 = np.abs(diff) > 2 * std_er[mask_k]
            mask3 = np.abs(diff) > 3 * std_er[mask_k]

            ax.plot(
                lam_base,
                mean_fc[mask_k],
                linestyle="--",
                label=f"Pangu mean ({format_suffix(sfx)})",
            )

            if mask1.any():
                # lam 越大代表越長波長，所以取 mask1 中最大的 lam
                lam1_cur = lam_base[mask1].max()
                lam1 = lam1_cur if lam1 is None else max(lam1, lam1_cur)
                ax.scatter(
                    lam_base[mask1],
                    mean_fc[mask_k][mask1],
                    s=10,
                    c="orange",
                    label=">1 std" if not added_1 else None,
                )
                added_1 = True

            if mask2.any():
                lam2_cur = lam_base[mask2].max()
                lam2 = lam2_cur if lam2 is None else max(lam2, lam2_cur)
                ax.scatter(
                    lam_base[mask2],
                    mean_fc[mask_k][mask2],
                    s=10,
                    c="green",
                    label=">2 std" if not added_2 else None,
                )
                added_2 = True

            if mask3.any():
                lam3_cur = lam_base[mask3].max()
                lam3 = lam3_cur if lam3 is None else max(lam3, lam3_cur)
                ax.scatter(
                    lam_base[mask3],
                    mean_fc[mask_k][mask3],
                    s=20,
                    c="red",
                    marker="*",
                    label=">3 std" if not added_3 else None,
                )
                added_3 = True

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("lambda (km)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, which="both", linestyle=":")

        lam_texts = []
        if lam1 is not None:
            lam_texts.append(f"lam1={int(round(lam1))}")
        if lam2 is not None:
            lam_texts.append(f"lam2={int(round(lam2))}")
        if lam3 is not None:
            lam_texts.append(f"lam3={int(round(lam3))}")
        lam_text = " | ".join(lam_texts) if lam_texts else "no exceed"

        ax.set_title(f"Wavelength: {var}\n({lam_text})")
        ax.legend(fontsize="x-small", loc="best")

    plt.tight_layout()


def main() -> None:
    npz_obj = np.load(DATA_FILE)
    k = npz_obj["k"]

    lat = np.load(LAT_PATH)
    lon = np.load(LON_PATH)
    if lat.ndim == 1 and lon.ndim == 1:
        lat, lon = np.meshgrid(lat, lon, indexing="ij")
    s_eff = effective_grid_km_equal_area(lat, lon)
    H_sub, W_sub = lat.shape
    L_eff = s_eff * min(H_sub, W_sub)

    suffixes = discover_suffixes(npz_obj)

    # 改成「一次畫完 ERA5 + 所有 lead」
    plot_wavenumber(npz_obj, suffixes, k)
    plot_wavelength(npz_obj, suffixes, k, L_eff)

    plt.show()


if __name__ == "__main__":
    main()
