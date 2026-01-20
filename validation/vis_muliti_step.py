from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from subset import extract_subset

# === 路徑設定（請依實際檔案位置調整）===
# results 底下的 run 資料夾，內含 output_upper_*.npy
RUN_DIR = Path(r"../results/2024-07-21-00-00")
# ERA5 真值根目錄
ERA5_ROOT = Path(r"../forecasts")
# 含 latitude/longitude 的 upper.nc，用來裁切 192x192 區域
REFERENCE_NC = ERA5_ROOT / "2020-05-05-00-00" / "upper.nc"

# 需要繪圖的變數（固定 5 個 + 衍生風速）
BASE_VARS = ("z500", "q700", "t850", "u850", "v850")
DERIVED_VARS = ("wind850",)  # 由 u850、v850 計算
VARS = BASE_VARS + DERIVED_VARS
CHANNEL_LEVEL_MAP = {
    "z500": (0, 5),
    "q700": (1, 3),
    "t850": (2, 2),
    "u850": (3, 2),
    "v850": (4, 2),
}

GRID_SHAPE = (192, 192)
SHOW_FIGURE = False
VALUE_CMAP = "jet"
ERROR_CMAP = "bwr"
LOSS_CMAP = "magma"
OUTPUT_DIR = Path("figures")
LEAD_HOURS = [24, 48, 72]  # to plot specific lead hours only
USE_CARTOPY = True
TERRAIN_ALPHA = 0.35
MAP_DATA_ALPHA = 0.9


def parse_dt(dt_str: str) -> datetime:
    return datetime.strptime(dt_str, "%Y-%m-%d-%H-%M")


def compute_wind_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """由東西向/南北向風分量計算風速。"""
    return np.sqrt(np.square(u) + np.square(v))


def to_hw(arr: np.ndarray) -> np.ndarray:
    """確保輸出為 192x192。"""
    if arr.ndim == 2 and arr.shape == GRID_SHAPE:
        return arr
    if arr.ndim == 1 and arr.size == GRID_SHAPE[0] * GRID_SHAPE[1]:
        return arr.reshape(GRID_SHAPE)
    if arr.ndim == 2 and (arr.shape[0] == GRID_SHAPE[0] * GRID_SHAPE[1] or arr.shape[1] == GRID_SHAPE[0] * GRID_SHAPE[1]):
        return arr.reshape(GRID_SHAPE)
    raise ValueError(f"無法 reshape 到 {GRID_SHAPE}，目前 shape={arr.shape}")


def load_lat_lon() -> Tuple[np.ndarray, np.ndarray] | None:
    lat_path = Path("lat.npy")
    lon_path = Path("lon.npy")
    if not (lat_path.exists() and lon_path.exists()):
        return None

    lat = np.load(lat_path)
    lon = np.load(lon_path)

    if lat.ndim == 1 and lon.ndim == 1:
        if lat.size != GRID_SHAPE[0] or lon.size != GRID_SHAPE[1]:
            return None
        lon2d, lat2d = np.meshgrid(lon, lat)
        return lon2d, lat2d

    if lat.ndim == 2 and lon.ndim == 2:
        if lat.shape != GRID_SHAPE or lon.shape != GRID_SHAPE:
            return None
        return lon, lat

    return None


def forecast_to_dict(forecast_data: np.ndarray) -> Dict[str, np.ndarray]:
    """從 (5, 13, 192, 192) 取得各變數 2D 網格。"""
    fc_dict = {}
    for name, (ch, lv) in CHANNEL_LEVEL_MAP.items():
        fc_dict[name] = to_hw(forecast_data[ch, lv])
    return fc_dict


def add_wind_speed(data_dict: Dict[str, np.ndarray]) -> None:
    """在原始 dict 上增加 wind850 = sqrt(u850^2 + v850^2)。"""
    u = to_hw(np.asarray(data_dict["u850"]))
    v = to_hw(np.asarray(data_dict["v850"]))
    data_dict["wind850"] = compute_wind_speed(u, v)


def load_step(forecast_file: Path) -> Tuple[str, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """讀取單一步 (forecast vs ERA5 truth)。"""
    target_dt = forecast_file.stem[len("output_upper_") :]
    truth_path = ERA5_ROOT / target_dt / "input_upper.npy"
    if not truth_path.exists():
        raise FileNotFoundError(f"找不到 ERA5 真值檔案：{truth_path}")

    fc_data = np.load(forecast_file)
    tr_data = np.load(truth_path)

    forecast_dict = forecast_to_dict(fc_data)
    truth_dict = extract_subset(tr_data, str(REFERENCE_NC), CHANNEL_LEVEL_MAP)
    add_wind_speed(forecast_dict)
    add_wind_speed(truth_dict)

    return target_dt, forecast_dict, truth_dict


def get_run_start_dt() -> datetime | None:
    run_name = RUN_DIR.name
    try:
        return parse_dt(run_name)
    except ValueError:
        pass
    if "to" in run_name:
        start_str = run_name.split("to")[0]
        try:
            return parse_dt(start_str)
        except ValueError:
            return None
    return None


def collect_steps() -> List[Tuple[str, Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
    if not RUN_DIR.exists():
        raise FileNotFoundError(f"找不到 RUN_DIR：{RUN_DIR}")

    files = sorted(RUN_DIR.glob("output_upper_*.npy"), key=lambda p: parse_dt(p.stem[len("output_upper_") :]))
    if not files:
        raise FileNotFoundError(f"{RUN_DIR} 裡沒有 output_upper_*.npy")

    lead_hours = set(LEAD_HOURS) if LEAD_HOURS else None
    start_dt = get_run_start_dt() if lead_hours else None
    if lead_hours and start_dt is None:
        raise ValueError("需要可解析的 RUN_DIR 起始時間才能篩選 lead hours。")

    steps = []
    for f in files:
        if lead_hours:
            target_dt = parse_dt(f.stem[len("output_upper_") :])
            lead_hour = int((target_dt - start_dt).total_seconds() // 3600)
            if lead_hour not in lead_hours:
                continue
        steps.append(load_step(f))
    return steps


def prepare_step_data(steps):
    prepared = []
    stats = {var: {"vmin": None, "vmax": None, "loss_max": None, "bias_max": None} for var in VARS}

    for target_dt, forecast, truth in steps:
        step_entry = {"target": target_dt, "vars": {}}
        for var in VARS:
            fc = forecast[var]
            tr = to_hw(np.asarray(truth[var]))
            err = fc - tr
            loss = np.square(err)
            mse = float(loss.mean())
            bias_value = float(err.mean())

            step_entry["vars"][var] = {
                "forecast": fc,
                "truth": tr,
                "error": err,
                "loss": loss,
                "mse": mse,
                "bias": bias_value,
            }

            # 色階僅依據 ERA5 真值
            vmin = float(tr.min())
            vmax = float(tr.max())
            loss_max = float(loss.max())
            err_abs = float(np.abs(err).max())

            var_stats = stats[var]
            var_stats["vmin"] = vmin if var_stats["vmin"] is None else min(var_stats["vmin"], vmin)
            var_stats["vmax"] = vmax if var_stats["vmax"] is None else max(var_stats["vmax"], vmax)
            var_stats["loss_max"] = loss_max if var_stats["loss_max"] is None else max(var_stats["loss_max"], loss_max)
            bias_abs = float(np.abs(err).max())  # spatial bias map shares range with error
            var_stats["bias_max"] = bias_abs if var_stats["bias_max"] is None else max(var_stats["bias_max"], bias_abs)

        prepared.append(step_entry)

    return prepared, stats


def visualize(prepared, stats):
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("需要 matplotlib 才能繪圖，請先安裝（pip install matplotlib）。") from exc

    if not SHOW_FIGURE:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    use_cartopy = False
    lon2d = lat2d = None
    if USE_CARTOPY:
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
        except ModuleNotFoundError:
            use_cartopy = False
        else:
            latlon = load_lat_lon()
            if latlon is not None:
                lon2d, lat2d = latlon
                use_cartopy = True

    n_steps = len(prepared)
    if n_steps == 0:
        print("沒有可視化的步驟資料。")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_name = RUN_DIR.name
    start_dt = run_name.split("to")[0] if "to" in run_name else run_name
    final_target = prepared[-1]["target"]
    map_extent = None
    if use_cartopy:
        map_extent = [
            float(np.nanmin(lon2d)),
            float(np.nanmax(lon2d)),
            float(np.nanmin(lat2d)),
            float(np.nanmax(lat2d)),
        ]

    for var in VARS:
        subplot_kw = {"projection": ccrs.PlateCarree()} if use_cartopy else None
        fig, axes = plt.subplots(4, n_steps, figsize=(4 * n_steps, 13), subplot_kw=subplot_kw)
        axes = np.asarray(axes).reshape(4, n_steps)

        vmin = stats[var]["vmin"]
        vmax = stats[var]["vmax"]
        loss_max = stats[var]["loss_max"] or 1e-9
        bias_max = stats[var]["bias_max"] or 1e-9
        colorbars = [None, None, None, None]

        for col, step in enumerate(prepared):
            data = step["vars"][var]
            fc = data["forecast"]
            tr = data["truth"]
            err = data["error"]
            loss = data["loss"]
            bias_val = data["bias"]

            if use_cartopy:
                for row in range(4):
                    ax = axes[row, col]
                    ax.add_feature(cfeature.LAND, facecolor="0.92", edgecolor="none", zorder=0)
                    ax.add_feature(cfeature.OCEAN, facecolor="white", edgecolor="none", zorder=0)
                    ax.add_feature(cfeature.COASTLINE, lw=0.4, zorder=2)
                    if map_extent:
                        ax.set_extent(map_extent, crs=ccrs.PlateCarree())

                im_fc = axes[0, col].pcolormesh(
                    lon2d,
                    lat2d,
                    fc,
                    cmap=VALUE_CMAP,
                    vmin=vmin,
                    vmax=vmax,
                    transform=ccrs.PlateCarree(),
                    shading="auto",
                    alpha=MAP_DATA_ALPHA,
                    zorder=1,
                )
                im_tr = axes[1, col].pcolormesh(
                    lon2d,
                    lat2d,
                    tr,
                    cmap=VALUE_CMAP,
                    vmin=vmin,
                    vmax=vmax,
                    transform=ccrs.PlateCarree(),
                    shading="auto",
                    alpha=MAP_DATA_ALPHA,
                    zorder=1,
                )
                im_bias = axes[2, col].pcolormesh(
                    lon2d,
                    lat2d,
                    err,
                    cmap=ERROR_CMAP,
                    vmin=-bias_max,
                    vmax=bias_max,
                    transform=ccrs.PlateCarree(),
                    shading="auto",
                    alpha=MAP_DATA_ALPHA,
                    zorder=1,
                )
                im_loss = axes[3, col].pcolormesh(
                    lon2d,
                    lat2d,
                    loss,
                    cmap=LOSS_CMAP,
                    vmin=0,
                    vmax=loss_max,
                    transform=ccrs.PlateCarree(),
                    shading="auto",
                    alpha=MAP_DATA_ALPHA,
                    zorder=1,
                )
            else:
                im_fc = axes[0, col].imshow(fc, cmap=VALUE_CMAP, vmin=vmin, vmax=vmax)
                im_tr = axes[1, col].imshow(tr, cmap=VALUE_CMAP, vmin=vmin, vmax=vmax)
                # Bias panel shows spatial bias (forecast - truth) with title indicating mean bias.
                im_bias = axes[2, col].imshow(err, cmap=ERROR_CMAP, vmin=-bias_max, vmax=bias_max)
                im_loss = axes[3, col].imshow(loss, cmap=LOSS_CMAP, vmin=0, vmax=loss_max)

            colorbars[0] = im_fc
            colorbars[1] = im_tr
            colorbars[2] = im_bias
            colorbars[3] = im_loss

            axes[0, col].set_title(f"{step['target']} Forecast")
            axes[1, col].set_title(f"{step['target']} Truth")
            axes[2, col].set_title(f"{step['target']} Bias (mean={bias_val:.4f})")
            axes[3, col].set_title(f"{step['target']} (RMSE={np.sqrt(data['mse']):.5f})")

            for row in range(4):
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])

        fig.suptitle(f"{run_name} — {var} multi-step 預報", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.965])

        labels = ["Value scale", "Value scale", "Bias scale", "MSE scale"]
        for row, im in enumerate(colorbars):
            if im is None:
                continue
            cbar = fig.colorbar(im, ax=axes[row, :].ravel().tolist(), shrink=0.8, pad=0.02)
            cbar.ax.set_ylabel(labels[row], rotation=270, labelpad=12)

        out_path = OUTPUT_DIR / f"{start_dt}_to_{final_target}_{var}_multistep.png"
        fig.savefig(out_path, dpi=150)
        print(f"已輸出 {var} 圖片：{out_path}")

        if SHOW_FIGURE:
            plt.show()
        plt.close(fig)


def main():
    steps = collect_steps()
    prepared, stats = prepare_step_data(steps)
    visualize(prepared, stats)


if __name__ == "__main__":
    main()
