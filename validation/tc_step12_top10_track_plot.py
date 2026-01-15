from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ------------------ Config ------------------
BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "tc_bst_z700_metrics35.csv"
TARGET_STEP = 12
TOP_N = 10
OUTPUT_PNG = BASE_DIR / "tc_step12_top10_track_plot.png"


def align_lon_to_ref(lon_value: float, ref_lon: float) -> float:
    v = float(lon_value)
    if ref_lon < 0 and v > 180:
        v = v - 360.0
    if ref_lon >= 0 and v < 0:
        v = v + 360.0
    return v


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlambda = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return float(2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))


def track_error_km(df: pd.DataFrame) -> np.ndarray:
    pred_lon = df["forecast_lon"].astype(float).to_numpy()
    bst_lon = df["bst_lon"].astype(float).to_numpy()
    pred_lon = np.array([align_lon_to_ref(p, r) for p, r in zip(pred_lon, bst_lon)])
    return np.array(
        [
            haversine_km(lat1, lon1, lat2, lon2)
            for lat1, lon1, lat2, lon2 in zip(
                df["bst_lat"].astype(float).to_numpy(),
                bst_lon,
                df["forecast_lat"].astype(float).to_numpy(),
                pred_lon,
            )
        ]
    )


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    df = df[(df['forecast_lat'] > 5) & (df['forecast_lat'] < 42.25)]
    df = df[(df['forecast_lon'] > 107.5) & (df['forecast_lon'] < 142.5)]
    print(len(df), "total rows in", INPUT_CSV)
    # df = df[df['CYCLONE_NUM']==5]
    # df = df[df['target_s10_max'] > 12]
    # print(len(df), "total rows in", INPUT_CSV)
    df = df[df['forecast_s10_max'] > 12]
    # # print(len(df), "total rows in", INPUT_CSV)
    # df = df[df['seed_distance_flag']==0]
    # print(len(df), "total rows in", INPUT_CSV)

    required_cols = {
        "time",
        "init_time",
        "step",
        "cyclone_num",
        "bst_lat",
        "bst_lon",
        "forecast_lat",
        "forecast_lon",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {INPUT_CSV}: {sorted(missing)}")

    df_step = df[df["step"] == TARGET_STEP].copy()
    if df_step.empty:
        raise ValueError(f"No rows found for step {TARGET_STEP}.")

    df_step["track_error_km"] = track_error_km(df_step)
    top_cases = (
        df_step.sort_values("track_error_km", ascending=False)
        .head(TOP_N)[["init_time", "cyclone_num", "track_error_km"]]
        .reset_index(drop=True)
    )

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("matplotlib is required. Install with `pip install matplotlib`.") from exc

    ncols = 2
    nrows = int(np.ceil(TOP_N / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.5 * nrows), squeeze=False)

    for idx, row in top_cases.iterrows():
        init_time = row["init_time"]
        cyclone_num = row["cyclone_num"]
        case_df = df[(df["init_time"] == init_time) & (df["cyclone_num"] == cyclone_num)].copy()
        case_df = case_df.sort_values("step")

        pred_lon = [
            align_lon_to_ref(p, r)
            for p, r in zip(case_df["forecast_lon"], case_df["bst_lon"])
        ]

        ax = axes[idx // ncols][idx % ncols]
        ax.plot(
            case_df["bst_lon"],
            case_df["bst_lat"],
            "-o",
            ms=3,
            label="BST",
        )
        ax.plot(
            pred_lon,
            case_df["forecast_lat"],
            "-o",
            ms=3,
            label="Forecast",
        )
        ax.scatter(
            case_df["bst_lon"].iloc[0],
            case_df["bst_lat"].iloc[0],
            marker="s",
            s=40,
            zorder=3,
            label="Start",
        )
        ax.scatter(
            case_df["bst_lon"].iloc[-1],
            case_df["bst_lat"].iloc[-1],
            marker="*",
            s=60,
            zorder=3,
            label="End",
        )
        ax.set_title(
            f"init={init_time} CYCLONE={cyclone_num} err@{TARGET_STEP}={row['track_error_km']:.1f} km"
        )
        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Hide any unused axes.
    for j in range(TOP_N, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150)
    print(f"Saved plot: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
