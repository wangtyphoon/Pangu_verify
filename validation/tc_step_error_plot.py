from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CSV_PATH = Path(__file__).resolve().parent / "tc_bst_z700_metrics35.csv"


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1r = np.deg2rad(lat1)
    lon1r = np.deg2rad(lon1)
    lat2r = np.deg2rad(lat2)
    lon2r = np.deg2rad(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def add_track_errors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lead_hours"] = pd.to_numeric(df["lead_hours"], errors="coerce")
    track_mask = df[["forecast_lat", "forecast_lon", "bst_lat", "bst_lon"]].apply(
        lambda col: np.isfinite(col), axis=0
    )
    track_valid = track_mask.all(axis=1)
    df.loc[track_valid, "track_error_km"] = haversine_km(
        df.loc[track_valid, "forecast_lat"].to_numpy(),
        df.loc[track_valid, "forecast_lon"].to_numpy(),
        df.loc[track_valid, "bst_lat"].to_numpy(),
        df.loc[track_valid, "bst_lon"].to_numpy(),
    )
    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = add_track_errors(df)
    df["s10_bias"] = df["forecast_s10_max"] - df["target_s10_max"]
    df["s850_bias"] = df["forecast_s850_max"] - df["target_s850_max"]
    df["msl_bias"] = df["forecast_msl_min_hPa"] - df["target_msl_min_hPa"]

    grouped = df.groupby("lead_hours", dropna=True)
    summary = grouped.agg(
        track_error_km_mean=("track_error_km", "mean"),
        track_error_km_median=("track_error_km", "median"),
        track_error_km_std=("track_error_km", "std"),
        s10_bias=("s10_bias", "mean"),
        s850_bias=("s850_bias", "mean"),
        msl_bias=("msl_bias", "mean"),
        track_count=("track_error_km", "count"),
        intensity_count=("s10_bias", "count"),
    ).reset_index()
    summary = summary.sort_values("lead_hours")
    return summary


def plot_summary(summary: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("matplotlib is required. Install with `pip install matplotlib`.") from exc

    lead = summary["lead_hours"].to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=True)
    axes[0].plot(lead, summary["track_error_km_mean"].to_numpy(), "-o", label="Mean track error")
    axes[0].plot(lead, summary["track_error_km_median"].to_numpy(), "--", label="Median track error")
    axes[0].set_ylabel("Track error (km)")
    axes[0].grid(alpha=0.4)
    axes[0].legend(loc="best")

    axes[1].plot(lead, summary["s10_bias"].to_numpy(), "-o", label="10m wind bias")
    axes[1].plot(lead, summary["s850_bias"].to_numpy(), "-o", label="850 hPa wind bias")
    axes[1].plot(lead, summary["msl_bias"].to_numpy(), "-o", label="MSL bias")
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].set_ylabel("Bias (forecast - target)")
    axes[1].set_xlabel("Lead hours")
    axes[1].grid(alpha=0.4)
    axes[1].legend(loc="best")

    fig.tight_layout()
    plt.show()


def plot_violin(df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("matplotlib is required. Install with `pip install matplotlib`.") from exc

    df = df.dropna(subset=["lead_hours", "track_error_km"])
    lead_hours = np.sort(df["lead_hours"].unique())
    if lead_hours.size == 0:
        print("No track_error_km data to plot.")
        return

    data = [df.loc[df["lead_hours"] == lead, "track_error_km"].to_numpy() for lead in lead_hours]

    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    ax.violinplot(data, showmeans=True, showmedians=True)
    ax.set_xticks(np.arange(1, len(lead_hours) + 1))
    ax.set_xticklabels(lead_hours.astype(int))
    ax.set_xlabel("Lead hours")
    ax.set_ylabel("Track error (km)")
    ax.set_title("Track error distribution by lead hour")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    plt.show()


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing CSV: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df =df.dropna(subset=[ 'bst_lat', 'bst_lon'])
    df = df[(df['forecast_lat'] > 5) & (df['forecast_lat'] < 42.25)]
    df = df[(df['forecast_lon'] > 107.5) & (df['forecast_lon'] < 142.5)]
    print(len(df), "total rows in CSV")
    # df = df[(df['bst_lat'] > 5) & (df['bst_lat'] < 42.25)]
    # df = df[(df['bst_lon'] > 107.5) & (df['bst_lon'] < 142.5)]
    # df = df[df['forecast_s10_max'] > 12]
    summary = build_summary(df)
    print(summary.to_string(index=False))
    plot_summary(summary)
    plot_violin(add_track_errors(df))


if __name__ == "__main__":
    main()
