#!/usr/bin/env python3
from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt

# =======================
# Configuration (defaults)
# =======================
SUMMARY_CSV_DEFAULT = Path("pangu_boundary_loss_summary.csv")
OUTPUT_DIR_DEFAULT = Path("pangu_boundary_loss_plots")
VARIABLES_DEFAULT = ["z500", "q700", "t850", "u850", "v850"]
VALUE_COLUMN_DEFAULT = "rmse"  # options: rmse or mean_mse
SIDES = ["top", "bottom", "left", "right"]
REGION_ORDER = ["outer", "inner"]
COLORS = {"outer": "#d62728", "inner": "#1f77b4"}
MARKERS = {"outer": "s", "inner": "o"}
FIGSIZE = (10, 6)


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot boundary-distance curves (Pangu, 5 variables)."
    )
    p.add_argument(
        "--summary",
        type=Path,
        default=SUMMARY_CSV_DEFAULT,
        help="Summary CSV produced by boundary_loss_analysis_pangu.py",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_DIR_DEFAULT,
        help="Output directory for figures",
    )
    p.add_argument(
        "--metric",
        choices=["rmse", "mean_mse"],
        default=VALUE_COLUMN_DEFAULT,
        help="Y-axis metric (rmse or mean_mse)",
    )
    p.add_argument(
        "--vars",
        nargs="+",
        default=VARIABLES_DEFAULT,
        help="Variables to plot (subset of z500 q700 t850 u850 v850)",
    )
    p.add_argument(
        "--yshared",
        action="store_true",
        help="Share Y-axis across sides for the same variable",
    )
    p.add_argument(
        "--tight",
        action="store_true",
        help="Use tight_layout on figures",
    )
    return p.parse_args()


def load_summary(csv_path: Path, value_col: str) -> pd.DataFrame:
    if not csv_path.exists():
        # 兼容舊檔名
        fallback = Path("boundary_loss_summary.csv")
        if fallback.exists():
            print(f"[Info] {csv_path} not found; using fallback {fallback}")
            csv_path = fallback
        else:
            raise FileNotFoundError(f"Summary CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = {"side", "distance", "region", "channel_name", value_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Summary CSV missing columns: {missing}")

    df = df.copy()
    df["side"] = pd.Categorical(df["side"], categories=SIDES, ordered=True)
    df["region"] = pd.Categorical(df["region"], categories=REGION_ORDER, ordered=True)
    return df


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_variable(
    df: pd.DataFrame,
    var_name: str,
    out_dir: Path,
    value_col: str,
    sharey: bool = False,
    use_tight: bool = False,
) -> None:
    var_df = df[df["channel_name"] == var_name]
    if var_df.empty:
        print(f"[Warn] variable '{var_name}' not present in summary CSV; skipping")
        return

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True, sharey=sharey)
    axes_flat = axes.flatten()

    global_ymin, global_ymax = None, None

    for ax, side in zip(axes_flat, SIDES):
        side_df = var_df[var_df["side"] == side].sort_values("distance")
        if side_df.empty:
            ax.set_title(f"{side.title()} (no data)")
            ax.axis("off")
            continue

        for region in REGION_ORDER:
            reg_df = side_df[side_df["region"] == region]
            if reg_df.empty:
                continue
            ax.plot(
                reg_df["distance"],
                reg_df[value_col],
                label=region,
                color=COLORS.get(region, None),
                marker=MARKERS.get(region, None),
                linewidth=2,
            )

        ax.set_title(side.title())
        ax.set_xlabel("Grid Distance")
        ax.set_ylabel(value_col.upper())
        ax.grid(True, linestyle="--", alpha=0.4)

        # 更新全域 y 範圍（若 sharey=False 仍計算，以便後續標註）
        ymin, ymax = side_df[value_col].min(), side_df[value_col].max()
        global_ymin = ymin if global_ymin is None else min(global_ymin, ymin)
        global_ymax = ymax if global_ymax is None else max(global_ymax, ymax)

    # 圖例（使用第一個軸的 handles）
    handles, labels = [], []
    for ax in axes_flat:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    if handles:
        fig.legend(handles, labels, loc="upper right")

    fig.suptitle(f"{var_name} {value_col.upper()} vs Boundary Distance")

    if use_tight:
        fig.tight_layout(rect=[0, 0, 0.9, 0.95])

    out_path = out_dir / f"{var_name}_{value_col}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved plot: {out_path}")


def main():
    args = parse_args()
    df = load_summary(args.summary, args.metric)
    ensure_output_dir(args.out)

    # 僅保留要求的五通道子集（容錯：如果傳入其他名稱則自動忽略）
    valid = set(["z500", "q700", "t850", "u850", "v850"])
    variables = [v for v in args.vars if v in valid]
    if not variables:
        print("[Warn] No valid variables requested; nothing to plot.")
        return

    for var in variables:
        plot_variable(
            df=df,
            var_name=var,
            out_dir=args.out,
            value_col=args.metric,
            sharey=args.yshared,
            use_tight=args.tight,
        )


if __name__ == "__main__":
    main()
