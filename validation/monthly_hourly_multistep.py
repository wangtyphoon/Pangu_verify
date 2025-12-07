"""
針對多 step RMSE 結果（rmse_multistep_prs.csv）計算月份/時段的平均與標準差。
- 平均採用 RMS（先平方後平均再開根號），與 weight_rmse_prs_multistep 的處理一致。
- 同時輸出整體分組與依 lead_hours 分組的結果，便於分析不同預報步長。
"""

from pathlib import Path
from typing import Iterable, List, Tuple, Union

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "rmse_multistep_prs.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV)
    dt = pd.to_datetime(df["datetime"], format="%Y-%m-%d-%H-%M", errors="coerce")
    df = df.loc[dt.notnull()].copy()
    df["datetime"] = dt.loc[dt.notnull()]
    df["month"] = df["datetime"].dt.month
    df["hour"] = (df["datetime"].dt.hour - 6) % 24  # 回退 6 小時
    return df


def rmse_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.endswith("_rmse")]
    if not cols:
        raise ValueError("找不到 *_rmse 欄位，請確認輸入檔案內容")
    return cols


def rms_mean(group: pd.DataFrame) -> pd.Series:
    return np.sqrt((group**2).mean())


def aggregate(
    df: pd.DataFrame, keys: Union[str, Iterable[str]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = rmse_columns(df)
    mean_df = (
        df.groupby(keys)[cols]
        .apply(rms_mean)
        .sort_index()
    )
    std_df = (
        df.groupby(keys)[cols]
        .std()
        .sort_index()
    )
    return mean_df, std_df


def save(df: pd.DataFrame, filename: str) -> None:
    out_path = ROOT / filename
    df.to_csv(out_path, float_format="%.8f", encoding="utf-8-sig")
    print(f"已輸出 {out_path.name}")


def main() -> None:
    df = load_data()

    # 整體（不分 lead）依月份/時段
    monthly_mean, monthly_std = aggregate(df, "month")
    hourly_mean, hourly_std = aggregate(df, "hour")
    save(monthly_mean, "monthly_mean_rmse_multistep.csv")
    save(monthly_std, "monthly_std_rmse_multistep.csv")
    save(hourly_mean, "hourly_mean_rmse_multistep.csv")
    save(hourly_std, "hourly_std_rmse_multistep.csv")

    # 依 lead_hours 分組的月份/時段
    if "lead_hours" in df.columns:
        monthly_mean_lead, monthly_std_lead = aggregate(df, ["month", "lead_hours"])
        hourly_mean_lead, hourly_std_lead = aggregate(df, ["hour", "lead_hours"])
        save(monthly_mean_lead, "monthly_mean_rmse_multistep_by_lead.csv")
        save(monthly_std_lead, "monthly_std_rmse_multistep_by_lead.csv")
        save(hourly_mean_lead, "hourly_mean_rmse_multistep_by_lead.csv")
        save(hourly_std_lead, "hourly_std_rmse_multistep_by_lead.csv")
    else:
        print("輸入檔案缺少 lead_hours 欄位，跳過依 lead 分組的輸出。")


if __name__ == "__main__":
    main()
