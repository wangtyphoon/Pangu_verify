"""Summarise minimal wavenumber spectral threshold exceedances from all_spectral_data2024.npz."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_THRESHOLDS: Sequence[float] = (1.0, 2.0, 3.0)


def infer_variables(npz: np.lib.npyio.NpzFile) -> list[str]:
    variables: list[str] = []
    for key in npz.files:
        if key.startswith("k_"):
            var = key[2:]
            required = [
                f"era5_mean_{var}",
                f"era5_std_{var}",
                f"pangu_mean_{var}",
            ]
            if all(r in npz for r in required):
                variables.append(var)
    return sorted(set(variables))


def summarise_variable(var: str, npz: np.lib.npyio.NpzFile, thresholds: Sequence[float]) -> dict[str, float | str]:
    k = np.asarray(npz[f"k_{var}"]).astype(float).ravel()
    era5_mean = np.asarray(npz[f"era5_mean_{var}"]).astype(float).ravel()
    era5_std = np.asarray(npz[f"era5_std_{var}"]).astype(float).ravel()
    pangu_mean = np.asarray(npz[f"pangu_mean_{var}"]).astype(float).ravel()

    if not (k.size and era5_mean.shape == era5_std.shape == pangu_mean.shape == k.shape):
        raise ValueError(f"Shape mismatch for variable '{var}'")

    std_safe = np.maximum(era5_std, 1e-12)
    abs_diff = np.abs(pangu_mean - era5_mean)

    row: dict[str, float | str] = {"variable": var}
    for thr in thresholds:
        mask = abs_diff > (thr * std_safe)
        row[f"k_min_ge_{thr:g}sigma"] = (
            float(k[mask].min()) if mask.any() else math.nan
        )
    return row


def main(
    npz_path: str = "all_spectral_data2024.npz",
    output_csv: str = "spectral_thresholds_2024.csv",
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> None:
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    with np.load(path) as npz:
        variables = infer_variables(npz)
        if not variables:
            raise ValueError("No variables inferred from NPZ contents")
        rows = [summarise_variable(var, npz, thresholds) for var in variables]

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv} with {len(df)} variables")


def cli(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", dest="npz_path", default="all_spectral_data2024.npz")
    parser.add_argument("--output", dest="output_csv", default="spectral_thresholds_2024.csv")
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=list(DEFAULT_THRESHOLDS),
        help="sigma thresholds (default: 1 2 3)",
    )
    args = parser.parse_args(argv)
    main(args.npz_path, args.output_csv, args.thresholds)


if __name__ == "__main__":
    cli()
