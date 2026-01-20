from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from netCDF4 import Dataset

# Variables and channels
BASE_VARS = ("z500", "q700", "t850", "u850", "v850")
DERIVED_VARS = ("wind850",)
VARS = BASE_VARS + DERIVED_VARS
CHANNEL_LEVEL_MAP = {
    "z500": (0, 5),
    "q700": (1, 3),
    "t850": (2, 2),
    "u850": (3, 2),
    "v850": (4, 2),
}

GRID_SHAPE = (192, 192)
DEFAULT_OUTPUT_DIR = Path(r"D:\paper\vis_multi_steps\pangu")

# Paths (edit these to match your environment)
RUN_DIR = Path(r"../results/2024-09-29-12-00")
ERA5_ROOT = Path(r"../forecasts")
REFERENCE_NC = ERA5_ROOT / "2020-05-05-00-00" / "upper.nc"
LEAD_HOURS = [24, 48, 72, 96,120,144,168]  # Set to None to process all available steps
OUTPUT_DIR = DEFAULT_OUTPUT_DIR


def parse_dt(dt_str: str) -> datetime:
    return datetime.strptime(dt_str, "%Y-%m-%d-%H-%M")


def compute_wind_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.sqrt(np.square(u) + np.square(v))


def to_hw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2 and arr.shape == GRID_SHAPE:
        return arr
    if arr.ndim == 1 and arr.size == GRID_SHAPE[0] * GRID_SHAPE[1]:
        return arr.reshape(GRID_SHAPE)
    if arr.ndim == 2 and (
        arr.shape[0] == GRID_SHAPE[0] * GRID_SHAPE[1]
        or arr.shape[1] == GRID_SHAPE[0] * GRID_SHAPE[1]
    ):
        return arr.reshape(GRID_SHAPE)
    raise ValueError(f"Cannot reshape to {GRID_SHAPE}, got shape={arr.shape}")


def extract_subset(
    data: np.ndarray,
    nc_path: str,
    channel_level_map: Dict[str, Tuple[int, int]],
    lat_range: Tuple[float, float] = (0.0, 47.75),
    lon_range: Tuple[float, float] = (102.25, 150.0),
) -> Dict[str, np.ndarray]:
    with Dataset(nc_path) as ds:
        lats = ds.variables["latitude"][:]
        lons = ds.variables["longitude"][:]
    lons = np.mod(lons, 360)

    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range

    lat_idx = np.where((lats >= lat_min) & (lats <= lat_max))[0]
    lon_idx = np.where((lons >= lon_min) & (lons <= lon_max))[0]
    if lat_idx.size == 0 or lon_idx.size == 0:
        raise ValueError("Lat/Lon range selects no grid points.")

    lat_sl = slice(lat_idx.min(), lat_idx.max() + 1)
    lon_sl = slice(lon_idx.min(), lon_idx.max() + 1)

    subset = {}
    for name, (ch, lv) in channel_level_map.items():
        subset[name] = data[ch, lv, lat_sl, lon_sl]
    return subset


def forecast_to_dict(forecast_data: np.ndarray) -> Dict[str, np.ndarray]:
    fc_dict = {}
    for name, (ch, lv) in CHANNEL_LEVEL_MAP.items():
        fc_dict[name] = to_hw(forecast_data[ch, lv])
    return fc_dict


def add_wind_speed(data_dict: Dict[str, np.ndarray]) -> None:
    u = to_hw(np.asarray(data_dict["u850"]))
    v = to_hw(np.asarray(data_dict["v850"]))
    data_dict["wind850"] = compute_wind_speed(u, v)


def load_step(
    forecast_file: Path, era5_root: Path, reference_nc: Path
) -> Tuple[str, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    target_dt = forecast_file.stem[len("output_upper_") :]
    truth_path = era5_root / target_dt / "input_upper.npy"
    if not truth_path.exists():
        raise FileNotFoundError(f"Missing ERA5 truth: {truth_path}")

    fc_data = np.load(forecast_file)
    tr_data = np.load(truth_path)

    forecast_dict = forecast_to_dict(fc_data)
    truth_dict = extract_subset(tr_data, str(reference_nc), CHANNEL_LEVEL_MAP)
    add_wind_speed(forecast_dict)
    add_wind_speed(truth_dict)

    return target_dt, forecast_dict, truth_dict


def get_run_start_dt(run_dir: Path) -> datetime | None:
    run_name = run_dir.name
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


def collect_steps(
    run_dir: Path,
    era5_root: Path,
    reference_nc: Path,
    lead_hours: List[int] | None,
) -> List[Tuple[str, Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
    if not run_dir.exists():
        raise FileNotFoundError(f"RUN_DIR not found: {run_dir}")

    files = sorted(
        run_dir.glob("output_upper_*.npy"),
        key=lambda p: parse_dt(p.stem[len("output_upper_") :]),
    )
    if not files:
        raise FileNotFoundError(f"No output_upper_*.npy under {run_dir}")

    lead_set = set(lead_hours) if lead_hours else None
    start_dt = get_run_start_dt(run_dir) if lead_set else None
    if lead_set and start_dt is None:
        raise ValueError("RUN_DIR name must contain a parsable start datetime.")

    steps = []
    for f in files:
        if lead_set:
            target_dt = parse_dt(f.stem[len("output_upper_") :])
            lead_hour = int((target_dt - start_dt).total_seconds() // 3600)
            if lead_hour not in lead_set:
                continue
        steps.append(load_step(f, era5_root, reference_nc))
    return steps


def prepare_step_data(
    steps: List[Tuple[str, Dict[str, np.ndarray], Dict[str, np.ndarray]]]
) -> List[Dict[str, object]]:
    prepared = []
    for target_dt, forecast, truth in steps:
        step_entry = {"target": target_dt, "vars": {}}
        for var in VARS:
            fc = forecast[var]
            tr = to_hw(np.asarray(truth[var]))
            err = fc - tr
            loss = np.square(err)
            step_entry["vars"][var] = {
                "forecast": fc,
                "truth": tr,
                "error": err,
                "loss": loss,
                "mse": float(loss.mean()),
                "bias": float(err.mean()),
            }
        prepared.append(step_entry)
    return prepared


def save_prepared(prepared: List[Dict[str, object]], output_dir: Path, run_dir: Path) -> None:
    if not prepared:
        print("No steps to save.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = run_dir.name

    for step in prepared:
        target = step["target"]
        step_dir = output_dir / run_name / target
        step_dir.mkdir(parents=True, exist_ok=True)

        for var in VARS:
            data = step["vars"][var]
            out_path = step_dir / f"{var}.npz"
            np.savez_compressed(
                out_path,
                forecast=data["forecast"],
                truth=data["truth"],
                error=data["error"],
                loss=data["loss"],
                mse=np.array(data["mse"]),
                bias=np.array(data["bias"]),
            )
        print(f"Saved {target} -> {step_dir}")


def main() -> None:
    steps = collect_steps(RUN_DIR, ERA5_ROOT, REFERENCE_NC, LEAD_HOURS)
    prepared = prepare_step_data(steps)
    save_prepared(prepared, OUTPUT_DIR, RUN_DIR)


if __name__ == "__main__":
    main()
