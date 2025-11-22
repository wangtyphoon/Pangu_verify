import numpy as np
import pandas as pd
import os
from loss import create_spatial_weights
from subset import extract_subset
from GCM_loss import radial_wavenumber_continuous

# Channel/level mapping (same as weight_rmse_prs.py)
z500_channel = 0
q700_channel = 1
t850_channel = 2
u850_channel = 3
v850_channel = 4
channel_level_map = {
    'z500': (z500_channel, 5),
    'q700': (q700_channel, 3),
    't850': (t850_channel, 2),
    'u850': (u850_channel, 2),
    'v850': (v850_channel, 2),
}

forecast_dir = '../results'
era5_dir     = '../forecasts'
nc_path      = r'../forecasts\\2020-05-05-00-00\\upper.nc'
boundary     = 8

def match_spectrum_amplitude_2d(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    調整 X 的振幅頻譜以在徑向上匹配 Y
    X, Y: (H, W) 的實數 numpy array
    回傳 X_adj，其 |FFT(X_adj)| 在徑向上匹配 |FFT(Y)|
    """
    assert X.shape == Y.shape
    H, W = X.shape

    # FFT
    Xf = np.fft.rfftn(X, axes=(-2, -1))  # (H, W//2+1)
    Yf = np.fft.rfftn(Y, axes=(-2, -1))

    magX = np.abs(Xf)
    magY = np.abs(Yf)

    # radial bins
    radial = radial_wavenumber_continuous(H, W,device='cpu')  # (H, W//2+1)
    k_flat = radial.flatten().long().to(Xf.device)
    kmax = int(radial.max()) + 1

    magX_sum = np.bincount(k_flat, weights=magX.flatten(), minlength=kmax)
    magY_sum = np.bincount(k_flat, weights=magY.flatten(), minlength=kmax)
    cnt     = np.bincount(k_flat, minlength=kmax)

    # mean amplitude per bin
    eps = 1e-12
    meanX = magX_sum / (cnt + eps)
    meanY = magY_sum / (cnt + eps)

    # scale ratio per bin
    ratio = meanY / (meanX + eps)
    ratio_grid = ratio[k_flat].reshape(magX.shape)

    # 調整 X 的頻譜 magnitude，保留相位
    Xf_adj = Xf * ratio_grid
    X_adj = np.fft.irfftn(Xf_adj, s=(H, W), axes=(-2, -1)).real
    return X_adj

def weighted_mse_prs_spectrum_matched(forecast_dict, era5_dict, b: int):
    # Weight for full grid from any variable
    any_var = next(iter(forecast_dict))
    H, W = forecast_dict[any_var].shape
    w_full = create_spatial_weights(H=H, W=W, boundary=b, device='cpu')
    y0, y1 = b, H - b
    x0, x1 = b, W - b
    w_crop = w_full[y0:y1, x0:x1]
    sum_w = np.sum(w_crop)

    mse_per_var = {}
    for var in forecast_dict:
        X = forecast_dict[var][y0:y1, x0:x1]
        Y = era5_dict[var][y0:y1, x0:x1]
        X_adj =  match_spectrum_amplitude_2d(X, Y)
        err2 = (X_adj - Y) ** 2
        mse_per_var[var] = float(np.sum(err2 * w_crop) / sum_w)

    mse_overall = float(np.mean(list(mse_per_var.values())))
    return mse_per_var, mse_overall


def main():
    records = []
    for folder in os.listdir(forecast_dir):
        dt = folder.split('o')[1]
        print(f"Processing {dt}...")

        fc_path = os.path.join(forecast_dir, folder, f'output_upper_{dt}.npy')
        er_path = os.path.join(era5_dir, dt, 'input_upper.npy')

        if not (os.path.exists(fc_path) and os.path.exists(er_path)):
            print(f"Skip {dt}: missing files")
            continue

        fc_data = np.load(fc_path)
        er_data = np.load(er_path)

        fc_sub = extract_subset(fc_data, nc_path, channel_level_map)
        er_sub = extract_subset(er_data, nc_path, channel_level_map)

        mse_dict, mse_all = weighted_mse_prs_spectrum_matched(fc_sub, er_sub, boundary)
        rmse_dict = {f"{var}_rmse": mse for var, mse in mse_dict.items()}
        rmse_dict["overall_rmse"] = mse_all
        rmse_dict["datetime"] = dt
        records.append(rmse_dict)

    if not records:
        print("No valid records found.")
        return

    df = pd.DataFrame(records).set_index("datetime").sort_index()
    mean_series = df.mean(axis=0).rename("mean_rmse")
    mean_series = np.sqrt(mean_series).rename("mean_rmse")
    df.loc["平均"] = mean_series

    out_csv = "rmse_summary_prs_spectrum_matched.csv"
    df.to_csv(out_csv, float_format="%.8f", encoding="utf-8-sig")
    print(f"結果已輸出到 {out_csv}")


if __name__ == "__main__":
    main()

