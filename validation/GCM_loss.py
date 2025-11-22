import torch, math
from torch.fft import rfftn, irfftn, fftfreq, rfftfreq
import matplotlib.pyplot as plt

# —— 1. 径向波数，只接收 H, W ——
def radial_wavenumber_2d(H: int, W: int, device):
    ky = torch.fft.fftfreq(H) * H
    kx = torch.fft.rfftfreq(W) * W
    kxg = kx.unsqueeze(0).expand(H, -1)
    kyg = ky.unsqueeze(1).expand(-1, kx.shape[0])
    wave_number = torch.round((kxg**2 + kyg**2).sqrt()).long()
    return wave_number
# —— 2. 2D 的 filtered MSE —— 
import torch
from torch.fft import rfftn, irfftn
import matplotlib.pyplot as plt

def radial_wavenumber_continuous(H: int, W: int, device):
    """
    回傳 shape=(H, W//2+1) 的頻域徑向波數（可帶小數）
    """
    ky = torch.fft.fftfreq(H, d=1.0) * H  # (-H/2 ~ H/2-1)
    kx = torch.fft.rfftfreq(W, d=1.0) * W # (0 ~ W/2)
    kxg = kx.unsqueeze(0).expand(H, -1)
    kyg = ky.unsqueeze(1).expand(-1, kx.shape[0])
    radial = torch.sqrt(kxg**2 + kyg**2)  # shape=(H, W//2+1)
    return radial

def filtered_mse_loss(X: torch.Tensor, Y: torch.Tensor, alpha=1.0, order=12, cutoff=21):
    assert X.ndim == 2 and Y.ndim == 2
    H, W = X.shape
    device = X.device

    # 頻域
    Xf = rfftn(X, dim=(-2, -1))
    Yf = rfftn(Y, dim=(-2, -1))

    # 徑向波數
    radial = radial_wavenumber_continuous(H, W, device)
    weight_mask = torch.exp(-alpha * (radial / cutoff) ** order)  # (H, W//2+1)

    # 權重濾波
    Xf_w = Xf * weight_mask
    Yf_w = Yf * weight_mask

    # 逆變換
    x_w = irfftn(Xf_w, s=(H, W), dim=(-2, -1))
    y_w = irfftn(Yf_w, s=(H, W), dim=(-2, -1))

    # 差值
    diff_orig = X - Y
    diff_filt = x_w.real - y_w.real

    # 2x3 視覺化
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(X.cpu(), cmap='bwr')
    plt.title('X Original')
    plt.colorbar()

    plt.subplot(2, 3, 2)
    plt.imshow(Y.cpu(), cmap='bwr')
    plt.title('Y Original')
    plt.colorbar()

    plt.subplot(2, 3, 3)
    plt.imshow(diff_orig.cpu(), cmap='bwr')
    plt.title('X - Y ')
    plt.colorbar()

    plt.subplot(2, 3, 4)
    plt.imshow(x_w.cpu().real, cmap='bwr')
    plt.title('X Weighted Filtered')
    plt.colorbar()

    plt.subplot(2, 3, 5)
    plt.imshow(y_w.cpu().real, cmap='bwr')
    plt.title('Y Weighted Filtered')
    plt.colorbar()

    plt.subplot(2, 3, 6)
    plt.imshow(diff_filt.cpu(), cmap='bwr')
    plt.title('X_filtered - Y_filtered')
    plt.colorbar()

    plt.tight_layout()
    plt.show()
    print(diff_filt.shape)

    return torch.mean(diff_filt ** 2)

# ---------- 3. Isotropic power spectrum ----------
def isotropic_spectrum(X: torch.Tensor):
    """
    X: [H, W] 實數場
    回傳 power spectrum spec，shape = [Lmax+1]，其中 Lmax = floor(max(radial))
    """
    H, W = X.shape
    # 頻域 FFT
    Xf = rfftn(X, dim=(-2, -1))                  # shape = [H, W//2+1], complex
    power = Xf.real**2 + Xf.imag**2             # [H, W_rfft]
    power /= (H * W)**2

    # 徑向波數 (可帶小數)
    k_mag = radial_wavenumber_continuous(H, W, X.device)  # [H, W_rfft]
    Lmax = int(k_mag.max().item())

    # 展平並累加到對應的波數 bin
    power_flat = power.flatten()                # [H*W_rfft]
    k_flat     = k_mag.flatten().long()         # [H*W_rfft]
    spec       = torch.zeros(Lmax+1, device=X.device)

    # bincount 累加
    spec = spec.index_add(0, k_flat, power_flat)
    return spec  # [Lmax+1]

def spectrum_mse_loss(X: torch.Tensor, Y: torch.Tensor, l_tilde: int, norm_power=None):
    """
    X, Y: [H, W]
    l_tilde: 計算到哪個波數
    回傳 mean((Sx[:l_tilde+1] - Sy[:l_tilde+1])**2)
    """
    Sx = isotropic_spectrum(X)[:l_tilde+1]
    Sy = isotropic_spectrum(Y)[:l_tilde+1]

    return torch.mean((Sx - Sy) ** 2)

# ---------- 5. Bias loss：spectral‑bias MSE ----------
def spectral_bias_mse_loss(X, Y, l_max=None):
    """
    先在 batch & lead‑time 方向做平均（此例只有 batch）。
    """
    diff_avg = (X - Y)
    Sb = isotropic_spectrum(diff_avg)             
    if l_max is not None:
        Sb = Sb[..., :l_max+1]
    # 對剩餘維度直接加總，相當於 \|·\|_spectral^2
    return Sb.mean()


# import numpy as np
# import pandas as pd
# import os
# from loss import create_spatial_weights
# from subset import extract_subset

# # 通道映射与路径配置
# z500_channel = 0
# q700_channel = 1
# t850_channel = 2
# u850_channel = 3
# v850_channel = 4
# channel_level_map = {
#     'z500': (z500_channel, 5),
#     'q700': (q700_channel, 3),
#     't850': (t850_channel, 2),
#     'u850': (u850_channel, 2),
#     'v850': (v850_channel, 2),
# }

# forecast_dir = '../results'
# era5_dir     = '../forecasts'
# nc_path      = r'../forecasts\2020-05-05-00-00\upper.nc'

# # 计算权重
# weight = create_spatial_weights(H=192, W=192, boundary=8, device='cpu')  # shape (H, W)

# # 用于存储每个时刻的 RMSE
# records = []

# def weighted_mse_dict(forecast_dict, era5_dict, weight):
#     sum_w = np.sum(weight)
#     mse_per_var = {}
#     for var in forecast_dict:
#         err2 = (forecast_dict[var] - era5_dict[var]) ** 2
#         mse_per_var[var] = np.sum(err2 * weight) / sum_w
#     mse_overall = np.mean(list(mse_per_var.values()))
#     return mse_per_var, mse_overall

# # 遍历所有时刻文件
# for folder in os.listdir(forecast_dir):
#     dt = folder.split('o')[1]
#     print(f"Processing {dt}...")
#     fc_path  = os.path.join(forecast_dir, folder,   f'output_upper_{dt}.npy')
#     er_path  = os.path.join(era5_dir,     dt,        'input_upper.npy')
    
#     # 载入并提取子集
#     fc_data  = np.load(fc_path)
#     er_data  = np.load(er_path)
#     fc_sub   = extract_subset(fc_data, nc_path, channel_level_map ,lat_range=(2.25, 45.50), lon_range=(104.5, 147.75))
#     er_sub   = extract_subset(er_data, nc_path, channel_level_map ,lat_range=(2.25, 45.50), lon_range=(104.5, 147.75))
#     x  = torch.tensor(fc_sub['u850'], dtype=torch.float32)
#     y  = torch.tensor(er_sub['u850'], dtype=torch.float32)
#     loss_acc = filtered_mse_loss(x, y, cutoff = 21)/36
#     print(f"Filtered MSE loss: {loss_acc.item()}")
#     mse_spec = spectrum_mse_loss(x, y, l_tilde=87)
#     print(f"Spectrum MSE loss: {mse_spec.item()}")
#     mse_bias = spectral_bias_mse_loss(x, y, l_max=87)
#     print(f"Spectral Bias MSE loss: {mse_bias.item()}")
#     break