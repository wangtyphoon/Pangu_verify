import numpy as np
import matplotlib.pyplot as plt

# 1. 载入所有频谱数据
data = np.load('all_energy_data.npz')
variables = ['z500', 'q700', 't850', 'u850', 'v850']

# 1. 基於波數的平均徑向譜（含 ±1σ，子標題顯示最大 k）
plt.figure(figsize=(20, 4))
for i, var in enumerate(variables, 1):
    k          = data[f'k_{var}']
    mean_er    = data[f'era5_mean_{var}']
    std_er     = data[f'era5_std_{var}']
    mean_fc    = data[f'pangu_mean_{var}']
    std_fc     = data[f'pangu_std_{var}']

    diff = mean_fc - mean_er
    mask1 = np.abs(diff) > 1 * std_er
    mask2 = np.abs(diff) > 2 * std_er
    mask3 = np.abs(diff) > 3 * std_er

    plt.subplot(1, len(variables), i)
    plt.plot(k, mean_er, linestyle='-',  label='ERA5 μ')
    plt.plot(k, mean_fc, linestyle='--', label='Pangu μ')

    plt.fill_between(k,
                     np.clip(mean_er - std_er, 1e-12, None),
                     mean_er + std_er,
                     alpha=0.25, linewidth=0, label='ERA5 ±1σ')
    plt.fill_between(k,
                     np.clip(mean_fc - std_fc, 1e-12, None),
                     mean_fc + std_fc,
                     alpha=0.25, linewidth=0, label='Pangu ±1σ')

    # 散點標記
    plt.scatter(k[mask1], mean_fc[mask1], s=10, c='orange', label='>1σ')
    plt.scatter(k[mask2], mean_fc[mask2], s=10, c='green', label='>2σ')
    plt.scatter(k[mask3], mean_fc[mask3], s=20, c='red', marker='*', label='>3σ')

    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('k'); plt.ylabel('Amplitude')
    plt.grid(True, which='both', linestyle=':')

    # === 子標題：只顯示最大 k ===
    k_texts = []
    if mask1.any():
        k_texts.append(f"k1={k[mask1].min():.1f}")
    if mask2.any():
        k_texts.append(f"k2={k[mask2].min():.1f}")
    if mask3.any():
        k_texts.append(f"k3={k[mask3].min():.1f}")
    k_text = " | ".join(k_texts) if k_texts else "no exceed"

    plt.title(f'Wavenumber: {var}\n({k_text})')
    plt.legend(fontsize='x-small', loc='best')

plt.tight_layout()
plt.show()



# =========================
# 基於波長的平均徑向譜（含 ±1σ）
# 使用 spec_ratio 的定義方式
# =========================
R = 6371.0  # km

def cell_area_equal_area(lat2d, lon2d):
    if lat2d.ndim == 1 and lon2d.ndim == 1:
        lat2d, lon2d = np.meshgrid(lat2d, lon2d, indexing='ij')
    lat1d = lat2d[:, 0]
    lon1d = lon2d[0, :]
    dlat = np.deg2rad(np.abs(np.diff(lat1d).mean()))
    dlon = np.deg2rad(np.abs(np.diff(lon1d).mean()))
    phi_c = np.deg2rad(lat1d)
    phi_n = phi_c + 0.5 * dlat
    phi_s = phi_c - 0.5 * dlat
    strip = (R**2) * (np.sin(phi_n) - np.sin(phi_s)) * dlon
    A = np.repeat(strip[:, None], lon1d.size, axis=1)
    return A

def effective_grid_km_equal_area(lat2d, lon2d):
    A = cell_area_equal_area(lat2d, lon2d)
    s_eff = np.sqrt(A.mean())   # 平均格距 (km)
    return s_eff

# 載入緯經度
lat = np.load("../lat.npy")
lon = np.load("../lon.npy")

s_eff = effective_grid_km_equal_area(lat, lon)
H_sub, W_sub = lat.shape
L_eff = s_eff * min(H_sub, W_sub)   # 有效邊長 (km)

# 2. 基於波長的平均徑向譜（含 ±1σ，子標題顯示最大 λ）
plt.figure(figsize=(20, 4))
for i, var in enumerate(variables, 1):
    k       = data[f'k_{var}']
    mean_er = data[f'era5_mean_{var}']
    std_er  = data[f'era5_std_{var}']
    mean_fc = data[f'pangu_mean_{var}']
    std_fc  = data[f'pangu_std_{var}']

    mask = k > 0
    lam = L_eff / k[mask]

    diff = mean_fc[mask] - mean_er[mask]
    mask1 = np.abs(diff) > 1 * std_er[mask]
    mask2 = np.abs(diff) > 2 * std_er[mask]
    mask3 = np.abs(diff) > 3 * std_er[mask]

    plt.subplot(1, len(variables), i)
    plt.plot(lam, mean_er[mask], linestyle='-',  label='ERA5 μ')
    plt.plot(lam, mean_fc[mask], linestyle='--', label='Pangu μ')

    plt.fill_between(lam,
                     np.clip(mean_er[mask] - std_er[mask], 1e-12, None),
                     mean_er[mask] + std_er[mask],
                     alpha=0.25, linewidth=0, label='ERA5 ±1σ')
    plt.fill_between(lam,
                     np.clip(mean_fc[mask] - std_fc[mask], 1e-12, None),
                     mean_fc[mask] + std_fc[mask],
                     alpha=0.25, linewidth=0, label='Pangu ±1σ')

    # 散點標記
    plt.scatter(lam[mask1], mean_fc[mask][mask1], s=10, c='orange', label='>1σ')
    plt.scatter(lam[mask2], mean_fc[mask][mask2], s=10, c='green', label='>2σ')
    plt.scatter(lam[mask3], mean_fc[mask][mask3], s=20, c='red', marker='*', label='>3σ')

    plt.xscale('log'); plt.yscale('log')
    plt.gca().invert_xaxis()
    plt.xlabel('λ (km)'); plt.ylabel('Amplitude')
    plt.grid(True, which='both', linestyle=':')

    # === 子標題：只顯示最大 λ ===
    lam_texts = []
    if mask1.any():
        lam_texts.append(f"λ1={int(round(lam[mask1].max()))}")
    if mask2.any():
        lam_texts.append(f"λ2={int(round(lam[mask2].max()))}")
    if mask3.any():
        lam_texts.append(f"λ3={int(round(lam[mask3].max()))}")
    lam_text = " | ".join(lam_texts) if lam_texts else "no exceed"

    plt.title(f'Wavelength: {var}\n({lam_text})')
    plt.legend(fontsize='x-small', loc='best')

plt.tight_layout()
plt.show()
