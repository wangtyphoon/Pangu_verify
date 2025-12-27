# =========================
# 4) Pangu / ERA5 功率譜比值 vs. 波數 k（含 ±1σ）
# =========================
import numpy as np
import matplotlib.pyplot as plt

R = 6371.0  # km

def cell_area_equal_area(lat2d, lon2d):
    """
    規則經緯格網的像素面積 A_ij（km^2，比例精確）。
    用條帶面積：A ≈ R^2 * (sin(phi_n) - sin(phi_s)) * dlon
    """
    if lat2d.ndim == 1 and lon2d.ndim == 1:
        lat2d, lon2d = np.meshgrid(lat2d, lon2d, indexing='ij')

    lat1d = lat2d[:, 0]
    lon1d = lon2d[0, :]

    dlat = np.deg2rad(np.abs(np.diff(lat1d).mean()))
    dlon = np.deg2rad(np.abs(np.diff(lon1d).mean()))

    phi_c = np.deg2rad(lat1d)
    phi_n = phi_c + 0.5 * dlat
    phi_s = phi_c - 0.5 * dlat

    strip = (R**2) * (np.sin(phi_n) - np.sin(phi_s)) * dlon  # 每條緯帶面積
    A = np.repeat(strip[:, None], lon1d.size, axis=1)        # 擴到 2D
    return A  # km^2

def effective_grid_km_equal_area(lat2d, lon2d):
    A = cell_area_equal_area(lat2d, lon2d)
    s_eff = np.sqrt(A.mean())            # km
    return s_eff
lat = np.load("../lat.npy")   # 可為 1D 或 2D
lon = np.load("../lon.npy")   # 可為 1D 或 2D

# LAT_SUB, LON_SUB 是你子區塊的經緯度 2D 陣列（與資料同大小）
s_eff = effective_grid_km_equal_area(lat, lon)  # 平均格距 (km)
H_sub, W_sub = lat.shape
L_eff = s_eff * min(H_sub, W_sub)                       # 有效邊長 (km)

data = np.load('all_energy_data.npz')
variables = ['z500', 'q700', 't850', 'u850', 'v850']

eps = 1e-12  # 避免除以零
# =========================
# 4) Pangu / ERA5 功率譜比值 vs. 波數 k（含 ±1σ）
# =========================
plt.figure(figsize=(20, 4))   # 單列多欄，圖拉寬
for i, var in enumerate(variables, 1):
    k       = data[f'k_{var}']
    A_e     = data[f'era5_mean_{var}']
    sA_e    = data[f'era5_std_{var}']
    A_f     = data[f'pangu_mean_{var}']
    sA_f    = data[f'pangu_std_{var}']

    P_e  = np.clip(A_e, eps, None)
    P_f  = np.clip(A_f, eps, None)
    sP_e = np.clip(sA_e, 0, None)
    sP_f = np.clip(sA_f, 0, None)

    R    = np.clip(P_f/(P_e + eps), eps, None)
    rel_e = np.clip(sP_e/(P_e + eps), 0, None)
    rel_f = np.clip(sP_f/(P_f + eps), 0, None)
    sR    = R * np.sqrt(rel_e**2 + rel_f**2)

    ax = plt.subplot(1, len(variables), i)  # ← 單排
    ax.plot(k, R, '-', label='Pangu/ERA5')
    ax.axhline(1.0, linestyle=':', linewidth=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e-1, 1e1)
    ax.set_xlabel('Wavenumber k')
    ax.set_ylabel('Power Ratio (Pangu/ERA5)')
    ax.set_title(f'{var}')
    ax.grid(True, which='both', linestyle=':')
    ax.legend(fontsize='small')

plt.tight_layout()
plt.show()


# =========================
# 5) Pangu / ERA5 功率譜比值 vs. 波長 λ（含 ±1σ）
# =========================
plt.figure(figsize=(20, 4))
for i, var in enumerate(variables, 1):
    k    = data[f'k_{var}']
    A_e  = data[f'era5_mean_{var}']
    sA_e = data[f'era5_std_{var}']
    A_f  = data[f'pangu_mean_{var}']
    sA_f = data[f'pangu_std_{var}']

    mask = k > 0
    lam  = L_eff / k[mask]

    P_e  = np.clip(A_e[mask], eps, None)
    P_f  = np.clip(A_f[mask], eps, None)
    sP_e = np.clip(sA_e[mask], 0, None)
    sP_f = np.clip(sA_f[mask], 0, None)

    R    = np.clip(P_f/(P_e + eps), eps, None)
    rel_e = np.clip(sP_e/(P_e + eps), 0, None)
    rel_f = np.clip(sP_f/(P_f + eps), 0, None)
    sR    = R * np.sqrt(rel_e**2 + rel_f**2)

    ax = plt.subplot(1, len(variables), i)  # ← 單排
    ax.plot(lam, R, '-', label='Pangu/ERA5')
    ax.axhline(1.0, linestyle=':', linewidth=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e-1, 1e1)
    ax.set_xlim(lam.max(), lam.min())
    ax.set_xlabel('Wavelength λ (km)')
    ax.set_ylabel('Power Ratio (Pangu/ERA5)')
    ax.set_title(f'{var}')
    ax.grid(True, which='both', linestyle=':')
    ax.legend(fontsize='small', loc='best')

plt.tight_layout()
plt.show()
