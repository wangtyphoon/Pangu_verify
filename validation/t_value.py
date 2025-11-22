import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 1. 载入所有频谱数据
data = np.load('all_spectral_data2024.npz')
variables = ['z500', 'q700', 't850', 'u850', 'v850']

# —— 请根据你的数据实际成员数修改下面两个值 —— 
n1, n2 = 1244, 1244  # 分别为 ERA5 和 Pangu 样本量

# 2. 基于波数的平均径向谱（含 ±1σ）并计算 t‑test
plt.figure(figsize=(20, 4))
for i, var in enumerate(variables, 1):
    # 从 NPZ 中取数组
    k       = data[f'k_{var}']              # shape (K,)
    m1      = data[f'era5_mean_{var}']      # ERA5 均值
    s1      = data[f'era5_std_{var}']       # ERA5 标准差
    m2      = data[f'pangu_mean_{var}']     # Pangu 均值
    s2      = data[f'pangu_std_{var}']      # Pangu 标准差

    # Welch t‑test（基于 summary stats）
    se      = np.sqrt(s1**2/n1 + s2**2/n2)  # 标准误差
    t_vals  = (m1 - m2) / se                # t 值
    df      = (s1**2/n1 + s2**2/n2)**2 / (
                (s1**4)/(n1**2*(n1-1)) +
                (s2**4)/(n2**2*(n2-1))
              )                            # 自由度
    p_vals  = 2 * stats.t.sf(np.abs(t_vals), df)  # 双侧 p 值

    # 打印显著差异的波数
    sig = p_vals < 0.05
    if np.any(sig):
        print(f"[{var}] 在 p<0.05 下显著差异的波数：")
        for kk, tt, pp in zip(k[sig], t_vals[sig], p_vals[sig]):
            print(f"  k={kk:.2f}, t={tt:.3f}, p={pp:.3e}")
    else:
        print(f"[{var}] 没有波数在 p<0.05 下显著差异")

    # 绘制 ±1σ 谱线
    plt.subplot(1, 5, i)
    plt.plot(k, m1, '-',  label='ERA5 μ')
    plt.plot(k, m2, '--', label='Pangu μ')
    plt.fill_between(k, np.clip(m1 - s1, 1e-12, None), m1 + s1,
                     alpha=0.25, linewidth=0, label='ERA5 ±1σ')
    plt.fill_between(k, np.clip(m2 - s2, 1e-12, None), m2 + s2,
                     alpha=0.25, linewidth=0, label='Pangu ±1σ')

    # 用星号标记统计显著点
    plt.scatter(k[sig], m2[sig],
                marker='*', s=40, color='k',
                label='p<0.05')

    plt.xscale('log'); plt.yscale('log')
    plt.title(f'Radial Spectrum (k): {var}')
    plt.xlabel('Wavenumber k'); plt.ylabel('Amplitude')
    plt.legend(fontsize='small'); plt.grid(True, which='both', linestyle=':')
plt.tight_layout()
plt.show()