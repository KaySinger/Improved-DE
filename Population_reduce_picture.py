import numpy as np
import matplotlib.pyplot as plt

# 参数设置
P_init = 1000     # 初始种群
P_min  = 4       # 最小种群
G_max  = 1000    # 最大迭代次数
gen = 0
g_over_G = np.linspace(0, 1, G_max)
P_linear = []
P_nonlinear = []
P_hybrid = []

for gen in range(G_max):
    cutoff = int(0.66 * G_max)
    P_mid = int(0.33 * P_init)
    P_linear.append(int(round(P_init - (P_init - P_min) * (gen / G_max))))
    P_nonlinear.append(int(round(P_init - (P_init - P_min) * (gen / G_max)**2)))
    if gen < cutoff:
        P_hybrid.append(int(round(P_init - (P_init - P_mid) * (gen / cutoff) ** 2)))
    else:
        P_hybrid.append(int(round(P_mid - (P_mid - P_min) * ((gen - cutoff) / (G_max - cutoff)))))

# 绘制
plt.figure(figsize=(10, 6))
plt.plot(g_over_G, P_linear, label='Linear Reduction', linestyle='--')
plt.plot(g_over_G, P_nonlinear, label='Parabolic Reduction', linestyle='-.')
plt.plot(g_over_G, P_hybrid, label='ACD-DE Hybrid Reduction', linewidth=2)

plt.xlabel('g / G', fontsize=13)
plt.ylabel('Population Size', fontsize=13)
plt.title('Population Size Reduction Schemes', fontsize=14)
plt.grid(True, linestyle=':')
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()