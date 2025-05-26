import numpy as np
from numba import jit
from matplotlib import pyplot as plt
from scipy.stats import ortho_group
from LSHADE import LSHADE
from LSHADE_cnEpsin import LSHADE_cnEpSin
from iLSHADE_RSP import iLSHADE_RSP
from mLSHADE_SPACMA import mLSHADE_SPACMA
from APSM_jSO import APSM_JSO
from LSHADE_SPACMA import LSHADE_SPACMA
from mLSHADE_RL import mLSHADE_RL
from ACD_DE import ACD_DE
from APDSDE import APDSDE
from cosineweight import LSHADE2

# ========================= 固定参数生成 =========================
np.random.seed(42)  # 固定随机种子保证可重复性
D = 100  # 10维问题

# 生成CEC2017 F1的固定偏移和旋转矩阵 (官方比赛中应为预定义值)
o = np.random.uniform(-100, 100, D)
R = ortho_group.rvs(D)


# ========================= CEC2017 F1目标函数 =========================
@jit(nopython=True)
def cec2017_f1(x):
    """CEC2017 F1: Shifted and Rotated Bent Cigar Function"""
    z = np.dot(R, x - o)
    return z[0] ** 2 + 1e6 * np.sum(z[1:] ** 2) + 100  # 原始定义，最优值100


# ========================= 优化参数设置 =========================
bounds = [(-100, 100)] * D  # 搜索范围调整为[-100, 100]
max_iter = 20 * D  # 最大迭代次数
pop_size = 18 * D  # 种群大小
optimal_value = 100  # 理论最优值


# ========================= 修改后的优化算法接口 =========================
def run_optimizer(optimizer_name, objective, bounds, max_iter, pop_size):
    """运行优化器并返回与最优值的差距历史"""

    if optimizer_name == "L-SHADE":
        # 假设L_SHADE返回适应度历史记录
        result = LSHADE(objective, bounds=bounds, pop_size=pop_size, max_gen=max_iter, H=6, tol=None)
        _, _, history = result.optimize()
        return np.array(history) - optimal_value  # 返回与最优值的差距

    elif optimizer_name == "L-SHADE-cnEpsin":
        # 假设L_SHADE返回适应度历史记录
        result = LSHADE_cnEpSin(objective, bounds=bounds, pop_size=pop_size, max_gen=max_iter, H=5, tol=None)
        _, _, history = result.optimize()
        return np.array(history) - optimal_value  # 返回与最优值的差距

    elif optimizer_name == "iL-SHADE-RSP":
        # 假设L_SHADE返回适应度历史记录
        result = iLSHADE_RSP(objective, bounds=bounds, pop_size=pop_size, max_gen=max_iter, H=5, tol=None)
        _, _, history = result.optimize()
        return np.array(history) - optimal_value  # 返回与最优值的差距

    elif optimizer_name == "mL-SHADE-SPACMA":
        # 假设L_SHADE返回适应度历史记录
        result = mLSHADE_SPACMA(objective, bounds=bounds, pop_size=pop_size, max_gen=max_iter, H=5, tol=None)
        _, _, history = result.optimize()
        return np.array(history) - optimal_value  # 返回与最优值的差距

    elif optimizer_name == "APSM-jSO":
        # 假设L_SHADE返回适应度历史记录
        result = APSM_JSO(objective, bounds=bounds, pop_size=pop_size, max_gen=max_iter, H=6, tol=None)
        _, _, history = result.optimize()
        return np.array(history) - optimal_value  # 返回与最优值的差距

    elif optimizer_name == "mL-SHADE-RL":
        # 假设L_SHADE返回适应度历史记录
        result = mLSHADE_RL(objective, bounds=bounds, pop_size=pop_size, max_gen=max_iter, H=5, tol=None)
        _, _, history = result.optimize()
        return np.array(history) - optimal_value  # 返回与最优值的差距

    elif optimizer_name == "ACD-DE":
        # 假设L_SHADE返回适应度历史记录
        result = ACD_DE(objective, bounds=bounds, pop_size=pop_size, max_gen=max_iter, H=5, tol=None)
        _, _, history = result.optimize()
        return np.array(history) - optimal_value  # 返回与最优值的差距

    elif optimizer_name == "APDSDE":
        # 假设L_SHADE返回适应度历史记录
        result = APDSDE(objective, bounds=bounds, pop_size=pop_size, max_gen=max_iter, H=6, tol=None)
        _, _, history = result.optimize()
        return np.array(history) - optimal_value  # 返回与最优值的差距

    elif optimizer_name == "L-SHADE-SPACMA":
        # 假设L_SHADE返回适应度历史记录
        result = LSHADE_SPACMA(objective, bounds=bounds, pop_size=pop_size, max_gen=max_iter, H=5, tol=None)
        _, _, history = result.optimize()
        return np.array(history) - optimal_value  # 返回与最优值的差距


# ========================= 执行优化比较 =========================
optimizers = ["L-SHADE", "iL-SHADE-RSP", "mL-SHADE-SPACMA", "APSM-jSO", "mL-SHADE-RL", "ACD-DE", "APDSDE"]

results = {}
for opt_name in optimizers:
    print(f"Running {opt_name}...")
    gap_history = run_optimizer(opt_name, cec2017_f1, bounds, max_iter, pop_size)
    results[opt_name] = gap_history
    print(f"{opt_name} completed. Final gap: {gap_history[-1]}")

# ========================= 结果可视化 =========================
plt.figure(figsize=(10, 6))
for opt_name, gaps in results.items():
    plt.plot(gaps, label=opt_name, lw=2)

plt.yscale('log')
plt.xlabel('Function Evaluations')
plt.ylabel('Fitness Gap (log scale)')
plt.title('Optimization Gap Comparison on CEC2017 F1')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()