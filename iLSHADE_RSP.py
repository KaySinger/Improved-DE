import numpy as np
from scipy.stats import cauchy

class iLSHADE_RSP:
    def __init__(self, func, bounds, pop_size=None, max_gen=None, H=None, tol=1e-6):
        """
        iL-SHADE-RSP优化算法类
        基于LSHADE-RSP的改进型

        参数：
        func: 目标函数（最小化）
        bounds: 变量边界的列表，例如 [(min1, max1), (min2, max2), ...]
        pop_size: 种群大小
        max_gen: 最大迭代次数
        H: 历史记忆大小
        tol: 收敛精度，达到时提前终止
        """
        self.func = func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.max_gen = max_gen
        self.H = H
        self.tol = tol
        self.k = 3
        self.p_j = 0.2 # 跳跃率参数

        # 自动计算初始种群大小
        self.N_init = int(round(np.sqrt(self.dim) * np.log(self.dim) * 25)) if pop_size is None else pop_size
        self.N_min = 4  # 最小种群大小
        self.pop_size = self.N_init  # 当前种群大小
        self.archive_size = 1.4

        # 初始化历史记忆
        self.F_memory = [0.3] * (self.H - 1) + [0.9]
        self.CR_memory = [0.8] * (self.H - 1) + [0.9]  # 初始CR=0.8
        self.hist_idx = 0

        # 初始化种群和存档
        self.pop = np.random.uniform(
            low=self.bounds[:, 0],
            high=self.bounds[:, 1],
            size=(self.N_init, self.dim)
        )
        self.fitness = np.apply_along_axis(self.func, 1, self.pop)
        self.archive = []
        self.iteration_log = []

    def _linear_pop_size_reduction(self, gen):
        """线性种群缩减策略"""
        return max(
            self.N_min,
            int(round(
                self.N_init - (self.N_init - self.N_min) * gen / self.max_gen
            ))
        )

    def mutation(self, i, F, gen):
        # current-to-pbest/1变异策略
        p_min, p_max = 0.085, 0.17
        p_i = p_min + (gen / self.max_gen) * (p_max - p_min)
        p_best_size = max(2, int(self.pop_size * p_i))
        sorted_idx = np.argsort(self.fitness)

        # 选择p_best个体
        p_best_indices = np.argsort(self.fitness)[:p_best_size]
        p_best_idx = np.random.choice(p_best_indices)
        p_best = self.pop[p_best_idx]

        # 计算排名及概率
        ranks = np.zeros(self.pop_size)
        for rank_order, idx in enumerate(sorted_idx):
            ranks[idx] = 3 * (self.pop_size - (rank_order + 1)) + 1
        # 计算概率
        prs = ranks / np.sum(ranks)

        combined_pop = np.vstack([self.pop, np.array(self.archive)]) if self.archive else self.pop
        combined_size = len(combined_pop)

        # 从种群或存档中选择 r1 和 r2（基于RSP策略）
        # 选择r1
        candidates_r1 = [idx for idx in range(self.pop_size) if idx != i and idx != p_best_idx]
        prs_r1 = prs[candidates_r1]
        prs_r1 /= np.sum(prs_r1)
        r1_idx = np.random.choice(candidates_r1, p=prs_r1)
        r1 = self.pop[r1_idx]

        # 选择r2
        candidates_r2 = [idx for idx in range(combined_size) if idx != i and idx != p_best_idx and idx != r1_idx]
        r2_idx = np.random.choice(candidates_r2)
        r2 = combined_pop[r2_idx]

        # 根据当前代数调整F的权重
        if gen < 0.2 * self.max_gen:
            Fw = 0.7 * F
        elif gen < 0.4 * self.max_gen:
            Fw = 0.8 * F
        else:
            Fw = 1.2 * F

        # 生成变异向量
        mutant = self.pop[i] + Fw * (p_best - self.pop[i]) + F * (r1 - r2)
        return mutant

    def cross(self, mutant, parent, CR):
        cross_chorm = parent.copy()
        j = np.random.randint(0, self.dim)  # 随机选择一个维度
        for k in range(self.dim):  # 对每个维度进行交叉
            if np.random.rand() < CR or k == j:  # 如果随机数小于交叉率或者维度为j
                cross_chorm[k] = mutant[k]  # 交叉
                # 边界处理
                if cross_chorm[k] > self.bounds[k, 1]:
                    cross_chorm[k] = (self.bounds[k, 1] + parent[k]) / 2
                elif cross_chorm[k] < self.bounds[k, 0]:
                    cross_chorm[k] = (self.bounds[k, 0] + parent[k]) / 2
        return cross_chorm

    def _cauchy_perturbation(self, target):
        """柯西扰动操作"""
        return np.random.standard_cauchy(size=target.shape) * 0.1 + target

    def optimize(self):
        """执行优化过程"""
        for gen in range(self.max_gen):
            S_F, S_CR, S_weights = [], [], []
            new_pop = []
            new_fitness = []

            # 记录当前最优值
            best_val = np.min(self.fitness)
            self.iteration_log.append(best_val)
            if self.tol is not None and best_val <= self.tol:
                print(f"Converged at generation {gen} with precision {best_val}")
                break
            elif gen >= self.max_gen - 1:
                print(f"Converged at generation {gen} with precision {best_val}")
                break

            for i in range(self.pop_size):
                r = np.random.randint(0, self.H)

                # 生成F和CR（jSO的调整）
                if np.isnan(self.CR_memory[r]):
                    CR = 0
                else:
                    CR = np.clip(np.random.normal(self.CR_memory[r], 0.1), 0.0, 1.0)
                F = cauchy.rvs(loc=self.F_memory[r], scale=0.1)
                while F > 1 or F < 0:
                    if F < 0:
                        F = cauchy.rvs(loc=self.F_memory[r], scale=0.1)
                    else:
                        F = 1

                # jSO的CR阶段式调整（根据当前评估次数）
                if gen < 0.25 * self.max_gen:
                    CR = max(CR, 0.7)
                elif gen < 0.5 * self.max_gen:
                    CR = max(CR, 0.6)

                # jSO的F时变调整
                if gen < 0.6 * self.max_gen and F > 0.7:
                    F = 0.7

                # ========================= 变异策略 =========================
                mutant = self.mutation(i, F, gen)

                # ========================= 交叉操作 =========================
                if np.random.rand() < self.p_j:
                    # 执行 mutant 与 Cauchy扰动parent 的交叉
                    cauchy_parent = self._cauchy_perturbation(self.pop[i])
                    trial = self.cross(mutant, cauchy_parent, CR)
                else:
                    # 执行 mutant 与原parent 的交叉
                    trial = self.cross(mutant, self.pop[i], CR)
                trial_fitness = self.func(trial)

                # 贪心选择
                if trial_fitness < self.fitness[i]:
                    new_pop.append(trial)
                    new_fitness.append(trial_fitness)
                    # 记录成功参数和权重
                    S_F.append(F)
                    S_CR.append(CR)
                    S_weights.append(np.abs(self.fitness[i] - trial_fitness))
                    # 更新适应度和存档
                    self.fitness[i] = trial_fitness
                    self.archive.append(self.pop[i].copy())
                    if len(self.archive) > self.archive_size * self.pop_size:
                        # 如果存档超过种群大小，则随机删除一些
                        for o in range(int(len(self.archive) - self.archive_size * self.pop_size)):
                            self.archive.pop(np.random.randint(0, len(self.archive)))
                else:
                    new_pop.append(self.pop[i])
                    new_fitness.append(self.fitness[i])

            # ======================== 种群和记忆更新 ========================
            # 更新种群大小
            new_N = self._linear_pop_size_reduction(gen)

            # 选择适应度最好的个体保留
            combined_fitness = np.array(new_fitness)
            survivor_indices = np.argsort(combined_fitness)[:new_N]

            # 更新种群和适应度
            self.pop = np.array([new_pop[i] for i in survivor_indices])
            self.fitness = np.array([new_fitness[i] for i in survivor_indices])
            self.pop_size = new_N

            # 更新历史记忆（加权Lehmer均值）
            if np.any(S_F):
                F_lehmer = np.sum(np.array(S_F) ** 2 * S_weights) / np.sum(np.array(S_F) * S_weights)
                self.F_memory[self.hist_idx] = (F_lehmer + self.F_memory[self.hist_idx]) / 2

            # CR 部分
            if np.any(S_CR):
                if np.max(S_CR) == 0:
                    self.CR_memory[self.hist_idx] = np.nan  # 置为 ⊥，表示未来采样的 CR 必为 0
                else:
                    CR_lehmer = np.sum(np.array(S_CR) ** 2 * S_weights) / np.sum(np.array(S_CR) * S_weights)
                    if self.CR_memory[self.hist_idx] is not np.nan:
                        self.CR_memory[self.hist_idx] = (CR_lehmer + self.CR_memory[self.hist_idx]) / 2
                    else:
                        self.CR_memory[self.hist_idx] = CR_lehmer  # 如果原来是 None，就直接赋值

            # 移动历史指针
            self.hist_idx = (self.hist_idx + 1) % (self.H - 1)

            # 输出迭代信息
            print(f"Iteration {gen + 1}, Best Fitness: {np.min(self.fitness)}, pop_size: {self.pop_size}")

        # 返回最优解
        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx], self.fitness[best_idx], self.iteration_log