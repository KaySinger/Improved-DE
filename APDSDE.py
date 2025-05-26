import numpy as np
from scipy.stats import norm, cauchy

class APDSDE:
    def __init__(self, func, bounds, pop_size=100, max_gen=1000, H=5, tol=1e-6):
        """
        APDSDE优化算法类
        后续改进的基础算法

        参数：
        func: 目标函数（最小化）
        bounds: 变量边界的列表，例如 [(min1, max1), (min2, max2), ...]
        pop_size: 初始种群大小
        max_gen: 最大迭代次数
        H: 历史记忆大小
        tol: 收敛精度，达到时提前终止
        N_min: 最小种群大小（默认18）
        """
        self.func = func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.N_init = pop_size
        self.N_min = 4
        self.max_gen = max_gen
        self.H = H
        self.tol = tol

        # 初始化历史记忆
        self.F_memory = [0.5] * H
        self.CR_memory = [0.5] * H
        self.hist_idx = 0

        # 初始化种群和存档
        self.N_current = self.N_init  # 当前种群大小
        self.archive_size = 2.6
        self.e = 0.5  # 存档比例系数（论文参数）
        self.a = 1.4  # F_w调整系数
        self.p = 0.11  # pBest比例
        self.pop = np.random.uniform(
            low=self.bounds[:, 0],
            high=self.bounds[:, 1],
            size=(self.N_init, self.dim)
        )
        self.fitness = np.apply_along_axis(self.func, 1, self.pop)
        self.archive = []
        self.iteration_log = []

    # 非线性种群缩减策略
    def _nonlinear_pop_size_reduction(self, gen):
        ratio = gen / self.max_gen

        return max(self.N_min, round(self.N_init - (self.N_init - self.N_min) * (ratio ** 2)))

    def _cosine_weight(self, parent, trial):
        """计算余弦相似度权重（公式22）"""
        dot_product = np.dot(parent, trial)
        norm_parent = np.linalg.norm(parent)
        norm_trial = np.linalg.norm(trial)
        if norm_parent == 0 or norm_trial == 0:
            return 0.0  # 避免除以零
        return dot_product / (norm_parent * norm_trial)

    def _calculate_SPG(self, gen):
        """计算策略选择概率SP_G（公式17）"""
        return 1 / (1 + np.exp(1 - (gen / self.max_gen) ** 2))

    def compute_Amean(self):
        if len(self.archive) == 0:
            return np.zeros(self.dim)
        m = max(1, round(self.e * len(self.archive)))
        archive_fitness = np.apply_along_axis(self.func, 1, self.archive)
        sorted_indices = np.argsort(archive_fitness)[:m]  # 假设你有 archive_fitness 存档适应度
        X_A = np.array([self.archive[i] for i in sorted_indices])
        weights = np.log(m + 0.5) - np.log(np.arange(1, m + 1))
        weights /= weights.sum()
        X_Amean = np.dot(weights, X_A)
        return X_Amean

    def mutantion(self, F, i, gen, SPG, X_Amean):
        # current-to-pbest-w/1变异策略
        p_best_size = max(2, int(self.N_current * self.p))
        p_best_idx = np.random.choice(np.argsort(self.fitness)[:p_best_size])
        p_best = self.pop[p_best_idx]

        # 选择r1
        r1 = self.pop[np.random.choice(np.delete(np.arange(self.N_current), i), 1, replace=False)].flatten()

        r2_idx = np.random.choice(np.arange(self.N_current + len(self.archive)), 1, replace=False)[0]  # 随机选择一个下标
        if r2_idx >= self.N_current:  # 如果下标大于种群大小，则从存档中选择
            r2_idx -= self.N_current
            r2 = self.archive[r2_idx]
        else:  # 否则从当前种群中选择
            r2 = self.pop[r2_idx].flatten()

        Fw = F * (0.7 + (self.a - 0.7) * gen / self.max_gen)

        # 变异操作
        if np.random.rand() < SPG:
            mutant = self.pop[i] + Fw * (p_best - self.pop[i]) + F * (r1 - r2)
        else:
            mutant = self.pop[i] + Fw * (X_Amean - self.pop[i]) + F * (r1 - r2)

        return mutant

    def cross(self, mutant, i, CR):
        cross_chorm = self.pop[i].copy()
        j = np.random.randint(0, self.dim)  # 随机选择一个维度
        for k in range(self.dim):  # 对每个维度进行交叉
            if np.random.rand() < CR or k == j:  # 如果随机数小于交叉率或者维度为j
                cross_chorm[k] = mutant[k]  # 交叉
                # 边界处理(反射)
                if cross_chorm[k] < self.bounds[k, 0]:
                    cross_chorm[k] = np.minimum(self.bounds[k, 1], 2 * self.bounds[k, 0] - cross_chorm[k])
                elif cross_chorm[k] > self.bounds[k, 1]:
                    cross_chorm[k] = np.maximum(self.bounds[k, 0], 2 * self.bounds[k, 1] - cross_chorm[k])
        return cross_chorm

    def optimize(self):
        """执行优化过程"""
        for gen in range(self.max_gen):
            S_F, S_CR, S_weights = [], [], []
            new_pop = []
            new_fitness = []

            # 记录当前最优值
            best_val = np.min(self.fitness)
            self.iteration_log.append(best_val)
            # 收敛检查
            if self.tol is not None and best_val <= self.tol:
                print(f"Converged at generation {gen + 1}: {best_val}")
                break
            elif gen >= self.max_gen - 1:
                print(f"Converged at generation {gen + 1}: {best_val}")
                break

            SPG = self._calculate_SPG(gen)

            for i in range(self.N_current):
                # 从历史记忆随机选择索引
                r = np.random.randint(0, self.H)

                # 生成F和CR
                CR = np.clip(np.random.normal(self.CR_memory[r], 0.1), 0.0, 1.0)
                F = cauchy.rvs(loc=self.F_memory[r], scale=0.1)
                while F > 1 or F < 0:
                    if F < 0:
                        F = cauchy.rvs(loc=self.F_memory[r], scale=0.1)
                    else:
                        F = 1

                # 变异
                X_Amean = self.compute_Amean()
                mutant = self.mutantion(F, i, gen, SPG, X_Amean)

                # 交叉操作
                trial = self.cross(mutant, i, CR)
                trial_fitness = self.func(trial)

                # 贪心选择
                if trial_fitness < self.fitness[i]:
                    new_pop.append(trial)
                    new_fitness.append(trial_fitness)
                    weight = self._cosine_weight(self.pop[i], trial)
                    # 记录成功参数和适应度改进量
                    S_F.append(F)
                    S_CR.append(CR)
                    S_weights.append(weight)
                    # 更新存档
                    self.archive.append(self.pop[i].copy())
                    if len(self.archive) > self.archive_size * self.N_current:
                        # 如果存档超过种群大小，则随机删除一些
                        for o in range(int(len(self.archive) - self.archive_size * self.N_current)):
                            self.archive.pop(np.random.randint(0, len(self.archive)))
                else:
                    new_pop.append(self.pop[i])
                    new_fitness.append(self.fitness[i])

            # --- LPSR关键步骤 ---
            # 更新种群大小
            new_N = self._nonlinear_pop_size_reduction(gen)

            # 选择适应度最好的个体保留
            combined_fitness = np.array(new_fitness)
            survivor_indices = np.argsort(combined_fitness)[:new_N]

            # 更新种群和适应度
            self.pop = np.array([new_pop[i] for i in survivor_indices])
            self.fitness = np.array([new_fitness[i] for i in survivor_indices])
            self.N_current = new_N

            # 更新历史记忆（加权Lehmer均值）
            if np.any(S_F):
                F_lehmer = np.sum(np.array(S_F) ** 2 * S_weights) / np.sum(np.array(S_F) * S_weights)
                self.F_memory[self.hist_idx] = F_lehmer

            # CR 部分
            if np.any(S_CR):
                CR_lehmer = np.sum(np.array(S_CR) ** 2 * S_weights) / np.sum(np.array(S_CR) * S_weights)
                self.CR_memory[self.hist_idx] = CR_lehmer

            # 移动历史指针
            self.hist_idx = (self.hist_idx + 1) % self.H

            print(f"Iteration {gen + 1}, Pop Size: {self.N_current}, Best: {np.min(self.fitness)}")

        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx], self.fitness[best_idx], self.iteration_log