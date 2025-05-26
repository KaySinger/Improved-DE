import numpy as np
from scipy.stats import cauchy

class mLSHADE_SPACMA:
    def __init__(self, func, bounds, pop_size=100, max_gen=1000, H=5, tol=1e-6):
        """
        mL-SHADE-SPACMA优化算法类

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
        self.rho = 0.11

        # 初始化历史记忆
        self.F_memory = [0.5] * H
        self.CR_memory = [0.5] * H
        self.hist_idx = 0

        # 混合算法参数
        self.FCP_memory = [0.5] * H  # First Class Probability (LSHADE分配概率)
        self.c = 0.8  # 学习率

        # 初始化种群和存档
        self.N_current = self.N_init  # 当前种群大小
        self.archive_size = 1.4 * self.N_init
        self.pop = np.random.uniform(
            low=self.bounds[:, 0],
            high=self.bounds[:, 1],
            size=(self.N_init, self.dim)
        )
        self.fitness = np.apply_along_axis(self.func, 1, self.pop)
        self.archive = []
        self.iteration_log = []

        # CMA-ES参数初始化
        self.sigma = 0.5
        self.xmean = np.mean(self.pop, axis=0)
        self.mu = self.N_current // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = 1 / np.sum(self.weights ** 2)

        # 协方差矩阵参数
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.dim + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs

        # 协方差矩阵
        self.B = np.eye(self.dim)
        self.D = np.ones(self.dim)
        self.C = self.B @ np.diag(self.D ** 2) @ self.B.T
        self.invsqrtC = np.eye(self.dim)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.eigeneval = 0
        self.chiN = self.dim ** 0.5 * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))

    def _linear_pop_size_reduction(self, gen):
        """线性种群缩减策略"""
        return max(
            self.N_min,
            int(round(
                self.N_init - (self.N_init - self.N_min) * gen / self.max_gen
            ))
        )

    def _elite_archive_update(self, candidate):
        """精英存档更新机制"""
        if len(self.archive) < self.archive_size:
            self.archive.append(candidate)
        else:
            # 找到存档中最差个体
            archive_fitness = np.array([self.func(ind) for ind in self.archive])
            worst_idx = np.argmax(archive_fitness)
            # 只保留更优解
            if self.func(candidate) < archive_fitness[worst_idx]:
                self.archive[worst_idx] = candidate

    def generate_mutant(self, i, F, FCP):
        """生成变异个体（包含CMA-ES混合逻辑）"""
        if np.random.rand() < FCP:
            # 采用LSHADE变异策略
            p_i = 0.11
            p_best_size = max(2, int(self.N_current * p_i))

            p_best_indices = np.argsort(self.fitness)[:p_best_size]
            p_best_idx = np.random.choice(p_best_indices)
            p_best = self.pop[p_best_idx]

            # 计算排名及概率
            ranks = np.zeros(self.N_current)
            for rank_order, idx in enumerate(np.argsort(self.fitness)):
                ranks[idx] = 3 * (self.N_current - (rank_order + 1)) + 1
            # 计算概率
            prs = ranks / np.sum(ranks)

            # 从种群或存档中选择 r1 和 r2（基于RSP策略）
            candidates_r1 = [idx for idx in range(self.N_current) if idx != i and idx != p_best_idx]
            prs_r1 = prs[candidates_r1]
            prs_r1 /= np.sum(prs_r1)
            r1_idx = np.random.choice(candidates_r1, p=prs_r1)
            r1 = self.pop[r1_idx]

            r2_idx = np.random.choice(np.arange(self.N_current + len(self.archive)), 1, replace=False)[0]  # 随机选择一个下标
            if r2_idx >= self.N_current:  # 如果下标大于种群大小，则从存档中选择
                r2_idx -= self.N_current
                r2 = self.archive[r2_idx]
            else:  # 否则从当前种群中选择
                r2 = self.pop[r2_idx].flatten()

            mutant = self.pop[i] + F * (p_best - self.pop[i]) + F * (r1 - r2)
            FCP_judge = 1
        else:
            # CMA-ES变异策略
            z = self.B @ (self.D * np.random.randn(self.dim))
            mutant = self.xmean + self.sigma * z
            FCP_judge = 2

        return mutant, FCP_judge

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

    def optimize(self):
        """执行优化过程"""
        for gen in range(self.max_gen):
            S_F, S_CR, S_weights = [], [], []
            S_FCP = []
            delta_alg1, delta_alg2 = [], []
            new_pop = []
            new_fitness = []

            # 记录当前最优值
            best_val = np.min(self.fitness)
            self.iteration_log.append(best_val)
            # 收敛检查
            if self.tol is not None and best_val <= self.tol:
                print(f"Converged at generation {gen + 1}: {best_val:.6e}")
                break
            elif gen >= self.max_gen - 1:
                print(f"Converged at generation {gen + 1}: {best_val:.6e}")
                break

            # if gen < self.max_gen // 2 and gen > 0:
            #     # 淘汰机制
            #     PE_m = int(np.ceil(self.rho * self.N_current))
            #     sorted_indices = np.argsort(self.fitness)
            #     eliminated_indices = sorted_indices[-PE_m:]
            #
            #     # 生成新个体
            #     best1 = self.pop[sorted_indices[0]]
            #     best2 = self.pop[sorted_indices[1]]
            #     new_individuals = [
            #         best1 + np.random.rand() * (best1 - best2)
            #         for _ in range(PE_m)
            #     ]
            #
            #     # 替换淘汰个体
            #     self.pop[eliminated_indices] = new_individuals
            #     self.fitness[eliminated_indices] = np.apply_along_axis(self.func, 1, new_individuals)

            for i in range(self.N_current):
                # 参数生成阶段
                r = np.random.randint(0, self.H)

                CR = np.clip(np.random.normal(self.CR_memory[r], 0.1), 0, 1)

                # SPA机制：前半段固定F范围，后半段自适应
                if gen < self.max_gen / 2:
                    F = 0.5 + 0.1 * np.random.rand()
                else:
                    F = cauchy.rvs(loc=self.F_memory[r], scale=0.1)
                    while F > 1 or F < 0:
                        if F < 0:
                            F = cauchy.rvs(loc=self.F_memory[r], scale=0.1)
                        else:
                            F = 1

                FCP = self.FCP_memory[r]

                # 生成变异个体
                mutant, FCP_judge = self.generate_mutant(i, F, FCP)
                trial = self.cross(mutant, self.pop[i], CR)
                trial_fitness = self.func(trial)

                # 选择操作
                if trial_fitness < self.fitness[i]:
                    new_pop.append(trial)
                    new_fitness.append(trial_fitness)
                    S_F.append(F)
                    S_CR.append(CR)
                    S_FCP.append(FCP)
                    S_weights.append(self.fitness[i] - trial_fitness)
                    if FCP_judge == 1:
                        delta_alg1.append(self.fitness[i] - trial_fitness)
                    else:
                        delta_alg2.append(self.fitness[i] - trial_fitness)
                    self._elite_archive_update(self.pop[i].copy())
                else:
                    new_pop.append(self.pop[i])
                    new_fitness.append(self.fitness[i])

            # 更新历史记忆
            if np.any(S_F):
                F_lehmer = np.sum(np.array(S_F) ** 2 * S_weights) / np.sum(np.array(S_F) * S_weights)
                self.F_memory[self.hist_idx] = F_lehmer

            # CR 部分
            if np.any(S_CR):
                if gen < self.max_gen / 2:
                    CR_lehmer = np.sum(np.array(S_CR) ** 2 * S_weights) / np.sum(np.array(S_CR) * S_weights)
                    self.CR_memory[self.hist_idx] = CR_lehmer

            if np.any(S_FCP):
                # 更新混合概率
                total_improve = np.sum(delta_alg1) + np.sum(delta_alg2)
                ratio = np.sum(delta_alg1) / total_improve
                # 平滑更新公式
                self.FCP_memory[self.hist_idx] = np.clip(self.c * self.FCP_memory[self.hist_idx] + (1 - self.c) * ratio, 0.2, 0.8)

            self.hist_idx = (self.hist_idx + 1) % self.H

            # 种群缩减
            new_N = self._linear_pop_size_reduction(gen)
            survivor_indices = np.argsort(new_fitness)[:new_N]
            self.pop = np.array([new_pop[i] for i in survivor_indices])
            self.fitness = np.array([new_fitness[i] for i in survivor_indices])
            self.N_current = new_N

            self.mu = max(1, self.N_current // 2)  # 防止mu为0
            self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
            self.weights /= np.sum(self.weights)  # 归一化权重
            self.mueff = 1 / np.sum(self.weights ** 2)  # 更新有效种群大小

            # 更新CMA-ES参数
            popold = np.copy(self.pop)  # CMA-ES部分用的旧种群

            # 按适应度排序
            popindex = np.argsort(self.fitness)
            if np.any(delta_alg2):
                xold = self.xmean.copy()
                self.xmean = np.dot(popold[popindex[:self.mu]].T, self.weights)

                # 演化路径ps更新
                y = (self.xmean - xold) / self.sigma
                self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * np.dot(self.invsqrtC, y)

                # hsig判别
                ps_norm_sq = np.sum(self.ps ** 2)
                hsig_cond = (1 - (1 - self.cs) ** (2 * (gen + 1) / self.N_current))
                hsig = ps_norm_sq / (self.dim * hsig_cond) < (2 + 4 / (self.dim + 1))

                # 演化路径pc更新
                self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y

                # 协方差矩阵C更新
                artmp = (popold[popindex[:self.mu]] - xold) / self.sigma  # mu个差分向量
                self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) + self.cmu * np.dot((artmp.T * self.weights), artmp)

                # 步长sigma更新
                self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))

                # C矩阵特征分解更新，O(d^2)复杂度
                if (gen + 1) - self.eigeneval > self.N_current / (self.c1 + self.cmu) / self.dim / 10:
                    self.eigeneval = (gen + 1)
                    self.C = np.triu(self.C) + np.triu(self.C, 1).T  # 保证对称性
                    if np.any(np.isnan(self.C)) or np.any(~np.isfinite(self.C)) or not np.isrealobj(self.C):
                        print("C matrix invalid, skipping CMA update")
                        continue  # 出问题跳过CMA-ES更新
                    D2, B = np.linalg.eigh(self.C)
                    if np.any(D2 < 0):
                        D2[D2 < 0] = 1e-10  # 数值安全
                    self.D = np.sqrt(D2)
                    self.B = B
                    self.invsqrtC = np.dot(self.B, np.dot(np.diag(self.D ** -1), self.B.T))

            print(f"Iteration {gen + 1}, Pop Size: {self.N_current}, Best: {np.min(self.fitness)}, FCP: {self.FCP_memory}")

        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx], self.fitness[best_idx], self.iteration_log