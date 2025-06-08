import numpy as np
from scipy.stats import norm, cauchy
from sklearn.cluster import KMeans

class ACD_DE:
    def __init__(self, func, bounds, pop_size=100, max_gen=1000, H=5, tol=1e-6):
        """
        ACD-DE优化算法类
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
        self.N_min = 5
        self.max_gen = max_gen
        self.gen = 0
        self.H = H
        self.tol = tol
        self.zeta = 0.01
        self.p = 0.15
        self.pa = 0.2

        # 初始化历史记忆
        self.F_memory = [0.5] * H
        self.CR_memory = [0.9] * H
        self.hist_idx = 0
        self.clustered = False

        # 初始化种群和存档
        self.N_current = self.N_init  # 当前种群大小
        self.archive_size = 1.3
        self.pop = np.random.uniform(
            low=self.bounds[:, 0],
            high=self.bounds[:, 1],
            size=(self.N_init, self.dim)
        )
        self.fitness = np.apply_along_axis(self.func, 1, self.pop)
        self.archive = []
        self.iteration_log = []
        self.stagnation_counter = np.zeros(self.N_init, dtype=int)

        # 计算初始多样性指标
        center = np.mean(self.pop, axis=0)
        self.DI_init = np.sqrt(np.sum((self.pop - center) ** 2))

        # 初始聚类标签
        self.cluster_labels = np.zeros(self.N_init, dtype=int)

    # 混合种群缩减策略
    def _hybrid_pop_size_reduction(self, gen):
        cutoff = int(0.66 * self.max_gen)
        P_mid = int(0.33 * self.N_init)
        if gen < cutoff:
            N = int(round(self.N_init - (self.N_init - P_mid) * (gen / cutoff) ** 2))
        else:
            N = int(round(P_mid - (P_mid - self.N_min) * ((gen - cutoff) / (self.max_gen - cutoff))))

        return max(self.N_min, N)

    def _cluster_population(self, k):
        kmeans = KMeans(n_clusters=k, n_init='auto')
        labels = kmeans.fit_predict(self.pop)
        return labels

    def _update_clusters(self):
        if self.gen == 0:
            self.cluster_labels = self._cluster_population(4)
        elif self.gen == int(0.66 * self.max_gen):
            self.cluster_labels = self._cluster_population(2)
        # 添加默认情况（保持当前聚类不变）
        else:
            pass  # 保持现有聚类标签


    def _build_Cpn_set(self):
        """构建精英集合Cpn（论文4.1节）"""
        best_idx = np.argmin(self.fitness)
        best_cluster = self.cluster_labels[best_idx]

        # 获取最优簇内个体
        cluster_indices = np.where(self.cluster_labels == best_cluster)[0]
        cluster_size = len(cluster_indices)

        # 计算Cpn大小（公式8）
        n_cpn = max(2, int(self.p * self.N_current))

        # 选择簇内前10%的个体
        n_select = max(1, int(0.1 * cluster_size))
        sorted_indices = np.argsort(self.fitness[cluster_indices])[:n_select]
        Cpn_indices = cluster_indices[sorted_indices]

        # 如果数量不足，从全局补充适应度高的个体
        if len(Cpn_indices) < n_cpn:
            remaining = n_cpn - len(Cpn_indices)
            all_indices = np.setdiff1d(np.arange(self.N_current), Cpn_indices)
            sorted_global = np.argsort(self.fitness[all_indices])[:remaining]
            Cpn_indices = np.concatenate([Cpn_indices, all_indices[sorted_global]])

        return Cpn_indices

    def _update_archive(self):
        unique_clusters = np.unique(self.cluster_labels)
        for c in unique_clusters:
            indices = np.where(self.cluster_labels == c)[0]
            n_select = max(1, int(len(indices) * self.pa))
            selected = np.random.choice(indices, n_select, replace=False)
            for s in selected:
                self.archive.append(self.pop[s].copy())
        max_archive = int(self.archive_size * self.N_current)
        if len(self.archive) > max_archive:
            excess = len(self.archive) - max_archive
            remove_indices = np.random.choice(len(self.archive), excess, replace=False)
            self.archive = [a for i, a in enumerate(self.archive) if i not in remove_indices]

    def cross(self, mutant, i, CR):
        cross_chorm = self.pop[i].copy()
        j = np.random.randint(0, self.dim)  # 随机选择一个维度
        for k in range(self.dim):  # 对每个维度进行交叉
            if np.random.rand() < CR or k == j:  # 如果随机数小于交叉率或者维度为j
                cross_chorm[k] = mutant[k]  # 交叉
                # 边界处理
                if cross_chorm[k] > self.bounds[k, 1]:
                    cross_chorm[k] = (self.bounds[k, 1] + self.pop[i][k]) / 2
                elif cross_chorm[k] < self.bounds[k, 0]:
                    cross_chorm[k] = (self.bounds[k, 0] + self.pop[i][k]) / 2
        return cross_chorm

    def _calculate_VIX(self, parent, offspring):
        """计算波动指数VIX（论文公式13）"""
        delta = offspring - parent
        if self.dim == 1:
            # 一维问题特殊处理
            return np.abs(delta[0])
        mean_delta = np.mean(delta)
        variance = np.sum((delta - mean_delta) ** 2) / (self.dim - 1)
        return np.sqrt(variance)

    def _update_parameters(self, S_F, S_CR, S_VIX):
        # 计算权重（VIX归一化）
        if not S_F or not S_CR or not S_VIX:
            return  # 无成功样本时不更新

            # 计算权重（公式14）
        total_VIX = sum(S_VIX)
        weights = [vix / total_VIX for vix in S_VIX]

        # 计算F的加权Lehmer均值（公式15）
        numerator_F = sum(w * f ** 2 for w, f in zip(weights, S_F))
        denominator_F = sum(w * f for w, f in zip(weights, S_F))
        F_lehmer = numerator_F / denominator_F if denominator_F != 0 else 0.5

        # 计算CR的加权Lehmer均值
        numerator_CR = sum(w * cr ** 2 for w, cr in zip(weights, S_CR))
        denominator_CR = sum(w * cr for w, cr in zip(weights, S_CR))
        CR_lehmer = numerator_CR / denominator_CR if denominator_CR != 0 else 0.9

        # 更新历史记忆
        self.F_memory[self.hist_idx] = F_lehmer
        self.CR_memory[self.hist_idx] = CR_lehmer

        # 移动历史指针
        self.hist_idx = (self.hist_idx + 1) % self.H

    def optimize(self):
        """执行优化过程"""
        for self.gen in range(self.max_gen):
            S_F, S_CR, S_VIX = [], [], []
            new_pop = []
            new_fitness = []

            # 记录当前最优值
            best_val = np.min(self.fitness)
            self.iteration_log.append(best_val)
            # 收敛检查
            if self.tol is not None and best_val <= self.tol:
                print(f"Converged at generation {self.gen + 1}: {best_val}")
                break
            elif self.gen >= self.max_gen - 1:
                print(f"Converged at generation {self.gen + 1}: {best_val}")
                break

            F_list = []

            self._update_clusters()

            # 构建Cpn集合（每个个体独立）
            Cpn_indices = self._build_Cpn_set()

            # 更新存档
            self._update_archive()

            self.p = 0.15 + (self.gen / self.max_gen) * 0.3
            self.pa = 0.2 - (self.gen / self.max_gen) * 0.05

            for i in range(self.N_current):
                # 从历史记忆随机选择索引
                r = np.random.randint(0, self.H)

                # 生成F和CR
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

                F_list.append(F)

                # current-to-pbest/1变异策略
                p_best = self.pop[np.random.choice(Cpn_indices)]

                available_indices = np.delete(np.arange(self.N_current), i)
                r1 = self.pop[np.random.choice(available_indices)]

                r2 = self.archive[np.random.randint(len(self.archive))]

                mutant = self.pop[i] + F * (p_best - self.pop[i]) + F * (r1 - r2)

                # 交叉操作
                trial = self.cross(mutant, i, CR)
                trial_fitness = self.func(trial)

                # 贪心选择
                if trial_fitness < self.fitness[i]:
                    new_pop.append(trial)
                    new_fitness.append(trial_fitness)
                    # 记录成功参数和适应度改进量
                    S_F.append(F)
                    S_CR.append(CR)
                    S_VIX.append(self._calculate_VIX(self.pop[i], trial))
                    self.stagnation_counter[i] = 0
                else:
                    new_pop.append(self.pop[i])
                    new_fitness.append(self.fitness[i])
                    self.stagnation_counter[i] += 1

            # STAR机制：检测多样性，个体停滞后扰动或重启
            center = np.mean(new_pop, axis=0)
            DI_current = np.sqrt(np.sum((np.array(new_pop) - center) ** 2))
            RDI = DI_current / self.DI_init

            best_idx_new = np.argmin(new_fitness)
            best_solution = new_pop[best_idx_new]

            if RDI < self.zeta:
                for i in range(self.N_current):
                    # 扰动阶段（停滞代数≥D）
                    if self.stagnation_counter[i] >= self.dim and i != best_idx_new:
                        # 选择维度（标准差最大的30%）
                        std_per_dim = np.std(new_pop, axis=0)
                        dim_indices = np.argsort(std_per_dim)[::-1][:int(0.3 * self.dim)]

                        # 随机选择个体
                        r1 = np.random.choice(np.delete(np.arange(self.N_current), i))
                        r2_archive = self.archive[np.random.randint(len(self.archive))] if self.archive else new_pop[r1]

                        # 公式11：扰动操作
                        new_pop[i][dim_indices] += F_list[i] * (best_solution[dim_indices] - new_pop[i][dim_indices]) \
                                                   + F_list[i] * (new_pop[r1][dim_indices] - r2_archive[dim_indices])

                        # 边界处理
                        new_pop[i] = np.clip(new_pop[i], self.bounds[:, 0], self.bounds[:, 1])
                        new_fitness[i] = self.func(new_pop[i])
                        self.stagnation_counter[i] = 0  # 重置计数器

                    # 重启阶段（停滞代数≥D+15）
                    elif self.stagnation_counter[i] >= self.dim + 15 and i != best_idx_new:
                        # 随机选择个体
                        r1 = np.random.choice(np.delete(np.arange(self.N_current), i))
                        r2_archive = self.archive[np.random.randint(len(self.archive))] if self.archive else new_pop[r1]

                        # 公式12：重启操作
                        new_pop[i] = best_solution + F_list[i] * (new_pop[r1] - r2_archive)

                        # 边界处理
                        new_pop[i] = np.clip(new_pop[i], self.bounds[:, 0], self.bounds[:, 1])
                        new_fitness[i] = self.func(new_pop[i])
                        self.stagnation_counter[i] = 0  # 重置计数器

            new_N = self._hybrid_pop_size_reduction(self.gen)
            survivor_indices = np.argsort(new_fitness)[:new_N]
            self.pop = np.array([new_pop[i] for i in survivor_indices])
            self.fitness = np.array([new_fitness[i] for i in survivor_indices])
            self.N_current = new_N
            self.stagnation_counter = self.stagnation_counter[survivor_indices]
            self.cluster_labels = self.cluster_labels[survivor_indices]

            # 更新历史记忆（加权Lehmer均值）
            self._update_parameters(S_F, S_CR, S_VIX)

            print(f"Iteration {self.gen + 1}, Pop Size: {self.N_current}, Best: {np.min(self.fitness)}")

        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx], self.fitness[best_idx], self.iteration_log
