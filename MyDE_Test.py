import numpy as np
from scipy.stats import cauchy
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from collections import deque

class MyDE:
    def __init__(self, func, bounds, pop_size=None, max_gen=None, H=6, tol=1e-6):
        self.func = func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.N_init = pop_size if pop_size else 18 * self.dim
        self.N_min = 5
        self.max_gen = max_gen
        self.gen = 0
        self.H = H
        self.tol = tol

        self.F_memory = [0.5] * H
        self.CR_memory = [0.5] * H
        self.freq_memory = [0.5] * H
        self.hist_idx = 0
        self.freq_judge = 0
        self.p1 = 0.15
        self.p2 = 0.1

        self.ps = 0.5
        self.pc = 0.4
        self.LP = 20
        self.ns1 = self.ns2 = 0
        self.nf1 = self.nf2 = 0
        self.S1 = self.S2 = 0.5

        self.N_current = self.N_init
        self.pop = np.random.uniform(
            low=self.bounds[:, 0], high=self.bounds[:, 1],
            size=(self.N_init, self.dim)
        )
        self.fitness = np.apply_along_axis(self.func, 1, self.pop)
        self.archive = deque(maxlen=int(1.4 * self.N_current))  # FIFO存档（最大长度1.4*NP）
        self.iteration_log = []

        # 初始聚类标签
        self.cluster_labels = np.zeros(self.N_init, dtype=int)

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
            self.p1 = 0.2
            self.p2 = 0.15
        # 添加默认情况（保持当前聚类不变）
        else:
            pass  # 保持现有聚类标签

    def _build_Cpn_set(self):
        # 获取所有簇的标签
        unique_clusters = np.unique(self.cluster_labels)
        best_idx = np.argmin(self.fitness)
        best_cluster = self.cluster_labels[best_idx]
        cpn_indices = []

        # 构建Xcpn候选池
        for cluster in unique_clusters:
            cluster_indices = np.where(self.cluster_labels == cluster)[0]

            # 最优个体所在簇使用p1比例
            if cluster == best_cluster:
                num_select = max(1, int(self.p1 * len(cluster_indices)))
            # 其他簇使用p2比例
            else:
                num_select = max(1, int(self.p2 * len(cluster_indices)))

            # 选择簇内最优的个体
            sorted_indices = cluster_indices[np.argsort(self.fitness[cluster_indices])[:num_select]]
            cpn_indices.extend(sorted_indices)

        return cpn_indices

    def mutation(self, F, i, cpn_indices):
        # 从候选池中随机选择Xcpn
        cpn_idx = np.random.choice(cpn_indices)
        Xcpn = self.pop[cpn_idx]

        # 计算排名及概率
        ranks = np.zeros(self.N_current)
        for rank_order, idx in enumerate(np.argsort(self.fitness)):
            ranks[idx] = 3 * (self.N_current - (rank_order + 1)) + 1
        # 计算概率
        prs = ranks / np.sum(ranks)

        combined_pop = np.vstack([self.pop, np.array(self.archive)]) if self.archive else self.pop
        combined_size = len(combined_pop)

        # 选择r1
        candidates_r1 = [idx for idx in range(self.N_current) if idx != i and idx != cpn_idx]
        prs_r1 = prs[candidates_r1]
        prs_r1 /= np.sum(prs_r1)
        r1_idx = np.random.choice(candidates_r1, p=prs_r1)
        r1 = self.pop[r1_idx]

        # 选择r2
        candidates_r2 = [idx for idx in range(combined_size) if idx != i and idx != cpn_idx and idx != r1_idx]
        if np.random.rand() < 0.5:
            r2_idx = np.random.choice(candidates_r1, p=prs_r1)
            r2 = self.pop[r2_idx]
        else:
            r2_idx = np.random.choice(candidates_r2)
            r2 = combined_pop[r2_idx]

        mutant = self.pop[i] + F * (Xcpn - self.pop[i]) + F * (r1 - r2)
        return mutant

    def _eigen_crossover(self, target, mutant, CR):
        sorted_indices = np.argsort(self.fitness)
        best_idx = sorted_indices[0]
        distances = np.linalg.norm(self.pop - self.pop[best_idx], axis=1)
        neighbor_indices = np.argsort(distances)[:int(self.ps * self.N_current)]
        neighbors = self.pop[neighbor_indices]

        cov_matrix = np.cov(neighbors, rowvar=False)
        eigenvalues, eigenvectors = eigh(cov_matrix)
        B = eigenvectors

        target_prime = B.T @ target
        mutant_prime = B.T @ mutant

        cross_mask = np.random.rand(self.dim) < CR
        cross_mask[np.random.randint(self.dim)] = True
        trial_prime = np.where(cross_mask, mutant_prime, target_prime)

        return B @ trial_prime

    def _binomial_crossover(self, target, mutant, CR):
        cross_mask = np.random.rand(self.dim) < CR
        if not np.any(cross_mask):
            cross_mask[np.random.randint(self.dim)] = True
        return np.where(cross_mask, mutant, target)

    def _update_success_rates(self, gen):
        if gen <= self.LP:
            self.S1 = self.S2 = 0.5
        else:
            total1 = self.ns1 + self.nf1 + 1e-20
            total2 = self.ns2 + self.nf2 + 1e-20
            self.S1 = (self.ns1 / total1) + 0.01
            self.S2 = (self.ns2 / total2) + 0.01
            sum_S = self.S1 + self.S2
            self.S1 /= sum_S
            self.S2 /= sum_S

    def optimize(self):
        for self.gen in range(self.max_gen):
            S_F, S_CR, S_weights = [], [], []
            S_freq, S_freq_weights = [], []
            new_pop, new_fitness = [], []

            best_val = np.min(self.fitness)
            self.iteration_log.append(best_val)
            # 终止条件
            if self.tol is not None and best_val <= self.tol:
                print(f"Converged at generation {self.gen} with precision {best_val}")
                break
            elif self.gen >= self.max_gen - 1:
                print(f"Converged at generation {self.gen} with precision {best_val}")
                break

            self._update_clusters()

            # 构建Cpn集合（每个个体独立）
            cpn_indices = self._build_Cpn_set()

            for i in range(self.N_current):
                r = np.random.randint(0, self.H)
                freq = cauchy.rvs(loc=self.freq_memory[r], scale=0.1)

                if self.gen <= self.max_gen / 2:
                    self._update_success_rates(self.gen)
                    if np.random.rand() < self.S1 / (self.S1 + self.S2):
                        F = 0.5 * (np.sin(2 * np.pi * 0.5 * self.gen + np.pi) * ((self.max_gen - self.gen) / self.max_gen) + 1)
                        self.freq_judge = 1
                    else:
                        F = 0.5 * np.sin(2 * np.pi * freq * self.gen) * (self.gen / self.max_gen) + 0.5
                        self.freq_judge = 2
                else:
                    F = cauchy.rvs(loc=self.F_memory[r], scale=0.1)
                    while F > 1 or F < 0:
                        if F < 0:
                            F = cauchy.rvs(loc=self.F_memory[r], scale=0.1)
                        else:
                            F = 1

                CR = np.clip(np.random.normal(self.CR_memory[r], 0.1), 0.0, 1.0)

                mutant = self.mutation(F, i, cpn_indices)

                if np.random.rand() < self.pc:
                    trial = self._eigen_crossover(self.pop[i], mutant, CR)
                else:
                    trial = self._binomial_crossover(self.pop[i], mutant, CR)
                trial = np.clip(trial, self.bounds[:, 0], self.bounds[:, 1])
                trial_fitness = self.func(trial)

                if trial_fitness < self.fitness[i]:
                    new_pop.append(trial)
                    new_fitness.append(trial_fitness)
                    S_F.append(F)
                    S_CR.append(CR)
                    delta = np.abs(self.fitness[i] - trial_fitness)
                    S_weights.append(delta)
                    if self.gen <= self.max_gen / 2 and self.freq_judge == 2:  # 只记录自适应配置的成功频率
                        S_freq.append(freq)
                        S_freq_weights.append(delta)
                    self.archive.append(self.pop[i].copy())
                    if self.gen > self.LP and self.gen <= self.max_gen / 2 and self.freq_judge != 0:
                        if self.freq_judge == 1:
                            self.ns1 += 1
                        elif self.freq_judge == 2:
                            self.ns2 += 1
                else:
                    new_pop.append(self.pop[i])
                    new_fitness.append(self.fitness[i])
                    if self.gen > self.LP and self.gen <= self.max_gen / 2 and self.freq_judge != 0:
                        if self.freq_judge == 1:
                            self.nf1 += 1
                        elif self.freq_judge == 2:
                            self.nf2 += 1

            # 更新种群
            new_N = self._hybrid_pop_size_reduction(self.gen)

            # 选择适应度最好的个体保留
            combined_fitness = np.array(new_fitness)
            survivor_indices = np.argsort(combined_fitness)[:new_N]

            # 更新种群和适应度
            self.pop = np.array([new_pop[i] for i in survivor_indices])
            self.fitness = np.array([new_fitness[i] for i in survivor_indices])
            self.N_current = new_N
            self.cluster_labels = self.cluster_labels[survivor_indices]

            # 更新历史记忆（加权Lehmer均值）
            if np.any(S_F):
                F_lehmer = np.sum(np.array(S_F) ** 2 * S_weights) / np.sum(np.array(S_F) * S_weights)
                self.F_memory[self.hist_idx] = F_lehmer

            # CR 部分
            if np.any(S_CR):
                if self.gen < self.max_gen // 2:
                    CR_lehmer = np.sum(np.array(S_CR) ** 2 * S_weights) / np.sum(np.array(S_CR) * S_weights)
                    self.CR_memory[self.hist_idx] = CR_lehmer
                else:
                    self.CR_memory[self.hist_idx] = self.CR_memory[self.hist_idx]

            if self.gen < self.max_gen // 2:
                if np.any(S_freq):
                    freq_lehmer = np.sum(np.array(S_freq)**2 * S_freq_weights) / np.sum(np.array(S_freq) * S_freq_weights)
                    self.freq_memory[self.hist_idx] = freq_lehmer

            # 移动历史指针
            self.hist_idx = (self.hist_idx + 1) % self.H

            print(f"Iteration {self.gen + 1}, Best: {best_val}, pop_size: {self.N_current}")

        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx], self.fitness[best_idx], self.iteration_log