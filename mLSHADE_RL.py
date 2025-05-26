import numpy as np
from scipy.stats import cauchy
from scipy.linalg import eigh
from scipy.optimize import minimize

class mLSHADE_RL:
    def __init__(self, func, bounds, pop_size=None, max_gen=None, H=5, tol=1e-6):
        self.func = func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.N_init = pop_size if pop_size else 18 * self.dim
        self.N_min = 4
        self.max_gen = max_gen
        self.H = H
        self.tol = tol

        self.F_memory = [0.5] * H
        self.CR_memory = [0.5] * H
        self.freq_memory = [0.5] * H
        self.hist_idx = 0
        self.freq_judge = 0

        self.ps = 0.5
        self.pc = 0.4
        self.LP = 20
        self.ns1 = self.ns2 = 0
        self.nf1 = self.nf2 = 0
        self.S1 = self.S2 = 0.5

        # 新增参数：重启机制、局部搜索、多策略概率
        self.P_LS = 0.1  # 局部搜索初始概率
        self.nfe_LS = 0  # 局部搜索评估次数
        self.count = np.zeros(self.N_init)  # 个体停滞计数器
        self.P_MS = np.array([1 / 3, 1 / 3, 1 / 3])  # 三种变异策略的初始概率

        self.N_current = self.N_init
        self.archive_size = 1.4
        self.pop = np.random.uniform(
            low=self.bounds[:, 0], high=self.bounds[:, 1],
            size=(self.N_init, self.dim)
        )
        self.fitness = np.apply_along_axis(self.func, 1, self.pop)
        self.archive = []
        self.iteration_log = []

    def _linear_pop_size_reduction(self, gen):
        return max(self.N_min, int(self.N_init - (self.N_init - self.N_min) * gen / self.max_gen))

    def mutation(self, F, i, gen):
        p = 0.11
        p_best_size = max(2, int(self.N_current * p))
        p_best_indices = np.argsort(self.fitness)[:p_best_size]
        p_best_idx = np.random.choice(p_best_indices)
        p_best = self.pop[p_best_idx]

        sorted_indices = np.argsort(self.fitness)
        ord_pbest = self.pop[sorted_indices[0]]
        ord_pmed = self.pop[sorted_indices[len(sorted_indices) // 2]]
        ord_pwst = self.pop[sorted_indices[-1]]

        # 选择r1
        r1 = self.pop[np.random.choice(np.delete(np.arange(self.N_current), i), 1, replace=False)].flatten()

        r2_idx = np.random.choice(np.arange(self.N_current + len(self.archive)), 1, replace=False)[0]  # 随机选择一个下标
        if r2_idx >= self.N_current:  # 如果下标大于种群大小，则从存档中选择
            r2_idx -= self.N_current
            r2 = self.archive[r2_idx]
        else:  # 否则从当前种群中选择
            r2 = self.pop[r2_idx].flatten()

        r3 = self.pop[np.random.choice(np.delete(np.arange(self.N_current), i), 1, replace=False)].flatten()

        # 根据当前代数调整F的权重
        if gen < 0.2 * self.max_gen:
            Fw = 0.7 * F
        elif gen < 0.4 * self.max_gen:
            Fw = 0.8 * F
        else:
            Fw = 1.2 * F

        if np.random.rand() < self.P_MS[0]:
            mutant = self.pop[i] + Fw * (p_best - self.pop[i]) + F * (r1 - r2)
            strategy = 1
        elif np.random.rand() < self.P_MS[0] + self.P_MS[1]:
            mutant = self.pop[i] + F * (p_best - self.pop[i] + r1 - r3)
            strategy = 2
        else:
            mutant = self.pop[i] + Fw * (ord_pbest - self.pop[i] + ord_pmed - ord_pwst)
            strategy = 3
        return mutant, strategy

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

    def _local_search(self, best_solution, gen):
        if gen >= 0.85 * self.max_gen and np.random.rand() < self.P_LS:
            res = minimize(self.func, best_solution, method='SLSQP',
                           bounds=self.bounds,
                           options={'maxiter': 50})
            if res.fun < self.fitness[0]:
                self.pop[0] = res.x
                self.fitness[0] = res.fun
                self.P_LS = 0.1  # 成功则保持概率
            else:
                self.P_LS = max(0.01, self.P_LS * 0.9)  # 失败则降低概率

    def _restart_mechanism(self):
        # 计算多样性指标 Vol（公式28）
        Vol_ind = np.sqrt(np.prod(np.maximum(self.bounds[:, 1] - self.bounds[:, 0], 1e-20)))
        pop_range = np.max(self.pop, axis=0) - np.min(self.pop, axis=0)
        Vol_pop = np.sqrt(np.sum(pop_range) / 2)
        Vol = np.sqrt(Vol_pop / Vol_ind)

        if Vol >= 0.001:
            return  # 不触发 restart

        for i in range(self.N_current):
            if self.count[i] > 2 * 50:
                if np.random.rand() < 0.5:
                    # 水平交叉 (公式26)
                    j = np.random.randint(self.N_current)
                    rd1 = np.random.rand()
                    rd2 = 1 - rd1
                    rnds = np.random.uniform(-1, 1)
                    trial = rd1 * self.pop[i] + rd2 * self.pop[j] + rnds * (self.pop[i] - self.pop[j])
                else:
                    # 垂直交叉 (公式27)
                    d1, d2 = np.random.choice(self.dim, 2, replace=False)
                    trial = self.pop[i].copy()
                    r = np.random.rand()
                    trial[d1] = r * self.pop[i, d1] + (1 - r) * self.pop[i, d2]

                # 边界约束
                trial = np.clip(trial, self.bounds[:, 0], self.bounds[:, 1])
                trial_fitness = self.func(trial)

                if trial_fitness < self.fitness[i]:
                    self.pop[i] = trial
                    self.fitness[i] = trial_fitness
                    self.count[i] = 0  # 重置计数器
                else:
                    self.count[i] += 1

    def optimize(self):
        for gen in range(self.max_gen):
            S_F, S_CR, S_weights = [], [], []
            S_freq, S_freq_weights = [], []
            new_pop, new_fitness = [], []
            strategy1_improvement = []  # 策略1的改进值
            strategy2_improvement = []  # 策略2的改进值
            strategy3_improvement = []  # 策略3的改进值
            strategy1_old_fitness = []  # 各策略的原始适应度值
            strategy2_old_fitness = []  # 各策略的原始适应度值
            strategy3_old_fitness = []  # 各策略的原始适应度值
            I_MSi = np.zeros(3)

            best_val = np.min(self.fitness)
            self.iteration_log.append(best_val)
            # 终止条件
            if self.tol is not None and best_val <= self.tol:
                print(f"Converged at generation {gen} with precision {best_val}")
                break
            elif gen >= self.max_gen - 1:
                print(f"Converged at generation {gen} with precision {best_val}")
                break

            for i in range(self.N_current):
                # 选择变异策略
                r = np.random.randint(0, self.H)
                freq = cauchy.rvs(loc=self.freq_memory[r], scale=0.1)

                if gen < self.max_gen // 2:
                    self._update_success_rates(gen)
                    if np.random.rand() < self.S1 / (self.S1 + self.S2):
                        F = 0.5 * (np.sin(2 * np.pi * 0.5 * gen + np.pi) * ((self.max_gen - gen) / self.max_gen) + 1)
                        self.freq_judge = 1
                    else:
                        freq = cauchy.rvs(self.freq_memory[r], 0.1)
                        F = 0.5 * np.sin(2 * np.pi * freq * gen) * (gen / self.max_gen) + 0.5
                        self.freq_judge = 2
                else:
                    F = cauchy.rvs(loc=self.F_memory[r], scale=0.1)
                    while F > 1 or F < 0:
                        if F < 0:
                            F = cauchy.rvs(loc=self.F_memory[r], scale=0.1)
                        else:
                            F = 1

                CR = np.clip(np.random.normal(self.CR_memory[r], 0.1), 0.0, 1.0)

                mutant, strategy = self.mutation(F, i, gen)

                if np.random.rand() < self.pc:
                    trial = self._eigen_crossover(self.pop[i], mutant, CR)
                else:
                    trial = self._binomial_crossover(self.pop[i], mutant, CR)
                trial = np.clip(trial, self.bounds[:, 0], self.bounds[:, 1])
                trial_fitness = self.func(trial)

                if trial_fitness < self.fitness[i]:
                    new_pop.append(trial)
                    new_fitness.append(trial_fitness)
                    delta = np.abs(self.fitness[i] - trial_fitness)
                    S_F.append(F)
                    S_CR.append(CR)
                    S_weights.append(delta)
                    self.count[i] = 0
                    if gen < self.max_gen // 2 and self.freq_judge == 2:  # 只记录自适应配置的成功频率
                        S_freq.append(freq)
                        S_freq_weights.append(delta)
                    self.archive.append(self.pop[i].copy())
                    if len(self.archive) > self.archive_size * self.N_current:
                        # 如果存档超过种群大小，则随机删除一些
                        for o in range(int(len(self.archive) - self.archive_size * self.N_current)):
                            self.archive.pop(np.random.randint(0, len(self.archive)))
                    if gen > self.LP and gen < self.max_gen // 2 and self.freq_judge != 0:
                        if self.freq_judge == 1:
                            self.ns1 += 1
                        elif self.freq_judge == 2:
                            self.ns2 += 1
                else:
                    new_pop.append(self.pop[i])
                    new_fitness.append(self.fitness[i])
                    self.count[i] += 1
                    if gen > self.LP and gen < self.max_gen // 2 and self.freq_judge != 0:
                        if self.freq_judge == 1:
                            self.nf1 += 1
                        elif self.freq_judge == 2:
                            self.nf2 += 1

                if strategy == 1:
                    strategy1_improvement.append(np.maximum(0, trial_fitness - self.fitness))
                    strategy1_old_fitness.append(self.fitness[i])
                elif strategy == 2:
                    strategy2_improvement.append(np.maximum(0, trial_fitness - self.fitness))
                    strategy2_old_fitness.append(self.fitness[i])
                else:
                    strategy3_improvement.append(np.maximum(0, trial_fitness - self.fitness))
                    strategy3_old_fitness.append(self.fitness[i])

            # 更新种群
            new_N = self._linear_pop_size_reduction(gen)

            # 选择适应度最好的个体保留
            combined_fitness = np.array(new_fitness)
            survivor_indices = np.argsort(combined_fitness)[:new_N]

            # 更新种群和适应度
            self.pop = np.array([new_pop[i] for i in survivor_indices])
            self.fitness = np.array([new_fitness[i] for i in survivor_indices])
            self.N_current = new_N

            self._local_search(self.pop[0], gen)

            self._restart_mechanism()

            for k in range(3):
                if k == 0:
                    num = np.sum(strategy1_improvement)
                    den = np.sum(strategy1_old_fitness)
                elif k == 1:
                    num = np.sum(strategy2_improvement)
                    den = np.sum(strategy2_old_fitness)
                else:
                    num = np.sum(strategy3_improvement)
                    den = np.sum(strategy3_old_fitness)

                I_MSi[k] = num / (den + 1e-20) if den > 0 else 0

            sum_IMSi = np.sum(I_MSi)
            if sum_IMSi > 0:
                for k in range(3):
                    self.P_MS[k] = max(0.1, min(0.9, float(I_MSi[k] / sum_IMSi)))
            else:
                self.P_MS[:] = 1 / 3

            self.P_MS /= np.sum(self.P_MS)  # 归一化

            # 更新历史记忆（加权Lehmer均值）
            if np.any(S_F):
                F_lehmer = np.sum(np.array(S_F) ** 2 * S_weights) / np.sum(np.array(S_F) * S_weights)
                self.F_memory[self.hist_idx] = F_lehmer

            # CR 部分
            if np.any(S_CR):
                CR_lehmer = np.sum(np.array(S_CR) ** 2 * S_weights) / np.sum(np.array(S_CR) * S_weights)
                self.CR_memory[self.hist_idx] = CR_lehmer

            if gen < self.max_gen // 2:
                if np.any(S_freq):
                    freq_lehmer = np.sum(np.array(S_freq) * S_freq_weights[:len(S_freq)]) / np.sum(np.array(S_freq) * S_freq_weights)
                    self.freq_memory[self.hist_idx] = freq_lehmer

            # 移动历史指针
            self.hist_idx = (self.hist_idx + 1) % self.H

            print(f"Iteration {gen + 1}, Best: {best_val}, pop_size: {self.N_current}, P_MS: {self.P_MS}")

        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx], self.fitness[best_idx], self.iteration_log