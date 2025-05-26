import numpy as np
from scipy.stats import cauchy
from scipy.linalg import eigh

class LSHADE_cnEpSin:
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

    def mutation(self, F, i):
        p = 0.11
        p_best_size = max(2, int(self.N_current * p))
        p_best_indices = np.argsort(self.fitness)[:p_best_size]
        p_best_idx = np.random.choice(p_best_indices)
        p_best = self.pop[p_best_idx]

        # 选择r1
        r1 = self.pop[np.random.choice(np.delete(np.arange(self.N_current), i), 1, replace=False)].flatten()

        r2_idx = np.random.choice(np.arange(self.N_current + len(self.archive)), 1, replace=False)[0]  # 随机选择一个下标
        if r2_idx >= self.N_current:  # 如果下标大于种群大小，则从存档中选择
            r2_idx -= self.N_current
            r2 = self.archive[r2_idx]
        else:  # 否则从当前种群中选择
            r2 = self.pop[r2_idx].flatten()

        mutant = self.pop[i] + F * (p_best - self.pop[i]) + F * (r1 - r2)
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
        for gen in range(self.max_gen):
            S_F, S_CR, S_weights = [], [], []
            S_freq, S_freq_weights = [], []
            new_pop, new_fitness = [], []

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
                r = np.random.randint(0, self.H)
                freq = cauchy.rvs(loc=self.freq_memory[r], scale=0.1)

                if gen <= self.max_gen / 2:
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

                mutant = self.mutation(F, i)

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
                    if gen <= self.max_gen / 2 and self.freq_judge == 2:  # 只记录自适应配置的成功频率
                        S_freq.append(freq)
                        S_freq_weights.append(delta)
                    self.archive.append(self.pop[i].copy())
                    if len(self.archive) > self.archive_size * self.N_current:
                        # 如果存档超过种群大小，则随机删除一些
                        for o in range(int(len(self.archive) - self.archive_size * self.N_current)):
                            self.archive.pop(np.random.randint(0, len(self.archive)))
                    if gen > self.LP and gen <= self.max_gen / 2 and self.freq_judge != 0:
                        if self.freq_judge == 1:
                            self.ns1 += 1
                        elif self.freq_judge == 2:
                            self.ns2 += 1
                else:
                    new_pop.append(self.pop[i])
                    new_fitness.append(self.fitness[i])
                    if gen > self.LP and gen <= self.max_gen / 2 and self.freq_judge != 0:
                        if self.freq_judge == 1:
                            self.nf1 += 1
                        elif self.freq_judge == 2:
                            self.nf2 += 1

            # 更新种群
            new_N = self._linear_pop_size_reduction(gen)

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

            if gen < self.max_gen // 2:
                if np.any(S_freq):
                    freq_lehmer = np.sum(np.array(S_freq) * S_freq_weights[:len(S_freq)]) / np.sum(np.array(S_freq) * S_freq_weights)
                    self.freq_memory[self.hist_idx] = freq_lehmer

            # 移动历史指针
            self.hist_idx = (self.hist_idx + 1) % self.H

            print(f"Iteration {gen + 1}, Best: {best_val}, pop_size: {self.N_current}")

        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx], self.fitness[best_idx], self.iteration_log