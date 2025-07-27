import numpy as np
import matplotlib
matplotlib.use('Agg')  
# 添加中文字体支持
import matplotlib.font_manager as fm
# 指定中文字体路径，这里使用系统自带的微软雅黑字体
font_path = 'C:/Windows/Fonts/msyh.ttc'  # 微软雅黑字体
prop = fm.FontProperties(fname=font_path)
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import matplotlib.pyplot as plt
import random
from typing import List, Tuple, Dict, Any

# 定义Schaffer问题参数
class SchafferProblem:
    def __init__(self, xl=-1000, xu=1000):
        self.n_var = 1  # 变量数量 (SCH问题只有一个决策变量)
        self.n_obj = 2  # 目标函数数量
        self.xl = xl    # 变量下界
        self.xu = xu    # 变量上界
        
        # 理论最优解范围
        self.optimal_xl = 0
        self.optimal_xu = 2
    
    def evaluate(self, x):
        # 计算目标函数值
        f1 = x[0]**2
        f2 = (x[0]-2)**2
        return np.array([f1, f2])
    
    def is_in_optimal_range(self, x):
        # 检查解是否在理论最优解范围内
        return self.optimal_xl <= x[0] <= self.optimal_xu

# 个体类
class Individual:
    def __init__(self, x=None):
        self.x = x          # 决策变量
        self.f = None       # 目标函数值
        self.strength = 0   # 强度值 (SPEA2特有)
        self.raw_fitness = 0  # 原始适应度 (SPEA2特有)
        self.density = 0    # 密度值 (SPEA2特有)
        self.fitness = 0    # 总适应度值 (SPEA2特有)
    
    def __eq__(self, other):
        if isinstance(other, Individual):
            return np.array_equal(self.x, other.x)
        return False

# SPEA2算法实现
class SPEA2:
    def __init__(self, problem, pop_size=100, archive_size=100, n_gen=200, 
                 crossover_prob=0.9, crossover_eta=15, mutation_eta=20):
        self.problem = problem
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.n_gen = n_gen
        self.crossover_prob = crossover_prob
        self.crossover_eta = crossover_eta
        self.mutation_eta = mutation_eta
        
    def run(self):
        # 初始化种群和归档集
        population = self.initialize_population()
        archive = []
        
        # 评估初始种群·
        self.evaluate_population(population)
        
        # 主循环
        for gen in range(self.n_gen):
            if gen % 10 == 0:
                print(f"Generation {gen}/{self.n_gen}")
            
            # 合并种群和归档集
            combined = population + archive
            
            # 计算适应度
            self.calculate_fitness(combined)
            
            # 环境选择 - 更新归档集
            archive = self.environmental_selection(combined)
            
            # 如果达到最大代数，退出循环
            if gen == self.n_gen - 1:
                break
            
            # 二元锦标赛选择
            mating_pool = self.binary_tournament_selection(archive, self.pop_size)
            
            # 产生子代
            offspring = self.create_offspring(mating_pool)
            
            # 评估子代
            self.evaluate_population(offspring)
            
            # 更新种群
            population = offspring
        
        # 提取归档集中的非支配解
        X = np.array([ind.x for ind in archive])
        F = np.array([ind.f for ind in archive])
        
        return X, F
    
    def initialize_population(self):
        # 随机初始化种群
        population = []
        for _ in range(self.pop_size):
            ind = Individual()
            ind.x = np.random.uniform(self.problem.xl, self.problem.xu, self.problem.n_var)
            population.append(ind)
        return population
    
    def evaluate_population(self, population):
        # 评估种群中每个个体的目标函数值
        for ind in population:
            ind.f = self.problem.evaluate(ind.x)
    
    def calculate_fitness(self, population):
        # SPEA2的适应度计算
        
        # 1. 计算每个个体的强度值（它支配的个体数量）
        for ind in population:
            ind.strength = sum(1 for other in population if self.dominates(ind, other))
        
        # 2. 计算每个个体的原始适应度（支配它的个体的强度值之和）
        for ind in population:
            ind.raw_fitness = sum(other.strength for other in population if self.dominates(other, ind))
        
        # 3. 计算密度值
        self.calculate_density(population)
        
        # 4. 计算总适应度
        for ind in population:
            ind.fitness = ind.raw_fitness + ind.density
    
    def calculate_density(self, population):
        # 计算每个个体的密度值（基于k-nearest neighbor）
        k = int(np.sqrt(len(population)))
        
        for i, ind in enumerate(population):
            # 计算个体间的距离
            distances = []
            for j, other in enumerate(population):
                if i != j:
                    dist = self.euclidean_distance(ind.f, other.f)
                    distances.append(dist)
            
            # 排序距离
            distances.sort()
            
            # 取第k个最近邻居的距离
            if len(distances) > k:
                kth_distance = distances[k]
                # 密度值定义为：1/(距离+2)，以避免距离为0的情况
                ind.density = 1.0 / (kth_distance + 2.0)
            else:
                ind.density = 0
    
    def euclidean_distance(self, p1, p2):
        # 计算欧几里得距离
        return np.sqrt(np.sum((p1 - p2) ** 2))
    
    def environmental_selection(self, population):
        # 环境选择，更新归档集
        
        # 1. 按适应度值排序
        population.sort(key=lambda x: x.fitness)
        
        # 2. 选择适应度值小于1的个体（非支配个体）
        archive = [ind for ind in population if ind.fitness < 1]
        
        # 3. 归档集大小调整
        if len(archive) <= self.archive_size:
            # 如果归档集不足，则从被支配的解中添加
            if len(archive) < self.archive_size:
                remaining = population[len(archive):]
                remaining.sort(key=lambda x: x.fitness)
                archive.extend(remaining[:self.archive_size - len(archive)])
        else:
            # 如果归档集过大，则移除距离最近的个体
            while len(archive) > self.archive_size:
                self.remove_closest_individual(archive)
        
        return archive
    
    def remove_closest_individual(self, archive):
        # 移除归档集中距离最近的个体
        min_dist = float('inf')
        idx_to_remove = -1
        
        for i in range(len(archive)):
            for j in range(i+1, len(archive)):
                dist = self.euclidean_distance(archive[i].f, archive[j].f)
                if dist < min_dist:
                    min_dist = dist
                    # 选择适应度较差的个体移除
                    if archive[i].fitness > archive[j].fitness:
                        idx_to_remove = i
                    else:
                        idx_to_remove = j
        
        if idx_to_remove != -1:
            archive.pop(idx_to_remove)
    
    def binary_tournament_selection(self, archive, size):
        # 二元锦标赛选择
        mating_pool = []
        
        for _ in range(size):
            # 随机选择两个个体
            if len(archive) >= 2:
                candidates = random.sample(archive, 2)
                # 选择适应度更好的个体（适应度值更小）
                if candidates[0].fitness <= candidates[1].fitness:
                    winner = candidates[0]
                else:
                    winner = candidates[1]
                mating_pool.append(winner)
            elif len(archive) == 1:
                # 如果归档集只有一个个体，直接添加
                mating_pool.append(archive[0])
        
        return mating_pool
    
    def create_offspring(self, parents):
        # 创建子代
        offspring = []
        
        # 确保父代数量是偶数
        if len(parents) % 2 == 1:
            parents = parents[:-1]
        
        # 随机打乱父代顺序
        random.shuffle(parents)
        
        # 两两配对进行交叉
        for i in range(0, len(parents), 2):
            if i+1 >= len(parents):  # 安全检查
                break
                
            p1, p2 = parents[i], parents[i+1]
            
            # SBX交叉
            if random.random() < self.crossover_prob:
                c1_x, c2_x = [], []
                for j in range(self.problem.n_var):
                    c1j, c2j = self.sbx_crossover(p1.x[j], p2.x[j])
                    c1_x.append(c1j)
                    c2_x.append(c2j)
                c1_x = np.array(c1_x)
                c2_x = np.array(c2_x)
            else:
                c1_x, c2_x = p1.x.copy(), p2.x.copy()
            
            # 多项式变异
            c1_x = self.polynomial_mutation(c1_x)
            c2_x = self.polynomial_mutation(c2_x)
            
            # 创建新个体
            o1, o2 = Individual(c1_x), Individual(c2_x)
            offspring.extend([o1, o2])
        
        return offspring
    
    def sbx_crossover(self, x1, x2):
        # 模拟二进制交叉(SBX)
        eta = self.crossover_eta
        
        # 确保x1 <= x2
        if x1 > x2:
            x1, x2 = x2, x1
        
        # 如果父代相同，直接返回
        if abs(x1 - x2) < 1e-14:
            return x1, x2
        
        # 随机数
        u = random.random()
        
        # 计算beta
        if u <= 0.5:
            beta = (2 * u) ** (1 / (eta + 1))
        else:
            beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
        
        # 计算子代
        c1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
        c2 = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
        
        # 边界处理
        c1 = max(min(c1, self.problem.xu), self.problem.xl)
        c2 = max(min(c2, self.problem.xu), self.problem.xl)
        
        return c1, c2
    
    def polynomial_mutation(self, x):
        # 多项式变异
        eta = self.mutation_eta
        
        # 变异概率
        p_m = 1.0 / self.problem.n_var
        
        for i in range(len(x)):
            if random.random() <= p_m:
                u = random.random()
                
                # 计算delta
                if u < 0.5:
                    delta = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                
                # 计算变异后的值
                x[i] = x[i] + delta * (self.problem.xu - self.problem.xl)
                
                # 边界处理
                x[i] = max(min(x[i], self.problem.xu), self.problem.xl)
        
        return x
    
    def dominates(self, p, q):
        # 判断p是否支配q
        better_in_any = False
        for i in range(len(p.f)):
            if p.f[i] > q.f[i]:  # 假设最小化问题
                return False
            if p.f[i] < q.f[i]:
                better_in_any = True
        return better_in_any

# 生成理论帕累托前沿点
def generate_pareto_front():
    # 对于Schaffer问题，理论帕累托前沿是x在[0,2]之间的解
    x = np.linspace(0, 2, 100)
    f1 = x**2
    f2 = (x-2)**2
    return np.column_stack((f1, f2))

# 计算解集与理论帕累托前沿的性能指标
def calculate_metrics(F, pareto_front):
    # 计算IGD (Inverted Generational Distance)
    # IGD衡量算法解集到理论帕累托前沿的平均距离
    igd = 0
    for pf_point in pareto_front:
        # 计算理论点到最近解的距离
        min_dist = min(np.sqrt(np.sum((F - pf_point)**2, axis=1)))
        igd += min_dist
    igd /= len(pareto_front)
    
    # 计算Spread (分布均匀性指标)
    # 首先计算相邻解之间的距离
    F_sorted = F[np.argsort(F[:, 0])]
    distances = np.sqrt(np.sum((F_sorted[1:] - F_sorted[:-1])**2, axis=1))
    
    # 计算极端解到理论极端解的距离
    d_f = np.min(np.sqrt(np.sum((F - np.array([0, 4]))**2, axis=1)))
    d_l = np.min(np.sqrt(np.sum((F - np.array([4, 0]))**2, axis=1)))
    
    # 计算平均距离
    if len(distances) > 0:
        d_mean = np.mean(distances)
        
        # 计算Spread
        spread = (d_f + d_l + np.sum(np.abs(distances - d_mean))) / (d_f + d_l + len(distances) * d_mean)
    else:
        spread = float('inf')
    
    return {"IGD": igd, "Spread": spread}

# 主函数
if __name__ == "__main__":
    # 定义问题
    problem = SchafferProblem(xl=-5, xu=5)  # 限制搜索空间在[-5,5]内
    
    # 创建并运行SPEA2算法
    spea2 = SPEA2(problem, pop_size=50, archive_size=50, n_gen=50)
    X, F = spea2.run()
    
    # 生成理论上的帕累托前沿
    pareto_front = generate_pareto_front()
    
    # 计算性能指标
    metrics = calculate_metrics(F, pareto_front)
    print(f"\n性能指标:")
    print(f"IGD (收敛性): {metrics['IGD']:.6f}")
    print(f"Spread (多样性): {metrics['Spread']:.6f}")
    
    # 可视化
    plt.figure(figsize=(10, 8))
    plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue', label='SPEA2解集')
    plt.plot(pareto_front[:, 0], pareto_front[:, 1], 'r-', label='理论帕累托前沿')
    plt.grid(True)
    plt.xlabel('$f_1(x) = x^2$')
    plt.ylabel('$f_2(x) = (x-2)^2$')
    plt.title('Schaffer问题的SPEA2求解结果', fontproperties=prop)
    plt.legend(prop=prop)
    plt.savefig('spea2_sch_pareto.png', dpi=300)
    
    # 打印一些解集
    print("\n决策变量和对应的目标函数值:")
    for i in range(min(10, len(X))):
        print(f"x = {X[i][0]:.4f}, f1 = {F[i][0]:.4f}, f2 = {F[i][1]:.4f}")
    
    # 检查解是否在理论最优解范围内
    in_range_count = sum(1 for x in X if problem.is_in_optimal_range(x))
    print(f"\n在理论最优解范围内的解数量: {in_range_count}/{len(X)} ({in_range_count/len(X)*100:.2f}%)")
    
    print("\n理论分析：")
    print("对于Schaffer问题，帕累托最优解应该满足：")
    print(f"x ∈ [{problem.optimal_xl}, {problem.optimal_xu}]")
    print("当x = 0时，f1达到最小值，f2达到最大值")
    print("当x = 2时，f1达到最大值，f2达到最小值")
    print("中间的解在这两点之间形成帕累托前沿") 