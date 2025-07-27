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

# 定义问题参数
class Problem:
    def __init__(self, n_var=1, n_obj=2, xl=-1000, xu=1000):
        self.n_var = n_var  # 变量数量
        self.n_obj = n_obj  # 目标函数数量
        self.xl = xl        # 变量下界
        self.xu = xu        # 变量上界
    
    def evaluate(self, x):
        # 计算目标函数值
        f1 = x**2
        f2 = (x-2)**2
        return np.array([f1, f2])

# 个体类
class Individual:
    def __init__(self, x=None):
        self.x = x          # 决策变量
        self.f = None       # 目标函数值
        self.rank = None    # 非支配排序的等级
        self.crowding_distance = 0.0  # 拥挤度距离，初始化为0而不是None
        self.dominated_solutions = []   # 被该个体支配的解集
        self.domination_count = 0       # 支配该个体的解的数量
    
    def __eq__(self, other):
        if isinstance(other, Individual):
            return np.array_equal(self.x, other.x)
        return False

# NSGA-II算法实现
class NSGA2:
    def __init__(self, problem, pop_size=100, n_gen=200, crossover_prob=0.9, 
                 crossover_eta=15, mutation_eta=20):
        self.problem = problem
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.crossover_prob = crossover_prob
        self.crossover_eta = crossover_eta
        self.mutation_eta = mutation_eta
    
    def run(self):
        # 初始化种群
        population = self.initialize_population()
        
        # 评估初始种群
        self.evaluate_population(population)
        
        # 非支配排序
        fronts = self.fast_non_dominated_sort(population)
        
        # 计算拥挤度
        for front in fronts:
            self.crowding_distance_assignment(front)
        
        # 主循环
        for gen in range(self.n_gen):
            if gen % 10 == 0:
                print(f"Generation {gen}/{self.n_gen}")
            
            # 选择父代
            parents = self.tournament_selection(population)
            
            # 交叉和变异生成子代
            offspring = self.create_offspring(parents)
            
            # 评估子代
            self.evaluate_population(offspring)
            
            # 合并父代和子代
            combined_pop = population + offspring
            
            # 非支配排序
            fronts = self.fast_non_dominated_sort(combined_pop)
            
            # 选择下一代种群
            population = []
            for front in fronts:
                if len(population) + len(front) <= self.pop_size:
                    # 如果可以完全添加当前前沿，则全部添加
                    population.extend(front)
                else:
                    # 否则根据拥挤度选择
                    self.crowding_distance_assignment(front)
                    front.sort(key=lambda x: (x.rank, -x.crowding_distance))
                    population.extend(front[:self.pop_size - len(population)])
                    break
        
        # 返回最终的非支配解集
        final_front = self.fast_non_dominated_sort(population)[0]
        
        # 提取决策变量和目标函数值
        X = np.array([ind.x for ind in final_front])
        F = np.array([ind.f for ind in final_front])
        
        return X, F
    
    def initialize_population(self):
        # 随机初始化种群
        population = []
        for _ in range(self.pop_size):
            ind = Individual()
            ind.x = np.random.uniform(self.problem.xl, self.problem.xu, self.problem.n_var)[0]
            population.append(ind)
        return population
    
    def evaluate_population(self, population):
        # 评估种群中每个个体的目标函数值
        for ind in population:
            ind.f = self.problem.evaluate(ind.x)
    
    def fast_non_dominated_sort(self, population):
        # 快速非支配排序
        fronts = [[]]  # 存储不同等级的前沿
        
        for p in population:#这里p是一个对象 由上面initialize_population函数返回 individual类
            p.dominated_solutions = []
            p.domination_count = 0
            
            for q in population:
                if self.dominates(p, q):
                    # p支配q
                    p.dominated_solutions.append(q)
                elif self.dominates(q, p):
                    # q支配p
                    p.domination_count += 1
            
            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)
        
        i = 0
        while i < len(fronts) and fronts[i]: 
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        
        # 移除空前沿
        return [front for front in fronts if front]
    
    def dominates(self, p, q):
        # 判断p是否支配q
        better_in_any = False
        for i in range(len(p.f)):
            if p.f[i] > q.f[i]:  # 假设最小化问题
                return False
            if p.f[i] < q.f[i]:
                better_in_any = True
        return better_in_any
    
    def crowding_distance_assignment(self, front):
        # 计算拥挤度距离
        if len(front) <= 2:
            for ind in front:
                ind.crowding_distance = float('inf')
            return
        
        for ind in front:
            ind.crowding_distance = 0
        
        # 对每个目标函数
        for m in range(self.problem.n_obj):
            # 按目标函数值排序
            front.sort(key=lambda x: x.f[m])
            
            # 边界点拥挤度设为无穷大
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # 计算中间点的拥挤度
            f_max = front[-1].f[m]
            f_min = front[0].f[m]
            
            if f_max == f_min:
                continue
                
            for i in range(1, len(front) - 1):
                front[i].crowding_distance += (front[i+1].f[m] - front[i-1].f[m]) / (f_max - f_min)
    
    def tournament_selection(self, population):
        # 锦标赛选择
        parents = []
        for _ in range(self.pop_size):
            # 随机选择两个个体
            a, b = random.sample(population, 2)
            
            # 确保rank不是None
            if a.rank is None:
                a.rank = float('inf')
            if b.rank is None:
                b.rank = float('inf')
                
            # 选择更好的个体
            if (a.rank < b.rank) or (a.rank == b.rank and a.crowding_distance > b.crowding_distance):
                parents.append(a)
            else:
                parents.append(b)
        
        return parents
    
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
            p1, p2 = parents[i], parents[i+1]
            
            # SBX交叉
            if random.random() < self.crossover_prob:
                c1, c2 = self.sbx_crossover(p1.x, p2.x)
            else:
                c1, c2 = p1.x, p2.x
            
            # 多项式变异
            c1 = self.polynomial_mutation(c1)
            c2 = self.polynomial_mutation(c2)
            
            # 创建新个体
            o1, o2 = Individual(c1), Individual(c2)
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
        
        if random.random() <= p_m:
            u = random.random()
            
            # 计算delta
            if u < 0.5:
                delta = (2 * u) ** (1 / (eta + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            
            # 计算变异后的值
            x = x + delta * (self.problem.xu - self.problem.xl)
            
            # 边界处理
            x = max(min(x, self.problem.xu), self.problem.xl)
        
        return x

# 主函数
if __name__ == "__main__":
    # 定义问题
    problem = Problem(n_var=1, n_obj=2, xl=-1000, xu=1000)
    
    # 创建并运行NSGA-II算法
    nsga2 = NSGA2(problem, pop_size=100, n_gen=200)
    X, F = nsga2.run()
    
    # 生成理论上的帕累托前沿
    x_theory = np.linspace(0, 2, 1000)
    f1_theory = x_theory**2
    f2_theory = (x_theory-2)**2
    
    # 可视化
    plt.figure(figsize=(10, 8))
    plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue', label='NSGA-II解集')
    plt.plot(f1_theory, f2_theory, 'r-', label='理论帕累托前沿')
    plt.grid(True)
    plt.xlabel('$f_1(x) = x^2$')
    plt.ylabel('$f_2(x) = (x-2)^2$')
    plt.title('NSGA-II求解的帕累托前沿', fontproperties=prop)
    plt.legend(prop=prop)
    plt.savefig('nsga2_sch.png', dpi=300)
    
    # 打印一些解集
    print("\n决策变量和对应的目标函数值:")
    for i in range(min(10, len(X))):
        print(f"x = {X[i]:.4f}, f1 = {F[i][0]:.4f}, f2 = {F[i][1]:.4f}")
    
    print("\n理论分析：")
    print("对于这个问题，帕累托最优解应该在x∈[0,2]范围内")
    print("当x=0时，f1=0, f2=4")
    print("当x=2时，f1=4, f2=0")
    print("中间的解在这两点之间形成帕累托前沿") 