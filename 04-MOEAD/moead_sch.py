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
import time
from typing import List, Tuple, Dict, Any

# 定义Schaffer问题参数
class SchafferProblem:
    def __init__(self, xl=-5, xu=5):
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
    
    def __eq__(self, other):
        if isinstance(other, Individual):
            return np.array_equal(self.x, other.x)
        return False

# MOEA/D算法实现
class MOEAD:
    def __init__(self, problem, pop_size=100, n_gen=200, neighborhood_size=20, 
                 cr=1.0, f=0.5, delta=0.9, nr=2):
    
        self.problem = problem
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.neighborhood_size = min(neighborhood_size, pop_size)
        self.cr = cr
        self.f = f
        self.delta = delta
        self.nr = nr
        
        # 生成均匀分布的权重向量
        self.weights = self.generate_weights()
        
        # 计算每个权重向量的邻域
        self.neighborhoods = self.compute_neighborhoods()
        
        # 理想点
        self.z = np.array([float('inf')] * problem.n_obj)
    
    def generate_weights(self):
        # 对于两个目标的问题，权重向量是(λ, 1-λ)形式
        weights = []
        for i in range(self.pop_size):
            w = i / (self.pop_size - 1) if self.pop_size > 1 else 0.5
            weights.append(np.array([w, 1-w]))
        return np.array(weights)
    
    def compute_neighborhoods(self):
        # 计算权重向量间的欧几里得距离
        distances = np.zeros((self.pop_size, self.pop_size))#初始化距离矩阵
        for i in range(self.pop_size):
            for j in range(i+1, self.pop_size):
                dist = np.linalg.norm(self.weights[i] - self.weights[j])#计算权重向量间的欧几里得距离
                distances[i, j] = distances[j, i] = dist
        
        # 为每个权重向量找到最近的T个邻居
        neighborhoods = []
        for i in range(self.pop_size):
            # argsort返回按距离排序的索引
            sorted_idx = np.argsort(distances[i])#按距离排序的索引,升序排序的索引
            # 选择最近的T个（包括自己）
            neighborhoods.append(sorted_idx[:self.neighborhood_size])#选择最近的T个邻居 构建领域
        
        return neighborhoods
    
    def run(self):

        start_time = time.time()
        
        # 初始化种群
        population = self.initialize_population()
        
        # 评估初始种群
        self.evaluate_population(population)
        
        # 更新理想点
        for ind in population:
            self.update_reference_point(ind.f)
        
        # 初始化外部档案（存储非支配解）
        archive = []
        
        # 主循环
        for gen in range(self.n_gen):
            if gen % 10 == 0:
                print(f"Generation {gen}/{self.n_gen}")
            
            # 更新计数器（每次迭代最多更新nr个解）
            c = 0
            
            # 随机排序权重向量索引
            perm = np.random.permutation(self.pop_size)
            
            for i in perm:
                # 选择邻域或整个种群
                if np.random.random() < self.delta:
                    # 从邻域中选择
                    pool = self.neighborhoods[i]
                else:
                    # 从整个种群中选择
                    pool = np.arange(self.pop_size)
                
                # 差分进化生成子代
                child = self.differential_evolution(population, pool, i)
                
                # 评估子代
                child.f = self.problem.evaluate(child.x)
                
                # 更新理想点
                self.update_reference_point(child.f)
                
                # 更新邻居解
                c = self.update_neighbors(population, child, pool, c)
                
                # 更新外部档案
                self.update_archive(archive, child)
                
                # 如果已达到更新上限，则停止本次迭代的更新
                if c >= self.nr:
                    break
        
        # 提取外部档案中的解
        if not archive:
            # 如果外部档案为空，则使用种群中的非支配解
            archive = self.extract_non_dominated_solutions(population)
        
        # 提取决策变量和目标函数值
        X = np.array([ind.x for ind in archive])
        F = np.array([ind.f for ind in archive])
        
        end_time = time.time()
        print(f"总运行时间: {end_time - start_time:.2f} 秒")
        
        return X, F
    
    def initialize_population(self):
      
        population = []
        for _ in range(self.pop_size):
            ind = Individual()
            ind.x = np.random.uniform(self.problem.xl, self.problem.xu, self.problem.n_var)
            population.append(ind)
        return population
    
    def evaluate_population(self, population):
      
        for ind in population:
            ind.f = self.problem.evaluate(ind.x)
    
    def update_reference_point(self, f):
       
        self.z = np.minimum(self.z, f)
    
    def differential_evolution(self, population, pool, i):
       
        # 从邻域中随机选择两个不同的个体
        if len(pool) >= 3:
            r = np.random.choice(pool, 3, replace=False)
        else:
            r = np.random.choice(range(self.pop_size), 3, replace=False)#如果邻域中个体数量小于3，则从整个种群中随机选择3个个体
        
        # 差分进化操作
        y = Individual()
        y.x = np.zeros(self.problem.n_var)
        
        # 差分变异
        v = population[r[0]].x + self.f * (population[r[1]].x - population[r[2]].x)
        
        # 交叉
        for j in range(self.problem.n_var):
            if np.random.random() < self.cr or j == np.random.randint(self.problem.n_var):
                y.x[j] = v[j]
            else:
                y.x[j] = population[i].x[j]
        
        # 边界处理
        y.x = np.clip(y.x, self.problem.xl, self.problem.xu)
        
        return y
    
    def update_neighbors(self, population, child, pool, c):
      
        for j in pool:
            # 使用切比雪夫分解方法计算聚合函数值
            g_old = self.compute_tchebycheff(population[j].f, self.weights[j])
            g_new = self.compute_tchebycheff(child.f, self.weights[j])
            
            # 如果子代更好，则替换
            if g_new <= g_old:
                population[j] = Individual()
                population[j].x = child.x.copy()
                population[j].f = child.f.copy()
                c += 1
        
        return c
    
    def compute_tchebycheff(self, f, w):
        
        return np.max(w * np.abs(f - self.z))
    
    def update_archive(self, archive, ind):
        # 检查新解是否被档案中的解支配
        dominated = False
        i = 0
        while i < len(archive):
            if self.dominates(archive[i].f, ind.f):
                dominated = True
                break
            elif self.dominates(ind.f, archive[i].f):
                archive.pop(i)
            else:
                i += 1
        
        # 如果新解不被支配，则添加到档案中
        if not dominated:
            archive.append(Individual())
            archive[-1].x = ind.x.copy()#将新解添加到档案中
            archive[-1].f = ind.f.copy()
    
    def dominates(self, f1, f2):
        better_in_any = False
        for i in range(len(f1)):
            if f1[i] > f2[i]:  # 假设最小化问题
                return False
            if f1[i] < f2[i]:
                better_in_any = True
        return better_in_any
    
    def extract_non_dominated_solutions(self, population):
        
        non_dominated = []
        for ind in population:
            dominated = False
            for other in population:
                if self.dominates(other.f, ind.f):
                    dominated = True
                    break
            if not dominated:
                new_ind = Individual()
                new_ind.x = ind.x.copy()
                new_ind.f = ind.f.copy()
                non_dominated.append(new_ind)
        return non_dominated

# 生成理论帕累托前沿点
def generate_pareto_front():
   
    x = np.linspace(0, 2, 100)
    f1 = x**2
    f2 = (x-2)**2
    return np.column_stack((f1, f2))

# 计算解集与理论帕累托前沿的性能指标
def calculate_metrics(F, pareto_front):

    # 计算IGD (Inverted Generational Distance)
    igd = 0
    for pf_point in pareto_front:
        # 计算理论点到最近解的距离
        min_dist = min(np.sqrt(np.sum((F - pf_point)**2, axis=1)))
        igd += min_dist
    igd /= len(pareto_front)
    
    # 计算Spread (分布均匀性指标)
    F_sorted = F[np.argsort(F[:, 0])]
    distances = np.sqrt(np.sum((F_sorted[1:] - F_sorted[:-1])**2, axis=1))
    
    # 计算极端解到理论极端解的距离
    d_f = np.min(np.sqrt(np.sum((F - np.array([0, 4]))**2, axis=1)))
    d_l = np.min(np.sqrt(np.sum((F - np.array([4, 0]))**2, axis=1)))
    
    # 计算平均距离
    if len(distances) > 0:
        d_mean = np.mean(distances)
        spread = (d_f + d_l + np.sum(np.abs(distances - d_mean))) / (d_f + d_l + len(distances) * d_mean)
    else:
        spread = float('inf')
    
    return {"IGD": igd, "Spread": spread}

# 主函数
if __name__ == "__main__":
    # 定义问题
    problem = SchafferProblem(xl=-5, xu=5)
    
    # 创建并运行MOEA/D算法
    moead = MOEAD(problem, pop_size=100, n_gen=200, neighborhood_size=20, cr=1.0, f=0.5, delta=0.9, nr=2)
    X, F = moead.run()
    
    # 生成理论上的帕累托前沿
    pareto_front = generate_pareto_front()
    
    # 计算性能指标
    metrics = calculate_metrics(F, pareto_front)
    print(f"\n性能指标:")
    print(f"IGD (收敛性): {metrics['IGD']:.6f}")
    print(f"Spread (多样性): {metrics['Spread']:.6f}")
    
    # 可视化
    plt.figure(figsize=(10, 8))
    plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue', label='MOEA/D解集')
    plt.plot(pareto_front[:, 0], pareto_front[:, 1], 'r-', label='理论帕累托前沿')
    plt.grid(True)
    plt.xlabel('$f_1(x) = x^2$')
    plt.ylabel('$f_2(x) = (x-2)^2$')
    plt.title('Schaffer问题的MOEA/D求解结果', fontproperties=prop)
    plt.legend(prop=prop)
    plt.savefig('moead_sch_pareto.png', dpi=300)
    
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