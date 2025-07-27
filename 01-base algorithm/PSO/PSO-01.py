import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Particle:
    def __init__(self, dimension, bounds=(0, 1)):
        """初始化粒子
        
        Args:
            dimension: 问题维度（物品数量）
            bounds: 位置的上下界
        """
        self.dimension = dimension
        self.bounds = bounds
        
        # 位置向量 - 对于01背包问题，表示选择物品的概率
        self.position = np.random.uniform(bounds[0], bounds[1], dimension)
        
        # 速度向量
        self.velocity = np.random.uniform(-0.1, 0.1, dimension)#保证下面初始化之后马上就有速度
        
        # 粒子最佳位置
        self.best_position = self.position.copy()
        
        # 粒子最佳适应度
        self.best_fitness = -np.inf
        
        # 当前解（二进制）
        self.solution = np.zeros(dimension, dtype=int)
        
        # 当前适应度
        self.fitness = -np.inf
    
    def update_solution(self):
        """更新二进制解，使用sigmoid函数将连续位置转换为二进制"""
        # sigmoid函数将位置值转换为概率
        probabilities = 1 / (1 + np.exp(-self.position))
        
        # 根据概率生成二进制解
        self.solution = (np.random.random(self.dimension) < probabilities).astype(int)
    
    def update_velocity(self, global_best_position, w=0.7, c1=1.5, c2=1.5):
        """更新粒子速度
            global_best_position: 全局最佳位置
            w: 惯性权重
            c1: 认知参数
            c2: 社会参数
        """
        # 认知部分 - 个体经验
        cognitive = c1 * np.random.random(self.dimension) * (self.best_position - self.position)
        
        # 社会部分 - 群体经验
        social = c2 * np.random.random(self.dimension) * (global_best_position - self.position)
        
        # 更新速度
        self.velocity = w * self.velocity + cognitive + social
        
        # 速度限制
        max_velocity = 4.0
        self.velocity = np.clip(self.velocity, -max_velocity, max_velocity)
    
    def update_position(self):
        """更新粒子位置"""
        self.position = self.position + self.velocity
        
        # 位置限制
        self.position = np.clip(self.position, self.bounds[0], self.bounds[1])
    
    def update_best(self, fitness):
        """更新粒子最佳位置和适应度"""
        self.fitness = fitness
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.position.copy()
            return True
        return False


class KnapsackPSO:
    def __init__(self, capacity, weights, values, 
                 num_particles=100, max_iter=500, w=0.7, c1=1.5, c2=1.5):
     
        self.capacity = capacity
        self.weights = np.array(weights)
        self.values = np.array(values)
        self.dimension = len(weights)
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # 计算价值密度（价值/重量）
        self.density = np.array([v/w if w > 0 else 0 for v, w in zip(values, weights)])
        
        # 创建粒子群
        self.particles = [Particle(self.dimension) for _ in range(num_particles)]
        
        # 全局最佳位置和适应度
        self.global_best_position = None
        self.global_best_solution = None
        self.global_best_fitness = -np.inf
        
        # 收敛历史
        self.history_best_fitness = []
        self.history_avg_fitness = []
        
        # 初始化粒子群
        self.initialize_particles()
    
    def initialize_particles(self):
        """初始化粒子群，使用多种策略"""
        # 为部分粒子使用贪心初始化
        greedy_num = int(self.num_particles * 0.3)
        
        # 对物品按价值密度排序
        sorted_indices = np.argsort(self.density)[::-1]  # 降序排序
        
        # 贪心初始化
        for i in range(greedy_num):
            # 创建基于贪心的解，加入随机扰动
            solution = np.zeros(self.dimension, dtype=int)
            remaining_capacity = self.capacity
            
            # 添加随机扰动因子
            noise = np.random.normal(1, 0.2, self.dimension)#
            density_with_noise = self.density * noise
            
            # 物品索引按扰动后的价值密度排序
            noisy_sorted = np.argsort(density_with_noise)[::-1]#返回的是索引，步长为负数，反向切片
            
            # 根据扰动后的顺序选择物品
            for idx in noisy_sorted:
                if self.weights[idx] <= remaining_capacity:
                    solution[idx] = 1
                    remaining_capacity -= self.weights[idx]
            
            # 将解转化为PSO中的位置，这里二进制和连续位置都存储了
            self.particles[i].solution = solution
            self.particles[i].position = self.solution_to_position(solution)
            self.particles[i].best_position = self.particles[i].position.copy()
            
            # 计算适应度
            fitness = self.calculate_fitness(solution)
            self.particles[i].fitness = fitness
            self.particles[i].best_fitness = fitness
            
            # 更新全局最佳解
            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = self.particles[i].position.copy()
                self.global_best_solution = solution.copy()
        
        # 随机初始化其余粒子
        for i in range(greedy_num, self.num_particles):
            self.particles[i].update_solution()
            self.particles[i].solution = self.repair_solution(self.particles[i].solution)
            self.particles[i].position = self.solution_to_position(self.particles[i].solution)
            
            fitness = self.calculate_fitness(self.particles[i].solution)
            self.particles[i].fitness = fitness
            self.particles[i].best_fitness = fitness
            self.particles[i].best_position = self.particles[i].position.copy()
            
            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = self.particles[i].position.copy()
                self.global_best_solution = self.particles[i].solution.copy()
    
    def solution_to_position(self, solution):
        """将二进制解转换为连续位置"""
        # 使用logit函数（sigmoid的反函数）
        # 为0的位置赋予较小的值，为1的位置赋予较大的值
        position = np.zeros(self.dimension)
        position[solution == 0] = -3 + np.random.random(np.sum(solution == 0)) * 2
        position[solution == 1] = 3 - np.random.random(np.sum(solution == 1)) * 2
        return position
    
    def calculate_fitness(self, solution):
        """计算解的适应度
        
        对于不可行解（超出容量），返回一个惩罚值
        """
        total_weight = np.sum(solution * self.weights)
        total_value = np.sum(solution * self.values)
        
        if total_weight <= self.capacity:
            return total_value
        else:
            # 惩罚超出容量的解
            return total_value * (self.capacity / total_weight) ** 2
    
    def repair_solution(self, solution):
        """修复不可行解"""
        repaired = solution.copy()
        total_weight = np.sum(repaired * self.weights)
        
        # 如果超过容量，移除物品
        if total_weight > self.capacity:
            # 按价值密度从低到高排序
            items_indices = np.where(repaired == 1)[0]
            if len(items_indices) == 0:
                return repaired
                
            density_indices = [(idx, self.density[idx]) for idx in items_indices]
            density_indices.sort(key=lambda x: x[1])  # 按价值密度升序排序
            
            # 移除价值密度最低的物品直到满足容量约束
            for idx, _ in density_indices:
                if total_weight <= self.capacity:
                    break
                repaired[idx] = 0
                total_weight -= self.weights[idx]
        
        # 如果还有剩余容量，尝试添加更多物品
        if total_weight < self.capacity:
            # 对未选物品按价值密度降序排序
            not_selected = np.where(repaired == 0)[0]
            if len(not_selected) > 0:
                density_indices = [(idx, self.density[idx]) for idx in not_selected]
                density_indices.sort(key=lambda x: x[1], reverse=True)  # 按价值密度降序排序
                
                # 尝试添加高价值密度的物品
                for idx, _ in density_indices:
                    if self.weights[idx] <= self.capacity - total_weight:
                        repaired[idx] = 1
                        total_weight += self.weights[idx]
        
        return repaired
    
    def run(self):
        """运行PSO算法"""
        start_time = time.time()
        no_improvement_count = 0
        
        for iteration in range(self.max_iter):
            iteration_fitness = []
            improved = False
            
            for particle in self.particles:
                # 更新速度和位置
                particle.update_velocity(self.global_best_position, self.w, self.c1, self.c2)
                particle.update_position()
                
                # 更新二进制解
                particle.update_solution()
                
                # 修复不可行解
                particle.solution = self.repair_solution(particle.solution)
                
                # 计算适应度
                fitness = self.calculate_fitness(particle.solution)
                iteration_fitness.append(fitness)
                
                # 更新粒子最佳位置
                if particle.update_best(fitness):
                    # 如果粒子找到更好的解，更新其位置以匹配实际解
                    particle.best_position = self.solution_to_position(particle.solution)
                
                # 更新全局最佳位置
                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()
                    self.global_best_solution = particle.solution.copy()
                    improved = True
                    no_improvement_count = 0  # 重置计数器
            
            # 记录历史数据
            self.history_best_fitness.append(self.global_best_fitness)
            self.history_avg_fitness.append(np.mean(iteration_fitness))
            
            # 如果长时间没有改进，提前终止
            if not improved:
                no_improvement_count += 1
                if no_improvement_count >= 50:  # 50次迭代没有改进
                    print(f"提前终止于第 {iteration} 次迭代：{no_improvement_count} 次未改进")
                    break
            
            # 动态调整参数
            if no_improvement_count > 20:
                # 如果长时间没有改进，增加扰动
                self.w = 0.9  # 增加惯性权重
                self.c1 = 0.5  # 减少个体认知
                self.c2 = 2.5  # 增加社会影响
            else:
                # 正常情况下的参数
                self.w = 0.7
                self.c1 = 1.5
                self.c2 = 1.5
            
            # 每隔10次迭代输出一次结果
            if iteration % 10 == 0 or iteration == self.max_iter - 1:
                total_weight = np.sum(self.global_best_solution * self.weights)
                print(f"迭代 {iteration}: 最佳价值 = {self.global_best_fitness}, 重量 = {total_weight}/{self.capacity}")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # 计算最终解的实际价值和重量
        final_weight = np.sum(self.global_best_solution * self.weights)
        final_value = np.sum(self.global_best_solution * self.values)
        
        print(f"\n优化完成!")
        print(f"PSO运行时间: {elapsed_time:.2f} 秒")
        print(f"最大价值: {final_value}")
        print(f"总重量: {final_weight}/{self.capacity}")
        
        return self.global_best_solution, final_value, final_weight, elapsed_time
    


def read_data(file_path):
    """从文件读取数据"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
        first_line = lines[0].strip().split()
        capacity = int(first_line[0])
        n_items = int(first_line[1])
        
        weights = []
        values = []
        for i in range(1, n_items + 1):
            w, v = map(int, lines[i].strip().split())
            weights.append(w)
            values.append(v)
    
    return capacity, weights, values


def solve_knapsack_pso(file_path, num_particles=100, max_iter=300, w=0.7, c1=1.5, c2=1.5, plot=False, num_runs=3):
    """使用PSO算法求解背包问题，多次运行取最优结果"""
    print(f"\n求解文件：{file_path}")
    capacity, weights, values = read_data(file_path)
    
    print(f"背包容量：{capacity}，物品数量：{len(weights)}")
    
    # 运行多次，取最优结果
    best_overall_value = 0
    best_overall_solution = None
    best_overall_weight = 0
    total_time = 0
    
    for run in range(num_runs):
        print(f"\n运行 {run+1}/{num_runs}:")
        pso = KnapsackPSO(
            capacity=capacity,
            weights=weights,
            values=values,
            num_particles=num_particles,
            max_iter=max_iter,
            w=w,
            c1=c1,
            c2=c2
        )
        
        solution, value, weight, elapsed_time = pso.run()
        total_time += elapsed_time
        
        if value > best_overall_value:
            best_overall_value = value
            best_overall_solution = solution
            best_overall_weight = weight
            
            # 只在最好的一次运行中绘制收敛曲线

    
    # 输出最终结果
    selected_items = [i+1 for i, x in enumerate(best_overall_solution) if x == 1]
    
    print("\n最终结果:")
    print(f"最大价值: {best_overall_value}")
    print(f"总重量: {best_overall_weight}/{capacity}")
    print(f"选择的物品编号: {selected_items}")
    print(f"选择的物品数量: {len(selected_items)}/{len(weights)}")
    print(f"平均运行时间: {total_time/num_runs:.2f} 秒")
    
    return best_overall_solution, best_overall_value, selected_items


if __name__ == "__main__":
    # 解决三个数据文件
    data_files = [
        "./data/data1.txt", 
        "./data/data5.txt", 
        "./data/data6.txt",
    ]
    
    results = {}
    
    for file_path in data_files:
        best_solution, best_value, selected_items = solve_knapsack_pso(
            file_path=file_path,
            num_particles=200,  # 增大粒子数
            max_iter=300,       # 最大迭代次数
            w=0.7,              # 惯性权重
            c1=1.5,             # 认知参数
            c2=1.5,             # 社会参数
            plot=False,         # 是否绘制收敛曲线
            num_runs=3          # 多次运行取最优
        )
        results[file_path] = {
            "best_value": best_value,
            "selected_items": selected_items
        }
    
    # 打印所有结果汇总
    print("\n========== 结果汇总 ==========")
    for file_path, result in results.items():
        print(f"\n文件: {file_path}")
        print(f"最大价值: {result['best_value']}")
        print(f"选择的物品: {result['selected_items']}")
