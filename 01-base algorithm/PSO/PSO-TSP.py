import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import time
import os
from scipy.spatial.distance import pdist, squareform
import random

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class TSP:
    """旅行商问题类，用于存储城市信息和计算距离"""
    def __init__(self, file_path):
        self.city_coords = []  # 存储城市坐标
        self.city_ids = []     # 存储城市ID
        self.distances = None  # 城市间距离矩阵
        self.num_cities = 0    # 城市数量
        
        # 从文件中加载数据
        self.load_data(file_path)
        
        # 计算城市间距离
        self.calculate_distances()
    
    def load_data(self, file_path):
        """从文件中加载城市数据"""
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line == "EOF":
                    break
                
                parts = line.split()
                if len(parts) == 3:
                    city_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    
                    self.city_ids.append(city_id)
                    self.city_coords.append((x, y))
        
        self.num_cities = len(self.city_coords)
        print(f"加载了 {self.num_cities} 个城市")
    
    def calculate_distances(self):
        """计算城市间的欧几里得距离"""
        # 使用scipy的pdist和squareform高效计算距离矩阵
        coords = np.array(self.city_coords)
        distances = pdist(coords, 'euclidean')
        self.distances = squareform(distances)
    
    def get_total_distance(self, route):
        """计算给定路线的总距离"""
        total_distance = 0
        for i in range(len(route) - 1):
            city1 = route[i]
            city2 = route[i + 1]
            total_distance += self.distances[city1][city2]
        
        # 加上从最后一个城市回到起点的距离
        total_distance += self.distances[route[-1]][route[0]]
        return total_distance
    
    def plot_route(self, route, title=None):
        """绘制TSP路线图"""
        plt.figure(figsize=(10, 8))
        
        # 提取路线中城市的坐标
        route_coords = [self.city_coords[city] for city in route]
        route_coords.append(route_coords[0])  # 闭合路线
        
        # 转换为x和y坐标数组
        x = [coord[0] for coord in route_coords]
        y = [coord[1] for coord in route_coords]
        
        # 绘制城市点
        plt.scatter([coord[0] for coord in self.city_coords], 
                   [coord[1] for coord in self.city_coords], c='blue', s=50)
        
        # 标记起点
        plt.scatter(route_coords[0][0], route_coords[0][1], c='red', s=100, marker='*')
        
        # 添加城市标签
        for i, city_id in enumerate(self.city_ids):
            plt.annotate(str(city_id), self.city_coords[i], xytext=(5, 5), 
                         textcoords='offset points')
        
        # 绘制路线
        plt.plot(x, y, 'r-', linewidth=1.5, alpha=0.8)
        
        if title:
            plt.title(title)
        else:
            plt.title(f"TSP 路线图 (总距离: {self.get_total_distance(route):.2f})")
        
        plt.xlabel('X 坐标')
        plt.ylabel('Y 坐标')
        plt.grid(True)
        plt.show()


class Particle:
    """PSO算法中的粒子类"""
    def __init__(self, num_cities):
        # 初始化随机路线
        self.position = list(range(num_cities))
        random.shuffle(self.position)#随机打乱
        
        # 当前最佳路线
        self.best_position = self.position.copy()
        
        # 当前和最佳距离
        self.distance = float('inf')
        self.best_distance = float('inf')
        
        # 速度（存储需要交换的城市对）
        self.velocity = []


class TSPPSO:
    """基于PSO的TSP求解器"""
    def __init__(self, tsp, num_particles=50, max_iter=500, w=0.8, c1=2.0, c2=2.0):
        self.tsp = tsp
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w            # 惯性权重
        self.c1 = c1          # 个体认知权重
        self.c2 = c2          # 社会认知权重
        
        # 初始化粒子群
        self.particles = [Particle(tsp.num_cities) for _ in range(num_particles)]
        
        # 全局最佳路线和距离
        self.global_best_position = None
        self.global_best_distance = float('inf')
        
        # 收敛历史
        self.history_best_distance = []
        
        # 初始化粒子群
        self.initialize_particles()
    
    def initialize_particles(self):
        """初始化粒子群"""
        # 一部分粒子使用贪心策略初始化
        num_greedy = min(int(self.num_particles * 0.2), self.tsp.num_cities)
        
        # 贪心初始化
        for i in range(num_greedy):
            # 随机选择起始城市
            start_city = i % self.tsp.num_cities
            self.particles[i].position = self.greedy_initialize(start_city)
        
        # 使用2-opt改进部分粒子的初始解
        for i in range(self.num_particles):
            if random.random() < 0.5:  # 50%的概率应用2-opt
                self.particles[i].position = self.two_opt(self.particles[i].position)
            
            # 计算初始距离
            self.particles[i].distance = self.tsp.get_total_distance(self.particles[i].position)
            self.particles[i].best_distance = self.particles[i].distance
            self.particles[i].best_position = self.particles[i].position.copy()
            
            # 更新全局最佳解
            if self.particles[i].distance < self.global_best_distance:
                self.global_best_distance = self.particles[i].distance
                self.global_best_position = self.particles[i].position.copy()
    
    def greedy_initialize(self, start_city):
        """使用贪心策略初始化路线"""
        route = [start_city]
        unvisited = set(range(self.tsp.num_cities))
        unvisited.remove(start_city)
        
        current_city = start_city
        while unvisited:
            # 找到最近的未访问城市
            next_city = min(unvisited, 
                            key=lambda city: self.tsp.distances[current_city][city])
            route.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city
        
        return route
    
    def two_opt(self, route, max_iterations=100):
        """使用2-opt算法改进路线"""
        improved = True
        iteration = 0
        best_route = route.copy()
        best_distance = self.tsp.get_total_distance(best_route)
        
        while improved and iteration < max_iterations:
            improved = False
            for i in range(1, len(route) - 1):
                for j in range(i + 1, len(route)):
                    if j - i == 1:
                        continue  # 相邻城市不交换
                    
                    # 尝试2-opt交换
                    new_route = best_route.copy()
                    new_route[i:j+1] = reversed(new_route[i:j+1])
                    new_distance = self.tsp.get_total_distance(new_route)
                    
                    if new_distance < best_distance:
                        best_distance = new_distance
                        best_route = new_route
                        improved = True
                        break
                if improved:
                    break
            iteration += 1
        
        return best_route
    
    def three_opt(self, route, max_iterations=100):
        """使用简化版的3-opt算法改进路线"""
        # 复制路线以避免修改原始路线
        best_route = route.copy()
        best_distance = self.tsp.get_total_distance(best_route)
        
        n = len(route)
        improved = True
        iteration = 0
        
        # 如果城市数量太少，直接返回
        if n < 5:
            return best_route
        
        while improved and iteration < max_iterations:
            improved = False
            
            # 对部分子路径进行2-opt优化
            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    # 2-opt交换：反转i到j之间的路径
                    new_route = best_route.copy()
                    # 反转从i到j的部分
                    new_route[i:j+1] = reversed(new_route[i:j+1])
                    
                    new_distance = self.tsp.get_total_distance(new_route)
                    
                    if new_distance < best_distance:
                        best_route = new_route
                        best_distance = new_distance
                        improved = True
                        break
                
                if improved:
                    break
            
            # 如果2-opt无法改进，尝试3-opt的一种简化形式
            if not improved and n > 5:
                # 随机选择几个位置进行检查
                tries = min(20, n)
                for _ in range(tries):
                    # 随机选择三个不同的切割点
                    i, j, k = sorted(random.sample(range(1, n), 3))
                    
                    
                    original_route = best_route.copy()
                    
                    # 变换1
                    new_route1 = original_route[:i] + original_route[j:k] + original_route[i:j] + original_route[k:]
                    new_distance1 = self.tsp.get_total_distance(new_route1)
                    
                    # 变换2
                    new_route2 = original_route[:i] + list(reversed(original_route[j:k])) + original_route[i:j] + original_route[k:]
                    new_distance2 = self.tsp.get_total_distance(new_route2)
                    
                    # 选择最佳变换
                    if new_distance1 < best_distance and new_distance1 <= new_distance2:
                        best_route = new_route1
                        best_distance = new_distance1
                        improved = True
                        break
                    elif new_distance2 < best_distance:
                        best_route = new_route2
                        best_distance = new_distance2
                        improved = True
                        break
            
            iteration += 1
            
            # 如果连续多次迭代没有改进，则随机扰动一下
            if iteration % 10 == 0 and not improved:
                # 随机交换两个城市
                i, j = random.sample(range(n), 2)
                best_route[i], best_route[j] = best_route[j], best_route[i]
                best_distance = self.tsp.get_total_distance(best_route)
        
        return best_route
    
    def update_velocity(self, particle):
        """更新粒子的速度"""
        new_velocity = []
        
        # 保留部分旧速度（惯性）
        if particle.velocity and random.random() < self.w:
            retain_count = int(len(particle.velocity) * self.w)
            new_velocity.extend(random.sample(particle.velocity, min(retain_count, len(particle.velocity))))
        
        # 向个体最佳位置移动
        if random.random() < self.c1:
            pbest_velocity = self.get_swap_operations(particle.position, particle.best_position)
            pbest_moves = int(len(pbest_velocity) * random.random() * self.c1)
            new_velocity.extend(random.sample(pbest_velocity, min(pbest_moves, len(pbest_velocity))))
        
        # 向全局最佳位置移动
        if random.random() < self.c2:
            gbest_velocity = self.get_swap_operations(particle.position, self.global_best_position)
            gbest_moves = int(len(gbest_velocity) * random.random() * self.c2)
            new_velocity.extend(random.sample(gbest_velocity, min(gbest_moves, len(gbest_velocity))))
        
        particle.velocity = new_velocity
    
    def get_swap_operations(self, source_route, target_route):
        """获取从source_route转换到target_route所需的交换操作"""
        operations = []
        source = source_route.copy()
        
        for i in range(len(source)):
            if source[i] != target_route[i]:
                j = source.index(target_route[i])
                source[i], source[j] = source[j], source[i]
                operations.append((i, j))
        
        return operations
    
    def update_position(self, particle, mutation_rate, local_search_prob):
        """更新粒子的位置"""
        new_position = particle.position.copy()
        
        # 应用速度（交换操作）
        for swap in particle.velocity:
            i, j = swap
            if 0 <= i < len(new_position) and 0 <= j < len(new_position):
                new_position[i], new_position[j] = new_position[j], new_position[i]
        
        # 随机变异以避免早熟收敛
        if random.random() < mutation_rate:  # 变异概率
            i, j = random.sample(range(len(new_position)), 2)
            new_position[i], new_position[j] = new_position[j], new_position[i]
        
        # 周期性应用2-opt局部优化
        if random.random() < local_search_prob:  # 局部搜索概率
            new_position = self.two_opt(new_position, max_iterations=20)
        
        particle.position = new_position
        particle.distance = self.tsp.get_total_distance(new_position)
        
        # 更新个体最佳位置
        if particle.distance < particle.best_distance:
            particle.best_distance = particle.distance
            particle.best_position = particle.position.copy()
            
            # 更新全局最佳位置
            if particle.distance < self.global_best_distance:
                self.global_best_distance = particle.distance
                self.global_best_position = particle.position.copy()
                return True  # 表示有改进
        
        return False  # 表示没有改进
    
    def run(self):
        """运行PSO算法"""
        start_time = time.time()
        no_improvement_count = 0
        
        print(f"开始PSO算法优化 - {self.tsp.num_cities}个城市，{self.num_particles}个粒子")
        
        # 初始参数
        initial_w = self.w
        initial_c1 = self.c1
        initial_c2 = self.c2
        
        best_iteration = 0
        
        for iteration in range(self.max_iter):
            improved = False
            
            # 动态调整参数
            progress = iteration / self.max_iter
            # 惯性权重从0.9线性减小到0.4，促进全局到局部的转换
            self.w = initial_w * (0.9 - 0.5 * progress)
            # 初始阶段增强个体认知，后期增强社会认知
            self.c1 = initial_c1 * (1.0 - 0.5 * progress)
            self.c2 = initial_c2 * (0.5 + 0.5 * progress)
            
            # 根据无改进代数调整参数，增强探索能力
            if no_improvement_count > 20:
                self.w = min(0.9, self.w * 1.1)  # 增大惯性权重
                mutation_rate = 0.2              # 增加变异率
            else:
                mutation_rate = 0.1              # 正常变异率
            
            # 周期性重启策略，每隔一定代数重新初始化部分粒子
            if no_improvement_count > 0 and no_improvement_count % 30 == 0:
                print(f"应用重启策略，重新初始化30%的粒子...")
                self.reinitialize_particles(0.3)
            
            # 遍历所有粒子进行更新
            for i, particle in enumerate(self.particles):
                # 对不同粒子应用不同强度的局部搜索
                local_search_prob = 0.05
                if i < self.num_particles * 0.1:  # 前10%的粒子应用更强的局部搜索
                    local_search_prob = 0.2
                
                self.update_velocity(particle)
                particle_improved = self.update_position(particle, mutation_rate, local_search_prob)
                
                if particle_improved:
                    improved = True
                    no_improvement_count = 0
                    best_iteration = iteration + 1
            
            # 应用社会交流机制：粒子之间交换信息
            if iteration % 5 == 0 and iteration > 0:
                self.apply_social_exchange()
            
            # 记录历史最佳距离
            self.history_best_distance.append(self.global_best_distance)
            
            # 如果长时间没有改进，提前终止
            if not improved:
                no_improvement_count += 1
                if no_improvement_count >= 150:  # 150次迭代无改进则终止
                    print(f"提前终止于第{iteration+1}次迭代：连续{no_improvement_count}次无改进")
                    break
            
            # 每10次迭代输出一次当前最佳路线长度
            if (iteration + 1) % 10 == 0 or iteration == 0:
                print(f"迭代 {iteration+1}/{self.max_iter}: 最佳距离 = {self.global_best_distance:.2f}, W={self.w:.2f}, C1={self.c1:.2f}, C2={self.c2:.2f}")
            
            # 应用强化局部搜索：每隔一段时间对全局最优解进行深度优化
            if iteration % 20 == 0 and iteration > 0:
                old_distance = self.global_best_distance
                self.global_best_position = self.three_opt(self.global_best_position, max_iterations=50)
                self.global_best_distance = self.tsp.get_total_distance(self.global_best_position)
                if self.global_best_distance < old_distance:
                    print(f"  局部搜索改进: {old_distance:.2f} -> {self.global_best_distance:.2f}")
        
        # 对最终结果应用更彻底的局部搜索优化
        print("对最终结果应用局部搜索优化...")
        old_distance = self.global_best_distance
        
        # 先应用2-opt
        self.global_best_position = self.two_opt(self.global_best_position, max_iterations=1000)
        # 然后应用3-opt
        self.global_best_position = self.three_opt(self.global_best_position, max_iterations=200)
        
        self.global_best_distance = self.tsp.get_total_distance(self.global_best_position)
        if self.global_best_distance < old_distance:
            print(f"  最终优化: {old_distance:.2f} -> {self.global_best_distance:.2f}")
        
        end_time = time.time()
        run_time = end_time - start_time
        
        print(f"\n优化完成!")
        print(f"最佳解于第 {best_iteration} 次迭代获得")
        print(f"最终路线长度: {self.global_best_distance:.2f}")
        print(f"运行时间: {run_time:.2f} 秒")
        
        return self.global_best_position, self.global_best_distance, run_time
    

    
    def apply_social_exchange(self):
        """粒子之间的社会交流机制，交换局部最优信息"""
        # 按照性能对粒子排序
        sorted_particles = sorted(self.particles, key=lambda p: p.best_distance)
        
        # 前50%的粒子与后50%的粒子交叉
        half = len(sorted_particles) // 2
        for i in range(half):
            if random.random() < 0.3:  # 30%的概率进行交叉
                good_particle = sorted_particles[i]
                bad_particle = sorted_particles[i + half]
                
                # 使用OX交叉算子（顺序交叉）
                crossover_point1 = random.randint(0, self.tsp.num_cities - 2)
                crossover_point2 = random.randint(crossover_point1 + 1, self.tsp.num_cities - 1)
                
                # 创建交叉子代
                offspring = [-1] * self.tsp.num_cities
                
                # 从好粒子复制片段
                for j in range(crossover_point1, crossover_point2 + 1):
                    offspring[j] = good_particle.best_position[j]
                
                # 从差粒子中按顺序填充剩余城市
                pos = 0
                for city in bad_particle.best_position:
                    if city not in offspring:
                        while pos < self.tsp.num_cities and offspring[pos] != -1:
                            pos += 1
                        if pos < self.tsp.num_cities:
                            offspring[pos] = city
                
                # 用交叉结果更新差粒子
                new_distance = self.tsp.get_total_distance(offspring)
                if new_distance < bad_particle.best_distance:
                    bad_particle.best_position = offspring
                    bad_particle.best_distance = new_distance
                    bad_particle.position = offspring.copy()
                    bad_particle.distance = new_distance
    
    def reinitialize_particles(self, percentage):
        """重启策略：重新初始化一部分表现差的粒子"""
        # 按照当前性能对粒子排序
        sorted_indices = sorted(range(self.num_particles), 
                               key=lambda i: self.particles[i].distance,
                               reverse=True)  # 降序，差的在前面
        
        # 确定要重新初始化的粒子数量
        num_reinit = int(self.num_particles * percentage)
        
        # 重新初始化性能最差的一部分粒子
        for idx in sorted_indices[:num_reinit]:
            particle = self.particles[idx]
            
            # 使用多样化策略重新初始化
            if random.random() < 0.5:
                # 从全局最佳解开始，但进行大幅度变异
                particle.position = self.global_best_position.copy()
                # 随机交换多次
                swaps = random.randint(5, self.tsp.num_cities // 3)
                for _ in range(swaps):
                    i, j = random.sample(range(self.tsp.num_cities), 2)
                    particle.position[i], particle.position[j] = particle.position[j], particle.position[i]
            else:
                # 完全随机初始化
                particle.position = list(range(self.tsp.num_cities))
                random.shuffle(particle.position)
            
            # 应用局部搜索优化初始解
            if random.random() < 0.3:
                particle.position = self.two_opt(particle.position, max_iterations=50)
            
            # 更新粒子状态
            particle.distance = self.tsp.get_total_distance(particle.position)
            
            # 更新个体最佳（如果新位置比粒子历史最佳还好）
            if particle.distance < particle.best_distance:
                particle.best_position = particle.position.copy()
                particle.best_distance = particle.distance


def solve_tsp_with_pso(file_path, num_particles=50, max_iter=300, w=0.8, c1=2.0, c2=2.0, 
                     run_times=3, plot_result=True):
    """使用PSO求解TSP问题，多次运行取最优结果"""
    print(f"\n求解文件: {file_path}")
    
    # 加载TSP问题
    tsp = TSP(file_path)
    
    # 根据问题规模调整参数
    if tsp.num_cities > 80:
        # 大规模问题增加粒子数和迭代次数
        num_particles = max(num_particles, 100)
        max_iter = max(max_iter, 500)
    elif tsp.num_cities > 50:
        # 中等规模问题
        num_particles = max(num_particles, 80)
        max_iter = max(max_iter, 400)
    
    print(f"参数设置: 粒子数={num_particles}, 最大迭代次数={max_iter}, w={w}, c1={c1}, c2={c2}")
    
    # 多次运行取最优结果
    best_overall_route = None
    best_overall_distance = float('inf')
    total_time = 0
    
    for run in range(run_times):
        print(f"\n运行 {run+1}/{run_times}:")
        
        pso = TSPPSO(
            tsp=tsp,
            num_particles=num_particles,
            max_iter=max_iter,
            w=w,
            c1=c1,
            c2=c2
        )
        
        route, distance, run_time = pso.run()
        total_time += run_time
        
        if distance < best_overall_distance:
            best_overall_distance = distance
            best_overall_route = route.copy()
            best_pso = pso
    
    # 输出最终结果
    print("\n最终结果:")
    print(f"最佳路线长度: {best_overall_distance:.2f}")
    print(f"平均运行时间: {total_time/run_times:.2f} 秒")
    
    # 绘制收敛曲线和最佳路线图
    if plot_result:
       
        
        city_ids = [tsp.city_ids[i] for i in best_overall_route]
        route_str = "->".join(map(str, city_ids)) + "->" + str(city_ids[0])
        title = f"最佳TSP路线 (长度: {best_overall_distance:.2f})"
        tsp.plot_route(best_overall_route, title)
    
    return best_overall_route, best_overall_distance


def simplified_solve_tsp_with_pso(file_path, max_iter=300, run_times=3, plot_result=True):
    """简化版的PSO求解TSP问题，专注于提高收敛性能"""
    print(f"\n求解文件: {file_path}")
    
    # 加载TSP问题
    tsp = TSP(file_path)
    
    # 根据问题规模调整参数
    num_particles = 100
    if tsp.num_cities > 80:
        num_particles = 120
        max_iter = max(max_iter, 400)
    
    # 使用更激进的参数设置
    w = 0.6    # 较小的惯性权重，促进局部搜索
    c1 = 1.5     # 减小个体认知
    c2 = 2.5     # 增强社会学习
    
    print(f"参数设置: 粒子数={num_particles}, 最大迭代次数={max_iter}, w={w}, c1={c1}, c2={c2}")
    
    # 多次运行取最优结果
    best_overall_route = None
    best_overall_distance = float('inf')
    total_time = 0
    
    for run in range(run_times):
        print(f"\n运行 {run+1}/{run_times}:")
        
        # 1. 使用贪心算法生成一个初始路径
        initial_path = list(range(tsp.num_cities))
        
        # 对初始路径应用2-opt改进
        improved_path = initial_path.copy()
        improved = True
        while improved:
            improved = False
            best_distance = tsp.get_total_distance(improved_path)
            
            for i in range(1, tsp.num_cities - 1):
                for j in range(i + 1, tsp.num_cities):
                    # 2-opt交换
                    new_path = improved_path.copy()
                    new_path[i:j+1] = reversed(new_path[i:j+1])
                    
                    new_distance = tsp.get_total_distance(new_path)
                    if new_distance < best_distance:
                        improved_path = new_path
                        best_distance = new_distance
                        improved = True
                        break
                if improved:
                    break
        
        # 2. 创建PSO实例并进行优化
        start_time = time.time()
        
        pso = TSPPSO(
            tsp=tsp,
            num_particles=num_particles,
            max_iter=max_iter,
            w=w,
            c1=c1,
            c2=c2
        )
        
        # 使用改进路径作为部分粒子的初始位置
        for i in range(min(10, pso.num_particles)):
            pso.particles[i].position = improved_path.copy()
            # 添加一些随机性
            for _ in range(5):
                idx1, idx2 = random.sample(range(tsp.num_cities), 2)
                pso.particles[i].position[idx1], pso.particles[i].position[idx2] = \
                    pso.particles[i].position[idx2], pso.particles[i].position[idx1]
                
            pso.particles[i].distance = tsp.get_total_distance(pso.particles[i].position)
            pso.particles[i].best_position = pso.particles[i].position.copy()
            pso.particles[i].best_distance = pso.particles[i].distance
            
            if pso.particles[i].distance < pso.global_best_distance or pso.global_best_position is None:
                pso.global_best_distance = pso.particles[i].distance
                pso.global_best_position = pso.particles[i].position.copy()
        
        # 运行PSO
        route, distance, run_time = pso.run()
        run_time = time.time() - start_time
        total_time += run_time
        
        if distance < best_overall_distance:
            best_overall_distance = distance
            best_overall_route = route.copy()
           
    
    # 输出最终结果
    print("\n最终结果:")
    print(f"最佳路线长度: {best_overall_distance:.2f}")
    print(f"平均运行时间: {total_time/run_times:.2f} 秒")
    
    # 绘制收敛曲线和最佳路线图
    if plot_result:
       
        
        city_ids = [tsp.city_ids[i] for i in best_overall_route]
        title = f"最佳TSP路线 (长度: {best_overall_distance:.2f})"
        tsp.plot_route(best_overall_route, title)
    
    return best_overall_route, best_overall_distance


if __name__ == "__main__":
    # 数据文件路径
    data_files = [
        "./data/eil51.tsp",
        "./data/eil76.tsp",
        "./data/eil101.tsp"
    ]
    
    # 检查文件是否存在
    existing_files = [f for f in data_files if os.path.exists(f)]
    
    if not existing_files:
        print("未找到任何指定的TSP数据文件！")
        exit(1)
    
    results = {}
    
    # 依次求解每个数据文件
    for file_path in existing_files:
        file_name = os.path.basename(file_path)
        print(f"\n{'='*50}")
        print(f"求解TSP问题: {file_name}")
        print(f"{'='*50}")
        
        # 使用简化版求解函数
        best_route, best_distance = simplified_solve_tsp_with_pso(
            file_path=file_path,
            max_iter=400,        # 增加最大迭代次数
            run_times=3,         # 运行次数
            plot_result=True     # 是否绘图
        )
        
        results[file_name] = best_distance
    
    # 打印所有结果汇总
    print("\n========== 结果汇总 ==========")
    for file_name, distance in results.items():
        print(f"问题: {file_name}, 最佳路线长度: {distance:.2f}")
