import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class TSP:
    def __init__(self, file_path):
        """初始化TSP问题，从文件中读取城市坐标"""
        self.cities = []  # 存储城市坐标
        self.city_names = []  # 存储城市名称
        self.distances = None  # 距离矩阵
        self.num_cities = 0  # 城市数量
        
        # 从文件中加载城市数据
        self.load_data(file_path)
        
        # 计算城市间距离矩阵
        self.calculate_distances()
    
    def load_data(self, file_path):
        """从文件中加载TSP数据"""
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    # 去除首尾空白，并按空格分割
                    line = line.strip()
                    
                    # 检查是否为EOF标记（文件结束标志）
                    if line == "EOF":
                        break
                        
                    parts = line.split()
                    # 确保行包含3个元素：城市编号、x坐标、y坐标
                    if len(parts) == 3:
                        try:
                            city_id = int(parts[0])
                            x = float(parts[1])
                            y = float(parts[2])
                            self.cities.append((x, y))
                            self.city_names.append(city_id)
                        except ValueError:
                            # 跳过无法解析为数字的行
                            continue
        except Exception as e:
            print(f"读取文件时出错: {e}")
            return
        
        self.num_cities = len(self.cities)
        print(f"已加载{self.num_cities}个城市")
    
    def calculate_distances(self):
        """计算城市间距离矩阵"""
        self.distances = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(i+1, self.num_cities):
                # 计算欧几里得距离
                dist = np.sqrt((self.cities[i][0] - self.cities[j][0])**2 + 
                               (self.cities[i][1] - self.cities[j][1])**2)
                self.distances[i][j] = dist
                self.distances[j][i] = dist  # 距离矩阵是对称的
    
    def get_path_distance(self, path):
        """计算路径总长度，包括返回起点的距离"""
        total_distance = 0
        for i in range(len(path) - 1):
            total_distance += self.distances[path[i]][path[i+1]]
        # 添加从最后一个城市返回起点的距离
        total_distance += self.distances[path[-1]][path[0]]
        return total_distance
    
    def plot_path(self, path, ax=None, title="TSP最优路径"):
        """绘制TSP路径"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # 提取所有城市的坐标
        xs = [self.cities[i][0] for i in path]
        ys = [self.cities[i][1] for i in path]
        
        # 将路径闭合（返回起点）
        xs.append(xs[0])
        ys.append(ys[0])
        
        # 绘制城市点
        ax.scatter(xs[:-1], ys[:-1], c='blue', s=50)
        
        # 标记起始城市
        ax.scatter(xs[0], ys[0], c='red', s=100, marker='*', label='起点')
        
        # 为城市添加标签
        for i, city_idx in enumerate(path):
            if i < len(path):  # 避免对闭合点重复标注
                ax.annotate(f"{self.city_names[city_idx]}", 
                           (xs[i], ys[i]),
                           xytext=(5, 5),
                           textcoords='offset points')
        
        # 绘制路径
        ax.plot(xs, ys, 'r-', alpha=0.7)
        
        ax.set_title(title)
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        ax.legend()
        
        return ax

class TSP_GA:
    def __init__(self, tsp, pop_size=100, crossover_rate=0.95, mutation_rate=0.1, 
                 max_generations=500, elite_size=10, visualize=True):
        """初始化TSP遗传算法"""
        self.tsp = tsp
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.elite_size = elite_size
        self.population = []
        self.best_individual = None
        self.best_fitness = 0
        self.best_distance = float('inf')
        self.history_best_distance = []
        self.history_best_individual = []
        self.no_improvement_count = 0  # 用于记录没有改进的代数
        self.visualize = visualize
        
        # 初始化种群
        self.initialize_population()
        
        # 设置可视化
        if self.visualize:
            self.setup_visualization()
    
    def initialize_population(self):
        """初始化种群，每个个体是一个城市访问顺序的排列"""
        # 使用贪心策略创建一部分初始解
        n_greedy = min(int(self.pop_size * 0.2), self.tsp.num_cities)  # 20%的种群使用贪心生成
        
        # 生成贪心解
        for i in range(n_greedy):
            # 从不同城市出发创建贪心路径
            start_city = i % self.tsp.num_cities
            individual = self.greedy_path(start_city)
            self.population.append(individual)
        
        # 其余个体随机生成
        for _ in range(self.pop_size - n_greedy):
            individual = list(range(self.tsp.num_cities))
            np.random.shuffle(individual)
            self.population.append(individual)
        
        # 应用2-opt局部搜索优化初始解
        for i in range(self.pop_size):
            self.population[i] = self.two_opt(self.population[i])
        
        # 评估初始种群
        self.evaluate_population()
    
    def greedy_path(self, start_idx):
        """从指定城市开始构建贪心路径"""
        path = [start_idx]
        unvisited = set(range(self.tsp.num_cities))
        unvisited.remove(start_idx)
        
        current = start_idx
        while unvisited:
            next_city = min(unvisited, 
                           key=lambda city: self.tsp.distances[current][city])
            path.append(next_city)
            unvisited.remove(next_city)
            current = next_city
            
        return path
    
    def two_opt(self, route):
        """使用2-opt算法优化路径"""
        # 安全检查：城市数量太少时不执行2-opt
        if len(route) < 4:
            return route.copy()
            
        improved = True
        best_route = route.copy()
        best_distance = self.tsp.get_path_distance(best_route)
        
        iteration_count = 0
        max_iterations = min(100, self.tsp.num_cities * 5)  # 限制最大迭代次数
        
        while improved and iteration_count < max_iterations:
            improved = False
            iteration_count += 1
            
            # 随机选择一些点进行检查，避免检查所有可能的交换
            check_points = min(self.tsp.num_cities, 20)  # 最多检查20个点
            available_i_points = list(range(1, len(route) - 2))
            i_sample_size = min(check_points, len(available_i_points))
            
            # 防止空列表抽样
            if i_sample_size <= 0:
                break
                
            i_points = np.random.choice(available_i_points, i_sample_size, replace=False)
            
            for i in i_points:
                available_j_points = list(range(i + 1, len(route)))
                j_sample_size = min(check_points, len(available_j_points))
                
                # 防止空列表抽样
                if j_sample_size <= 0:
                    continue
                    
                j_points = np.random.choice(available_j_points, j_sample_size, replace=False)
                
                for j in j_points:
                    if j - i == 1:
                        continue  # 相邻边不交换
                    
                    # 计算当前边的长度
                    current_distance = (self.tsp.distances[best_route[i-1]][best_route[i]] + 
                                       self.tsp.distances[best_route[j]][best_route[j+1 if j < len(route)-1 else 0]])
                    
                    # 计算交叉边的长度
                    new_distance = (self.tsp.distances[best_route[i-1]][best_route[j]] + 
                                   self.tsp.distances[best_route[i]][best_route[j+1 if j < len(route)-1 else 0]])
                    
                    # 如果交叉后的路径更短，则执行2-opt交换
                    if new_distance < current_distance:
                        # 执行交换
                        best_route = self.two_opt_swap(best_route.copy(), i, j)
                        best_distance = self.tsp.get_path_distance(best_route)
                        improved = True
                        break
                    
                
                if improved:
                    break
        
        return best_route
    
    def two_opt_swap(self, route, i, j):
        """执行2-opt交换"""
        # 翻转从位置i到位置j的路径段
        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
        return new_route
    
    def evaluate_population(self):
        """评估种群中每个个体的适应度（路径长度的倒数）"""
        distances = []
        for individual in self.population:
            distance = self.tsp.get_path_distance(individual)
            distances.append(distance)
        
        # 将距离转换为适应度（距离越短，适应度越高）
        scale_factor = np.mean(distances) / 50  # 动态调整缩放因子
        fitness_values = np.exp(-np.array(distances) / scale_factor)
        
        # 找到最佳个体
        best_idx = np.argmin(distances)
        current_best_distance = distances[best_idx]
        current_best_fitness = fitness_values[best_idx]
        
        # 更新全局最佳解
        if current_best_distance < self.best_distance:
            improvement = self.best_distance - current_best_distance
            
            self.best_fitness = current_best_fitness
            self.best_distance = current_best_distance
            self.best_individual = self.population[best_idx].copy()
            self.no_improvement_count = 0  # 重置无改进计数
        else:
            self.no_improvement_count += 1  # 增加无改进计数
            
        # 记录历史最佳解
        self.history_best_distance.append(self.best_distance)
        self.history_best_individual.append(self.best_individual.copy())
            
        return fitness_values, distances
    
    def select_parents(self, fitness_values, distances):
        """选择父代 - 使用锦标赛选择"""
        # 精英保留
        elite_indices = np.argsort(distances)[:self.elite_size]
        elite_population = [self.population[i].copy() for i in elite_indices]
        
        # 锦标赛选择
        tournament_size = max(3, int(self.pop_size * 0.1))
        selected_population = elite_population.copy()
        
        while len(selected_population) < self.pop_size:
            # 锦标赛选择
            tournament_indices = np.random.choice(range(self.pop_size), size=tournament_size, replace=False)
            # 从锦标赛中选出最优个体
            winner_idx = tournament_indices[np.argmin([distances[i] for i in tournament_indices])]
            selected_population.append(self.population[winner_idx].copy())
        
        # 随机打乱非精英个体的顺序
        np.random.shuffle(selected_population[self.elite_size:])
        
        return selected_population
    
    def order_crossover(self, parent1, parent2):
        """实施顺序交叉 (Order Crossover, OX)"""
        if np.random.rand() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1 = [-1] * len(parent1)
        child2 = [-1] * len(parent2)
        
        # 随机选择连续的子路径
        route_len = len(parent1)
        segment_len = max(2, int(route_len * np.random.beta(2, 2)))  # Beta分布倾向于中间值
        start = np.random.randint(0, route_len - segment_len + 1)
        end = start + segment_len - 1
        
        # 复制子路径
        child1[start:end+1] = parent1[start:end+1]
        child2[start:end+1] = parent2[start:end+1]
        
        # 填充剩余位置
        # 对于child1，使用parent2的顺序
        p2_idx = 0
        for i in range(len(child1)):
            if child1[i] == -1:  # 如果位置尚未填充
                # 找到下一个不在child1中的parent2中的城市
                while parent2[p2_idx] in child1:
                    p2_idx = (p2_idx + 1) % len(parent2)
                child1[i] = parent2[p2_idx]
                p2_idx = (p2_idx + 1) % len(parent2)
        
        # 对于child2，使用parent1的顺序
        p1_idx = 0
        for i in range(len(child2)):
            if child2[i] == -1:  # 如果位置尚未填充
                # 找到下一个不在child2中的parent1中的城市
                while parent1[p1_idx] in child2:
                    p1_idx = (p1_idx + 1) % len(parent1)
                child2[i] = parent1[p1_idx]
                p1_idx = (p1_idx + 1) % len(parent1)
        
        return child1, child2
    
    def mutation(self, individual):
        """变异操作 - 使用多种变异策略"""
        if np.random.rand() < self.mutation_rate:
            # 随机选择变异类型
            mutation_type = np.random.choice(["swap", "inversion", "insertion"])
            
            if mutation_type == "swap":
                # 交换变异 - 随机交换两个城市
                pos1, pos2 = np.random.choice(range(len(individual)), size=2, replace=False)
                individual[pos1], individual[pos2] = individual[pos2], individual[pos1]
            
            elif mutation_type == "inversion":
                # 反转变异 - 反转路径的一个片段
                pos1, pos2 = sorted(np.random.choice(range(len(individual)), size=2, replace=False))
                individual[pos1:pos2+1] = individual[pos1:pos2+1][::-1]
            
            elif mutation_type == "insertion":
                # 插入变异 - 取出一个城市并插入到其他位置
                pos1, pos2 = np.random.choice(range(len(individual)), size=2, replace=False)
                if pos1 < pos2:
                    city = individual[pos1]
                    individual = individual[:pos1] + individual[pos1+1:pos2+1] + [city] + individual[pos2+1:]
                else:
                    city = individual[pos1]
                    individual = individual[:pos2] + [city] + individual[pos2:pos1] + individual[pos1+1:]
            
            # 概率性地应用2-opt优化
            if np.random.rand() < 0.5:  # 10%的概率
                individual = self.two_opt(individual)
                
        return individual
    
    def crossover_population(self, population):
        """对种群进行交叉操作"""
        children = []
        
        # 确保精英直接传递到下一代
        elite_size = min(self.elite_size, len(population))
        children.extend(population[:elite_size])
        
        # 对剩余个体进行交叉
        for i in range(elite_size, len(population), 2):
            if i + 1 < len(population):
                child1, child2 = self.order_crossover(population[i], population[i+1])
                children.append(child1)
                children.append(child2)
            else:
                children.append(population[i].copy())  # 如果剩余奇数个，最后一个直接复制
        
        return children[:self.pop_size]  # 确保种群大小不变
    
    def mutate_population(self, population):
        """对种群进行变异操作"""
        # 精英保留，不进行变异
        elite_size = min(self.elite_size, len(population))
        for i in range(elite_size, len(population)):
            population[i] = self.mutation(population[i])
        
        return population
    
    def evolve(self):
        """进化一代种群"""
        # 评估当前种群
        fitness_values, distances = self.evaluate_population()
        
        # 根据无改进代数动态调整变异率
        if self.no_improvement_count > 20:
            # 如果连续20代没有改进，增加变异率
            self.mutation_rate = min(0.5, self.mutation_rate * 1.05)
            
            # 重置计数器
            if self.no_improvement_count > 50:
                # 重置一部分种群
                n_reset = int(self.pop_size * 0.3)  # 重置30%的种群
                for i in range(n_reset):
                    # 保留精英个体
                    if i >= self.elite_size:
                        # 使用随机方法重新生成
                        new_individual = list(range(self.tsp.num_cities))
                        np.random.shuffle(new_individual)
                        self.population[-(i+1)] = new_individual
                
                self.no_improvement_count = 0
        else:
            # 正常运行时，逐渐减小变异率
            self.mutation_rate = max(0.01, self.mutation_rate * 0.999)
        
        # 选择父代
        selected_population = self.select_parents(fitness_values, distances)
        
        # 交叉
        crossovered_population = self.crossover_population(selected_population)
        
        # 变异
        mutated_population = self.mutate_population(crossovered_population)
        
        # 更新种群
        self.population = mutated_population
    
    def setup_visualization(self):
        """设置简化的可视化界面（低质量模式）"""
        # 创建图形，设置较低的DPI以提高性能
        self.fig = plt.figure(figsize=(12, 5), dpi=80)
        
        # 路径图
        self.ax1 = self.fig.add_subplot(121)
        self.ax1.set_title('当前最优路径')
        
        # 收敛曲线
        self.ax2 = self.fig.add_subplot(122)
        self.ax2.set_title('收敛曲线')
        self.ax2.set_xlabel('代数')
        self.ax2.set_ylabel('最短路径长度')
        
        # 初始化收敛曲线
        self.line, = self.ax2.plot([], [], 'g-')
        
        plt.tight_layout()
        plt.ion()  # 打开交互模式
        
        # 启用更快的渲染器
        plt.rcParams['path.simplify'] = True
        plt.rcParams['path.simplify_threshold'] = 0.8
        plt.rcParams['agg.path.chunksize'] = 10000
        
        plt.show(block=False)
        
    def _update_plot(self, generation):
        """更新可视化图表 - 简化版本，减少卡顿"""
        if not self.visualize:
            return
            
        try:
            # 更新收敛曲线（只更新数据，不重绘整个图形）
            self.line.set_data(range(len(self.history_best_distance)), self.history_best_distance)
            self.ax2.relim()
            self.ax2.autoscale_view()
            
            # 更新收敛曲线标题（只在整50代更新一次）
            if generation % 50 == 0 or generation == self.max_generations - 1:
                self.ax2.set_title(f'收敛曲线 (当前最短距离: {self.best_distance:.2f})')
            
            # 不频繁更新路径图（每20代更新一次）
            if generation % 20 == 0 or generation == self.max_generations - 1:
                # 清除当前路径图
                self.ax1.clear()
                
                # 提取所有城市的坐标
                path = self.best_individual
                xs = [self.tsp.cities[i][0] for i in path]
                ys = [self.tsp.cities[i][1] for i in path]
                
                # 将路径闭合（返回起点）
                xs.append(xs[0])
                ys.append(ys[0])
                
                # 绘制城市点
                self.ax1.scatter(xs[:-1], ys[:-1], c='blue', s=30)
                
                # 标记起始城市
                self.ax1.scatter(xs[0], ys[0], c='red', s=80, marker='*')
                
                # 为所有城市显示编号标签，无论城市数量多少
                for i, city_idx in enumerate(path):
                    if i < len(path):  # 避免对闭合点重复标注
                        self.ax1.annotate(f"{self.tsp.city_names[city_idx]}", 
                                   (xs[i], ys[i]),
                                   xytext=(2, 2),
                                   textcoords='offset points',
                                   fontsize=6)  # 使用更小的字体
                
                # 绘制路径
                self.ax1.plot(xs, ys, 'r-', alpha=0.6)
                
                self.ax1.set_title(f"当前最优路径 (代数 {generation+1}/{self.max_generations})")
            
            # 减少重绘频率
            if generation % 20 == 0 or generation == self.max_generations - 1:
                self.fig.canvas.draw_idle()
            else:
                # 轻量级更新，只更新收敛曲线
                self.ax2.draw_artist(self.line)
                self.fig.canvas.blit(self.ax2.bbox)
            
            # 控制刷新率
            if generation % 5 == 0:
                self.fig.canvas.flush_events()
                plt.pause(0.001)  # 使用更小的暂停时间
        except Exception as e:
            print(f"可视化更新错误 (可忽略): {e}")
    
    def run(self):
        """运行遗传算法"""
        # 记录起始时间，用于计算性能
        generation_start_time = time.time()
        visual_update_time = 0
        
        # 初始化可视化
        if self.visualize:
            self._update_plot(0)
        
        for generation in range(self.max_generations):
            # 进化一代
            self.evolve()
            
            # 输出进度
            if generation % 50 == 0:
                current_time = time.time()
                elapsed = current_time - generation_start_time
                gen_per_sec = (generation + 1) / max(elapsed, 0.001)
                remaining = (self.max_generations - generation - 1) / max(gen_per_sec, 0.001)
                print(f"Generation {generation}/{self.max_generations} - Best: {self.best_distance:.2f} - Speed: {gen_per_sec:.1f} gen/s - Time left: {remaining:.1f}s")
            
            # 优化可视化更新 - 根据代数决定是否更新
            if self.visualize and (generation % 5 == 0 or generation == self.max_generations - 1):
                # 测量可视化时间占比
                vis_start = time.time()
                self._update_plot(generation)
                visual_update_time += (time.time() - vis_start)
            
            # 提前停止条件：如果超过100代没有改进，且已经至少运行了总代数的30%
            if self.no_improvement_count > 100 and generation > self.max_generations * 0.3:
                print(f"提前停止于代数 {generation}: {self.no_improvement_count} 代未改进")
                break
        
        # 输出性能统计
        total_time = time.time() - generation_start_time
        print(f"\n总运行时间: {total_time:.2f}秒")
        if self.visualize and generation > 0:
            visual_percentage = (visual_update_time / total_time) * 100
            print(f"可视化占用时间: {visual_update_time:.2f}秒 ({visual_percentage:.1f}%)")
        print(f"平均每代耗时: {total_time/(generation+1):.4f}秒")
        
        # 显示最终结果
        if self.visualize:
            plt.ioff()  # 关闭交互模式
            self.plot_best_path()  # 绘制最终的最优路径
        
        return self.best_individual, self.best_distance
    
    def plot_best_path(self):
        """绘制最佳路径"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 提取所有城市的坐标
        path = self.best_individual
        xs = [self.tsp.cities[i][0] for i in path]
        ys = [self.tsp.cities[i][1] for i in path]
        
        # 将路径闭合（返回起点）
        xs.append(xs[0])
        ys.append(ys[0])
        
        # 绘制城市点
        ax.scatter(xs[:-1], ys[:-1], c='blue', s=50)
        
        # 标记起始城市
        ax.scatter(xs[0], ys[0], c='red', s=100, marker='*', label='起点')
        
        # 为所有城市添加标签，字体加大以便更清晰显示
        for i, city_idx in enumerate(path):
            if i < len(path):  # 避免对闭合点重复标注
                ax.annotate(f"{self.tsp.city_names[city_idx]}", 
                           (xs[i], ys[i]),
                           xytext=(5, 5),
                           textcoords='offset points',
                           fontsize=9,  # 增大字体大小
                           weight='bold')  # 加粗字体
        
        # 绘制路径
        ax.plot(xs, ys, 'r-', alpha=0.7)
        
        ax.set_title(f"最优路径 (总长度: {self.best_distance:.2f})")
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        ax.legend()
        
        plt.tight_layout()
        plt.show()

def run_tsp_ga(file_path):
    """运行TSP遗传算法"""
    # 从文件名中提取问题名称
    file_name = os.path.basename(file_path)
    problem_name = os.path.splitext(file_name)[0]
    
    print(f"正在解决TSP问题: {problem_name}")
    print("="*50)
    
    # 加载TSP问题
    tsp = TSP(file_path)
    
    # 根据城市数量调整参数
    num_cities = tsp.num_cities
    
    # 调整参数
    if num_cities <= 60:  # eil51
        pop_size = int(max(200, num_cities * 5))
        max_generations = 500
        elite_size = int(pop_size * 0.15)
        mutation_rate = 0.20
    elif num_cities <= 80:  # eil76
        pop_size = int(max(250, num_cities * 4))
        max_generations = 800
        elite_size = int(pop_size * 0.12)
        mutation_rate = 0.15
    else:  # eil101 或更大规模
        pop_size = int(max(300, num_cities * 3.5))
        max_generations = 1000
        elite_size = int(pop_size * 0.10)
        mutation_rate = 0.12
    
    # 创建并运行遗传算法
    ga = TSP_GA(
        tsp=tsp,
        pop_size=pop_size,
        crossover_rate=0.95,
        mutation_rate=mutation_rate,
        max_generations=max_generations,
        elite_size=elite_size,
        visualize=True  # 启用可视化
    )
    
    print(f"参数设置: 种群大小={pop_size}, 最大代数={max_generations}")
    print(f"          交叉概率=0.95, 变异概率={mutation_rate}, 精英数量={elite_size}")
    print(f"          城市数量={tsp.num_cities}")
    print(f"          已启用低质量可视化模式")
    print("-"*50)
    
    print("开始GA优化...")
    start_time = time.time()
    
    # 运行遗传算法
    best_path, best_distance = ga.run()
    
    # 计算运行时间
    elapsed_time = time.time() - start_time
    
    print("\n优化完成！")
    print(f"运行时间: {elapsed_time:.2f}秒")
    print(f"最短路径长度: {best_distance:.2f}")
    print("-"*50)
    
    return ga

def create_sample_data_file(file_name, num_cities):
    """创建示例数据文件，当找不到实际文件时使用"""
    print(f"正在创建示例数据文件: {file_name}...")
    
    # 确保目标目录存在
    file_dir = os.path.dirname(file_name)
    if file_dir and not os.path.exists(file_dir):
        try:
            os.makedirs(file_dir)
            print(f"创建目录: {file_dir}")
        except Exception as e:
            print(f"创建目录时出错: {e}")
            return False
    
    # 生成随机城市坐标
    np.random.seed(42)
    cities = []
    
    for i in range(1, num_cities + 1):
        # 生成0-100范围内的随机坐标
        x = np.random.randint(0, 100)
        y = np.random.randint(0, 100)
        cities.append((i, x, y))
    
    # 写入文件
    try:
        with open(file_name, 'w') as f:
            for city_id, x, y in cities:
                f.write(f"{city_id} {x} {y}\n")
        print(f"示例数据文件 {file_name} 创建成功！")
        return True
    except Exception as e:
        print(f"创建文件时出错: {e}")
        return False

if __name__ == "__main__":
    # 显示当前工作目录
    print(f"当前工作目录: {os.getcwd()}")
    
    # 先尝试创建data目录（如果不存在）
    data_dir = "./data"
    if not os.path.exists(data_dir):
        try:
            os.makedirs(data_dir)
            print(f"已创建data目录: {data_dir}")
        except Exception as e:
            print(f"创建data目录时出错: {e}")
    
    # 检测是否存在数据文件
    possible_locations = [
        # 同级目录下的data文件夹（优先检查）
        "./data/eil51.tsp", "./data/eil76.tsp", "./data/eil101.tsp",
        # 当前目录
        "./eil51.tsp", "./eil76.tsp", "./eil101.tsp", 
        # 父目录
        "../eil51.tsp", "../eil76.tsp", "../eil101.tsp",
        # 没有路径前缀
        "eil51.tsp", "eil76.tsp", "eil101.tsp",
        # 父目录的data文件夹
        "../data/eil51.tsp", "../data/eil76.tsp", "../data/eil101.tsp",
    ]
    
    available_files = []
    for file_name in possible_locations:
        if os.path.exists(file_name):
            available_files.append(file_name)
            print(f"找到文件: {file_name}")
    
    if not available_files:
        print("未找到任何TSP数据文件")
        print("请选择操作:")
        print("1. 创建示例数据文件")
        print("2. 退出程序")
        
        while True:
            try:
                choice = int(input("\n请选择 (1/2): "))
                if choice == 1:
                    # 创建示例数据文件（保存到data目录）
                    files_to_create = [
                        ("./data/eil51.tsp", 51),
                        ("./data/eil76.tsp", 76),
                        ("./data/eil101.tsp", 101)
                    ]
                    
                    print("正在创建示例数据文件到data目录...")
                    for file_name, num_cities in files_to_create:
                        success = create_sample_data_file(file_name, num_cities)
                        if success:
                            available_files.append(file_name)
                    
                    if available_files:
                        print("已创建示例数据文件，继续运行程序...")
                        break
                    else:
                        print("创建文件失败，程序退出")
                        exit(1)
                        
                elif choice == 2:
                    print("程序退出")
                    exit(0)
                else:
                    print("无效的选择，请重新输入")
            except ValueError:
                print("请输入有效的数字")
    
    # 选择要解决的问题
    print("可用的TSP问题：")
    for i, file_name in enumerate(available_files):
        print(f"{i+1}. {file_name}")
    
    while True:
        try:
            choice = int(input("\n请选择要解决的TSP问题 (输入序号): "))
            if 1 <= choice <= len(available_files):
                selected_file = available_files[choice-1]
                break
            else:
                print("无效的选择，请重新输入")
        except ValueError:
            print("请输入有效的数字")
    
    # 运行算法
    run_tsp_ga(selected_file) 