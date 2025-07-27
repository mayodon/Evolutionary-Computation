import numpy as np
import random
import time
import matplotlib.pyplot as plt
from operator import itemgetter

class KnapsackGA:
    def __init__(self, capacity, weights, values, pop_size=200, max_gen=500, 
                 crossover_rate=0.9, mutation_rate=0.05, elite_rate=0.2):
        
        self.capacity = capacity
        self.weights = weights
        self.values = values
        self.n_items = len(weights)
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_rate = elite_rate
        self.elite_num = int(pop_size * elite_rate)
        
        # 计算价值密度（性价比）
        self.density = [v/w if w > 0 else 0 for v, w in zip(values, weights)]
        
        # 初始化种群
        self.population = self.initialize_population()
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_solution = None
        self.best_fitness = 0
        self.no_improvement_count = 0  # 用于提前终止
    
    def initialize_population(self):
        """初始化种群，使用多种策略生成初始解"""
        population = np.zeros((self.pop_size, self.n_items), dtype=int)
        
        # 1. 生成一部分贪心解（根据性价比）
        greedy_num = int(self.pop_size * 0.2)
        sorted_items = sorted([(i, self.density[i]) for i in range(self.n_items)], 
                             key=itemgetter(1), reverse=True)#降序排列 itemgetter(1)指的性价比
        sorted_indices = [item[0] for item in sorted_items]
        
        for i in range(greedy_num):
            chromosome = np.zeros(self.n_items, dtype=int)
            remaining_capacity = self.capacity
            noise_factor = np.random.normal(1, 0.1)  # 加入随机扰动，生成多样的贪心解
            
            # 根据性价比选择物品
            for idx in sorted_indices:
                if self.weights[idx] * noise_factor <= remaining_capacity:
                    chromosome[idx] = 1
                    remaining_capacity -= self.weights[idx]
            population[i] = chromosome
        
        # 2. 生成一部分随机解但倾向选择价值较高的物品
        random_with_bias_num = int(self.pop_size * 0.3)
        for i in range(greedy_num, greedy_num + random_with_bias_num):
            chromosome = np.zeros(self.n_items, dtype=int)
            remaining_capacity = self.capacity
            
            # 根据物品价值给予概率偏向
            item_probs = np.array(self.values) / sum(self.values)
            items_order = np.random.choice(range(self.n_items), size=self.n_items, 
                                          replace=False, p=item_probs)#根据物品价值给予概率偏向 不放回抽样
            
            for idx in items_order:
                if self.weights[idx] <= remaining_capacity:
                    chromosome[idx] = 1
                    remaining_capacity -= self.weights[idx]
            population[i] = chromosome
        
        # 3. 生成完全随机解，保证多样性
        for i in range(greedy_num + random_with_bias_num, self.pop_size):
            chromosome = np.random.randint(0, 2, self.n_items)#随机生成0-1的二进制数
            population[i] = self.repair_solution(chromosome)#修复不可行解
        
        return population
    
    def calculate_fitness(self, chromosome):
        """计算个体的适应度（背包中物品的总价值）"""
        total_weight = np.sum(chromosome * self.weights)
        if total_weight > self.capacity:
            return 0  # 超过背包容量，适应度为0 相当于罚函数
        return np.sum(chromosome * self.values)
    
    def select_parents(self, fitness):
        """使用锦标赛选择父代"""
        parents_indices = []
        tournament_size = 5  # 锦标赛大小
        
        for _ in range(self.pop_size):
            # 随机选择锦标赛选手
            tournament_indices = np.random.choice(range(self.pop_size), tournament_size, replace=False)
            tournament_fitness = [fitness[i] for i in tournament_indices]
            # 选择锦标赛中的最佳个体
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parents_indices.append(winner_idx)
        
        return parents_indices
    
    def crossover(self, parent1, parent2):
        """实现均匀交叉和启发式交叉的混合策略"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # 随机选择交叉策略
        crossover_type = np.random.choice(['uniform', 'heuristic'], p=[0.7, 0.3])
        
        if crossover_type == 'uniform':
            # 均匀交叉：随机决定从哪个父代继承每个基因
            mask = np.random.randint(0, 2, self.n_items)
            child1 = np.where(mask, parent1, parent2)#类似于三元运算符
            child2 = np.where(mask, parent2, parent1)
        else:
            # 启发式交叉：根据背包问题特性生成子代
            p1_fitness = self.calculate_fitness(parent1)
            p2_fitness = self.calculate_fitness(parent2)
            
            # 选择更好的父代作为基础
            better_parent = parent1 if p1_fitness >= p2_fitness else parent2
            worse_parent = parent2 if p1_fitness >= p2_fitness else parent1
            
            # 创建子代，倾向于继承更好父代的基因
            child1 = better_parent.copy()#复制父代1
            child2 = np.zeros_like(better_parent)#创建一个和父代1一样大小的数组 元素为0
            
            # 第一个子代：保留好的父代，并随机接受另一个父代的一些基因
            for i in range(self.n_items):
                if worse_parent[i] == 1 and better_parent[i] == 0:
                    if random.random() < 0.3:  # 有30%概率接受弱父代的1
                        child1[i] = 1
            
            # 第二个子代：从两个父代中随机选择，但更偏向于好的父代
            for i in range(self.n_items):
                prob = 0.7 if better_parent[i] == 1 else 0.3
                if random.random() < prob:
                    child2[i] = 1
        
        # 修复不可行解
        child1 = self.repair_solution(child1)
        child2 = self.repair_solution(child2)
        
        return child1, child2
    
    def mutation(self, chromosome):
        """智能变异操作"""
        if random.random() > self.mutation_rate:
            return chromosome.copy()
        
        mutated_chromosome = chromosome.copy()
        mutation_type = np.random.choice(['simple', 'swap', 'intelligent'], p=[0.4, 0.3, 0.3])
        
        if mutation_type == 'simple':
            # 简单变异：随机翻转少数几个比特
            num_mutations = max(1, int(0.05 * self.n_items))
            mutation_points = np.random.choice(range(self.n_items), num_mutations, replace=False)
            for point in mutation_points:
                mutated_chromosome[point] = 1 - mutated_chromosome[point]
        
        elif mutation_type == 'swap':
            # 交换变异：交换一个选中物品和一个未选中物品
            selected = np.where(mutated_chromosome == 1)[0]#selected是选中物品的索引
            not_selected = np.where(mutated_chromosome == 0)[0]#not_selected是未选中物品的索引
            
            if len(selected) > 0 and len(not_selected) > 0:
                idx1 = np.random.choice(selected)#随机选择一个选中物品
                idx2 = np.random.choice(not_selected)
                mutated_chromosome[idx1] = 0
                mutated_chromosome[idx2] = 1
        
        else:
            # 智能变异：根据价值密度进行有针对性的变异
            current_weight = np.sum(mutated_chromosome * self.weights)
            remaining_capacity = self.capacity - current_weight
            
            # 如果有剩余容量，尝试添加高价值物品
            if remaining_capacity > 0:
                not_selected = np.where(mutated_chromosome == 0)[0]
                if len(not_selected) > 0:
                    # 根据价值密度排序可能添加的物品  (i, self.density[i]) 是列表的元素 元组
                    candidates = [(i, self.density[i]) for i in not_selected #这里是生成一个列表 列表的元素是物品的索引和价值密度
                                  if self.weights[i] <= remaining_capacity]#candidates是可能添加的物品的索引和价值密度
                    if candidates:
                        candidates.sort(key=lambda x: x[1], reverse=True)#根据价值密度降序排列
                        # 选择最高价值密度的物品添加
                        mutated_chromosome[candidates[0][0]] = 1#将最高价值密度的物品添加到mutated_chromosome中
            
            # 随机移除一个低价值物品
            selected = np.where(mutated_chromosome == 1)[0]
            if len(selected) > 0:
                # 根据价值密度逆序排序已选物品
                candidates = [(i, self.density[i]) for i in selected]
                candidates.sort(key=lambda x: x[1])#根据价值密度升序排列
                # 有一定概率移除最低价值密度的物品
                if random.random() < 0.3:
                    mutated_chromosome[candidates[0][0]] = 0
        
        # 修复不可行解
        mutated_chromosome = self.repair_solution(mutated_chromosome)
        return mutated_chromosome
    
    def repair_solution(self, chromosome):
        """智能修复不可行解"""
        repaired_chromosome = chromosome.copy()
        total_weight = np.sum(repaired_chromosome * self.weights) #计算总重量 括号得到的是一个数组 元素是0-1 乘以weights得到的是一个数组 元素是0-weights 相加得到的是一个数
        
        # 如果超过背包容量，优先移除价值密度低的物品
        if total_weight > self.capacity:
            # 计算所有被选中物品的价值密度
            selected_indices = np.where(repaired_chromosome == 1)[0]
            if len(selected_indices) == 0:
                return repaired_chromosome
                
            # 根据价值密度排序选中的物品
            selected_density = [(i, self.values[i] / self.weights[i]) 
                              for i in selected_indices]
            selected_density.sort(key=lambda x: x[1])  # 按价值密度升序排序
            
            # 从价值密度最低的物品开始移除
            for idx, _ in selected_density:
                if total_weight <= self.capacity:
                    break
                repaired_chromosome[idx] = 0
                total_weight -= self.weights[idx]
        
        # 如果还有剩余容量，尝试贪心地添加更多物品
        remaining_capacity = self.capacity - total_weight
        if remaining_capacity > 0:
            # 对未选中的物品按照价值密度排序
            not_selected = np.where(repaired_chromosome == 0)[0]
            not_selected_density = [(i, self.values[i] / self.weights[i], self.weights[i]) 
                                 for i in not_selected]
            not_selected_density.sort(key=lambda x: x[1], reverse=True)  # 按价值密度降序排序
            
            # 尝试添加更多物品
            for idx, _, weight in not_selected_density:
                if weight <= remaining_capacity:
                    repaired_chromosome[idx] = 1
                    remaining_capacity -= weight
        
        return repaired_chromosome
    
    def run(self):
        """运行遗传算法"""
        start_time = time.time()
        patience = 50  # 提前终止的耐心值
        early_stop = False
        
        for generation in range(self.max_gen):
            # 计算当前种群的适应度
            fitness = np.array([self.calculate_fitness(chrom) for chrom in self.population])
            
            # 记录最佳个体和平均适应度
            current_best_fitness = np.max(fitness)
            avg_fitness = np.mean(fitness)
            
            # 更新全局最佳解
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                best_idx = np.argmax(fitness)
                self.best_solution = self.population[best_idx].copy()
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
            
            # 记录历史
            self.best_fitness_history.append(self.best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            
            # 提前终止条件
            if self.no_improvement_count >= patience:
                print(f"提前终止于第 {generation} 代：{patience} 代未改进")
                early_stop = True
                break
            
            # 选择精英直接进入下一代
            elite_indices = np.argsort(fitness)[-self.elite_num:]#精英个体的索引
            next_population = [self.population[i].copy() for i in elite_indices]
            
            # 选择父代
            parents_indices = self.select_parents(fitness)
            
            # 交叉和变异生成新一代种群
            for i in range(0, self.pop_size - self.elite_num, 2):
                if i + 1 >= self.pop_size - self.elite_num:
                    next_population.append(self.mutation(self.population[parents_indices[i]]))
                    continue
                    
                parent1 = self.population[parents_indices[i]]
                parent2 = self.population[parents_indices[i + 1]]
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                next_population.extend([child1, child2])
            
            self.population = np.array(next_population[:self.pop_size])
            
            # 动态调整变异率，随着代数增加适当提高变异率以增加多样性
            if self.no_improvement_count > 20:
                self.mutation_rate = min(0.2, self.mutation_rate * 1.05)
            else:
                self.mutation_rate = max(0.01, self.mutation_rate * 0.99)
            
            # 打印每10代的进度
            if generation % 10 == 0 or generation == self.max_gen - 1:
                print(f"第 {generation} 代：最佳适应度 = {self.best_fitness}")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # 最终输出
        if not early_stop:
            print(f"完成 {self.max_gen} 代训练")
        
        return self.best_solution, self.best_fitness, elapsed_time
    
    def plot_fitness_history(self):
        """绘制适应度历史变化图"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.best_fitness_history, label='最佳适应度')
        plt.plot(self.avg_fitness_history, label='平均适应度')
        plt.xlabel('代数')
        plt.ylabel('适应度')
        plt.title('遗传算法适应度变化')
        plt.legend()
        plt.grid(True)
        plt.show()

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

def solve_knapsack(file_path, pop_size=200, max_gen=500, crossover_rate=0.9, 
                  mutation_rate=0.05, elite_rate=0.2, plot=False, num_runs=5):
    """求解指定数据文件的背包问题，进行多次运行取最优"""
    print(f"\n求解文件：{file_path}")
    capacity, weights, values = read_data(file_path)
    
    print(f"背包容量：{capacity}，物品数量：{len(weights)}")
    
    # 运行多次，取最优结果
    best_overall_fitness = 0
    best_overall_solution = None
    total_time = 0
    
    for run in range(num_runs):
        print(f"\n运行 {run+1}/{num_runs}:")
        ga = KnapsackGA(
            capacity=capacity,
            weights=weights,
            values=values,
            pop_size=pop_size,
            max_gen=max_gen,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elite_rate=elite_rate
        )
        
        best_solution, best_fitness, elapsed_time = ga.run()
        total_time += elapsed_time
        
        print(f"本次运行最佳适应度: {best_fitness}")
        
        if best_fitness > best_overall_fitness:
            best_overall_fitness = best_fitness
            best_overall_solution = best_solution
            
            if plot:
                ga.plot_fitness_history()
    
    # 输出最终结果
    selected_items = [i+1 for i, x in enumerate(best_overall_solution) if x == 1]
    total_weight = sum(weights[i] for i in range(len(weights)) if best_overall_solution[i] == 1)
    
    print("\n最终结果:")
    print(f"最大价值: {best_overall_fitness}")
    print(f"总重量: {total_weight}/{capacity}")
    print(f"选择的物品编号: {selected_items}")
    print(f"选择的物品数量: {len(selected_items)}/{len(weights)}")
    print(f"平均运行时间: {total_time/num_runs:.2f} 秒")
    
    return best_overall_solution, best_overall_fitness, selected_items

if __name__ == "__main__":
    # 解决三个数据文件
    data_files = [
        "./data/data1.txt", 
        "./data/data5.txt", 
        "./data/data6.txt"
    ]
    
    results = {}
    
    for file_path in data_files:
        best_solution, best_fitness, selected_items = solve_knapsack(
            file_path=file_path,
            pop_size=300,       # 增大种群规模
            max_gen=300,        # 增加最大代数
            crossover_rate=0.9, # 提高交叉率
            mutation_rate=0.05, # 降低基础变异率，但会动态调整
            elite_rate=0.2,     # 提高精英比例
            plot=False,
            num_runs=3          # 多次运行取最优
        )
        results[file_path] = {
            "best_fitness": best_fitness,
            "selected_items": selected_items
        }
    
    # 打印所有结果汇总
    print("\n========== 结果汇总 ==========")
    for file_path, result in results.items():
        print(f"\n文件: {file_path}")
        print(f"最大价值: {result['best_fitness']}")
        print(f"选择的物品: {result['selected_items']}")
