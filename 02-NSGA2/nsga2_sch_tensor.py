import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import time  
# 添加中文字体支持
import matplotlib.font_manager as fm
font_path = 'C:/Windows/Fonts/msyh.ttc' 
prop = fm.FontProperties(fname=font_path)
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.family'] = ['Microsoft YaHei']  # 设置全局默认字体为微软雅黑

# 设置GPU设备（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 问题参数
pop_size = 100  # 种群大小
num_generations = 200  # 迭代代数
mutation_rate = 0.1  # 变异率
crossover_rate = 0.9  # 交叉率
x_min, x_max = -2.0, 4.0  # 修改决策变量范围，确保包含[0,2]区间

# 目标函数（SCH问题）
def evaluate_objectives(x_tensor):
   
    f1 = x_tensor ** 2
    f2 = (x_tensor - 2) ** 2
    return torch.stack([f1, f2], dim=1)

# 初始化种群
def initialize_population(pop_size):
   
    # 70%的解在[0,2]区间
    in_optimal = int(0.7 * pop_size)
    optimal_range = torch.rand(in_optimal, 1, device=device) * 2.0  # [0,2]
    
    # 30%的解在整个搜索空间
    explore_range = torch.rand(pop_size - in_optimal, 1, device=device) * (x_max - x_min) + x_min
    
    # 合并并随机打乱
    population = torch.cat([optimal_range, explore_range])
    indices = torch.randperm(pop_size)
    return population[indices]

# 非支配排序 - 张量化实现
def fast_non_dominated_sort_tensor(objectives):
    n = objectives.shape[0] # 种群大小
    # 使用张量计算支配关系矩阵 - 一次性计算所有支配关系
    X = objectives.unsqueeze(1)  # [n, 1, num_obj]
    Y = objectives.unsqueeze(0)  # [1, n, num_obj]
    # 计算支配关系矩阵：i支配j的条件是所有目标不劣于且至少一个目标严格优于
    dominates = torch.logical_and(
        torch.all(X <= Y, dim=2),
        torch.any(X < Y, dim=2)
    )
    # 计算每个解被支配的次数
    domination_count = torch.sum(dominates, dim=0)
    # 初始化前沿等级
    rank = torch.zeros(n, dtype=torch.int32, device=device)
    k = 0  # 当前等级
    # 分配前沿等级
    while torch.any(domination_count == 0):
        # 找出当前前沿的解（不被任何解支配）
        current_front = (domination_count == 0) 
        if not torch.any(current_front):
            break
        # 分配当前等级
        k += 1
        rank[current_front] = k
        # 更新被当前前沿支配的解的支配计数 - 使用张量操作
        d = torch.sum(current_front.unsqueeze(1) * dominates, dim=0)
        domination_count = domination_count - d - current_front.to(torch.int32)
    return rank

# 拥挤度距离计算 - 优化版本
def crowding_distance_tensor_optimized(objectives, front_indices):
   
    if len(front_indices) <= 2:
        return torch.ones(len(front_indices), device=device) * float('inf')
    
    front_size = len(front_indices)
    front_objectives = objectives[front_indices]
    num_objectives = objectives.shape[1]
    
    # 初始化距离
    distances = torch.zeros(front_size, device=device)
    
    for obj in range(num_objectives):
        # 按当前目标排序
        sorted_indices = torch.argsort(front_objectives[:, obj])
        sorted_obj_values = front_objectives[sorted_indices, obj]
        
        # 边界点设为无穷
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')
        
        # 计算范围
        obj_range = sorted_obj_values[-1] - sorted_obj_values[0]
        if obj_range > 0:
            # 一次性计算所有中间点的距离
            normalized_diffs = (sorted_obj_values[2:] - sorted_obj_values[:-2]) / obj_range
            distances[sorted_indices[1:-1]] += normalized_diffs
    
    return distances

# 简化的锦标赛选择
def tournament_selection_optimized(rank, crowding_dist, tournament_size=2):
   
    pop_size = rank.shape[0]
    selected_indices = torch.zeros(pop_size, dtype=torch.long, device=device)
    
    for i in range(pop_size):
        # 随机选择参赛者
        candidates = torch.randperm(pop_size, device=device)[:tournament_size]
        
        # 获取候选者的排名和拥挤度
        candidate_ranks = rank[candidates]
        candidate_crowding = crowding_dist[candidates]
        
        # 找出最佳候选者 - 首先比较排名，然后比较拥挤度
        best_by_rank = torch.argmin(candidate_ranks)
        min_rank = candidate_ranks[best_by_rank]
        
        # 找出具有相同最小排名的候选者
        same_rank_mask = (candidate_ranks == min_rank)
        
        if torch.sum(same_rank_mask) == 1:
            # 如果只有一个最小排名，直接选择它
            best_idx = candidates[best_by_rank]
        else:
            # 如果有多个相同排名，选择拥挤度最大的
            same_rank_indices = candidates[same_rank_mask]
            same_rank_crowding = crowding_dist[same_rank_indices]
            best_by_crowding = torch.argmax(same_rank_crowding)
            best_idx = same_rank_indices[best_by_crowding]
        
        selected_indices[i] = best_idx
    
    return selected_indices

# 模拟二进制交叉(SBX)
def simulated_binary_crossover(parent1, parent2, eta=20):
   
    # 生成随机数
    u = torch.rand(1, device=device)
    
    # 计算beta值
    if u <= 0.5:
        beta = (2 * u) ** (1 / (eta + 1))
    else:
        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
    
    # 生成子代
    child1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
    child2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)
    
    # 边界处理
    child1 = torch.clamp(child1, x_min, x_max)
    child2 = torch.clamp(child2, x_min, x_max)
    
    return child1, child2

# 多项式变异
def polynomial_mutation(individual, eta=20):
   
    u = torch.rand(individual.shape, device=device)
    delta = torch.zeros_like(individual)
    
    # 计算delta值
    mask_le = u <= 0.5
    mask_gt = ~mask_le
    
    delta[mask_le] = ((2 * u[mask_le]) ** (1 / (eta + 1))) - 1
    delta[mask_gt] = 1 - ((2 * (1 - u[mask_gt])) ** (1 / (eta + 1)))
    
    # 应用变异
    mutated = individual + delta * (x_max - x_min) * mutation_rate
    
    # 边界处理
    return torch.clamp(mutated, x_min, x_max)

# 主NSGA-II算法 - 优化版本
def nsga2_optimized():
    # 初始化计时器
    start_time = time.time()
    generation_times = []
    
    # 初始化种群
    population = initialize_population(pop_size)
    
    # 评估初始种群
    objectives = evaluate_objectives(population.squeeze())
    
    for generation in range(num_generations):
        gen_start_time = time.time()  # 记录每代开始时间
        
        # 非支配排序 - 使用优化版本
        rank = fast_non_dominated_sort_tensor(objectives)
        
        # 计算每个个体的拥挤度距离
        crowding_dist = torch.zeros(pop_size, device=device)
        
        # 对每个前沿计算拥挤度距离
        max_rank = torch.max(rank).item()
        for current_rank in range(1, max_rank + 1):
            front_indices = torch.where(rank == current_rank)[0]
            if len(front_indices) == 0:
                break
                
            front_distances = crowding_distance_tensor_optimized(objectives, front_indices)
            crowding_dist[front_indices] = front_distances
        
        # 优化的锦标赛选择
        selected_indices = tournament_selection_optimized(rank, crowding_dist)
        
        # 简化的交叉操作
        parents = population[selected_indices]
        offspring = torch.zeros_like(population)
        
        for i in range(0, pop_size, 2):
            if torch.rand(1, device=device).item() < crossover_rate:
                offspring[i], offspring[i+1] = simulated_binary_crossover(parents[i], parents[i+1])
            else:
                offspring[i], offspring[i+1] = parents[i], parents[i+1]
        
        # 简化的变异操作
        mutation_mask = torch.rand(pop_size, 1, device=device) < mutation_rate
        if mutation_mask.any():
            offspring[mutation_mask.squeeze()] = polynomial_mutation(offspring[mutation_mask.squeeze()])
        
        # 评估子代
        offspring_objectives = evaluate_objectives(offspring.squeeze())
        
        # 合并父代和子代
        combined_pop = torch.cat([population, offspring])
        combined_obj = torch.cat([objectives, offspring_objectives])
        
        # 对合并种群进行非支配排序
        combined_rank = fast_non_dominated_sort_tensor(combined_obj)
        
        # 计算合并种群的拥挤度距离
        combined_crowding_dist = torch.zeros(pop_size*2, device=device)
        
        max_rank = torch.max(combined_rank).item()
        for current_rank in range(1, max_rank + 1):
            front_indices = torch.where(combined_rank == current_rank)[0]
            if len(front_indices) == 0:
                break
                
            front_distances = crowding_distance_tensor_optimized(combined_obj, front_indices)
            combined_crowding_dist[front_indices] = front_distances
        
        # 环境选择 - 优化实现
        next_generation = torch.zeros(pop_size, dtype=torch.long, device=device)
        count = 0
        current_rank = 1
        
        while True:
            front_indices = torch.where(combined_rank == current_rank)[0]
            front_size = len(front_indices)
            
            if count + front_size <= pop_size:
                # 如果整个前沿都可以被包含
                next_generation[count:count+front_size] = front_indices
                count += front_size
                if count == pop_size:
                    break
            else:
                # 需要根据拥挤度选择部分个体
                remaining = pop_size - count
                front_crowding = combined_crowding_dist[front_indices]
                _, sorted_indices = torch.sort(front_crowding, descending=True)
                selected_from_front = front_indices[sorted_indices[:remaining]]
                next_generation[count:] = selected_from_front
                break
                
            current_rank += 1
        
        # 更新种群
        population = combined_pop[next_generation]
        objectives = combined_obj[next_generation]
        
        # 记录每代运行时间
        gen_time = time.time() - gen_start_time
        generation_times.append(gen_time)
        
        # 打印当前代的信息
        if (generation + 1) % 10 == 0:
            print(f"第 {generation+1} 代完成，用时: {gen_time:.4f} 秒")
    
    # 计算总运行时间
    total_time = time.time() - start_time
    avg_time = sum(generation_times) / len(generation_times)
    
    print(f"\n算法总运行时间: {total_time:.4f} 秒")
    print(f"每代平均运行时间: {avg_time:.4f} 秒")
    
    # 返回最终种群和其目标函数值
    return population, objectives

# 修改主程序调用优化版本的算法
print("开始运行优化的NSGA-II算法...")
final_pop, final_obj = nsga2_optimized()

# 将结果转移到CPU以便绘图
final_pop_cpu = final_pop.cpu().numpy()
final_obj_cpu = final_obj.cpu().numpy()

# 绘制Pareto前沿
plt.figure(figsize=(10, 6))
plt.scatter(final_obj_cpu[:, 0], final_obj_cpu[:, 1], c='blue', s=30)
plt.xlabel('f1(x) = x^2', fontproperties=prop)
plt.ylabel('f2(x) = (x-2)^2', fontproperties=prop)
plt.title('NSGA-II求解SCH问题的Pareto前沿', fontproperties=prop)
plt.grid(True)

# 绘制真实Pareto前沿
x_true = np.linspace(0, 2, 100)
f1_true = x_true ** 2
f2_true = (x_true - 2) ** 2
plt.plot(f1_true, f2_true, 'r-', linewidth=2, label='真实Pareto前沿')
plt.legend(prop=prop)

# 设置坐标轴范围，聚焦在Pareto前沿区域
plt.xlim(-0.5, 4.5)
plt.ylim(-0.5, 4.5)

plt.savefig('pareto_front.png', dpi=300, bbox_inches='tight')
print("图像已保存为'pareto_front.png'")

# 打印部分解
print("\n部分非支配解:")
# 找出第一个前沿的解
first_front_indices = torch.where(fast_non_dominated_sort_tensor(final_obj) == 0)[0].cpu().numpy()
first_front_pop = final_pop_cpu[first_front_indices]
first_front_obj = final_obj_cpu[first_front_indices]

# 按f1排序
sorted_indices = np.argsort(first_front_obj[:, 0])
for i in range(min(10, len(first_front_pop))):
    idx = sorted_indices[i]
    print(f"x = {first_front_pop[idx][0]:.4f}, f1 = {first_front_obj[idx][0]:.4f}, f2 = {first_front_obj[idx][1]:.4f}")