import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import time
from torch import jit


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# SCH问题参数
pop_size = 100  # 种群大小
max_gen = 200   # 最大迭代次数
T = 20          # 邻居数量
x_min, x_max = -6.0, 6.0  # 决策变量范围

# 算法参数
crossover_rate = 1.0  # 交叉率
mutation_rate = 1.0 / 1  # 变异率 (1/决策变量维度)
mutation_eta = 20  # 多项式变异参数
crossover_eta = 20  # 模拟二进制交叉参数
pbi_theta = 5.0  # PBI方法的惩罚参数

# 使用JIT编译优化的目标函数计算
@jit.script
def evaluate_objectives(x_tensor):

    f1 = x_tensor ** 2
    f2 = (x_tensor - 2) ** 2
    return torch.stack([f1, f2], dim=1)

# 初始化种群
def initialize_population():
  
    population = torch.rand(pop_size, 1, device=device) * (x_max - x_min) + x_min
    return population

# 初始化权重向量
def initialize_weights():
   
    # 使用arange和stack代替循环
    w1 = torch.arange(pop_size, dtype=torch.float32, device=device) / (pop_size - 1)
    w2 = 1 - w1
    weights = torch.stack([w1, w2], dim=1)
    return weights

# 计算欧氏距离并获取邻居
def compute_neighbors(weights):
  
    # 计算权重向量之间的欧氏距离
    distances = torch.cdist(weights, weights, p=2)
    
    # 对每个子问题找出T个最近的邻居
    _, neighbors = torch.topk(distances, T, largest=False)
    
    return neighbors

# PBI聚合函数
@jit.script
def pbi_aggregation(objectives, weights, z_min, theta: float):
  
    # 归一化目标值
    normalized_obj = objectives - z_min
    
    # 计算权重向量的范数
    weights_norm = torch.norm(weights, dim=1, keepdim=True)
    weights_normalized = weights / (weights_norm + 1e-10)
    
    # 计算d1 (投影距离)
    d1 = torch.sum(normalized_obj * weights_normalized, dim=1)
    
    # 计算投影点
    proj = d1.unsqueeze(1) * weights_normalized
    
    # 计算d2 (垂直距离)
    d2 = torch.norm(normalized_obj - proj, dim=1)
    
    # 计算PBI值
    pbi = d1 + theta * d2
    
    return pbi

# 模拟二进制交叉(SBX)
@jit.script
def simulated_binary_crossover(parent1, parent2, eta: float, x_min: float, x_max: float):
   
    # 生成随机数
    u = torch.rand(parent1.shape, device=parent1.device)
    
    # 计算beta值
    beta = torch.where(
        u <= 0.5,
        (2 * u) ** (1 / (eta + 1)),
        (1 / (2 * (1 - u))) ** (1 / (eta + 1))
    )
    
    # 生成子代
    child1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
    child2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)
    
    # 边界处理
    child1 = torch.clamp(child1, x_min, x_max)
    child2 = torch.clamp(child2, x_min, x_max)
    
    return child1, child2

# 多项式变异
@jit.script
def polynomial_mutation(individual, eta: float, mutation_rate: float, x_min: float, x_max: float):
   
    # 生成随机数
    u = torch.rand(individual.shape, device=individual.device)
    mutation_mask = torch.rand(individual.shape, device=individual.device) < mutation_rate
    
    # 计算delta值
    delta = torch.zeros_like(individual)
    mask_le = u <= 0.5
    mask_gt = ~mask_le
    
    delta[mask_le] = ((2 * u[mask_le]) ** (1 / (eta + 1))) - 1
    delta[mask_gt] = 1 - ((2 * (1 - u[mask_gt])) ** (1 / (eta + 1)))
    
    # 应用变异
    mutated = individual + delta * (x_max - x_min) * mutation_mask.float()
    
    # 边界处理
    return torch.clamp(mutated, x_min, x_max)

# f_op1函数 - 详细实现
def f_op1(population, objectives, offspring, offspring_obj, weights, neighbors, z_min):

    # 初始化更新标记矩阵 (所有位置初始化为其索引值，表示保留原解)
    i_new = torch.arange(pop_size, device=device).repeat(pop_size, 1)
    
    # 为每个子问题计算
    for i in range(pop_size):
        # 获取邻居索引
        nb = neighbors[i]
        
        # 提取邻居的目标值和权重
        nb_objectives = objectives[nb]
        nb_weights = weights[nb]
        
        # 计算当前解的PBI值 - 批量计算
        g_old = pbi_aggregation(nb_objectives, nb_weights, z_min, pbi_theta)
        
        # 计算后代解的PBI值 - 批量计算
        offspring_obj_expanded = offspring_obj[i].unsqueeze(0).expand(T, -1)
        g_new = pbi_aggregation(offspring_obj_expanded, nb_weights, z_min, pbi_theta)
        
        # 找出需要更新的邻居
        better_mask = g_new < g_old
        
        # 更新标记矩阵
        for j, is_better in enumerate(better_mask):
            if is_better:
                neighbor_idx = nb[j].item()
                i_new[i, neighbor_idx] = -1  # 标记为使用后代解
    
    return i_new

# f_op2函数 - 详细实现
def f_op2(population, objectives, offspring, offspring_obj, weights, i_new, z_min):
 
    # 创建新种群和新目标函数值
    new_population = population.clone()
    new_objectives = objectives.clone()
    # 对每个位置j进行处理
    for j in range(pop_size):
        # 获取位置j的更新标记
        update_marks = i_new[:, j]
        # 检查是否有子问题标记为更新位置j
        if torch.any(update_marks == -1):
            # 找出所有标记为更新位置j的子问题
            update_subproblems = torch.where(update_marks == -1)[0]
            # 收集所有候选后代
            candidate_offspring = offspring[update_subproblems]
            candidate_obj = offspring_obj[update_subproblems]
            
            # 计算每个候选后代在位置j对应权重下的PBI值
            pbi_values = pbi_aggregation(candidate_obj, weights[j].unsqueeze(0).expand(len(update_subproblems), -1), z_min, pbi_theta)
            # 计算当前解在位置j对应权重下的PBI值
            current_pbi = pbi_aggregation(objectives[j].unsqueeze(0), weights[j].unsqueeze(0), z_min, pbi_theta)
            # 如果当前解的PBI值小于所有候选后代的PBI值，保留当前解
            if current_pbi < torch.min(pbi_values):
                continue
            # 否则选择PBI值最小的后代
            best_idx = torch.argmin(pbi_values).item()
            best_subproblem = update_subproblems[best_idx].item()
            # 更新位置j
            new_population[j] = offspring[best_subproblem]
            new_objectives[j] = offspring_obj[best_subproblem]
    return new_population, new_objectives

# TensorMOEA/D主算法 - 完整版本
def tensor_moead_full():
  
    # 初始化计时器
    start_time = time.time()
    
    # 初始化种群
    population = initialize_population()
    
    # 评估初始种群
    objectives = evaluate_objectives(population.squeeze())
    
    # 初始化权重向量
    weights = initialize_weights()
    
    # 计算邻居
    neighbors = compute_neighbors(weights)
    
    # 初始化理想点
    z_min = torch.min(objectives, dim=0)[0]
    
    for gen in range(max_gen):
        gen_start_time = time.time()
        
        # 生成后代
        offspring = torch.zeros_like(population)
        
        for i in range(pop_size):
            # 随机选择两个邻居作为父代
            k = torch.randperm(T)[:2]
            parent_indices = neighbors[i, k]
            parent1 = population[parent_indices[0]]
            parent2 = population[parent_indices[1]]
            
            # 交叉
            if torch.rand(1).item() < crossover_rate:
                child1, _ = simulated_binary_crossover(parent1, parent2, crossover_eta, x_min, x_max)
                offspring[i] = child1
            else:
                offspring[i] = parent1.clone()
            
            # 变异
            if torch.rand(1).item() < mutation_rate:
                offspring[i] = polynomial_mutation(offspring[i], mutation_eta, mutation_rate, x_min, x_max)
        
        # 评估后代
        offspring_obj = evaluate_objectives(offspring.squeeze())
        
        # 更新理想点
        z_min = torch.min(torch.cat([z_min.unsqueeze(0), offspring_obj]), dim=0)[0]
        
        # 环境选择 - 详细实现f_op1和f_op2
        # 1. 执行f_op1，确定哪些位置需要被更新
        i_new = f_op1(population, objectives, offspring, offspring_obj, weights, neighbors, z_min)
        
        # 2. 执行f_op2，根据更新标记选择最佳解
        population, objectives = f_op2(population, objectives, offspring, offspring_obj, weights, i_new, z_min)
        
        # 打印当前代的信息
        if (gen + 1) % 10 == 0:
            gen_time = time.time() - gen_start_time
            print(f"第 {gen+1} 代完成，用时: {gen_time:.4f} 秒")
            
            # 计算当前非支配解数量
            non_dominated_mask = identify_non_dominated_solutions(objectives)
            print(f"  当前非支配解数量: {torch.sum(non_dominated_mask).item()}/{pop_size}")
    
    # 计算总运行时间
    total_time = time.time() - start_time
    print(f"\n算法总运行时间: {total_time:.4f} 秒")
    print(f"每代平均运行时间: {total_time/max_gen:.4f} 秒")
    
    # 提取非支配解
    non_dominated_mask = identify_non_dominated_solutions(objectives)
    non_dominated_pop = population[non_dominated_mask]
    non_dominated_obj = objectives[non_dominated_mask]
    
    print(f"非支配解数量: {torch.sum(non_dominated_mask).item()}/{pop_size}")
    
    return non_dominated_pop, non_dominated_obj

# 使用张量操作识别非支配解
def identify_non_dominated_solutions(objectives):
    
    pop_size = objectives.shape[0]
    is_non_dominated = torch.ones(pop_size, dtype=torch.bool, device=device)
    
    # 扩展维度以便批量比较
    obj_i = objectives.unsqueeze(1)  # shape: [pop_size, 1, 2]
    obj_j = objectives.unsqueeze(0)  # shape: [1, pop_size, 2]
    
    # 检查支配关系
    # i被j支配: j的所有目标都不差于i，且至少一个目标更好
    dominates = ((obj_j <= obj_i).all(dim=2) & (obj_j < obj_i).any(dim=2))
    
    # 如果任何其他解支配当前解，则当前解不是非支配的
    is_non_dominated = ~dominates.any(dim=1)
    
    return is_non_dominated

# 运行算法
print("开始运行完整版TensorMOEA/D算法解决SCH问题...")
final_pop, final_obj = tensor_moead_full()

# 将结果转移到CPU以便绘图
final_pop_cpu = final_pop.cpu().numpy()
final_obj_cpu = final_obj.cpu().numpy()

# 绘制Pareto前沿
plt.figure(figsize=(10, 6))
plt.scatter(final_obj_cpu[:, 0], final_obj_cpu[:, 1], c='blue', s=30)
plt.xlabel('f1(x) = x^2')
plt.ylabel('f2(x) = (x-2)^2')
plt.title('Full TensorMOEA/D for SCH Problem - Pareto Front')
plt.grid(True)

# 绘制真实Pareto前沿
x_true = np.linspace(0, 2, 100)
f1_true = x_true ** 2
f2_true = (x_true - 2) ** 2
plt.plot(f1_true, f2_true, 'r-', linewidth=2, label='True Pareto Front')
plt.legend()

# 设置坐标轴范围，聚焦在Pareto前沿区域
plt.xlim(-0.5, 4.5)
plt.ylim(-0.5, 4.5)

plt.savefig('tensor_moead_full_pareto.png', dpi=300, bbox_inches='tight')
print("图像已保存为'tensor_moead_full_pareto.png'")

# 打印部分非支配解
print("\n部分非支配解:")
# 按f1排序
sorted_indices = np.argsort(final_obj_cpu[:, 0])
for i in range(min(10, len(final_pop_cpu))):
    idx = sorted_indices[i]
    print(f"x = {final_pop_cpu[idx][0]:.4f}, f1 = {final_obj_cpu[idx][0]:.4f}, f2 = {final_obj_cpu[idx][1]:.4f}")
