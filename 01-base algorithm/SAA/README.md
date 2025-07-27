# 模拟退火算法（Simulated Annealing Algorithm）实现

本文件夹包含一个高度优化的模拟退火算法实现，专门用于求解复杂函数的全局最优化问题。该算法具有多种增强机制，能够有效逃离局部最优解。


## SAA.py - 增强型模拟退火算法

### 问题描述
求解复杂多峰函数的全局最大值：
```
f(x) = |x·sin(x)·cos(2x) - 2x·sin(3x) + 3x·sin(4x)|
```

该函数具有多个局部最优解，是测试全局优化算法性能的经典函数。

### 核心特性

#### 1. 多重重启机制
- **探索性重启**：前几次重启使用高温度和大步长，强调全局探索
- **智能重启**：基于已发现的局部最优解选择远离区域作为新起点
- **局部最优记录**：自动记录和避免重复陷入相同的局部最优

#### 2. 自适应参数调整
- **动态步长调整**：根据接受率自动调整搜索步长
  - 接受率过高 → 增大步长（加强探索）
  - 接受率过低 → 减小步长（加强开发）
- **温度控制**：指数降温 + 重新退火机制
- **停滞检测**：长期无改进时触发重启或重新退火

#### 3. 多样化搜索策略
```python
# 搜索策略分配
85% - 局部搜索：在当前解附近随机扰动
10% - 定向搜索：朝全局最优方向移动
5%  - 全局探索：完全随机搜索
```

#### 4. 增强的接受准则
```python
# 动态退火准则
T_factor = max(0.01, current_temp / initial_temp)
accept_prob = exp(-delta_energy / (temp * (1 - 0.5 * T_factor)))
```

### 算法流程

#### 主要步骤
1. **初始化**：随机生成初始解，设置初始温度
2. **搜索循环**：
   - 生成邻域解（多种策略）
   - 计算能量差
   - 应用接受准则
   - 更新当前解和最优解
3. **参数调整**：
   - 降低温度
   - 调整步长
   - 检测停滞
4. **重启判断**：
   - 记录局部最优
   - 选择新起点
   - 重置参数
5. **终止条件**：达到最大迭代次数或重启次数

#### 关键算法
```python
# 温度更新（指数降温）
temperature = initial_temp * (alpha ** iteration)

# 步长自适应
if accept_rate > 0.6:
    step_size *= (1 / step_adjust_rate)  # 增大步长
elif accept_rate < 0.2:
    step_size *= step_adjust_rate        # 减小步长

# 智能重启点选择
max_distance = 0
for candidate in random_candidates:
    min_distance = min([abs(candidate - opt) for opt in known_optima])
    if min_distance > max_distance:
        best_restart_point = candidate
```


### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `initial_temp` | 1000 | 初始温度，影响初期接受差解的概率 |
| `final_temp` | 1e-7 | 终止温度，算法收敛判断条件 |
| `alpha` | 0.95 | 降温系数，控制温度下降速度 |
| `max_iter` | 1000 | 最大迭代次数 |
| `step_size` | 1.0 | 初始搜索步长 |
| `restart_temp` | 100 | 重启时的温度设置 |
| `max_stagnation` | 20 | 触发重启的最大停滞次数 |
| `n_restarts` | 3 | 最大重启次数 |
| `step_adjust_rate` | 0.95 | 步长调整的变化率 |
| `reannealing_threshold` | 0.1 | 重新退火的接受率阈值 |

### 使用方法

#### 基本使用
```python
# 直接运行文件
python SAA.py

# 自定义参数运行
sa = SimulatedAnnealing(
    objective_func=f,
    bounds=[0, 50],
    initial_temp=1000,
    max_iter=1000,
    visualize=True
)
best_x, best_value, history = sa.run()
```

#### 高级配置
```python
# 针对特定问题的参数调优
sa = SimulatedAnnealing(
    objective_func=your_function,
    bounds=[x_min, x_max],
    initial_temp=2000,      # 复杂函数用更高初始温度
    alpha=0.98,             # 更慢的降温速度
    step_size=5.0,          # 更大的初始步长
    n_restarts=10,          # 更多重启次数
    max_stagnation=30,      # 更大的停滞容忍度
    visualize=True
)
```

