# 粒子群优化算法（Particle Swarm Optimization）实现集合

本文件夹包含三个不同的粒子群优化算法实现，分别解决不同类型的优化问题。每个算法都具有完整的可视化功能，展示了PSO算法在不同问题域的应用。

## 文件概览

| 文件名 | 问题类型 | 主要特点 | 适用场景 |
|--------|----------|----------|----------|
| `PSO-01.py` | 0-1背包问题 | Sigmoid转换、修复机制 | 离散优化问题 |
| `PSO-TSP.py` | 旅行商问题 | 交换操作、局部搜索 | 排列优化问题 |
| `PSO-fmax.py` | 函数最大值 | 动态权重、实时可视化 | 连续函数优化 |

---

## PSO-01.py - 0-1背包问题求解器

### 问题描述
使用粒子群优化算法求解0-1背包问题，通过连续空间搜索找到最优的物品选择组合。

### 核心特性
- **Sigmoid位置转换**：
  - 使用sigmoid函数将连续位置转换为选择概率
  - 概率化的二进制解生成机制
- **混合初始化策略**：
  - 30% 基于贪心策略的高质量初始解
  - 70% 随机初始化保证多样性
- **智能修复机制**：
  - 自动修复超出容量限制的不可行解
  - 基于价值密度的贪心修复策略
- **动态参数调整**：
  - 根据收敛情况自适应调整惯性权重
  - 长期无改进时增强探索能力

### 算法流程
```python
# 粒子位置更新
velocity = w * velocity + c1 * r1 * (pbest - position) + c2 * r2 * (gbest - position)
position = position + velocity

# Sigmoid转换
probability = 1 / (1 + exp(-position))
solution = (random() < probability)

# 修复不可行解
solution = repair_solution(solution)
```

### 使用方法
```python
# 直接运行文件
python PSO-01.py

# 或导入使用
from PSO-01 import solve_knapsack_pso
result = solve_knapsack_pso("data/data1.txt", num_particles=200, max_iter=300)
```

---

## PSO-TSP.py - 旅行商问题求解器

### 问题描述
使用粒子群优化算法求解旅行商问题，在离散的排列空间中寻找最短路径。

### 核心特性
- **离散PSO设计**：
  - 速度表示为交换操作序列
  - 位置更新通过交换操作实现
- **混合初始化**：
  - 贪心算法生成高质量起始解
  - 2-opt局部搜索优化初始种群
- **多层次局部搜索**：
  - 2-opt算法进行边交换优化
  - 3-opt算法处理更复杂的路径重组
- **社会交流机制**：
  - 粒子间信息交换
  - 顺序交叉（OX）算子
- **重启策略**：
  - 长期无改进时重新初始化部分粒子
  - 保持种群多样性

### 速度和位置更新
```python
# 速度更新（交换操作）
velocity = w * old_velocity + c1 * pbest_operations + c2 * gbest_operations

# 位置更新（应用交换）
for (i, j) in velocity:
    position[i], position[j] = position[j], position[i]

# 局部搜索优化
position = two_opt(position)
```

### 使用方法
```python
# 直接运行文件
python PSO-TSP.py

# 自定义参数
best_route, best_distance = solve_tsp_with_pso(
    file_path="data/eil51.tsp",
    num_particles=100,
    max_iter=400,
    run_times=3
)
```

---

## PSO-fmax.py - 函数最大值优化器

### 问题描述
寻找复杂函数的全局最大值：
```
f(x) = |x·sin(x)·cos(2x) - 2x·sin(3x) + 3x·sin(4x)|
```

### 核心特性
- **标准PSO实现**：
  - 经典的速度-位置更新公式
  - 个体认知和社会学习平衡
- **动态惯性权重**：
  - 线性递减策略：w_max → w_min
  - 促进全局搜索到局部搜索的转换
- **实时可视化**：
  - 粒子分布动态显示
  - 收敛过程实时追踪
  - 函数曲线背景参考
- **动画回放功能**：
  - 完整记录优化历史
  - 可视化粒子运动轨迹

### PSO更新公式
```python
# 惯性权重线性递减
w = w_max - (w_max - w_min) * iter / max_iter

# 速度更新
v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)

# 位置更新
x = x + v
```

### 使用方法
```python
# 直接运行文件
python PSO-fmax.py

# 自定义优化
pso = PSO(num_particles=50, max_iter=100, bounds=[0, 50], visualize=True)
best_x, best_value = pso.run()
```

---

## 快速开始

1. **安装依赖**：
```bash
pip install numpy matplotlib scipy tkinter
```

2. **选择算法**：根据问题类型选择对应的文件

3. **运行示例**：
```bash
# 背包问题
python PSO-01.py

# 旅行商问题  
python PSO-TSP.py

# 函数优化
python PSO-fmax.py
```

4. **查看结果**：每个算法都会输出最优解和可视化结果

---

## 算法比较

| 特性 | PSO-01 | PSO-TSP | PSO-fmax |
|------|--------|---------|----------|
| 搜索空间 | 连续→离散 | 离散排列 | 连续实数 |
| 位置表示 | 概率向量 | 城市序列 | 实数坐标 |
| 速度含义 | 位置变化 | 交换操作 | 位置变化 |
| 约束处理 | 修复算子 | 合法性保证 | 边界限制 |
| 局部搜索 | 贪心修复 | 2-opt/3-opt | 无 |
| 特殊机制 | Sigmoid转换 | 社会交流 | 动态权重 |

---

## 核心算法原理

### PSO基本思想
粒子群优化算法模拟鸟群觅食行为，每个粒子代表问题空间中的一个候选解：

1. **个体认知**：粒子向自己历史最佳位置移动
2. **社会学习**：粒子向群体最佳位置移动  
3. **惯性保持**：保持原有运动方向

### 参数说明
- **w（惯性权重）**：控制粒子保持原有速度的程度
- **c1（认知参数）**：控制粒子向个体最佳位置移动的强度
- **c2（社会参数）**：控制粒子向全局最佳位置移动的强度
- **粒子数量**：影响算法的探索能力和计算复杂度
- **最大迭代次数**：控制算法运行时间和收敛精度




