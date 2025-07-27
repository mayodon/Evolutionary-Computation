# MOEA/D算法求解Schaffer问题实现说明

本项目包含两个不同的MOEA/D（Multi-Objective Evolutionary Algorithm based on Decomposition）算法实现，用于求解经典的Schaffer多目标优化问题。

## 项目概述

### Schaffer问题定义
Schaffer问题是一个经典的双目标优化测试问题：
- **目标函数1**: f₁(x) = x²
- **目标函数2**: f₂(x) = (x-2)²
- **决策变量范围**: x ∈ [-5, 5]
- **理论最优解范围**: x ∈ [0, 2]

该问题的帕累托前沿是连续的，当x从0变化到2时，形成一条从(0,4)到(4,0)的曲线。

## 代码文件说明

### 1. moead_sch.py - 经典MOEA/D实现

这是一个完整的面向对象的MOEA/D算法实现，包含以下主要特性：

#### 核心类结构
- **SchafferProblem类**: 定义Schaffer问题的参数和评估函数
- **Individual类**: 表示个体，包含决策变量和目标函数值
- **MOEAD类**: 主算法实现

#### 算法特点
- **分解方法**: 使用切比雪夫分解方法
- **邻域策略**: 基于权重向量欧几里得距离的邻域构建
- **变异算子**: 差分进化（Differential Evolution）
- **选择策略**: 基于聚合函数值的邻域更新
- **外部档案**: 维护非支配解集合

#### 主要参数
```python
pop_size = 100          # 种群大小
n_gen = 200            # 迭代次数
neighborhood_size = 20  # 邻域大小
cr = 1.0               # 交叉率
f = 0.5                # 差分进化缩放因子
delta = 0.9            # 邻域选择概率
nr = 2                 # 每代最大更新数量
```

#### 性能评估
- **IGD指标**: 衡量解集与理论帕累托前沿的收敛性
- **Spread指标**: 衡量解集的分布均匀性

### 2. moead_sch_tensor.py - 基于PyTorch的张量化实现

这是一个高度优化的MOEA/D实现，利用PyTorch的张量操作和GPU加速：

#### 核心特性
- **GPU加速**: 支持CUDA设备，显著提升计算速度
- **JIT编译**: 使用@jit.script装饰器优化关键函数
- **张量化操作**: 批量处理，减少循环开销
- **PBI聚合**: 使用Penalty-based Boundary Intersection方法

#### 算法组件
- **交叉算子**: 模拟二进制交叉（SBX）
- **变异算子**: 多项式变异
- **选择策略**: 两阶段选择机制（f_op1和f_op2）
- **非支配排序**: 基于张量操作的高效实现

#### 主要参数
```python
pop_size = 100         # 种群大小
max_gen = 200          # 最大迭代次数
T = 20                 # 邻居数量
crossover_rate = 1.0   # 交叉率
mutation_rate = 1.0    # 变异率
crossover_eta = 20     # SBX分布指数
mutation_eta = 20      # 多项式变异分布指数
pbi_theta = 5.0        # PBI惩罚参数
```

## 两种实现的对比

| 特性 | moead_sch.py | moead_sch_tensor.py |
|------|-------------|-------------------|
| **编程范式** | 面向对象 | 函数式+张量化 |
| **计算平台** | CPU | CPU/GPU |
| **性能优化** | 标准Python | PyTorch JIT编译 |
| **分解方法** | 切比雪夫分解 | PBI分解 |
| **交叉算子** | 差分进化 | 模拟二进制交叉 |
| **变异算子** | 差分进化 | 多项式变异 |
| **代码复杂度** | 中等 | 较高 |
| **可扩展性** | 良好 | 优秀 |
| **运行速度** | 较慢 | 快速 |

## 算法流程

### 通用MOEA/D流程
1. **初始化**
   - 生成均匀分布的权重向量
   - 初始化种群
   - 计算邻域关系
   - 初始化理想点

2. **主循环**
   - 对每个子问题：
     - 从邻域中选择父代
     - 生成子代（交叉+变异）
     - 更新理想点
     - 更新邻域解

3. **结果输出**
   - 提取非支配解
   - 计算性能指标
   - 可视化结果

## 运行方式

### 运行经典版本
```bash
python moead_sch.py
```

### 运行张量化版本
```bash
python moead_sch_tensor.py
```

## 技术细节

### 经典版本关键算法

#### 权重向量生成
```python
def generate_weights(self):
    weights = []
    for i in range(self.pop_size):
        w = i / (self.pop_size - 1) if self.pop_size > 1 else 0.5
        weights.append(np.array([w, 1-w]))
    return np.array(weights)
```

#### 切比雪夫分解
```python
def compute_tchebycheff(self, f, w):
    return np.max(w * np.abs(f - self.z))
```

#### 差分进化操作
```python
def differential_evolution(self, population, pool, i):
    # 选择三个不同个体
    r = np.random.choice(pool, 3, replace=False)
    # 差分变异: v = x_r0 + F * (x_r1 - x_r2)
    v = population[r[0]].x + self.f * (population[r[1]].x - population[r[2]].x)
    # 交叉操作生成子代
    # ...
```

### 张量化版本关键算法

#### PBI聚合函数
```python
@jit.script
def pbi_aggregation(objectives, weights, z_min, theta: float):
    normalized_obj = objectives - z_min
    weights_normalized = weights / (torch.norm(weights, dim=1, keepdim=True) + 1e-10)
    d1 = torch.sum(normalized_obj * weights_normalized, dim=1)
    proj = d1.unsqueeze(1) * weights_normalized
    d2 = torch.norm(normalized_obj - proj, dim=1)
    pbi = d1 + theta * d2
    return pbi
```

#### 模拟二进制交叉
```python
@jit.script
def simulated_binary_crossover(parent1, parent2, eta: float, x_min: float, x_max: float):
    u = torch.rand(parent1.shape, device=parent1.device)
    beta = torch.where(
        u <= 0.5,
        (2 * u) ** (1 / (eta + 1)),
        (1 / (2 * (1 - u))) ** (1 / (eta + 1))
    )
    child1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
    child2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)
    return torch.clamp(child1, x_min, x_max), torch.clamp(child2, x_min, x_max)
```


## 结果分析

### 收敛性指标
- **IGD值**: 越小表示解集越接近真实帕累托前沿
- **典型IGD值**: 0.001-0.01之间为良好收敛

### 多样性指标
- **Spread值**: 越小表示解集分布越均匀
- **典型Spread值**: 0.1-0.5之间为良好分布



## 依赖环境

### 经典版本依赖
```
numpy>=1.19.0
matplotlib>=3.3.0
```

### 张量化版本依赖
```
torch>=1.8.0
numpy>=1.19.0
matplotlib>=3.3.0
```

## 许可证

本项目采用MIT许可证，详见LICENSE文件。
