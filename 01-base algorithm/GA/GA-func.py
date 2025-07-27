import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import time
from matplotlib.animation import FuncAnimation

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 目标函数定义
def f(x):
    return abs(-x**3 + 7*x + 13)

# 遗传算法类
class GeneticAlgorithm:
    def __init__(self, pop_size, bounds, crossover_rate, mutation_rate, max_iter, visualize=False):
        self.pop_size = pop_size
        self.bounds = bounds
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_iter = max_iter
        self.population = np.random.uniform(bounds[0], bounds[1], pop_size)
        self.best_solution = None
        self.best_fitness = float('inf')
        self.history = []
        self.visualize = visualize
        self.history_populations = []  # 记录历史种群状态
        
        # 初始化种群
        self.initialize_population()
        
        # 设置可视化
        if self.visualize:
            self.setup_visualization()

    def initialize_population(self):
        """初始化种群"""
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], self.pop_size)#均匀分布里随机采样
        # 初始评估
        self.evaluate()
        
        # 存储初始状态
        if self.visualize:
            self.history_populations.append(self.population.copy())
            self.history.append(self.best_fitness)

    def setup_visualization(self):
        """设置可视化界面"""
        self.fig = plt.figure(figsize=(12, 5))
        
        # 计算函数在搜索范围内的最大值用于设置纵轴范围
        x_vals = np.linspace(self.bounds[0], self.bounds[1], 1000)
        y_vals = [f(x) for x in x_vals]
        max_y = max(y_vals) * 1.2  # 留出20%的额外空间
        
        # 个体分布子图
        self.ax1 = self.fig.add_subplot(121)
        self.ax1.set_xlim(self.bounds[0], self.bounds[1])
        self.ax1.set_ylim(0, max_y)  # 自适应函数值范围
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('函数值f(x)')
        self.ax1.set_title('种群分布')
        
        # 适应度曲线子图
        self.ax2 = self.fig.add_subplot(122)
        self.ax2.set_xlabel('迭代次数')
        self.ax2.set_ylabel('最优适应度 |f(x)|')
        self.ax2.set_title('收敛曲线')
        
        # 绘制函数曲线
        self.ax1.plot(x_vals, y_vals, 'b-', alpha=0.3)
        
        # 获取初始位置和函数值
        values = [f(ind) for ind in self.population]
        
        # 初始散点图和曲线
        self.scatter = self.ax1.scatter(self.population, values, c='blue', alpha=0.7, label='个体')
        if self.best_solution is not None:
            self.best_scatter = self.ax1.scatter(self.best_solution, f(self.best_solution),
                                          c='red', s=100, marker='*', label='最优个体')
        else:
            self.best_scatter = self.ax1.scatter([], [], c='red', s=100, marker='*', label='最优个体')
            
        self.line, = self.ax2.plot([], [], 'g-')
        
        self.ax1.legend()
        plt.tight_layout()
        plt.ion()  # 打开交互模式
        plt.show(block=False)

    def _update_plot(self, generation):
        """更新可视化图表"""
        if not self.visualize:
            return
        
        # 获取当前个体的函数值
        values = [f(ind) for ind in self.population]
            
        # 更新个体位置
        self.scatter.set_offsets(np.column_stack((self.population, values)))
        
        # 更新全局最优位置
        if self.best_solution is not None:
            best_value = f(self.best_solution)
            self.best_scatter.set_offsets([[self.best_solution, best_value]])
        
        # 更新收敛曲线
        self.line.set_data(range(len(self.history)), self.history)
        self.ax2.relim()
        self.ax2.autoscale_view()
        
        # 动态调整收敛曲线的纵轴，确保所有点都可见
        if len(self.history) > 1:
            min_val = min(self.history)
            max_val = max(self.history)
            margin = (max_val - min_val) * 0.1  # 10%的边距
            if margin == 0:  # 处理所有值相同的情况
                margin = max_val * 0.1 if max_val != 0 else 1.0
            self.ax2.set_ylim(min_val - margin, max_val + margin)
        
        # 更新标题
        self.ax1.set_title(f'种群分布 (迭代 {generation+1}/{self.max_iter})')
        self.ax2.set_title(f'收敛曲线 (当前最优值: {self.best_fitness:.6f})')
        
        # 刷新图形
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.05)  # 暂停一小段时间以便观察

    def evaluate(self):
        """评估种群适应度"""
        fitness = np.array([f(x) for x in self.population])
        best_idx = np.argmin(fitness)#最小适应度的个体
        current_best_fitness = fitness[best_idx]
        if current_best_fitness < self.best_fitness:#更新最优适应度
            self.best_fitness = current_best_fitness
            self.best_solution = self.population[best_idx]
        return fitness

    def select(self, fitness):
        """轮盘赌选择"""
        probabilities = 1 / (fitness + 1e-6)  # 避免除零
        probabilities /= probabilities.sum()#归一化
        selected_indices = np.random.choice(
            range(self.pop_size), size=self.pop_size, p=probabilities
        )
        return self.population[selected_indices]

    def crossover(self, parents):
        """算术交叉"""
        new_population = []
        for i in range(0, self.pop_size, 2):
            parent1 = parents[i]
            parent2 = parents[i+1] if i+1 < self.pop_size else parent1
            if np.random.rand() < self.crossover_rate:
                alpha = np.random.rand()
                child1 = alpha * parent1 + (1 - alpha) * parent2
                child2 = alpha * parent2 + (1 - alpha) * parent1
                new_population.extend([child1, child2])
            else:
                new_population.extend([parent1, parent2])
        return np.array(new_population[:self.pop_size])

    def mutate(self, population):
        """高斯变异"""
        for i in range(self.pop_size):
            if np.random.rand() < self.mutation_rate:
                mutation_strength = (self.bounds[1] - self.bounds[0]) * 0.1#变异强度
                population[i] += np.random.normal(0, mutation_strength)#高斯分布
                population[i] = np.clip(population[i], self.bounds[0], self.bounds[1])#限制在搜索范围内，若不在范围内，则取边界值
        return population

    def run(self):
        """执行优化"""
        # 可视化初始状态
        if self.visualize:
            self._update_plot(0)
            
        for generation in range(self.max_iter):
            fitness = self.evaluate()
            self.history.append(self.best_fitness)
            
            # 收集历史数据用于动画
            if self.visualize:
                self.history_populations.append(self.population.copy())
                self._update_plot(generation)
                
            parents = self.select(fitness)
            offspring = self.crossover(parents)
            mutated_offspring = self.mutate(offspring)
            self.population = mutated_offspring
            
        return self.best_solution, self.best_fitness
        
    def plot_convergence(self):
        """绘制收敛曲线（静态图）"""
        if not self.history:
            print("没有可用的优化历史数据")
            return
            
        # 创建Tkinter窗口
        root = tk.Tk()
        root.title("GA收敛曲线")
        root.geometry("800x600")
        
        # 创建Figure和Axes
        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(range(len(self.history)), self.history)
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('最优适应度 |f(x)|')
        ax.set_title('GA收敛曲线')
        
        # 创建Canvas并将图形嵌入Tkinter窗口
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # 启动Tkinter主循环
        root.mainloop()
    
    def animate_convergence(self):
        """创建遗传算法进化过程的动画"""
        if not self.history_populations:
            print("没有可用的历史数据进行动画显示")
            return
            
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        # 计算函数在搜索范围内的最大值用于设置纵轴范围
        x_vals = np.linspace(self.bounds[0], self.bounds[1], 1000)
        y_vals = [f(x) for x in x_vals]
        max_y = max(y_vals) * 1.2  # 留出20%的额外空间
        
        # 绘制函数图像
        ax1.plot(x_vals, y_vals, 'b-', alpha=0.3)
        
        ax1.set_xlim(self.bounds[0], self.bounds[1])
        ax1.set_ylim(0, max_y)  # 自适应函数值范围
        ax1.set_xlabel('X')
        ax1.set_ylabel('函数值f(x)')
        ax1.set_title('种群分布随迭代变化')
        
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('最优适应度 |f(x)|')
        ax2.set_title('收敛曲线')
        
        scatter = ax1.scatter([], [], c='blue', alpha=0.7)
        best_scatter = ax1.scatter([], [], c='red', s=100, marker='*')
        line, = ax2.plot([], [], 'g-')
        
        def init():
            scatter.set_offsets(np.empty((0, 2)))
            best_scatter.set_offsets(np.empty((0, 2)))
            line.set_data([], [])
            return scatter, best_scatter, line
        
        def update(frame):
            # 更新种群位置
            population = self.history_populations[frame]
            values = [f(ind) for ind in population]
            scatter.set_offsets(np.column_stack((population, values)))
            
            # 得到当前帧对应的最佳位置
            if frame < len(self.history):
                # 简化：使用最终的最优个体
                best_ind = self.best_solution
                best_val = f(best_ind)
                
                best_scatter.set_offsets([[best_ind, best_val]])
                
                # 更新收敛曲线
                line.set_data(range(frame + 1), self.history[:frame + 1])
                ax2.relim()
                ax2.autoscale_view()
                
                # 动态调整收敛曲线的纵轴，确保所有点都可见
                if frame > 0:
                    current_data = self.history[:frame + 1]
                    min_val = min(current_data)
                    max_val = max(current_data)
                    margin = (max_val - min_val) * 0.1  # 10%的边距
                    if margin == 0:  # 处理所有值相同的情况
                        margin = max_val * 0.1 if max_val != 0 else 1.0
                    ax2.set_ylim(min_val - margin, max_val + margin)
                
                # 更新标题
                ax2.set_title(f'收敛曲线 (当前最优值: {self.history[frame]:.6f})')
            
            ax1.set_title(f'种群分布 (迭代 {frame+1}/{len(self.history_populations)})')
            
            return scatter, best_scatter, line
        
        ani = FuncAnimation(fig, update, frames=len(self.history_populations),
                           init_func=init, blit=False, repeat=True, interval=200)
        
        # 保存动画对象的引用，防止被垃圾回收
        self._animation = ani
        
        plt.tight_layout()
        plt.show()

# 主函数
if __name__ == "__main__":
    print("遗传算法(GA)优化演示程序")
    print("="*50)
    
    # 参数设置
    pop_size = 50
    bounds = [-5, 5]
    crossover_rate = 0.8
    mutation_rate = 0.1
    max_iter = 150
    
    print(f"参数设置: 种群大小={pop_size}, 最大迭代次数={max_iter}")
    print(f"          交叉概率={crossover_rate}, 变异概率={mutation_rate}")
    print(f"          搜索范围={bounds}")
    print("-"*50)
    
    # 创建GA优化器，并启用可视化
    ga = GeneticAlgorithm(
        pop_size=pop_size, 
        bounds=bounds,
        crossover_rate=crossover_rate, 
        mutation_rate=mutation_rate, 
        max_iter=max_iter,
        visualize=True
    )
    
    print("开始GA优化...")
    # 记录开始时间
    start_time = time.time()
    
    # 运行遗传算法
    best_x, best_fitness = ga.run()
    
    # 计算运行时间
    elapsed_time = time.time() - start_time
    
    print("\n优化完成！")
    print(f"运行时间: {elapsed_time:.2f}秒")
    print(f"最优解 x = {best_x:.6f}, 方程值 |f(x)| = {best_fitness:.6f}")
    print("-"*50)
    
    # 展示动画或静态图
    while True:
        choice = input("选择查看方式: 1=动画回放 2=静态收敛曲线 q=退出: ")
        if choice == '1':
            ga.animate_convergence()
        elif choice == '2':
            ga.plot_convergence()
        elif choice.lower() == 'q':
            break
        else:
            print("无效的选择，请重新输入")
    
    print("程序结束")