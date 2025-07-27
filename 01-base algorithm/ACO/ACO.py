import numpy as np
import math
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

def f(x):
    return abs(x * math.sin(x) * math.cos(2*x) - 2*x*math.sin(3*x) + 3*x*math.sin(4*x))

class ACO:
    def __init__(self, num_ants=50, max_iter=100, bounds=(0,50), k=10, q_init=0.5, q_final=0.01, 
                 xi=0.85, stagnation_limit=20, restart_prob=0.05, diversity_weight=0.3, visualize=False):
        self.num_ants = num_ants      # 蚂蚁数量
        self.max_iter = max_iter      # 最大迭代次数
        self.bounds = bounds          # 搜索范围
        self.k = k                    # 档案大小（保留的优质解数量）
        self.q_init = q_init          # 初始局部搜索强度参数 (增大初始范围)
        self.q_final = q_final        # 最终局部搜索强度参数 (允许更精细的最终搜索)
        self.q = q_init               # 当前局部搜索强度参数
        self.xi = xi                  # 信息素挥发率
        self.archive = []             # 解决方案档案（保存位置和适应度）
        self.visualize = visualize    # 是否可视化
        self.best_fitness = -float('inf')  # 最优适应度
        self.best_solution = None     # 最优解
        self.history_best_fitness = []  # 记录每次迭代的最优适应度
        self.history_positions = []   # 记录历史蚂蚁位置
        
        # 改进参数
        self.stagnation_counter = 0   # 停滞计数器
        self.stagnation_limit = stagnation_limit  # 停滞限制 (新增)
        self.restart_prob = restart_prob  # 重启概率 (新增)
        self.diversity_weight = diversity_weight  # 多样性权重 (新增)
        self.prev_best_fitness = -float('inf')  # 上一次最优适应度
        self.global_best_fitness = -float('inf')  # 全局最优适应度
        self.global_best_solution = None  # 全局最优解
        
        # 初始化蚂蚁位置
        self.initialize_ants()
        
        # 设置可视化
        if self.visualize:
            self.setup_visualization()
            
    def setup_visualization(self):
        """设置可视化界面"""
        self.fig = plt.figure(figsize=(12, 5))
        
        # 计算函数在搜索范围内的最大值用于设置纵轴范围
        x_vals = np.linspace(self.bounds[0], self.bounds[1], 1000)
        y_vals = [f(x) for x in x_vals]
        max_y = max(y_vals) * 1.2  # 留出20%的额外空间
        
        # 蚂蚁位置子图
        self.ax1 = self.fig.add_subplot(121)
        self.ax1.set_xlim(self.bounds[0], self.bounds[1])
        self.ax1.set_ylim(0, max_y)  # 自适应函数值范围
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('函数值f(x)')
        self.ax1.set_title('蚂蚁分布')
        
        # 适应度曲线子图
        self.ax2 = self.fig.add_subplot(122)
        self.ax2.set_xlabel('迭代次数')
        self.ax2.set_ylabel('最优适应度')
        self.ax2.set_title('收敛曲线')
        
        # 绘制函数曲线
        self.ax1.plot(x_vals, y_vals, 'b-', alpha=0.3)
        
        # 获取初始位置和函数值
        positions = [ant[0] for ant in self.archive]
        values = [ant[1] for ant in self.archive]
        all_ants_positions = [pos for pos, _ in self.current_solutions]
        all_ants_values = [val for _, val in self.current_solutions]
        
        # 初始散点图和曲线
        self.scatter = self.ax1.scatter(all_ants_positions, all_ants_values, c='blue', alpha=0.5, label='蚂蚁')
        self.elite_scatter = self.ax1.scatter(positions, values, c='green', alpha=0.7, s=50, label='精英蚂蚁')
        if self.best_solution is not None:
            self.best_scatter = self.ax1.scatter(self.best_solution, f(self.best_solution),
                                            c='red', s=100, marker='*', label='当前最优解')
        else:
            self.best_scatter = self.ax1.scatter([], [], c='red', s=100, marker='*', label='当前最优解')
            
        # 添加全局最优解的可视化
        self.global_best_scatter = self.ax1.scatter([], [], c='purple', s=120, marker='*', label='全局最优解')
            
        self.line, = self.ax2.plot([], [], 'g-')
        
        self.ax1.legend()
        plt.tight_layout()
        plt.ion()  # 打开交互模式
        plt.show(block=False)

    def initialize_ants(self):
        """初始化蚂蚁位置"""
        positions = np.random.uniform(self.bounds[0], self.bounds[1], self.num_ants)
        fitness = [f(x) for x in positions]
        self.current_solutions = list(zip(positions, fitness))
        self.archive = sorted(self.current_solutions, key=lambda x: x[1], reverse=True)[:self.k]
        
        # 更新最优解
        if self.archive[0][1] > self.best_fitness:
            self.best_fitness = self.archive[0][1]
            self.best_solution = self.archive[0][0]
            
            # 更新全局最优解
            if self.best_fitness > self.global_best_fitness:
                self.global_best_fitness = self.best_fitness
                self.global_best_solution = self.best_solution
        
        # 存储初始状态
        if self.visualize:
            all_positions = [pos for pos, _ in self.current_solutions]
            self.history_positions.append(all_positions)
            self.history_best_fitness.append(self.best_fitness)

    def calculate_diversity(self):
        """计算种群多样性"""
        # 使用位置标准差作为多样性指标
        positions = [ant[0] for ant in self.archive]
        return np.std(positions) / (self.bounds[1] - self.bounds[0])

    def update_search_parameters(self, iter_num):
        """更新搜索参数"""
        # 动态调整搜索强度参数q
        progress = iter_num / self.max_iter
        self.q = self.q_init * (1 - progress) + self.q_final * progress
        
        # 检查是否停滞
        if abs(self.best_fitness - self.prev_best_fitness) < 1e-6:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
            
        self.prev_best_fitness = self.best_fitness
        
        # 如果停滞超过限制，增加搜索范围
        if self.stagnation_counter >= self.stagnation_limit:
            self.q = min(self.q * 3.0, 1.0)  # 临时增大搜索范围
            self.stagnation_counter = 0
            return True  # 表示需要多样化
        
        return False

    def generate_new_solutions(self):
        """生成新的解决方案"""
        need_diversify = self.update_search_parameters(len(self.history_best_fitness))
        
        # 计算种群多样性
        diversity = self.calculate_diversity()
        
        new_solutions = []
        for i in range(self.num_ants):
            # 随机重启机制: 有一定概率随机生成解，而不是根据档案生成
            if np.random.rand() < self.restart_prob or need_diversify:
                # 全局探索: 完全随机位置
                new_x = np.random.uniform(self.bounds[0], self.bounds[1])
                new_fitness = f(new_x)
                new_solutions.append((new_x, new_fitness))
                continue
                
            # 选择参考解（基于信息素权重）
            # 根据多样性调整权重
            if diversity < 0.1:  # 多样性低时，增加对低排名解的选择概率
                weights = np.ones(self.k)  # 均匀选择
            else:
                # 正常情况下优先选择高排名解 因为前面是升序 要想概率大这里要降序来赋值
                weights = np.arange(self.k, 0, -1) ** (1 - self.diversity_weight * (1 - diversity))
                
            weights = weights / weights.sum()
            selected_idx = np.random.choice(self.k, p=weights)
            mean = self.archive[selected_idx][0]#选择参考解 选择了概率获得的x值
            
            # 计算标准差（局部搜索范围）
            sigma = self.q * sum(abs(s1[0]-s2[0]) for s1, s2 in zip(self.archive, self.archive[1:] + [self.archive[0]])) / self.k
            
            # 偶尔参考全局最优解
            if np.random.rand() < 0.1 and self.global_best_solution is not None:
                mean = (mean + self.global_best_solution) / 2
            
            # 生成新解并限制在范围内
            new_x = np.random.normal(mean, sigma)#正态分布
            new_x = np.clip(new_x, self.bounds[0], self.bounds[1])#限制在范围内
            new_fitness = f(new_x)
            new_solutions.append((new_x, new_fitness))
        
        self.current_solutions = new_solutions
        
        # 合并新旧解并更新档案
        combined = self.archive + new_solutions
        
        # 排序并保留精英解
        self.archive = sorted(combined, key=lambda x: x[1], reverse=True)[:self.k]
        
        # 更新最优解
        if self.archive[0][1] > self.best_fitness:
            self.best_fitness = self.archive[0][1]
            self.best_solution = self.archive[0][0]
            
            # 更新全局最优解
            if self.best_fitness > self.global_best_fitness:
                self.global_best_fitness = self.best_fitness
                self.global_best_solution = self.best_solution
    
    def _update_plot(self, iter_num):
        """更新可视化图表"""
        if not self.visualize:
            return
        
        # 获取当前蚂蚁位置和函数值
        positions = [ant[0] for ant in self.archive]
        values = [ant[1] for ant in self.archive]
        all_ants_positions = [pos for pos, _ in self.current_solutions]
        all_ants_values = [val for _, val in self.current_solutions]
            
        # 更新蚂蚁位置
        self.scatter.set_offsets(np.column_stack((all_ants_positions, all_ants_values)))
        self.elite_scatter.set_offsets(np.column_stack((positions, values)))
        
        # 更新当前最优位置
        if self.best_solution is not None:
            best_value = f(self.best_solution)
            self.best_scatter.set_offsets([[self.best_solution, best_value]])
            
        # 更新全局最优位置
        if self.global_best_solution is not None:
            global_best_value = f(self.global_best_solution)
            self.global_best_scatter.set_offsets([[self.global_best_solution, global_best_value]])
        
        # 更新收敛曲线
        self.line.set_data(range(len(self.history_best_fitness)), self.history_best_fitness)
        self.ax2.relim()
        self.ax2.autoscale_view()
        
        # 动态调整收敛曲线的纵轴，确保所有点都可见
        if len(self.history_best_fitness) > 1:
            min_val = min(self.history_best_fitness)
            max_val = max(self.history_best_fitness)
            margin = (max_val - min_val) * 0.1  # 10%的边距
            if margin == 0:  # 处理所有值相同的情况
                margin = max_val * 0.1 if max_val != 0 else 1.0
            self.ax2.set_ylim(min_val - margin, max_val + margin)
        
        # 更新标题并显示停滞情况
        stagnation_info = f" (停滞: {self.stagnation_counter}/{self.stagnation_limit})" if self.stagnation_counter > 0 else ""
        diversity = self.calculate_diversity()
        
        self.ax1.set_title(f'蚂蚁分布 (迭代 {iter_num+1}/{self.max_iter}, 多样性: {diversity:.2f})')
        self.ax2.set_title(f'收敛曲线 (当前值: {self.best_fitness:.6f}, 全局最优: {self.global_best_fitness:.6f}){stagnation_info}')
        
        # 刷新图形
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.05)  # 暂停一小段时间以便观察

    def run(self):
        """运行蚁群算法"""
        # 可视化初始状态
        if self.visualize:
            self._update_plot(0)
        
        for iter in range(1, self.max_iter + 1):
            # 生成新解决方案
            self.generate_new_solutions()
            
            # 记录历史
            if self.visualize:
                all_positions = [pos for pos, _ in self.current_solutions]
                self.history_positions.append(all_positions)
                self.history_best_fitness.append(self.best_fitness)
                
                # 更新可视化
                self._update_plot(iter)
                
        return self.global_best_solution, self.global_best_fitness
    
    def plot_convergence(self):
        """绘制收敛曲线（静态图）"""
        if not self.history_best_fitness:
            print("没有可用的优化历史数据")
            return
            
        # 创建Tkinter窗口
        root = tk.Tk()
        root.title("ACO收敛曲线")
        root.geometry("800x600")
        
        # 创建Figure和Axes
        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(range(len(self.history_best_fitness)), self.history_best_fitness)
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('最优适应度')
        ax.set_title('ACO收敛曲线')
        
        # 创建Canvas并将图形嵌入Tkinter窗口
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # 启动Tkinter主循环
        root.mainloop()
    
    def animate_convergence(self):
        """创建蚁群优化过程的动画"""
        if not self.history_positions:
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
        ax1.set_title('蚂蚁分布随时间变化')
        
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('最优适应度')
        ax2.set_title('收敛曲线')
        
        scatter = ax1.scatter([], [], c='blue', alpha=0.5)
        best_scatter = ax1.scatter([], [], c='red', s=100, marker='*')
        global_best_scatter = ax1.scatter([], [], c='purple', s=120, marker='*')
        line, = ax2.plot([], [], 'g-')
        
        def init():
            scatter.set_offsets(np.empty((0, 2)))
            best_scatter.set_offsets(np.empty((0, 2)))
            global_best_scatter.set_offsets(np.empty((0, 2)))
            line.set_data([], [])
            return scatter, best_scatter, global_best_scatter, line
        
        def update(frame):
            # 更新蚂蚁位置
            positions = self.history_positions[frame]
            values = [f(pos) for pos in positions]
            scatter.set_offsets(np.column_stack((positions, values)))
            
            # 得到当前帧对应的最佳位置
            if frame < len(self.history_best_fitness):
                # 使用历史记录中对应帧的最优解
                current_best_fitness = self.history_best_fitness[frame]
                current_best_pos = next((pos for pos in positions if abs(f(pos) - current_best_fitness) < 1e-6), self.best_solution)
                best_val = f(current_best_pos)
                
                best_scatter.set_offsets([[current_best_pos, best_val]])
                
                # 全局最优
                global_best_scatter.set_offsets([[self.global_best_solution, f(self.global_best_solution)]])
                
                # 更新收敛曲线
                line.set_data(range(frame + 1), self.history_best_fitness[:frame + 1])
                ax2.relim()
                ax2.autoscale_view()
                
                # 动态调整收敛曲线的纵轴，确保所有点都可见
                if frame > 0:
                    current_data = self.history_best_fitness[:frame + 1]
                    min_val = min(current_data)
                    max_val = max(current_data)
                    margin = (max_val - min_val) * 0.1  # 10%的边距
                    if margin == 0:  # 处理所有值相同的情况
                        margin = max_val * 0.1 if max_val != 0 else 1.0
                    ax2.set_ylim(min_val - margin, max_val + margin)
                
                # 更新标题
                ax2.set_title(f'收敛曲线 (当前最优值: {current_best_fitness:.6f}, 全局最优: {self.global_best_fitness:.6f})')
            
            ax1.set_title(f'蚂蚁分布 (迭代 {frame+1}/{len(self.history_positions)})')
            
            return scatter, best_scatter, global_best_scatter, line
        
        ani = FuncAnimation(fig, update, frames=len(self.history_positions),
                           init_func=init, blit=False, repeat=True, interval=200)
        
        # 保存动画对象的引用，防止被垃圾回收
        self._animation = ani
        
        plt.tight_layout()
        plt.show()

# 运行蚁群算法
if __name__ == "__main__":
    print("改进蚁群算法(ACO)演示程序")
    print("="*50)
    
    # 参数设置
    num_ants = 50           # 蚂蚁数量
    max_iterations = 100    # 最大迭代次数
    bounds = (0, 50)        # 搜索空间范围
    archive_size = 10       # 精英解档案大小
    q_init = 0.5            # 初始局部搜索范围参数 (增大)
    q_final = 0.01          # 最终局部搜索范围参数
    xi = 0.85               # 信息素挥发率
    stagnation_limit = 20   # 停滞限制
    restart_prob = 0.05     # 重启概率
    diversity_weight = 0.3  # 多样性权重
    
    print(f"参数设置: 蚂蚁数量={num_ants}, 最大迭代次数={max_iterations}")
    print(f"          搜索范围={bounds}, 精英解数量={archive_size}")
    print(f"          局部搜索参数={q_init}→{q_final}, 信息素挥发率={xi}")
    print(f"          停滞限制={stagnation_limit}, 重启概率={restart_prob}")
    print(f"          多样性权重={diversity_weight}")
    print("-"*50)
    
    # 创建ACO优化器，并启用可视化
    aco = ACO(
        num_ants=num_ants, 
        max_iter=max_iterations, 
        bounds=bounds,
        k=archive_size,
        q_init=q_init,
        q_final=q_final,
        xi=xi,
        stagnation_limit=stagnation_limit,
        restart_prob=restart_prob,
        diversity_weight=diversity_weight,
        visualize=True
    )
    
    print("开始ACO优化...")
    # 记录开始时间
    start_time = time.time()
    
    # 运行优化
    best_x, best_val = aco.run()
    
    # 计算运行时间
    elapsed_time = time.time() - start_time
    
    print("\n优化完成！")
    print(f"运行时间: {elapsed_time:.2f}秒")
    print(f"ACO最大值：{best_val:.6f}，x={best_x:.6f}")
    print("-"*50)
    
    # 展示动画或静态图
    while True:
        choice = input("选择查看方式: 1=动画回放 2=静态收敛曲线 q=退出: ")
        if choice == '1':
            aco.animate_convergence()
        elif choice == '2':
            aco.plot_convergence()
        elif choice.lower() == 'q':
            break
        else:
            print("无效的选择，请重新输入")
    
    print("程序结束")