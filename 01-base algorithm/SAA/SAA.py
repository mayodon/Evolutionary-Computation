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
    """计算目标函数值"""
    return abs(x * math.sin(x) * math.cos(2*x) - 2*x * math.sin(3*x) + 3*x * math.sin(4*x))


class SimulatedAnnealing:
    def __init__(self, 
                objective_func,        # 目标函数
                bounds,                # 变量范围，格式为 [x_min, x_max]
                initial_temp=1000,     # 初始温度
                final_temp=1e-7,       # 终止温度
                alpha=0.95,            # 降温系数（指数降温）
                max_iter=1000,         # 最大迭代次数
                step_size=1.0,         # 邻域搜索步长
                visualize=False,       # 是否可视化
                restart_temp=100,      # 重启温度
                max_stagnation=20,     # 最大停滞次数
                n_restarts=3,          # 最大重启次数
                step_adjust_rate=0.95, # 步长调整率
                reannealing_threshold=0.1): # 重新退火阈值
        
        self.objective_func = objective_func
        self.bounds = bounds
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.alpha = alpha
        self.max_iter = max_iter
        self.step_size_init = step_size
        self.step_size = step_size
        self.visualize = visualize
        
        # 新增参数
        self.restart_temp = restart_temp        # 重启时的温度
        self.max_stagnation = max_stagnation    # 允许的最大停滞次数
        self.n_restarts = n_restarts            # 最大重启次数
        self.step_adjust_rate = step_adjust_rate # 步长调整率
        self.reannealing_threshold = reannealing_threshold # 重新退火阈值（接受率过低时重新退火）
        
        # 初始化解
        self.current_solution = np.random.uniform(bounds[0], bounds[1])
        self.current_energy = objective_func(self.current_solution)
        self.best_solution = self.current_solution
        self.best_energy = self.current_energy
        self.global_best_solution = self.best_solution
        self.global_best_energy = self.best_energy
        
        # 记录历史最优值和温度
        self.history = {
            'current_solutions': [self.current_solution],
            'current_energies': [self.current_energy],
            'best_energies': [self.best_energy],
            'temperatures': [initial_temp],
            'accept_rates': [1.0],  # 接受率历史
            'step_sizes': [step_size]  # 步长历史
        }
        
        self.temp = initial_temp  # 当前温度
        self.accept_count = 0     # 接受新解的计数
        self.iter_count = 0       # 迭代计数
        self.stagnation_count = 0 # 停滞计数
        self.restart_count = 0    # 重启计数
        
        # 设置可视化
        if self.visualize:
            self.setup_visualization()
    
    def setup_visualization(self):
        """设置可视化界面"""
        self.fig = plt.figure(figsize=(12, 6))
        
        # 计算函数在搜索范围内的最大值用于设置纵轴范围
        x_vals = np.linspace(self.bounds[0], self.bounds[1], 1000)
        y_vals = [self.objective_func(x) for x in x_vals]
        max_y = max(y_vals) * 1.2  # 留出20%的额外空间
        
        # 搜索位置子图
        self.ax1 = self.fig.add_subplot(121)
        self.ax1.set_xlim(self.bounds[0], self.bounds[1])
        self.ax1.set_ylim(0, max_y)  # 自适应函数值范围
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('函数值f(x)')
        self.ax1.set_title('搜索过程')
        
        # 适应度曲线子图
        self.ax2 = self.fig.add_subplot(122)
        self.ax2.set_xlabel('迭代次数')
        self.ax2.set_ylabel('最优适应度')
        self.ax2.set_title('收敛曲线')
        
        # 绘制函数曲线
        self.ax1.plot(x_vals, y_vals, 'b-', alpha=0.3, label='目标函数')
        
        # 初始散点图和曲线
        self.current_scatter = self.ax1.scatter(
            self.current_solution, 
            self.current_energy, 
            c='blue', s=80, alpha=0.7, label='当前解'
        )
        self.best_scatter = self.ax1.scatter(
            self.best_solution, 
            self.best_energy, 
            c='red', s=100, marker='*', label='当前最优解'
        )
        self.global_best_scatter = self.ax1.scatter(
            self.global_best_solution, 
            self.global_best_energy, 
            c='purple', s=120, marker='*', label='全局最优解'
        )
        
        # 收敛曲线
        self.line, = self.ax2.plot([], [], 'g-', label='最优值')
        self.temp_line, = self.ax2.plot([], [], 'r--', alpha=0.5, label='温度')
        self.step_line, = self.ax2.plot([], [], 'b:', alpha=0.5, label='步长')
        
        # 添加图例
        self.ax1.legend(loc='upper right')
        self.ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.ion()  # 打开交互模式
        plt.show(block=False)
    
    def _update_plot(self, iter_num):
        """更新可视化图表"""
        if not self.visualize:
            return
        
        # 更新当前解位置
        self.current_scatter.set_offsets([[self.current_solution, self.current_energy]])
        
        # 更新当前最优解位置
        self.best_scatter.set_offsets([[self.best_solution, self.best_energy]])
        
        # 更新全局最优解位置
        self.global_best_scatter.set_offsets([[self.global_best_solution, self.global_best_energy]])
        
        # 更新收敛曲线
        self.line.set_data(range(len(self.history['best_energies'])), self.history['best_energies'])
        
        # 更新温度曲线 - 归一化到与适应度相同的比例尺度
        temps = np.array(self.history['temperatures'])
        # 避免除以零：如果所有温度都相同，则使用一个小常数
        temp_range = temps.max() - temps.min() if temps.max() != temps.min() else 1.0
        
        # 获取当前的最优适应度范围
        best_energies = np.array(self.history['best_energies'])
        energy_min = best_energies.min()
        energy_max = best_energies.max()
        energy_range = energy_max - energy_min if energy_max != energy_min else 1.0
        
        # 将温度归一化到适应度相同的比例尺度
        normalized_temps = energy_min + (temps - temps.min()) / temp_range * energy_range * 0.8
        
        self.temp_line.set_data(range(len(normalized_temps)), normalized_temps)
        
        # 绘制步长变化曲线
        if hasattr(self, 'step_line'):
            # 归一化步长
            steps = np.array(self.history['step_sizes'])
            step_range = steps.max() - steps.min() if steps.max() != steps.min() else 1.0
            normalized_steps = energy_min + (steps - steps.min()) / step_range * energy_range * 0.6
            self.step_line.set_data(range(len(normalized_steps)), normalized_steps)
        
        # 更新坐标轴范围
        self.ax2.relim()
        self.ax2.autoscale_view()
        
        # 更新标题
        accept_rate = self.history['accept_rates'][-1]
        stagnation_info = f", 停滞: {self.stagnation_count}/{self.max_stagnation}" if self.stagnation_count > 0 else ""
        self.ax1.set_title(f'搜索过程 (迭代: {iter_num+1}/{self.max_iter}, 温度: {self.temp:.2f}, 步长: {self.step_size:.4f})')
        self.ax2.set_title(f'收敛曲线 (当前值: {self.best_energy:.6f}, 接受率: {accept_rate:.2f}{stagnation_info})')
        
        # 刷新图形
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)  # 暂停一小段时间以便观察
    
    def run(self):
        """运行模拟退火算法"""
        # 可视化初始状态
        if self.visualize:
            self._update_plot(0)
        
        global_iter = 0
        self.restart_count = 0
        reached_max_iter = False
        exploration_runs = 2  # 前几次运行强调探索
        
        # 记录已知的局部最优解，避免多次重启陷入相同的局部最优
        known_local_optima = []
        
        # 主循环，允许多次重启
        while self.restart_count <= self.n_restarts and global_iter < self.max_iter:
            # 重启机制
            if self.restart_count > 0:
                # 策略：前几次随机重启，后面根据已知局部最优设置更合理的起点
                if self.restart_count <= exploration_runs:
                    # 完全随机重启，增强探索
                    self.current_solution = np.random.uniform(self.bounds[0], self.bounds[1])
                else:
                    # 基于已知局部最优解选择远离的区域
                    if known_local_optima:
                        # 确定远离已知局部最优的起点
                        max_distance = 0
                        best_candidate = None
                        
                        # 生成多个候选点，选择离已知局部最优最远的点
                        for _ in range(20):
                            candidate = np.random.uniform(self.bounds[0], self.bounds[1])
                            min_distance = min([abs(candidate - opt) for opt in known_local_optima])
                            if min_distance > max_distance:
                                max_distance = min_distance
                                best_candidate = candidate
                        
                        self.current_solution = best_candidate
                    else:
                        # 没有已知局部最优，随机选择
                        self.current_solution = np.random.uniform(self.bounds[0], self.bounds[1])
                
                # 计算新起点的函数值
                self.current_energy = self.objective_func(self.current_solution)
                
                # 重启参数
                if self.restart_count <= exploration_runs:
                    # 前几次用较高温度和较大步长，鼓励广泛探索
                    self.temp = self.restart_temp * 2
                    self.step_size = self.step_size_init * 1.5
                    print(f"第{self.restart_count}次探索性重启 - 温度重置为{self.temp:.2f}, 步长设为{self.step_size:.2f}, 新起点为{self.current_solution:.6f}")
                else:
                    # 后续用正常温度
                    self.temp = self.restart_temp
                    self.step_size = self.step_size_init
                    print(f"第{self.restart_count}次重启 - 温度重置为{self.temp:.2f}, 步长设为{self.step_size:.2f}, 新起点为{self.current_solution:.6f}")
                
                # 记录重启状态
                self.history['current_solutions'].append(self.current_solution)
                self.history['current_energies'].append(self.current_energy)
                self.history['best_energies'].append(self.best_energy)
                self.history['temperatures'].append(self.temp)
                self.history['accept_rates'].append(1.0)
                self.history['step_sizes'].append(self.step_size)
                
                # 重置停滞计数
                self.stagnation_count = 0
                
                # 可视化重启状态
                if self.visualize:
                    self._update_plot(global_iter)
            
            # 模拟退火搜索过程
            local_improvement = False
            prev_best_energy = self.best_energy
            
            # 当前迭代的最优值和对应解
            current_local_best = self.current_energy
            current_local_best_solution = self.current_solution
            
            # 为了确保搜索充分，设置最小迭代次数
            min_iterations = min(30, self.max_iter // 10)
            search_finished = False
            
            for i in range(self.max_iter):
                if global_iter >= self.max_iter:
                    reached_max_iter = True
                    break
                    
                global_iter += 1
                self.iter_count = global_iter
                
                # 计算当前温度（指数降温）
                self.temp = self.initial_temp * (self.alpha ** i)
                
                # 跟踪接受新解的次数
                accept_count_iter = 0
                trial_count = 0
                
                # 在每个温度下进行多次尝试
                max_trials = 20  # 每个温度下的尝试次数
                for _ in range(max_trials):
                    trial_count += 1
                    
                    # 随机选择搜索策略
                    rand_val = np.random.rand()
                    if rand_val < 0.85:  # 85%概率做局部搜索
                        # 局部搜索：在当前解附近随机扰动，使用自适应步长
                        new_solution = self.current_solution + np.random.uniform(-self.step_size, self.step_size)
                    elif rand_val < 0.95:  # 10%概率做定向搜索
                        # 如果有全局最优解，朝全局最优方向搜索
                        if self.global_best_solution is not None:
                            direction = self.global_best_solution - self.current_solution
                            # 添加随机扰动，避免直线搜索
                            perturbation = np.random.uniform(-0.5, 0.5) * self.step_size
                            new_solution = self.current_solution + 0.5 * direction + perturbation
                        else:
                            new_solution = self.current_solution + np.random.uniform(-self.step_size, self.step_size)
                    else:  # 5%概率做全局随机探索
                        new_solution = np.random.uniform(self.bounds[0], self.bounds[1])
                    
                    # 确保新解在搜索范围内
                    new_solution = np.clip(new_solution, self.bounds[0], self.bounds[1])
                    new_energy = self.objective_func(new_solution)
                    
                    # 计算能量差（最大化问题，能量差为负表示更优）
                    delta_energy = self.current_energy - new_energy
                    
                    # 动态退火准则：温度低时，更严格地拒绝差解
                    T_factor = max(0.01, self.temp / self.initial_temp)  # 温度比例因子
                    accept_prob = np.exp(-delta_energy / (self.temp * (1 - 0.5 * T_factor)))
                    
                    # 接受新解的条件：1.新解更好 或 2.满足概率接受条件
                    if delta_energy < 0 or np.random.rand() < accept_prob:
                        self.current_solution = new_solution
                        self.current_energy = new_energy
                        accept_count_iter += 1
                        
                        # 更新当前迭代的最优解
                        if self.current_energy > current_local_best:
                            current_local_best = self.current_energy
                            current_local_best_solution = self.current_solution
                        
                        # 更新全局最优解
                        if self.current_energy > self.best_energy:
                            local_improvement = True
                            self.stagnation_count = 0  # 找到更好解，重置停滞计数
                            self.best_solution = self.current_solution
                            self.best_energy = self.current_energy
                            
                            # 更新全局最优解
                            if self.best_energy > self.global_best_energy:
                                self.global_best_solution = self.best_solution
                                self.global_best_energy = self.best_energy
                                print(f"找到新的全局最优解: {self.global_best_energy:.6f} at x = {self.global_best_solution:.6f}")
                
                # 计算接受率
                accept_rate = accept_count_iter / trial_count if trial_count > 0 else 0
                
                # 动态调整步长：根据接受率调整步长
                if accept_rate > 0.6:
                    # 接受率太高，增大步长以加强探索
                    self.step_size = min(self.step_size * (1/self.step_adjust_rate), (self.bounds[1] - self.bounds[0]) * 0.5)
                elif accept_rate < 0.2:
                    # 接受率太低，减小步长以加强开发
                    self.step_size = max(self.step_size * self.step_adjust_rate, (self.bounds[1] - self.bounds[0]) * 0.001)
                
                # 记录当前状态
                self.history['current_solutions'].append(self.current_solution)
                self.history['current_energies'].append(self.current_energy)
                self.history['best_energies'].append(self.best_energy)
                self.history['temperatures'].append(self.temp)
                self.history['accept_rates'].append(accept_rate)
                self.history['step_sizes'].append(self.step_size)
                
                # 更新可视化
                if self.visualize:
                    self._update_plot(global_iter)
                
                # 检查停滞情况
                if not local_improvement:
                    self.stagnation_count += 1
                else:
                    local_improvement = False
                
                # 停滞处理：如果长时间没有改进，考虑重新退火或重启
                if self.stagnation_count >= self.max_stagnation:
                    # 如果已经进行了足够的迭代，可以考虑本次搜索结束
                    if i >= min_iterations:
                        # 优先考虑重新退火（提高温度）
                        if accept_rate < self.reannealing_threshold and self.restart_count < self.n_restarts:
                            new_temp = self.temp * 10  # 提高温度
                            print(f"搜索停滞：温度从{self.temp:.6f}提高到{new_temp:.6f}")
                            self.temp = new_temp
                            self.stagnation_count = 0  # 重置停滞计数
                        else:
                            # 如果重新退火不足以解决，记录当前局部最优，然后考虑重启
                            print(f"搜索停滞{self.stagnation_count}次，记录当前局部最优点")
                            if current_local_best_solution is not None:
                                # 检查是否与已知局部最优足够接近
                                is_new_optimum = True
                                for opt in known_local_optima:
                                    if abs(current_local_best_solution - opt) < 0.1 * (self.bounds[1] - self.bounds[0]):
                                        is_new_optimum = False
                                        break
                                
                                if is_new_optimum:
                                    known_local_optima.append(current_local_best_solution)
                                    print(f"发现新的局部最优点: x = {current_local_best_solution:.6f}")
                            
                            search_finished = True
                            self.restart_count += 1
                            break
                
                # 收敛判断
                if self.temp <= self.final_temp:
                    # 温度降至最终温度，但仍要确保已经进行了足够的迭代
                    if i >= min_iterations:
                        print(f"温度降至{self.final_temp}以下，当前迭代结束")
                        search_finished = True
                        break
            
            # 如果到达了迭代次数但没有触发搜索完成，确保记录局部最优
            if not search_finished and not reached_max_iter:
                # 记录当前局部最优
                if current_local_best_solution is not None:
                    # 检查是否与已知局部最优足够接近
                    is_new_optimum = True
                    for opt in known_local_optima:
                        if abs(current_local_best_solution - opt) < 0.1 * (self.bounds[1] - self.bounds[0]):
                            is_new_optimum = False
                            break
                    
                    if is_new_optimum:
                        known_local_optima.append(current_local_best_solution)
                        print(f"迭代结束，发现新的局部最优点: x = {current_local_best_solution:.6f}")
            
            # 检查本次运行是否有提升
            if self.best_energy <= prev_best_energy and self.restart_count < self.n_restarts:
                print(f"本次搜索未改进解，当前最优值：{self.best_energy:.6f}")
                self.restart_count += 1
            elif self.best_energy > prev_best_energy:
                print(f"本次搜索找到更优解：{self.best_energy:.6f}")
            
            # 如果达到最大迭代次数，退出循环
            if reached_max_iter:
                print(f"达到最大迭代次数({self.max_iter})，结束搜索")
                break
        
        # 最终输出
        if known_local_optima:
            print(f"\n发现的所有局部最优点:")
            for i, opt in enumerate(known_local_optima):
                opt_val = self.objective_func(opt)
                print(f"  - 局部最优 #{i+1}: x = {opt:.6f}, f(x) = {opt_val:.6f}")
        
        print(f"\n总迭代次数：{global_iter}/{self.max_iter}，重启次数：{self.restart_count}/{self.n_restarts}")
        return self.global_best_solution, self.global_best_energy, self.history
    
    def plot_convergence(self):
        """绘制收敛曲线（静态图）"""
        if not self.history['best_energies']:
            print("没有可用的优化历史数据")
            return
            
        # 创建Tkinter窗口
        root = tk.Tk()
        root.title("模拟退火收敛曲线")
        root.geometry("800x600")
        
        # 创建Figure和Axes
        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(range(len(self.history['best_energies'])), self.history['best_energies'], 'g-', label='最优值')
        
        # 添加温度曲线（使用次坐标轴）
        ax2 = ax.twinx()
        ax2.plot(range(len(self.history['temperatures'])), self.history['temperatures'], 'r--', label='温度')
        ax2.set_ylabel('温度')
        ax2.set_yscale('log')  # 温度通常以对数形式更容易观察
        
        # 绘制步长变化
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('outward', 60))  # 将第三个y轴放在右侧，并偏移60个点
        ax3.plot(range(len(self.history['step_sizes'])), self.history['step_sizes'], 'b:', label='步长')
        ax3.set_ylabel('步长')
        
        # 标记重启点
        restart_indices = []
        for i in range(1, len(self.history['temperatures'])):
            if self.history['temperatures'][i] > self.history['temperatures'][i-1] * 5:
                restart_indices.append(i)
                ax.axvline(x=i, color='orange', linestyle='--', alpha=0.5)
                ax.text(i, min(self.history['best_energies']), "重启", rotation=90, va='bottom')
        
        # 设置标题和标签
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('最优适应度')
        ax.set_title('模拟退火收敛曲线 (包含重启和步长自适应)')
        
        # 添加图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')
        
        # 创建Canvas并将图形嵌入Tkinter窗口
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # 启动Tkinter主循环
        root.mainloop()
    
    def animate_search(self):
        """创建模拟退火优化过程的动画"""
        if not self.history['current_solutions']:
            print("没有可用的历史数据进行动画显示")
            return
            
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        # 计算函数在搜索范围内的值
        x_vals = np.linspace(self.bounds[0], self.bounds[1], 1000)
        y_vals = [self.objective_func(x) for x in x_vals]
        max_y = max(y_vals) * 1.2  # 留出20%的额外空间
        
        # 绘制函数图像
        ax1.plot(x_vals, y_vals, 'b-', alpha=0.3, label='目标函数')
        
        ax1.set_xlim(self.bounds[0], self.bounds[1])
        ax1.set_ylim(0, max_y)  # 自适应函数值范围
        ax1.set_xlabel('X')
        ax1.set_ylabel('函数值f(x)')
        ax1.set_title('搜索过程随时间变化')
        
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('最优适应度')
        ax2.set_title('收敛曲线')
        
        # 创建散点和线条对象
        current_scatter = ax1.scatter([], [], c='blue', s=80, alpha=0.7, label='当前解')
        best_scatter = ax1.scatter([], [], c='red', s=100, marker='*', label='当前最优解')
        global_best_scatter = ax1.scatter([], [], c='purple', s=120, marker='*', label='全局最优解')
        line, = ax2.plot([], [], 'g-', label='最优值')
        temp_line, = ax2.plot([], [], 'r--', alpha=0.5, label='温度')
        step_line, = ax2.plot([], [], 'b:', alpha=0.5, label='步长')
        
        # 添加图例
        ax1.legend(loc='upper right')
        ax2.legend(loc='upper right')
        
        # 查找重启点
        restart_indices = []
        for i in range(1, len(self.history['temperatures'])):
            if self.history['temperatures'][i] > self.history['temperatures'][i-1] * 5:
                restart_indices.append(i)
        
        def init():
            current_scatter.set_offsets(np.empty((0, 2)))
            best_scatter.set_offsets(np.empty((0, 2)))
            global_best_scatter.set_offsets(np.empty((0, 2)))
            line.set_data([], [])
            temp_line.set_data([], [])
            step_line.set_data([], [])
            return current_scatter, best_scatter, global_best_scatter, line, temp_line, step_line
        
        def update(frame):
            # 获取当前帧的解和能量
            current_solution = self.history['current_solutions'][frame]
            current_energy = self.history['current_energies'][frame]
            
            # 更新当前解位置
            current_scatter.set_offsets([[current_solution, current_energy]])
            
            # 获取当前帧的最优解
            best_energy = self.history['best_energies'][frame]
            # 从历史中找到对应的解（假设最优解总是由当前解产生）
            best_solutions = [
                self.history['current_solutions'][i] 
                for i in range(frame+1) 
                if abs(self.history['current_energies'][i] - best_energy) < 1e-6
            ]
            best_solution = best_solutions[-1] if best_solutions else self.history['current_solutions'][frame]
            
            best_scatter.set_offsets([[best_solution, best_energy]])
            
            # 全局最优
            global_best_value = max(self.history['best_energies'][:frame+1])
            global_best_idx = self.history['best_energies'][:frame+1].index(global_best_value)
            global_best_solution = best_solutions[-1] if best_solutions else self.history['current_solutions'][global_best_idx]
            
            global_best_scatter.set_offsets([[global_best_solution, global_best_value]])
            
            # 更新收敛曲线
            line.set_data(range(frame + 1), self.history['best_energies'][:frame + 1])
            
            # 更新温度曲线 - 归一化到与适应度相同的比例尺度
            temps = np.array(self.history['temperatures'][:frame + 1])
            # 避免除以零
            temp_range = temps.max() - temps.min() if temps.max() != temps.min() else 1.0
            
            # 获取当前的最优适应度范围
            best_energies = np.array(self.history['best_energies'][:frame + 1])
            energy_min = best_energies.min()
            energy_max = best_energies.max()
            energy_range = energy_max - energy_min if energy_max != energy_min else 1.0
            
            # 将温度归一化到适应度相同的比例尺度
            normalized_temps = energy_min + (temps - temps.min()) / temp_range * energy_range * 0.8
            
            temp_line.set_data(range(frame + 1), normalized_temps)
            
            # 更新步长曲线
            steps = np.array(self.history['step_sizes'][:frame + 1])
            step_range = steps.max() - steps.min() if steps.max() != steps.min() else 1.0
            normalized_steps = energy_min + (steps - steps.min()) / step_range * energy_range * 0.6
            
            step_line.set_data(range(frame + 1), normalized_steps)
            
            # 更新坐标轴范围
            ax2.relim()
            ax2.autoscale_view()
            
            # 标记重启点
            for idx in restart_indices:
                if idx == frame:
                    ax1.axvline(x=current_solution, color='orange', linestyle='--', alpha=0.7)
                    ax2.axvline(x=frame, color='orange', linestyle='--', alpha=0.7)
            
            # 更新标题
            current_temp = self.history['temperatures'][frame]
            accept_rate = self.history['accept_rates'][frame]
            step_size = self.history['step_sizes'][frame]
            
            # 判断是否是重启点
            restart_info = " [重启]" if frame in restart_indices else ""
            
            ax1.set_title(f'搜索过程 (迭代: {frame+1}/{len(self.history["current_solutions"])}{restart_info})')
            ax2.set_title(f'收敛曲线 (值: {best_energy:.6f}, 温度: {current_temp:.2f}, 步长: {step_size:.4f})')
            
            return current_scatter, best_scatter, global_best_scatter, line, temp_line, step_line
        
        ani = FuncAnimation(fig, update, frames=len(self.history['current_solutions']),
                           init_func=init, blit=False, repeat=True, interval=200)
        
        # 保存动画对象的引用，防止被垃圾回收
        self._animation = ani
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print("改进模拟退火算法(SA)演示程序 - 增强逃离局部最优版")
    print("="*60)
    
    # 定义搜索范围和参数
    bounds = [0, 50]
    initial_temp = 1000
    final_temp = 1e-8
    alpha = 0.98          # 降温系数（增大以减缓降温速度）
    max_iter = 100         # 最大迭代次数
    step_size = 4.0        # 初始步长较大以增强探索
    restart_temp = 200     # 提高重启温度
    max_stagnation = 20    # 最大停滞次数
    n_restarts = 6         # 增加重启次数
    step_adjust_rate = 0.9 # 步长调整更缓慢
    reannealing_threshold = 0.15 # 提高重新退火阈值
    
    print(f"参数设置: 初始温度={initial_temp}, 终止温度={final_temp}")
    print(f"          最大迭代次数={max_iter}, 降温系数={alpha}")
    print(f"          搜索范围={bounds}, 初始步长={step_size}")
    print(f"          重启温度={restart_temp}, 最大停滞次数={max_stagnation}")
    print(f"          最大重启次数={n_restarts}, 步长调整率={step_adjust_rate}")
    print(f"          重新退火阈值={reannealing_threshold}")
    print("-"*60)
    
    # 创建模拟退火优化器，并启用可视化
    sa = SimulatedAnnealing(
        objective_func=f,
        bounds=bounds,
        initial_temp=initial_temp,
        final_temp=final_temp,
        alpha=alpha,
        max_iter=max_iter,
        step_size=step_size,
        visualize=True,
        restart_temp=restart_temp,
        max_stagnation=max_stagnation,
        n_restarts=n_restarts,
        step_adjust_rate=step_adjust_rate,
        reannealing_threshold=reannealing_threshold
    )
    
    print("开始模拟退火优化...")
    # 记录开始时间
    start_time = time.time()
    
    # 运行优化
    best_x, best_val, history = sa.run()
    
    # 计算运行时间
    elapsed_time = time.time() - start_time
    
    print("\n优化完成！")
    print(f"运行时间: {elapsed_time:.2f}秒")
    print(f"模拟退火找到的最大值：{best_val:.6f}，位置 x = {best_x:.6f}")
    
    # 查看函数在该范围内的理论最优值（仅用于验证）
    x_check = np.linspace(bounds[0], bounds[1], 10000)
    y_check = [f(xi) for xi in x_check]
    max_idx = np.argmax(y_check)
    theoretical_best_x = x_check[max_idx]
    theoretical_best_y = y_check[max_idx]
    print(f"理论最优值参考: {theoretical_best_y:.6f}，位置 x = {theoretical_best_x:.6f}")
    print(f"与理论最优值差距: {(theoretical_best_y - best_val)/theoretical_best_y*100:.4f}%")
    print("-"*60)
    
    # 展示动画或静态图
    while True:
        choice = input("选择查看方式: 1=动画回放 2=静态收敛曲线 q=退出: ")
        if choice == '1':
            sa.animate_search()
        elif choice == '2':
            sa.plot_convergence()
        elif choice.lower() == 'q':
            break
        else:
            print("无效的选择，请重新输入")
    
    print("程序结束")