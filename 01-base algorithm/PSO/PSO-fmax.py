import numpy as np
import math
import random
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

class PSO:
    def __init__(self, num_particles, max_iter, bounds, visualize=False):
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.bounds = bounds
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 2
        self.c2 = 2
        self.w = self.w_max  # 当前惯性权重
        self.v_max = (bounds[1] - bounds[0]) * 0.1
        self.particles = []
        self.gbest_value = -float('inf')
        self.gbest_position = (bounds[0] + bounds[1]) / 2  # 初始化为中点
        self.visualize = visualize
        self.history_gbest = []  # 记录每次迭代的全局最优值
        self.history_positions = []  # 记录粒子位置历史
        
        # 初始化粒子
        self.initialize_particles()
        
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
        
        # 粒子位置子图
        self.ax1 = self.fig.add_subplot(121)
        self.ax1.set_xlim(self.bounds[0], self.bounds[1])
        self.ax1.set_ylim(0, max_y)  # 自适应函数值范围
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('函数值f(x)')
        self.ax1.set_title('粒子分布')
        
        # 适应度曲线子图
        self.ax2 = self.fig.add_subplot(122)
        self.ax2.set_xlabel('迭代次数')
        self.ax2.set_ylabel('全局最优值')
        self.ax2.set_title('收敛曲线')
        
        # 绘制函数曲线
        self.ax1.plot(x_vals, y_vals, 'b-', alpha=0.3)
        
        # 获取初始位置和函数值
        positions = [p['position'] for p in self.particles]
        values = [f(pos) for pos in positions]
        
        # 初始散点图和曲线
        self.scatter = self.ax1.scatter(positions, values, c='blue', alpha=0.7, label='粒子')
        self.best_scatter = self.ax1.scatter(self.gbest_position, f(self.gbest_position), 
                                          c='red', s=100, marker='*', label='全局最优')
        self.line, = self.ax2.plot([], [], 'g-')
        
        self.ax1.legend()
        plt.tight_layout()
        plt.ion()  # 打开交互模式
        plt.show(block=False)

    def initialize_particles(self):
        """初始化粒子群"""
        for _ in range(self.num_particles):
            position = np.random.uniform(self.bounds[0], self.bounds[1])
            velocity = np.random.uniform(-self.v_max, self.v_max)
            self.particles.append({
                'position': position,
                'velocity': velocity,
                'pbest_value': -float('inf'),
                'pbest_position': position
            })
        
        # 初始评估
        self.evaluate()
        
        # 存储初始状态
        if self.visualize:
            positions = [p['position'] for p in self.particles]
            self.history_positions.append(positions.copy())
            self.history_gbest.append(self.gbest_value)

    def evaluate(self):
        """评估粒子群中每个粒子的适应度"""
        for particle in self.particles:
            current_value = f(particle['position'])
            if current_value > particle['pbest_value']:
                particle['pbest_value'] = current_value
                particle['pbest_position'] = particle['position']
            if current_value > self.gbest_value:
                self.gbest_value = current_value
                self.gbest_position = particle['position']

    def update_particles(self):
        """更新粒子的速度和位置"""
        for particle in self.particles:
            v = particle['velocity']
            pbest_pos = particle['pbest_position']
            pos = particle['position']
            r1, r2 = random.random(), random.random()
            new_v = self.w * v + self.c1 * r1 * (pbest_pos - pos) + self.c2 * r2 * (self.gbest_position - pos)
            new_v = np.clip(new_v, -self.v_max, self.v_max)
            new_pos = pos + new_v
            new_pos = np.clip(new_pos, self.bounds[0], self.bounds[1])
            particle['velocity'] = new_v
            particle['position'] = new_pos
    
    def _update_plot(self, iter_num):
        """更新可视化图表"""
        if not self.visualize:
            return
        
        # 获取当前位置和函数值
        positions = [p['position'] for p in self.particles]
        values = [f(pos) for pos in positions]
            
        # 更新粒子位置
        self.scatter.set_offsets(np.column_stack((positions, values)))
        
        # 更新全局最优位置
        best_value = f(self.gbest_position)
        self.best_scatter.set_offsets([[self.gbest_position, best_value]])
        
        # 更新收敛曲线
        self.line.set_data(range(len(self.history_gbest)), self.history_gbest)
        self.ax2.relim()
        self.ax2.autoscale_view()
        
        # 动态调整收敛曲线的纵轴，确保所有点都可见
        if len(self.history_gbest) > 1:
            min_val = min(self.history_gbest)
            max_val = max(self.history_gbest)
            margin = (max_val - min_val) * 0.1  # 10%的边距
            if margin == 0:  # 处理所有值相同的情况
                margin = max_val * 0.1 if max_val != 0 else 1.0
            self.ax2.set_ylim(min_val - margin, max_val + margin)
        
        # 更新标题
        self.ax1.set_title(f'粒子分布 (迭代 {iter_num+1}/{self.max_iter})')
        self.ax2.set_title(f'收敛曲线 (当前最优值: {self.gbest_value:.6f})')
        
        # 刷新图形
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.05)  # 暂停一小段时间以便观察

    def run(self):
        """运行PSO算法"""
        # 可视化初始状态
        if self.visualize:
            self._update_plot(0)
        
        for iter in range(1, self.max_iter + 1):
            # 更新惯性权重
            self.w = self.w_max - (self.w_max - self.w_min) * iter / self.max_iter
            
            # 更新粒子位置
            self.update_particles()
            
            # 评估新位置
            self.evaluate()
            
            # 记录历史
            self.history_gbest.append(self.gbest_value)
            
            # 更新可视化
            if self.visualize:
                positions = [p['position'] for p in self.particles]
                self.history_positions.append(positions.copy())
                self._update_plot(iter)
        
        return self.gbest_position, self.gbest_value
    
    def plot_convergence(self):
        """绘制收敛曲线（静态图）"""
        if not self.history_gbest:
            print("没有可用的优化历史数据")
            return
            
        # 创建Tkinter窗口
        root = tk.Tk()
        root.title("PSO收敛曲线")
        root.geometry("800x600")
        
        # 创建Figure和Axes
        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(range(len(self.history_gbest)), self.history_gbest)
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('全局最优值')
        ax.set_title('PSO收敛曲线')
        
        # 创建Canvas并将图形嵌入Tkinter窗口
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # 启动Tkinter主循环
        root.mainloop()
    
    def animate_convergence(self):
        """创建粒子优化过程的动画"""
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
        ax1.set_title('粒子分布随时间变化')
        
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('全局最优值')
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
            # 更新粒子位置
            positions = self.history_positions[frame]
            values = [f(pos) for pos in positions]
            scatter.set_offsets(np.column_stack((positions, values)))
            
            # 得到当前帧对应的最佳位置
            if frame < len(self.history_gbest):
                # 使用当前帧的全局最优值
                best_pos = self.gbest_position  # 简化：使用最终的全局最优位置
                best_val = f(best_pos)
                
                best_scatter.set_offsets([[best_pos, best_val]])
                
                # 更新收敛曲线
                line.set_data(range(frame + 1), self.history_gbest[:frame + 1])
                ax2.relim()
                ax2.autoscale_view()
                
                # 动态调整收敛曲线的纵轴，确保所有点都可见
                if frame > 0:
                    current_data = self.history_gbest[:frame + 1]
                    min_val = min(current_data)
                    max_val = max(current_data)
                    margin = (max_val - min_val) * 0.1  # 10%的边距
                    if margin == 0:  # 处理所有值相同的情况
                        margin = max_val * 0.1 if max_val != 0 else 1.0
                    ax2.set_ylim(min_val - margin, max_val + margin)
                
                # 更新标题
                ax2.set_title(f'收敛曲线 (当前最优值: {self.history_gbest[frame]:.6f})')
            
            ax1.set_title(f'粒子分布 (迭代 {frame+1}/{len(self.history_positions)})')
            
            return scatter, best_scatter, line
        
        ani = FuncAnimation(fig, update, frames=len(self.history_positions),
                           init_func=init, blit=False, repeat=True, interval=200)
        
        # 保存动画对象的引用，防止被垃圾回收
        self._animation = ani
        
        plt.tight_layout()
        plt.show()

# 运行PSO
if __name__ == "__main__":
    print("粒子群优化算法(PSO)演示程序")
    print("="*50)
    
    # 参数设置
    num_particles = 50      # 粒子数量
    max_iterations = 100    # 最大迭代次数
    bounds = [0, 50]        # 搜索空间范围
    
    print(f"参数设置: 粒子数量={num_particles}, 最大迭代次数={max_iterations}")
    print(f"          搜索范围={bounds}, 惯性权重={0.9}→{0.4}")
    print("-"*50)
    
    # 创建PSO优化器，并启用可视化
    pso = PSO(
        num_particles=num_particles, 
        max_iter=max_iterations, 
        bounds=bounds,
        visualize=True
    )
    
    print("开始PSO优化...")
    # 记录开始时间
    start_time = time.time()
    
    # 运行优化
    pso_x, pso_val = pso.run()
    
    # 计算运行时间
    elapsed_time = time.time() - start_time
    
    print("\n优化完成！")
    print(f"运行时间: {elapsed_time:.2f}秒")
    print(f"PSO最大值：{pso_val}，x={pso_x}")
    print("-"*50)
    
    # 展示动画或静态图
    while True:
        choice = input("选择查看方式: 1=动画回放 2=静态收敛曲线 q=退出: ")
        if choice == '1':
            pso.animate_convergence()
        elif choice == '2':
            pso.plot_convergence()
        elif choice.lower() == 'q':
            break
        else:
            print("无效的选择，请重新输入")
    
    print("程序结束")