# algorithms/coa.py
import numpy as np
import time
from scipy.special import gamma

class CuckooOptimizationAlgorithm:
    """
    正确的杜鹃优化算法 (Cuckoo Optimization Algorithm)
    基于杜鹃鸟的巢寄生行为：
    1. 每只杜鹃鸟在宿主巢中产卵（生成新解）
    2. 宿主有一定概率发现并抛弃外来蛋
    3. 使用Lévy飞行进行长距离搜索
    """
    
    def __init__(self, objective_func, dim, pop_size=50, pa=0.25, 
                 alpha=0.01, beta=1.5, iter_max=100):
        """
        参数：
            objective_func: 目标函数
            dim: 问题维度
            pop_size: 种群大小
            pa: 宿主发现概率 (0-1)
            alpha: 步长缩放因子
            beta: Lévy飞行参数
            iter_max: 最大迭代次数
        """
        self.objective_func = objective_func
        self.dim = dim
        self.pop_size = pop_size
        self.pa = pa
        self.alpha = alpha
        self.beta = beta
        self.iter_max = iter_max
        
        self.population = None
        self.fitness = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.bounds = None
        self.history = []
        
    def levy_flight(self, size):
        """生成Lévy飞行步长"""
        sigma = (gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2) / 
                 (gamma((1 + self.beta) / 2) * self.beta * 
                  2**((self.beta - 1) / 2)))**(1/self.beta)
        
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / (np.abs(v)**(1/self.beta))
        
        return self.alpha * step * np.random.randn(*size)
    
    def initialize(self, bounds):
        """初始化种群"""
        self.bounds = bounds
        lower, upper = bounds
        
        # 随机初始化种群
        self.population = np.random.uniform(lower, upper, (self.pop_size, self.dim))
        self.fitness = np.zeros(self.pop_size)
        
        # 评估初始种群
        for i in range(self.pop_size):
            self.fitness[i] = self.objective_func(self.population[i])
            
        # 找到最优解
        best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]
        
        return self.best_solution, self.best_fitness
    
    def get_cuckoo(self):
        """生成新的杜鹃解（通过Lévy飞行）"""
        # 随机选择一只杜鹃
        i = np.random.randint(0, self.pop_size)
        
        # 使用Lévy飞行生成新解
        step_size = self.levy_flight(self.dim)
        new_solution = self.population[i] + step_size
        
        # 边界处理
        new_solution = np.clip(new_solution, self.bounds[0], self.bounds[1])
        
        # 评估新解
        new_fitness = self.objective_func(new_solution)
        
        return new_solution, new_fitness, i
    
    def abandon_nests(self, fitness):
        """宿主发现并抛弃劣质解"""
        # 按适应度排序
        sorted_indices = np.argsort(fitness)
        num_abandon = int(self.pa * self.pop_size)
        
        # 抛弃最差的解
        worst_indices = sorted_indices[-num_abandon:]
        
        # 重新初始化被抛弃的解
        for idx in worst_indices:
            self.population[idx] = np.random.uniform(
                self.bounds[0], self.bounds[1], self.dim
            )
            self.fitness[idx] = self.objective_func(self.population[idx])
    
    def run_iteration(self, iteration):
        """执行单次迭代"""
        iteration_start = time.time()
        
        # 保存当前最优解
        current_best = self.best_fitness
        
        # 生成新的杜鹃解并替换
        for _ in range(self.pop_size):
            new_solution, new_fitness, i = self.get_cuckoo()
            
            # 如果新解更好，替换旧解
            if new_fitness < self.fitness[i]:
                self.population[i] = new_solution
                self.fitness[i] = new_fitness
                
                # 更新全局最优
                if new_fitness < self.best_fitness:
                    self.best_solution = new_solution.copy()
                    self.best_fitness = new_fitness
        
        # 宿主发现并抛弃劣质解
        self.abandon_nests(self.fitness)
        
        # 记录迭代信息
        iteration_time = time.time() - iteration_start
        improvement = current_best - self.best_fitness
        
        self.history.append({
            'iteration': iteration,
            'best_fitness': self.best_fitness,
            'time': iteration_time,
            'improvement': improvement,
            'population_diversity': np.std(self.population),
            'discovery_rate': self.pa
        })
        
        return self.best_solution, self.best_fitness
    
    def run(self, bounds):
        """运行完整的优化过程"""
        self.initialize(bounds)
        
        print("\n" + "="*60)
        print("杜鹃优化算法 (COA) 开始执行")
        print(f"种群大小: {self.pop_size}")
        print(f"宿主发现概率: {self.pa}")
        print(f"最大迭代次数: {self.iter_max}")
        print("="*60)
        
        for iter_num in range(self.iter_max):
            self.run_iteration(iter_num)
            
            # 打印进度
            progress = (iter_num + 1) / self.iter_max * 100
            bar_length = 40
            filled = int(bar_length * progress / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            print(f"\rCOA 迭代 {iter_num+1:3d}/{self.iter_max} |{bar}| "
                  f"{progress:.1f}% | 最佳适应度: {self.best_fitness:.6f}", end="")
        
        print("\n\n" + "="*60)
        print("杜鹃优化算法完成!")
        print(f"最终最佳适应度: {self.best_fitness:.6f}")
        print("="*60)
        
        return self.best_solution, self.best_fitness