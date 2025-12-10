# algorithms/fpa.py
import numpy as np
import time
from scipy.special import gamma

class FlowerPollinationAlgorithm:
    """
    改进的花授粉算法 (Flower Pollination Algorithm)
    包含自适应参数调整和精英策略
    """
    
    def __init__(self, objective_func, dim, pop_size=50, p=0.8,
                 lambda_=1.5, alpha=0.1, iter_max=100):
        self.objective_func = objective_func
        self.dim = dim
        self.pop_size = pop_size
        self.p = p
        self.lambda_ = lambda_
        self.alpha = alpha
        self.iter_max = iter_max
        
        self.population = None
        self.fitness = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.bounds = None
        self.history = []
        
    def levy_flight(self, size):
        """改进的Lévy飞行生成"""
        if self.lambda_ <= 0:
            self.lambda_ = 1.5
            
        try:
            sigma = (gamma(1 + self.lambda_) * np.sin(np.pi * self.lambda_ / 2) /
                    (gamma((1 + self.lambda_) / 2) * self.lambda_ *
                     2**((self.lambda_ - 1) / 2)))**(1/self.lambda_)
        except:
            sigma = 1.0
            
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        
        # 避免除以零
        v_abs = np.abs(v)
        v_abs[v_abs < 1e-10] = 1e-10
        
        step = u / (v_abs**(1/self.lambda_))
        return self.alpha * step * np.random.randn(size)
    
    def initialize(self, bounds):
        """初始化种群"""
        self.bounds = bounds
        lower, upper = bounds
        
        self.population = np.random.uniform(lower, upper, (self.pop_size, self.dim))
        self.fitness = np.zeros(self.pop_size)
        
        # 评估种群
        for i in range(self.pop_size):
            self.fitness[i] = self.objective_func(self.population[i])
        
        # 找到最优
        best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]
        
        return self.best_solution, self.best_fitness
    
    def adaptive_p(self, iteration):
        """自适应调整转换概率"""
        base_p = self.p
        # 随着迭代增加，增加局部搜索概率
        adaptive_p = base_p * (1 - 0.5 * iteration / self.iter_max)
        return max(0.3, min(0.95, adaptive_p))
    
    def run_iteration(self, iteration):
        """执行单次迭代"""
        iteration_start = time.time()
        current_best = self.best_fitness
        
        # 自适应参数
        current_p = self.adaptive_p(iteration)
        
        for i in range(self.pop_size):
            if np.random.rand() < current_p:
                # 全局授粉 (Lévy飞行)
                L = self.levy_flight(self.dim)
                scale = np.random.rand()
                new_solution = self.population[i] + scale * L * (
                    self.best_solution - self.population[i])
            else:
                # 局部授粉
                j, k = np.random.choice(self.pop_size, 2, replace=False)
                epsilon = np.random.rand()
                new_solution = self.population[i] + epsilon * (
                    self.population[j] - self.population[k])
            
            # 边界处理
            new_solution = np.clip(new_solution, self.bounds[0], self.bounds[1])
            new_fitness = self.objective_func(new_solution)
            
            # 更新个体
            if new_fitness < self.fitness[i]:
                self.population[i] = new_solution
                self.fitness[i] = new_fitness
                
                # 更新全局最优
                if new_fitness < self.best_fitness:
                    self.best_solution = new_solution.copy()
                    self.best_fitness = new_fitness
        
        # 记录历史
        iteration_time = time.time() - iteration_start
        improvement = current_best - self.best_fitness
        
        self.history.append({
            'iteration': iteration,
            'best_fitness': self.best_fitness,
            'time': iteration_time,
            'improvement': improvement,
            'p_value': current_p,
            'population_mean': np.mean(self.fitness),
            'population_std': np.std(self.fitness)
        })
        
        return self.best_solution, self.best_fitness
    
    def run(self, bounds):
        """运行优化"""
        self.initialize(bounds)
        
        print("\n" + "="*60)
        print("花授粉算法 (FPA) 开始执行")
        print(f"种群大小: {self.pop_size}")
        print(f"初始转换概率: {self.p}")
        print(f"最大迭代次数: {self.iter_max}")
        print("="*60)
        
        for iter_num in range(self.iter_max):
            self.run_iteration(iter_num)
            
            # 打印进度
            progress = (iter_num + 1) / self.iter_max * 100
            bar_length = 40
            filled = int(bar_length * progress / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            print(f"\rFPA 迭代 {iter_num+1:3d}/{self.iter_max} |{bar}| "
                  f"{progress:.1f}% | 最佳适应度: {self.best_fitness:.6f}", end="")
        
        print("\n\n" + "="*60)
        print("花授粉算法完成!")
        print(f"最终最佳适应度: {self.best_fitness:.6f}")
        print("="*60)
        
        return self.best_solution, self.best_fitness