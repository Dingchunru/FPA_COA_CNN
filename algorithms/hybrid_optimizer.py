# algorithms/hybrid_optimizer.py - 修复版
import numpy as np
import time
from typing import Callable, Dict, Tuple
from .fpa import FlowerPollinationAlgorithm
from .coa import CuckooOptimizationAlgorithm

class HybridFPA_COA_Optimizer:
    """
    改进的FPA-COA混合优化器
    实现真正的协同优化策略
    """
    
    def __init__(self, objective_func: Callable, dim: int,
                 pop_size: int = 50, iter_max: int = 100,
                 fpa_params: Dict = None, coa_params: Dict = None,
                 hybrid_params: Dict = None):
        
        self.objective_func = objective_func
        self.dim = dim
        self.pop_size = pop_size
        self.iter_max = iter_max
        
        # 设置默认参数
        self.fpa_params = fpa_params or {'p': 0.8, 'lambda_': 1.5, 'alpha': 0.1}
        self.coa_params = coa_params or {'pa': 0.25, 'alpha': 0.01, 'beta': 1.5}
        self.hybrid_params = hybrid_params or {
            'elite_rate': 0.1,
            'migration_rate': 0.2,
            'adaptive_weight': True,
            'collaboration_frequency': 5
        }
        
        # 过滤掉算法不支持的参数
        self._filter_unsupported_params()
        
        # 初始化算法
        self.fpa = FlowerPollinationAlgorithm(
            objective_func, dim, pop_size,
            **self.fpa_params, iter_max=iter_max
        )
        
        self.coa = CuckooOptimizationAlgorithm(
            objective_func, dim, pop_size,
            **self.coa_params, iter_max=iter_max
        )
        
        # 混合优化状态
        self.best_solution = None
        self.best_fitness = float('inf')
        self.history = []
        self.collaboration_history = []
    
    def _filter_unsupported_params(self):
        """过滤掉算法不支持的参数"""
        
        # FlowerPollinationAlgorithm支持的参数
        fpa_supported_params = {
            'p', 'lambda_', 'alpha'
        }
        
        # CuckooOptimizationAlgorithm支持的参数
        coa_supported_params = {
            'pa', 'alpha', 'beta'
        }
        
        # 过滤fpa_params
        filtered_fpa_params = {}
        for key, value in self.fpa_params.items():
            if key in fpa_supported_params:
                filtered_fpa_params[key] = value
            else:
                print(f"警告: FPA不支持参数 '{key}'，已忽略")
        self.fpa_params = filtered_fpa_params
        
        # 过滤coa_params
        filtered_coa_params = {}
        for key, value in self.coa_params.items():
            if key in coa_supported_params:
                filtered_coa_params[key] = value
            else:
                print(f"警告: COA不支持参数 '{key}'，已忽略")
        self.coa_params = filtered_coa_params
        
        # 打印过滤后的参数
        if len(filtered_fpa_params) < len(self.fpa_params) or len(filtered_coa_params) < len(self.coa_params):
            print(f"FPA参数: {filtered_fpa_params}")
            print(f"COA参数: {filtered_coa_params}")
    
    def dynamic_weight_adjustment(self, iteration: int, 
                                 fpa_performance: float, 
                                 coa_performance: float) -> Tuple[float, float]:
        """动态调整算法权重"""
        if not self.hybrid_params.get('adaptive_weight', True):
            return 0.5, 0.5
        
        # 根据性能调整权重
        total_perf = fpa_performance + coa_performance
        if total_perf == 0:
            return 0.5, 0.5
            
        fpa_weight = fpa_performance / total_perf
        coa_weight = coa_performance / total_perf
        
        # 添加探索偏向（早期偏向FPA，后期偏向COA）
        exploration_bias = 0.7 - 0.4 * (iteration / self.iter_max)
        fpa_weight = exploration_bias * fpa_weight
        coa_weight = (1 - exploration_bias) * coa_weight
        
        # 归一化
        total = fpa_weight + coa_weight
        if total > 0:
            fpa_weight /= total
            coa_weight /= total
            
        return fpa_weight, coa_weight
    
    def population_migration(self, fpa_pop: np.ndarray, fpa_fit: np.ndarray,
                           coa_pop: np.ndarray, coa_fit: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """种群迁移策略"""
        migration_rate = self.hybrid_params.get('migration_rate', 0.2)
        num_migrants = int(self.pop_size * migration_rate)
        
        if num_migrants == 0:
            return fpa_pop, coa_pop
        
        # 选择要迁移的个体
        fpa_migrants_idx = np.random.choice(self.pop_size, num_migrants, replace=False)
        coa_migrants_idx = np.random.choice(self.pop_size, num_migrants, replace=False)
        
        # 交换种群
        fpa_pop[fpa_migrants_idx], coa_pop[coa_migrants_idx] = \
            coa_pop[coa_migrants_idx].copy(), fpa_pop[fpa_migrants_idx].copy()
        
        return fpa_pop, coa_pop
    
    def elite_selection(self, fpa_pop: np.ndarray, fpa_fit: np.ndarray,
                       coa_pop: np.ndarray, coa_fit: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """精英选择策略"""
        elite_rate = self.hybrid_params.get('elite_rate', 0.1)
        num_elites = int(self.pop_size * elite_rate)
        
        if num_elites == 0:
            return fpa_pop, coa_pop
        
        # 选择精英个体
        all_pop = np.vstack([fpa_pop, coa_pop])
        all_fit = np.concatenate([fpa_fit, coa_fit])
        
        elite_indices = np.argsort(all_fit)[:num_elites]
        elites = all_pop[elite_indices]
        
        # 用精英替换最差个体
        fpa_worst = np.argsort(fpa_fit)[-num_elites:]
        coa_worst = np.argsort(coa_fit)[-num_elites:]
        
        fpa_pop[fpa_worst] = elites[:num_elites]
        coa_pop[coa_worst] = elites[:num_elites]
        
        return fpa_pop, coa_pop
    
    def collaborative_optimization(self, iteration: int):
        """协同优化步骤"""
        collaboration_frequency = self.hybrid_params.get('collaboration_frequency', 5)
        if collaboration_frequency == 0 or iteration % collaboration_frequency != 0:
            return
        
        # 种群迁移
        self.fpa.population, self.coa.population = self.population_migration(
            self.fpa.population, self.fpa.fitness,
            self.coa.population, self.coa.fitness
        )
        
        # 精英选择
        self.fpa.population, self.coa.population = self.elite_selection(
            self.fpa.population, self.fpa.fitness,
            self.coa.population, self.coa.fitness
        )
        
        # 重新评估适应度
        for i in range(self.pop_size):
            self.fpa.fitness[i] = self.objective_func(self.fpa.population[i])
            self.coa.fitness[i] = self.objective_func(self.coa.population[i])
        
        # 更新各自的最优解
        fpa_best_idx = np.argmin(self.fpa.fitness)
        coa_best_idx = np.argmin(self.coa.fitness)
        
        self.fpa.best_solution = self.fpa.population[fpa_best_idx].copy()
        self.fpa.best_fitness = self.fpa.fitness[fpa_best_idx]
        self.coa.best_solution = self.coa.population[coa_best_idx].copy()
        self.coa.best_fitness = self.coa.fitness[coa_best_idx]
        
        # 记录协作信息
        self.collaboration_history.append({
            'iteration': iteration,
            'migration_count': int(self.pop_size * self.hybrid_params.get('migration_rate', 0.2)),
            'elite_count': int(self.pop_size * self.hybrid_params.get('elite_rate', 0.1))
        })
    
    def run(self, bounds: Tuple[float, float]) -> Tuple[np.ndarray, float]:
        """运行混合优化"""
        print("\n" + "="*60)
        print("FPA-COA混合优化器开始执行")
        print(f"问题维度: {self.dim}")
        print(f"种群大小: {self.pop_size}")
        print(f"最大迭代次数: {self.iter_max}")
        print("="*60 + "\n")
        
        # 初始化两个算法
        print("初始化FPA和COA种群...")
        self.fpa.initialize(bounds)
        self.coa.initialize(bounds)
        
        # 初始最优解
        self.best_solution = self.fpa.best_solution.copy()
        self.best_fitness = self.fpa.best_fitness
        
        if self.coa.best_fitness < self.best_fitness:
            self.best_solution = self.coa.best_solution.copy()
            self.best_fitness = self.coa.best_fitness
        
        print(f"初始最佳适应度: {self.best_fitness:.6f}")
        
        total_start_time = time.time()
        
        for iteration in range(self.iter_max):
            iter_start_time = time.time()
            
            # 动态权重调整
            fpa_weight, coa_weight = self.dynamic_weight_adjustment(
                iteration, 
                1.0 / (self.fpa.best_fitness + 1e-10),
                1.0 / (self.coa.best_fitness + 1e-10)
            )
            
            # 并行运行两个算法
            fpa_sol, fpa_fit = self.fpa.run_iteration(iteration)
            coa_sol, coa_fit = self.coa.run_iteration(iteration)
            
            # 协同优化
            self.collaborative_optimization(iteration)
            
            # 更新混合最优解
            improved_by = "None"
            if fpa_fit < self.best_fitness:
                self.best_solution = fpa_sol.copy()
                self.best_fitness = fpa_fit
                improved_by = "FPA"
                
            if coa_fit < self.best_fitness:
                self.best_solution = coa_sol.copy()
                self.best_fitness = coa_fit
                improved_by = "COA"
            
            # 记录历史
            iter_time = time.time() - iter_start_time
            self.history.append({
                'iteration': iteration,
                'best_fitness': self.best_fitness,
                'fpa_fitness': fpa_fit,
                'coa_fitness': coa_fit,
                'fpa_weight': fpa_weight,
                'coa_weight': coa_weight,
                'time': iter_time,
                'improved_by': improved_by,
                'population_diversity': np.std(np.vstack([
                    self.fpa.population, 
                    self.coa.population
                ]))
            })
            
            # 打印进度
            progress = (iteration + 1) / self.iter_max * 100
            bar_length = 50
            filled = int(bar_length * progress / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            info = (f"\r混合迭代 {iteration+1:3d}/{self.iter_max} |{bar}| {progress:.1f}% | "
                   f"最佳: {self.best_fitness:.6f} | FPA: {fpa_fit:.6f} | COA: {coa_fit:.6f}")
            print(info, end="")
        
        total_time = time.time() - total_start_time
        
        # 打印最终报告
        print("\n\n" + "="*60)
        print("FPA-COA混合优化完成!")
        print("="*60)
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"平均迭代时间: {total_time/self.iter_max:.2f} 秒")
        print(f"最终最佳适应度: {self.best_fitness:.6f}")
        
        # 统计信息
        if self.history:
            improvements = [h['improved_by'] for h in self.history]
            fpa_improvements = improvements.count('FPA')
            coa_improvements = improvements.count('COA')
            
            print(f"\n算法贡献统计:")
            print(f"  FPA改进次数: {fpa_improvements} ({fpa_improvements/len(improvements):.1%})")
            print(f"  COA改进次数: {coa_improvements} ({coa_improvements/len(improvements):.1%})")
            print(f"  协作次数: {len(self.collaboration_history)}")
        
        print("="*60)
        return self.best_solution, self.best_fitness