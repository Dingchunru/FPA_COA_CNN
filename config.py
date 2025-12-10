# config.py - GPU优化版本（修复版）
import torch

class Config:
    # ========== GPU优化配置 ==========
    # 设备配置 - 强制使用GPU
    USE_GPU = True  # 强制使用GPU
    FORCE_CPU = False  # 强制使用CPU（调试时使用）
    
    if FORCE_CPU:
        DEVICE = torch.device('cpu')
        print("强制使用CPU模式")
    elif USE_GPU and torch.cuda.is_available():
        DEVICE = torch.device('cuda:0')
        # 设置CUDA优化标志
        torch.backends.cudnn.benchmark = True  # 加速卷积运算
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = False  # 为了速度牺牲一点确定性
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎯 使用GPU加速: {gpu_name}")
        print(f"  内存: {gpu_memory:.2f} GB")
        print(f"  CUDA版本: {torch.version.cuda}")
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
    else:
        DEVICE = torch.device('cpu')
        if USE_GPU:
            print("⚠ GPU不可用，自动回退到CPU模式")
        else:
            print("ℹ 使用CPU模式")
    
    # ========== 数据配置 ==========
    DATA_PATH = "data/CICIDS2017"
    SAMPLE_FRACTION = 0.3  # 数据采样比例
    TEST_SIZE = 0.3
    VAL_SIZE = 0.2
    
    # ========== CNN模型配置（GPU优化） ==========
    CNN_CONFIG = {
        'hidden_channels': [64, 128, 256],  # 卷积层通道数
        'kernel_sizes': [3, 3, 3],  # 卷积核大小
        'pool_sizes': [2, 2, 2],  # 池化大小
        'fc_sizes': [256, 128],  # 增加全连接层大小（GPU有足够内存）
        'dropout_rate': 0.3,
        'batch_norm': True,
        'activation': 'relu'  # 使用ReLU激活函数
    }
    
    # ========== 训练配置（GPU优化） ==========
    TRAIN_CONFIG = {
        'batch_size': 512 if DEVICE.type == 'cuda' else 64,  # GPU用大batch
        'learning_rate': 0.001,
        'epochs': 100,
        'patience': 10,
        'weight_decay': 1e-4,
        'gradient_accumulation_steps': 1,  # 梯度累积步数
        'mixed_precision': True if DEVICE.type == 'cuda' else False,  # GPU用混合精度
        'pin_memory': True if DEVICE.type == 'cuda' else False,  # GPU固定内存
        'num_workers': 4 if DEVICE.type == 'cuda' else 2,  # GPU用更多workers
        'persistent_workers': True,  # 保持工作进程活跃
        'prefetch_factor': 2,  # 预取因子
        'scheduler_type': 'plateau',  # 学习率调度器类型
        'warmup_epochs': 3,  # 学习率预热轮数
        'max_grad_norm': 1.0,  # 梯度裁剪
    }
    
    # ========== FPA-COA混合优化配置 ==========
    OPTIMIZER_CONFIG = {
        'pop_size': 30,
        'iter_max': 20,
        'fpa_params': {
            'p': 0.8,      # 转换概率
            'lambda_': 1.5, # Lévy飞行参数
            'alpha': 0.1,   # 步长缩放因子
            # 删除 'use_gpu': True  # FPA不支持此参数
        },
        'coa_params': {
            'pa': 0.25,    # 宿主发现概率
            'alpha': 0.01, # 步长因子
            'beta': 1.5,   # 幂律分布参数
            # 删除 'use_gpu': True  # COA不支持此参数
        },
        'hybrid_params': {
            'elite_rate': 0.1,     # 精英保留比例
            'migration_rate': 0.2, # 种群迁移比例
            'adaptive_weight': True, # 自适应权重
            'collaboration_frequency': 5,
            'gpu_accelerated': True if DEVICE.type == 'cuda' else False,  # GPU加速标志
        }
    }
    
    # ========== 特征选择配置（GPU优化） ==========
    FEATURE_SELECTION = {
        'threshold': 0.5,
        'min_features': 10,
        'max_features_ratio': 0.8,
        'fast_eval': True,  # 快速评估模式
        'eval_batch_size': 256 if DEVICE.type == 'cuda' else 64,  # GPU用大batch
        'eval_epochs': 3,  # 评估时训练轮数
        'use_gpu_for_eval': True if DEVICE.type == 'cuda' else False,  # 评估时使用GPU
    }
    
    # ========== GPU内存管理配置 ==========
    GPU_CONFIG = {
        'empty_cache_frequency': 50,  # 每50个batch清理一次缓存
        'memory_monitor': True,  # 监控GPU内存使用
        'max_memory_usage': 0.8,  # 最大GPU内存使用率（80%）
    }
    
    # ========== 日志和保存配置 ==========
    LOG_CONFIG = {
        'log_interval': 10,  # 日志间隔
        'save_checkpoints': True,  # 保存检查点
        'checkpoint_frequency': 5,  # 检查点保存频率（epoch）
        'experiment_name': 'fpa_coa_cnn_gpu',  # 实验名称
    }
    
    # ========== 性能优化配置 ==========
    PERFORMANCE_CONFIG = {
        'use_amp': True if DEVICE.type == 'cuda' else False,  # GPU用自动混合精度
        'channels_last': True if DEVICE.type == 'cuda' else False,  # GPU用channels_last格式
    }
    
    def __init__(self):
        """初始化配置"""
        self._setup_directories()
        self._print_config_summary()
        self.optimize_for_gpu()
    
    def _setup_directories(self):
        """创建必要的目录"""
        import os
        
        # 基础目录
        os.makedirs("model", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # GPU专用目录
        if self.DEVICE.type == 'cuda':
            os.makedirs("model/gpu_models", exist_ok=True)
            os.makedirs("results/gpu_results", exist_ok=True)
            os.makedirs("logs/gpu_logs", exist_ok=True)
    
    def _print_config_summary(self):
        """打印配置摘要"""
        print("\n" + "="*60)
        print("配置摘要")
        print("="*60)
        print(f"设备: {self.DEVICE}")
        
        if self.DEVICE.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"批量大小: {self.TRAIN_CONFIG['batch_size']}")
            print(f"混合精度: {self.TRAIN_CONFIG['mixed_precision']}")
            print(f"数据加载工作进程: {self.TRAIN_CONFIG['num_workers']}")
            print(f"特征选择评估batch: {self.FEATURE_SELECTION['eval_batch_size']}")
        else:
            print(f"批量大小: {self.TRAIN_CONFIG['batch_size']}")
            print(f"特征选择评估batch: {self.FEATURE_SELECTION['eval_batch_size']}")
        
        print(f"CNN层数: {len(self.CNN_CONFIG['hidden_channels'])}")
        print(f"FPA-COA种群大小: {self.OPTIMIZER_CONFIG['pop_size']}")
        print(f"特征选择最小特征数: {self.FEATURE_SELECTION['min_features']}")
        print(f"数据采样比例: {self.SAMPLE_FRACTION}")
        print("="*60 + "\n")
    
    def get_gpu_memory_info(self):
        """获取GPU内存信息"""
        if self.DEVICE.type != 'cuda':
            return None
        
        try:
            info = {
                'total': torch.cuda.get_device_properties(0).total_memory / 1024**3,
                'allocated': torch.cuda.memory_allocated() / 1024**3,
                'reserved': torch.cuda.memory_reserved() / 1024**3,
                'free': (torch.cuda.get_device_properties(0).total_memory - 
                        torch.cuda.memory_allocated()) / 1024**3
            }
            return info
        except:
            return None
    
    def optimize_for_gpu(self):
        """应用GPU优化"""
        if self.DEVICE.type != 'cuda':
            return
        
        try:
            # 设置Tensor核心优化（如果可用）
            if hasattr(torch.cuda, 'set_float32_matmul_precision'):
                torch.cuda.set_float32_matmul_precision('high')
            
            # 启用TF32（如果GPU支持）
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            # 清理GPU缓存
            torch.cuda.empty_cache()
            
            print("✅ GPU优化已应用")
        except Exception as e:
            print(f"⚠ GPU优化应用失败: {e}")
    
    def print_current_gpu_status(self):
        """打印当前GPU状态"""
        if self.DEVICE.type == 'cuda':
            memory_info = self.get_gpu_memory_info()
            if memory_info:
                print(f"\n📊 当前GPU状态:")
                print(f"  已分配内存: {memory_info['allocated']:.2f} GB")
                print(f"  已缓存内存: {memory_info['reserved']:.2f} GB")
                print(f"  可用内存: {memory_info['free']:.2f} GB")
                print(f"  总内存: {memory_info['total']:.2f} GB")
    
    def get_training_params(self):
        """获取训练参数"""
        return {
            'device': str(self.DEVICE),
            'batch_size': self.TRAIN_CONFIG['batch_size'],
            'learning_rate': self.TRAIN_CONFIG['learning_rate'],
            'epochs': self.TRAIN_CONFIG['epochs'],
            'mixed_precision': self.TRAIN_CONFIG['mixed_precision'],
            'num_workers': self.TRAIN_CONFIG['num_workers'],
            'use_gpu': self.DEVICE.type == 'cuda',
            'gpu_name': torch.cuda.get_device_name(0) if self.DEVICE.type == 'cuda' else 'CPU'
        }

# 创建全局配置实例
config = Config()

# 测试代码
if __name__ == "__main__":
    print("配置测试完成!")
    
    # 显示配置信息
    params = config.get_training_params()
    print(f"\n训练参数:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # 显示GPU内存信息
    config.print_current_gpu_status()